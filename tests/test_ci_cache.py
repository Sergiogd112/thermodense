"""Coverage for the durable SQLite CI-test cache (within-cell resumption).

The cache accelerates repeated Tigramite conditional-independence tests across
isolated process restarts, so the tests here verify the two contracts that
matter: exact result parity with the uncached path, and key sensitivity to
data, configuration, and the caller-supplied identity salt.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from thermodense.benchmarks import ci_cache
from thermodense.benchmarks import gpdctorch_gates
from thermodense.benchmarks import pcmci_real
from tigramite import data_processing as pp
from tigramite.independence_tests.independence_tests_base import CondIndTest
from tigramite.independence_tests.parcorr import ParCorr

IDENTITY = {"stage": "primary", "seed": 20260802}


def _data(seed: int = 0, rows: int = 80) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(rows, 4))


def _parcorr(data: np.ndarray) -> ParCorr:
    test = ParCorr(significance="analytic")
    test.set_dataframe(pp.DataFrame(data))
    return test


def _run_test(test: CondIndTest, **kwargs: Any) -> tuple[float, float]:
    return test.run_test(X=[(0, -1)], Y=[(1, 0)], tau_max=1, **kwargs)


@pytest.fixture(autouse=True)
def _restore_patch() -> None:
    yield
    ci_cache.disable_ci_cache()


def test_enable_disable_roundtrip_records_identity(tmp_path: Path) -> None:
    db = tmp_path / "cache.sqlite3"
    original = CondIndTest.run_test
    ci_cache.enable_ci_cache(db, IDENTITY)
    assert CondIndTest.run_test is ci_cache._wrapped_run_test
    assert ci_cache.stats()["enabled"] is True
    assert ci_cache.stats()["db_path"] == str(db)
    assert ci_cache.stats()["identity"] == IDENTITY
    connection = sqlite3.connect(db)
    row = connection.execute("SELECT v FROM cache_meta WHERE k = 'identity'").fetchone()
    connection.close()
    assert json.loads(row[0]) == IDENTITY
    ci_cache.disable_ci_cache()
    assert CondIndTest.run_test is original
    assert ci_cache.stats()["enabled"] is False


def test_repeated_in_process_test_never_touches_the_durable_cache(
    tmp_path: Path,
) -> None:
    db = tmp_path / "cache.sqlite3"
    ci_cache.enable_ci_cache(db, IDENTITY)
    test = _parcorr(_data())
    first = _run_test(test)
    second = _run_test(test)
    assert first == second
    assert ci_cache.stats()["misses"] == 1
    assert ci_cache.stats()["hits"] == 0  # in-process repeats hit the memory cache
    _run_test(test)
    assert ci_cache.stats()["hits"] == 0 and ci_cache.stats()["misses"] == 1


def test_results_replay_across_enable_cycles(tmp_path: Path) -> None:
    db = tmp_path / "cache.sqlite3"
    data = _data()
    ci_cache.enable_ci_cache(db, IDENTITY)
    test = _parcorr(data)
    expected_a = _run_test(test)
    other = ParCorr(significance="analytic")
    other.set_dataframe(pp.DataFrame(data))
    expected_b = other.run_test(X=[(1, -1)], Y=[(2, 0)], tau_max=1)
    ci_cache.disable_ci_cache()
    ci_cache.enable_ci_cache(db, IDENTITY)
    fresh = _parcorr(data)
    assert _run_test(fresh) == expected_a
    assert fresh.run_test(X=[(1, -1)], Y=[(2, 0)], tau_max=1) == expected_b
    assert ci_cache.stats()["hits"] == 2
    assert ci_cache.stats()["misses"] == 0


def test_config_change_invalidates_the_key(tmp_path: Path) -> None:
    db = tmp_path / "cache.sqlite3"
    ci_cache.enable_ci_cache(db, IDENTITY)
    _run_test(_parcorr(_data()))
    assert ci_cache.stats()["misses"] == 1
    changed = _parcorr(_data())
    changed.sig_samples = 999  # result-affecting setting fingerprint
    _run_test(changed)
    assert ci_cache.stats()["misses"] == 2
    assert ci_cache.stats()["hits"] == 0


def test_data_change_invalidates_the_key(tmp_path: Path) -> None:
    db = tmp_path / "cache.sqlite3"
    ci_cache.enable_ci_cache(db, IDENTITY)
    _run_test(_parcorr(_data(seed=0)))
    _run_test(_parcorr(_data(seed=1)))
    assert ci_cache.stats()["misses"] == 2
    assert ci_cache.stats()["hits"] == 0


def test_identity_change_invalidates_all_keys(tmp_path: Path) -> None:
    db = tmp_path / "cache.sqlite3"
    ci_cache.enable_ci_cache(db, {"stage": "primary", "seed": 1})
    _run_test(_parcorr(_data()))
    ci_cache.disable_ci_cache()
    ci_cache.enable_ci_cache(db, {"stage": "primary", "seed": 2})
    _run_test(_parcorr(_data()))
    assert ci_cache.stats()["misses"] == 1
    assert ci_cache.stats()["hits"] == 0


def test_duplicate_enable_raises(tmp_path: Path) -> None:
    ci_cache.enable_ci_cache(tmp_path / "cache.sqlite3", IDENTITY)
    with pytest.raises(RuntimeError, match="already enabled"):
        ci_cache.enable_ci_cache(tmp_path / "other.sqlite3", IDENTITY)


def test_original_error_parity_is_preserved(tmp_path: Path) -> None:
    db = tmp_path / "cache.sqlite3"
    ci_cache.enable_ci_cache(db, IDENTITY)
    fixed = ParCorr(significance="fixed_thres")
    fixed.set_dataframe(pp.DataFrame(_data()))
    with pytest.raises(ValueError, match="fixed_thres"):
        _run_test(fixed)

    nan_triggered = ParCorr(significance="analytic")
    nan_triggered.set_dataframe(pp.DataFrame(_data()))
    array = np.ones((20, 3))
    array[0, 0] = np.nan
    nan_triggered._get_array = lambda *a, **k: (  # type: ignore[method-assign]
        array,
        np.array([0, 1, 2]),
        ([0], [1], []),
        "continuous",
        None,
        None,
        ([0], [1], []),
        "continuous",
    )
    with pytest.raises(ValueError, match="nans in the array"):
        _run_test(nan_triggered)

    ci_cache.disable_ci_cache()
    with pytest.raises(ValueError, match="nans in the array"):
        _run_test(nan_triggered)
    assert ci_cache.stats()["hits"] == 0 and ci_cache.stats()["misses"] == 0


def test_corrupt_database_degrades_to_passthrough(tmp_path: Path) -> None:
    db = tmp_path / "corrupt.sqlite3"
    db.write_bytes(b"this is not a sqlite database")
    ci_cache.enable_ci_cache(db, IDENTITY)
    assert ci_cache.stats()["enabled"] is True
    assert ci_cache.stats()["degraded"] is True
    expected = _run_test(_parcorr(_data()))
    assert isinstance(expected, tuple) and len(expected) == 2
    assert ci_cache.stats()["hits"] == 0 and ci_cache.stats()["misses"] == 0


def test_config_digest_ignores_volatile_state(tmp_path: Path) -> None:
    a = _parcorr(_data())
    b = _parcorr(_data())
    assert ci_cache._config_digest(a) == ci_cache._config_digest(b)
    _run_test(a)
    assert ci_cache._config_digest(a) == ci_cache._config_digest(b)
    a.sig_samples = 7
    assert ci_cache._config_digest(a) != ci_cache._config_digest(b)


def test_cache_key_binds_data_hash_and_identity(tmp_path: Path) -> None:
    ci_cache.enable_ci_cache(tmp_path / "cache.sqlite3", {"stage": "primary"})
    test = _parcorr(_data())
    hash_a = test._get_array_hash(
        np.arange(16, dtype=float).reshape(4, 4), np.array([0, 1, 2, 2]),
        ([0], [1], [2, 3]),
    )
    hash_b = test._get_array_hash(
        np.arange(16, dtype=float).reshape(4, 4), np.array([0, 1, 2, 2]),
        ([3], [1], [2, 0]),
    )
    key_a = ci_cache._cache_key(hash_a, test)
    assert key_a != ci_cache._cache_key(hash_b, test)
    assert key_a == ci_cache._cache_key(hash_a, test)
    ci_cache.disable_ci_cache()
    ci_cache.enable_ci_cache(tmp_path / "cache.sqlite3", {"stage": "interaction"})
    assert ci_cache._cache_key(hash_a, test) != key_a


def test_child_main_enables_cache_for_gated_gpdctorch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = tmp_path / "input.csv"
    input_path.write_bytes(b"date,f107,x1,x2,x3,x4\n")
    state_path = tmp_path / "state" / "gpdctorch-gates.json"
    output_path = tmp_path / "results.jsonl"
    artifact_path = tmp_path / "artifacts" / "matrix.npz"
    args = pcmci_real.parser().parse_args(
        [
            "case",
            "--input", str(input_path),
            "--method", "gpdctorch",
            "--tau-max", "10",
            "--artifact", str(artifact_path),
            "--timing-variant", "raw_observed_daily",
            "--preprocessing-profile", "detrended_anomaly",
            "--gate-state", str(state_path),
            "--gate-threads", "1",
            "--gate-output", str(output_path),
        ]
    )
    stage = next(
        item for item in gpdctorch_gates.stages() if item.name == "primary"
    )
    context = pcmci_real._GatedChildContext(stage)
    calls: list[tuple[Path, dict[str, Any]]] = []
    monkeypatch.setattr(pcmci_real, "_validate_gated_child", lambda a: context)
    monkeypatch.setattr(pcmci_real, "load_input", lambda *a, **k: None)
    monkeypatch.setattr(pcmci_real, "_run_pcmciplus_gated", lambda *a, **k: {"status": "succeeded"})
    monkeypatch.setattr(
        pcmci_real.ci_cache,
        "enable_ci_cache",
        lambda db, identity=None: calls.append((db, identity)),
    )

    rc = pcmci_real._child_main(args)

    assert rc == 0
    assert len(calls) == 1
    db_path, identity = calls[0]
    assert db_path == state_path.parent / "ci-cache-primary.sqlite3"
    assert identity["stage"] == "primary"
    assert identity["method"] == "gpdctorch"
    assert identity["tau_max"] == 10
    assert identity["timing_variant"] == "raw_observed_daily"
    assert identity["preprocessing_profile"] == "detrended_anomaly"
    assert identity["seed"] == pcmci_real._method_seed("gpdctorch")
    assert identity["input_sha256"] == hashlib.sha256(
        input_path.read_bytes()
    ).hexdigest()
