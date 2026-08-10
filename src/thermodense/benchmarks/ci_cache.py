"""Durable SQLite-backed cache for Tigramite conditional-independence tests.

The expensive part of a production GPDCtorch gate stage is the repeated
PCMCI+ conditional-independence tests.  Tigramite already caches ``(val,
pval)`` per test *in memory* keyed by a hash of the data array
(``CondIndTest.cached_ci_results``), but that cache dies with the isolated
child process.  This module makes the same cache durable so an interrupted
stage can be resumed: on restart the algorithm re-runs, every previously
completed test is replayed from SQLite, and only the unfinished tests are
actually computed.

Correctness notes
-----------------
The cache key binds three things:

* ``data`` -- the order-independent sha1 fingerprint Tigramite computes for
  the (X, Y, Z) data slices (``_get_array_hash``).  Different data,
  different lags, different ``cut_off`` or ``tau_max`` all change the key.
* ``config`` -- a canonical fingerprint of the CI test configuration
  (measure, significance, ``sig_samples``, ``null_dist_filename``, ...).
  Different test settings can never reuse each other's results.
* ``identity`` -- a salt supplied by the caller (stage, seed, input digest,
  ...).  Changing any identity component invalidates the whole cache.

The p-value of GPDCtorch's analytic significance test derives from a null
distribution that is generated once per sample size using the test's seeded
``random_state``.  Because the generator is seeded and the null distribution
is generated at most once per run, cached ``(val, pval)`` pairs stay valid
across restarts without replaying the random stream.  Tests that consume
``random_state`` per test (e.g. shuffle significance) are cached the same
way; their p-values are position-dependent, so such caches are only reused
when the caller keeps the full run configuration identical (the identity
salt is designed exactly for that).

Failure behavior
----------------
The cache is an accelerator, never a correctness gate.  If the SQLite
database is corrupt or unreadable the module degrades to a no-op (warning
once on stderr) and the run continues exactly as if the cache had not been
enabled.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

# Keys that carry per-instance mutable state rather than configuration.  They
# must never appear in the config fingerprint.
_VOLATILE_KEYS = frozenset(
    {
        "cached_ci_results",  # in-memory (val, pval) cache, grows during a run
        "ci_results",  # full CI result records, grows during a run
        "dataframe",  # the data object itself; the data is hashed separately
        "random_state",  # stateful RNG stream; the seed lives in the identity
        "verbosity",  # output verbosity never changes results
        "residuals",  # recycled-residual cache, grows during a run
    }
)

_SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS ci_cache (
    cache_key TEXT PRIMARY KEY,
    result_json TEXT NOT NULL,
    created_utc TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS cache_meta (
    k TEXT PRIMARY KEY,
    v TEXT NOT NULL
);
"""

_ORIGINAL_RUN_TEST: Any = None
_CONN: sqlite3.Connection | None = None
_DB_PATH: Path | None = None
_IDENTITY_SALT: str | None = None
_DEGRADED = False
_WARNED = False
_STATS = {"hits": 0, "misses": 0}
_LOCK = threading.Lock()


def _tigramite_version() -> str:
    """Return the installed Tigramite version or ``"unknown"``."""
    try:
        from importlib.metadata import version

        return version("tigramite")
    except Exception:
        return "unknown"


def _leaf_config(obj: Any, seen: set[int], depth: int = 0) -> dict[str, Any]:
    """Extract a canonical, cycle-safe, JSON-serializable config fingerprint.

    Walks ``__dict__`` namespaces (at most ``depth`` levels) and keeps only
    scalar values: strings, numbers, booleans, ``None`` and containers of
    scalars.  Nested config objects such as ``GPDCtorch.gauss_pr`` are
    traversed so that result-affecting settings like ``null_dist_filename``
    are captured.  Objects that are not configuration (data, RNG streams,
    caches) are skipped via ``_VOLATILE_KEYS`` or by not being scalar.
    """
    result: dict[str, Any] = {}
    if depth > 4 or obj is None or id(obj) in seen:
        return result
    namespace = getattr(obj, "__dict__", None)
    if not isinstance(namespace, dict):
        return result
    seen.add(id(obj))
    for key, value in namespace.items():
        if key in _VOLATILE_KEYS:
            continue
        if value is None or isinstance(value, (str, bool, int)):
            result[key] = value
        elif isinstance(value, (float, np.generic)):
            result[key] = value.item() if isinstance(value, np.generic) else value
        elif isinstance(value, dict):
            inner = {
                str(inner_key): (
                    inner_value.item()
                    if isinstance(inner_value, np.generic)
                    else inner_value
                )
                for inner_key, inner_value in value.items()
                if isinstance(inner_value, (str, bool, int, float, np.generic))
                or inner_value is None
            }
            if inner:
                result[key] = inner
        elif isinstance(value, (list, tuple)):
            inner = [
                item.item() if isinstance(item, np.generic) else item
                for item in value
                if isinstance(item, (str, bool, int, float, np.generic)) or item is None
            ]
            if inner:
                result[key] = inner
        else:
            sub = _leaf_config(value, seen, depth + 1)
            if sub:
                result[key] = sub
    seen.discard(id(obj))
    return result


def _config_digest(test: Any) -> str:
    """Return a stable fingerprint of every result-affecting test setting."""
    leaves = _leaf_config(test, set())
    canonical = json.dumps(leaves, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _json_scalar(value: Any) -> Any:
    """Convert a numpy scalar to a plain JSON-serializable Python value."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"not a JSON-serializable scalar: {type(value).__name__}")


def _cache_key(combined_hash: tuple[str, str, str], test: Any) -> str:
    """Return the durable SQLite key for one CI test."""
    payload = {
        "data": list(combined_hash),
        "config": _config_digest(test),
        "identity": hashlib.sha256(
            (_IDENTITY_SALT or "").encode("utf-8")
        ).hexdigest(),
        "tigramite": _tigramite_version(),
    }
    canonical = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _connect(db_path: Path) -> sqlite3.Connection | None:
    """Open (or create) the SQLite database; return None on structural failure."""
    try:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(db_path), timeout=60, check_same_thread=False)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute("PRAGMA busy_timeout=60000")
        connection.executescript(_SQLITE_SCHEMA)
        connection.commit()
        return connection
    except (sqlite3.Error, OSError):
        return None


def _degrade(reason: str) -> None:
    global _DEGRADED, _WARNED
    _DEGRADED = True
    if not _WARNED:
        _WARNED = True
        print(f"ci-cache degraded, continuing without caching: {reason}", file=sys.stderr)


def _lookup(key: str) -> tuple[Any, Any] | None:
    if _CONN is None:
        return None
    try:
        row = _CONN.execute(
            "SELECT result_json FROM ci_cache WHERE cache_key = ?", (key,)
        ).fetchone()
        if row is None:
            return None
        entry = json.loads(row[0])
        return (_json_scalar(entry["val"]), _json_scalar(entry["pval"]))
    except (sqlite3.Error, KeyError, json.JSONDecodeError, TypeError) as error:
        _degrade(str(error))
        return None


def _store(key: str, val: Any, pval: Any) -> None:
    if _CONN is None:
        return
    try:
        result_json = json.dumps(
            {"val": _json_scalar(val), "pval": _json_scalar(pval)}, sort_keys=True
        )
        _CONN.execute(
            "INSERT OR REPLACE INTO ci_cache (cache_key, result_json, created_utc)"
            " VALUES (?, ?, ?)",
            (key, result_json, datetime.now(timezone.utc).isoformat()),
        )
        _CONN.commit()
    except (sqlite3.Error, TypeError, ValueError) as error:
        _degrade(str(error))


def _store_meta(key: str, value: str) -> None:
    if _CONN is None:
        return
    try:
        _CONN.execute(
            "INSERT OR REPLACE INTO cache_meta (k, v) VALUES (?, ?)", (key, value)
        )
        _CONN.commit()
    except sqlite3.Error:
        _degrade("metadata write failed")


def _wrapped_run_test(
    self: Any,
    X: Any,
    Y: Any,
    Z: Any = None,
    tau_max: int = 0,
    cut_off: str = "2xtau_max",
    alpha_or_thres: float | None = None,
) -> Any:
    """Replacement for ``CondIndTest.run_test`` with a durable cache.

    Replays the validation prefix of the original method exactly so a failed
    cache lookup never changes observable errors, then either serves the
    cached ``(val, pval)`` (injecting it into the in-memory cache so the
    original method reproduces its decision logic) or computes and persists.
    """
    # Mirror of the original run_test prefix (validation, array construction,
    # data hashing).  Errors raised here are identical to the uncached path.
    if self.significance == "fixed_thres" and alpha_or_thres is None:
        raise ValueError(
            "significance == 'fixed_thres' requires setting alpha_or_thres"
        )
    (array, xyz, XYZ, _data_type, _nonzero_array, _nonzero_xyz, _nonzero_XYZ,
     _nonzero_data_type) = self._get_array(
        X=X, Y=Y, Z=Z, tau_max=tau_max, cut_off=cut_off,
        remove_constant_data=True, verbosity=self.verbosity,
    )
    cleaned_X, cleaned_Y, cleaned_Z = XYZ
    if np.any(np.isnan(array)):
        raise ValueError("nans in the array!")
    combined_hash = self._get_array_hash(array, xyz, XYZ)

    # Only the first occurrence of a test in this process consults the
    # durable cache.  Repeated in-process tests hit the in-memory cache and
    # must not rewind any RNG state recorded for the first occurrence.
    first_in_process = combined_hash not in self.cached_ci_results
    key: str | None = None
    if first_in_process and not _DEGRADED:
        key = _cache_key(combined_hash, self)
        cached = _lookup(key)
        if cached is not None:
            val, pval = cached
            self.cached_ci_results[combined_hash] = (val, pval)
            with _LOCK:
                _STATS["hits"] += 1
            return _ORIGINAL_RUN_TEST(
                self, X, Y, Z=Z, tau_max=tau_max, cut_off=cut_off,
                alpha_or_thres=alpha_or_thres,
            )

    result = _ORIGINAL_RUN_TEST(
        self, X, Y, Z=Z, tau_max=tau_max, cut_off=cut_off,
        alpha_or_thres=alpha_or_thres,
    )
    if first_in_process and not _DEGRADED and key is not None:
        entry = self.cached_ci_results.get(combined_hash)
        if entry is not None:
            _store(key, entry[0], entry[1])
            with _LOCK:
                _STATS["misses"] += 1
    return result


def enable_ci_cache(db_path: Path, identity: dict[str, Any] | None = None) -> None:
    """Enable the durable CI-test cache for this process.

    Patches ``CondIndTest.run_test`` process-wide and opens the SQLite
    database at ``db_path``.  ``identity`` is a JSON-serializable dict that
    becomes the key salt: any change (stage, seed, input digest, ...)
    invalidates the cache.  Raises if already enabled in this process; a
    corrupt or unreadable database degrades to a no-op instead of failing.
    """
    global _CONN, _DB_PATH, _IDENTITY_SALT, _DEGRADED, _WARNED, _STATS
    global _ORIGINAL_RUN_TEST
    with _LOCK:
        if _ORIGINAL_RUN_TEST is not None:
            raise RuntimeError("ci cache is already enabled in this process")
        if not isinstance(db_path, Path):
            raise TypeError("db_path must be a pathlib.Path")
        identity = identity or {}
        _IDENTITY_SALT = json.dumps(identity, sort_keys=True, default=str)
        _STATS = {"hits": 0, "misses": 0}
        _WARNED = False
        _CONN = _connect(db_path)
        _DEGRADED = _CONN is None
        if _DEGRADED:
            _degrade(f"cannot open cache database {db_path}")
        else:
            _DB_PATH = db_path
            _store_meta("identity", _IDENTITY_SALT)
            _store_meta("tigramite_version", _tigramite_version())
            _store_meta("enabled_utc", datetime.now(timezone.utc).isoformat())
        from tigramite.independence_tests.independence_tests_base import (
            CondIndTest,
        )

        _ORIGINAL_RUN_TEST = CondIndTest.run_test
        CondIndTest.run_test = _wrapped_run_test


def disable_ci_cache() -> None:
    """Restore the original ``run_test`` and close the database (testing)."""
    global _CONN, _DB_PATH, _IDENTITY_SALT, _DEGRADED, _WARNED, _STATS
    global _ORIGINAL_RUN_TEST
    with _LOCK:
        if _ORIGINAL_RUN_TEST is not None:
            from tigramite.independence_tests.independence_tests_base import (
                CondIndTest,
            )

            CondIndTest.run_test = _ORIGINAL_RUN_TEST
            _ORIGINAL_RUN_TEST = None
        if _CONN is not None:
            try:
                _CONN.close()
            except sqlite3.Error:
                pass
        _CONN = None
        _DB_PATH = None
        _IDENTITY_SALT = None
        _DEGRADED = False
        _WARNED = False
        _STATS = {"hits": 0, "misses": 0}


def stats() -> dict[str, Any]:
    """Return a snapshot of the current cache state for diagnostics."""
    with _LOCK:
        return {
            "enabled": _ORIGINAL_RUN_TEST is not None,
            "db_path": str(_DB_PATH) if _DB_PATH is not None else None,
            "hits": _STATS["hits"],
            "misses": _STATS["misses"],
            "degraded": _DEGRADED,
            "identity": (
                json.loads(_IDENTITY_SALT) if _IDENTITY_SALT is not None else None
            ),
        }
