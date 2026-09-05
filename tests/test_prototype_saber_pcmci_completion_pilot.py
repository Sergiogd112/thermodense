import json

import numpy as np
import pytest

import scripts.prototype_saber_pcmci_completion_pilot as pilot


def write_sources(tmp_path, exact=True):
    stream = tmp_path / "stream"
    stream.mkdir()
    time_ns = np.array([1, 2, 3], dtype=np.int64)
    np.savez_compressed(
        stream / "final_arrays.npz",
        timestamps=time_ns.astype("datetime64[ns]"),
        channels=np.array(pilot.CHANNELS),
        values=np.ones((5, 3)),
        counts=np.ones((5, 3)),
        masks=np.ones((5, 3), dtype=bool),
        joint_all_five_mask=np.ones(3, dtype=bool),
    )
    (stream / "final_report.json").write_text(
        json.dumps(
            {
                "exact_full_cell": exact,
                "output_status": "exact_complete" if exact else "partial_not_for_pcmci",
            }
        )
    )
    bundle = tmp_path / "bundle.npz"
    np.savez_compressed(
        bundle,
        hasdm_time_ns=time_ns,
        hasdm_targets=np.ones((3, 27)),
        hasdm_f107=np.ones(3),
        hasdm_ap=np.ones(3),
        hasdm_kp=np.ones(3),
    )
    return stream, bundle


def test_load_sources_refuses_non_exact_final(tmp_path, monkeypatch):
    stream, bundle = write_sources(tmp_path, exact=False)
    monkeypatch.setattr(pilot, "STREAM", stream)
    monkeypatch.setattr(pilot, "BUNDLE", bundle)
    with pytest.raises(ValueError, match="exact_full_cell"):
        pilot.load_sources()


def test_load_sources_preserves_five_channel_order_and_exact_times(
    tmp_path, monkeypatch
):
    stream, bundle = write_sources(tmp_path)
    monkeypatch.setattr(pilot, "STREAM", stream)
    monkeypatch.setattr(pilot, "BUNDLE", bundle)
    source, note = pilot.load_sources()
    assert source["saber"].shape == (3, 5)
    assert source["targets"].shape == (3, 3)
    assert note["partial_input"] is False
    assert pilot.policy()["max_conds_dim"] is None
