"""Tests for checkpoint compatibility and custom-calendar validation."""

import numpy as np
import pytest

import scripts.prototype_saber_vertical_totals_40x60_stream as stream


def test_legacy_checkpoint_without_path_keys_matches_default_paths():
    current = {"schema": 1, **stream.LEGACY_PATH_DEFAULTS}
    assert stream.with_legacy_path_defaults({"schema": 1}, current) == current


def test_legacy_checkpoint_does_not_match_changed_paths():
    current = {"schema": 1, "samples_path": "other.parquet",
               "baseline_path": stream.LEGACY_PATH_DEFAULTS["baseline_path"]}
    assert stream.with_legacy_path_defaults({"schema": 1}, current) != current


def test_calendar_accepts_unique_three_hour_grid():
    times = np.array(["2014-01-01T00", "2014-01-01T03", "2014-01-01T06"], dtype="datetime64[ns]")
    np.testing.assert_array_equal(stream.validate_calendar_ns(times.astype(np.int64)), times)


@pytest.mark.parametrize(
    "times, message",
    [
        (["2014-01-01T00", "2014-01-01T00"], "duplicate"),
        (["2014-01-01T00", "2014-01-01T04"], "three-hour"),
        (["2014-01-01T00", "NaT"], "missing"),
    ],
)
def test_calendar_rejects_inputs_that_would_drop_scans(times, message):
    with pytest.raises(ValueError, match=message):
        stream.validate_calendar_ns(np.array(times, dtype="datetime64[ns]").astype(np.int64))
