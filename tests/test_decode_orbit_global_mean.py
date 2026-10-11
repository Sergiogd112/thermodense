"""Tests for the orbit-derived global-mean table converter."""

from datetime import date

import pytest

from scripts.decode_orbit_global_mean import convert

TEXT = """ year day log10rho_250  sigma_250 log10rho_400  sigma_400
 1967   1     -10.2317     0.0616     -11.5940     0.0578
 1968  60     -10.2035     0.0618     -999.0000     0.0579
"""


def test_convert_maps_day_of_year_and_keeps_values():
    df = convert(TEXT)
    assert df.columns == ["date", "log10rho_250", "sigma_250", "log10rho_400", "sigma_400"]
    assert df["date"].to_list() == [date(1967, 1, 1), date(1968, 2, 29)]
    assert df["log10rho_400"].to_list() == [-11.594, -999.0]


def test_convert_rejects_ragged_rows_and_duplicate_dates():
    with pytest.raises(ValueError, match="field count"):
        convert(TEXT + " 1969 1 -10.0\n")
    with pytest.raises(ValueError, match="duplicate"):
        convert(TEXT + " 1967   1     -10.2     0.06     -11.5     0.05\n")
