"""Tests for WACCM-X density-trend inputs (synthetic extraction directories)."""

import json

import numpy as np
import pytest

from scripts import waccmx_density_trends as wdt


def write_years(directory, years, base=19970101):
    heights = [400]
    for year in years:
        days = np.arange(np.datetime64(f"{year}-01-01"), np.datetime64(f"{year + 1}-01-01"))
        days = np.array([d for d in days if d not in wdt.DOCUMENTED_MISSING])
        n = len(days)
        drivers = np.zeros((n, 3))
        drivers[:, 0] = 4e-4  # co2vmr
        drivers[:, 1] = 120.0  # f107a
        np.savez(directory / f"{year}.npz", dates=days.astype(str), base_date=np.full(n, base),
                 drivers=drivers, fixed_heights=np.full((n, 1, 1, 2), 1e-12))
        (directory / f"{year}.json").write_text(json.dumps(
            {"heights_km_geopotential": heights, "height_vars": ["DEN"], "drivers": ["co2vmr", "f107a", "ap"]}))


def test_incomplete_extraction_is_rejected(tmp_path):
    write_years(tmp_path, [2006, 2024])  # 2007-2021 and 2023 absent
    with pytest.raises(ValueError, match="incomplete"):
        wdt.specs(tmp_path)


def test_complete_window_with_only_documented_gaps_is_accepted(tmp_path, monkeypatch):
    monkeypatch.setattr(wdt, "RUN_WINDOWS", {"WACCM-X run 3 (2006-2024)": (19970101, "2021-01-01", "2023-12-31")})
    write_years(tmp_path, [2021, 2023])
    specs = wdt.specs(tmp_path)
    assert specs and all(coverage == 1.0 for _, coverage in specs)
