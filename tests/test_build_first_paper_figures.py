from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.build_first_paper_figures import (
    FIGURES,
    FIG2_REQUIRED_SERIES,
    FIG5_LEAKAGE_METRICS,
    PrerequisiteError,
    SABER_139_COLUMN,
    TUDELFT_MISSIONS,
    coverage_record,
    density_columns,
    nearest_saber_altitude,
    require,
    validate_summary,
)


def test_coverage_record_does_not_parse_model_version_as_altitude(tmp_path: Path) -> None:
    import polars as pl

    path = tmp_path / "models.parquet"
    pl.DataFrame(
        {
            "date": ["2020-01-01", "2020-01-02"],
            "log10rho_250": [5.0, 6.0],
            "log10rho_175_daily_mean": [3.0, 4.0],
            "nrlmsise_00_log10rho_daily_mean_125km": [1.0, 2.0],
            "nrlmsis_2p1_log10rho_daily_range_825km": [0.1, 0.2],
        }
    ).with_columns(pl.col("date").str.to_date()).write_parquet(path)

    record, _ = coverage_record("models", path)

    assert record["altitude_km"] == [125.0, 825.0]


def test_require_reports_missing_prerequisite(tmp_path: Path) -> None:
    with pytest.raises(PrerequisiteError, match="Missing daily artifact"):
        require(tmp_path / "missing.csv", "daily artifact")


def test_density_columns_only_exposes_contracted_diagnostic() -> None:
    import polars as pl

    frame = pl.DataFrame(
        {
            "log10rho_175_daily_mean": [1.0],
            "log10rho_825_daily_range": [2.0],
            "log10rho_300_daily_other": [3.0],
        }
    )
    assert density_columns(frame, "mean") == {175: "log10rho_175_daily_mean"}
    assert density_columns(frame, "range") == {825: "log10rho_825_daily_range"}


def test_summary_schema_requires_exact_inventory_and_accessible_text() -> None:
    figures = {
        name: {
            "files": [f"{name}.png", f"{name}.pdf"],
            "sources": [],
            "caption": "caption",
            "alt_text": "alt",
        }
        for name in FIGURES
    }
    summary = {"schema_version": 1, "figures": figures}
    validate_summary(summary)
    del figures["fig6"]
    with pytest.raises(ValueError, match="exactly"):
        validate_summary(summary)


def test_summary_is_machine_readable_with_per_figure_contract_fields(tmp_path: Path) -> None:
    path = tmp_path / "summary.json"
    path.write_text(json.dumps({"schema_version": 1, "figures": {}}))
    assert json.loads(path.read_text())["schema_version"] == 1


def test_figure_contract_inventories_lock_canonical_products_and_missions() -> None:
    assert set(FIG2_REQUIRED_SERIES) == {
        "Global mean", "HASDM", "NRLMSISE-00", "NRLMSIS 2.0", "NRLMSIS 2.1", "JB2006", "JB2008",
    }
    assert TUDELFT_MISSIONS == (
        "CHAMP", "GOCE", "GRACE-A", "GRACE-B", "GRACE-FO", "Swarm-A", "Swarm-B", "Swarm-C",
    )


def test_saber_selection_uses_nearest_available_139_km_channel() -> None:
    assert nearest_saber_altitude([100.0, 119.0, 140.0]) == 140.0
    assert SABER_139_COLUMN == "SABER_CO2_COOLING_139KM"


def test_figure5_locks_centered_f107_and_ap_not_raw_f107() -> None:
    assert FIG5_LEAKAGE_METRICS == ("forcing_f107_obs_center81", "forcing_ap_avg")
