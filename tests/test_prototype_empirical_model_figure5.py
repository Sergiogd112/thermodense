"""Focused contracts for the Figure 5 PCMCI+ prototype helpers."""

import numpy as np
import pandas as pd

from scripts.prototype_empirical_model_figure5 import (
    AVAILABILITY,
    FORCING_RUNS,
    PCMCI_PC_ALPHA,
    SUPERVISOR_CAPTIONS,
    build_link_assumptions,
    complete_daily_calendar,
    fdr_bh_qvalues,
    forcing_metric,
    pcmci_fingerprint,
    prepare_saber_window,
    retain_fdr_links,
    retained_link_table,
    seasonal_detrended_standardized,
    strongest_links,
)


def test_link_assumptions_forbid_peer_error_and_target_to_driver_links() -> None:
    columns = ["epsilon_a", "epsilon_b", "F10.7_OBS", "AP_AVG"]
    assumptions = build_link_assumptions(columns, columns[:2], tau_max=2)

    assert (0, -1) in assumptions[0]
    assert (2, -1) in assumptions[0]
    assert (0, -1) not in assumptions[1]
    assert (0, -1) not in assumptions[2]
    assert (3, -2) in assumptions[2]


def test_six_forcing_display_mapping_uses_declared_distinct_runs() -> None:
    assert set(FORCING_RUNS) == {
        "F10.7_OBS",
        "F10.7_OBS_CENTER81",
        "AP_AVG",
        "KP_SUM",
        "CO2_ppm",
        "SABER_CO2_COOLING_139KM",
    }
    assert FORCING_RUNS["F10.7_OBS_CENTER81"] == "centered_f107_robustness"
    assert FORCING_RUNS["KP_SUM"] == "kp_geomagnetic_variant"
    assert FORCING_RUNS["SABER_CO2_COOLING_139KM"] == "saber_extension"
    assert len({forcing_metric(source) for source in FORCING_RUNS}) == 6
    assert set(SUPERVISOR_CAPTIONS) == {"A", "C"}


def test_saber_window_preserves_daily_calendar_and_gapped_missing_values() -> None:
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    data = pd.DataFrame({"date": dates, "base": range(10)})
    saber = pd.DataFrame(
        {
            "date": [dates[2], dates[4], dates[7]],
            "SABER_CO2_COOLING_139KM": [1.0, 2.0, 3.0],
        }
    )
    window = prepare_saber_window(data, saber)

    assert list(window.date) == list(pd.date_range(dates[2], dates[7], freq="D"))
    assert window.SABER_CO2_COOLING_139KM.isna().sum() == 3


def test_complete_daily_calendar_exposes_missing_target_without_losing_dates() -> None:
    dates = pd.to_datetime(["2020-01-01", "2020-01-03"])
    completed = complete_daily_calendar(
        pd.DataFrame({"date": dates, "target": [1.0, 3.0]}),
        pd.Timestamp("2020-01-01"),
        pd.Timestamp("2020-01-03"),
    )

    assert list(completed.date) == list(pd.date_range("2020-01-01", periods=3))
    assert np.isnan(completed.loc[1, "target"])


def test_fdr_summary_excludes_graph_links_above_q_threshold() -> None:
    columns = ["epsilon_a", "F10.7_OBS"]
    graph = np.full((2, 2, 2), "", dtype="<U3")
    values = np.zeros((2, 2, 2))
    raw_pvalues = np.full((2, 2, 2), 0.9)
    qvalues = np.full((2, 2, 2), 0.9)
    graph[1, 0, 1], values[1, 0, 1] = "-->", 0.4
    raw_pvalues[1, 0, 1] = 0.001
    qvalues[1, 0, 1] = PCMCI_PC_ALPHA + 0.01

    retained = retained_link_table(
        retain_fdr_links(graph, qvalues),
        values,
        raw_pvalues,
        qvalues,
        columns,
        ["epsilon_a"],
        "primary",
        175,
    )
    summary = strongest_links(retained, ["F10.7_OBS"], ["epsilon_a"], "primary", 175)
    assert not summary.iloc[0].detected
    assert summary.iloc[0].display_value == 0.0


def test_fdr_includes_lag_zero_and_link_table_rejects_reverse_orientation() -> None:
    class FakePcmci:
        def get_corrected_pvalues(self, pvalues, **kwargs):
            assert kwargs["exclude_contemporaneous"] is False
            return pvalues

    pvalues = np.ones((2, 2, 2))
    assert fdr_bh_qvalues(FakePcmci(), pvalues, 1, {}) is pvalues

    graph = np.full((2, 2, 2), "", dtype="<U3")
    graph[1, 0, 0] = "<--"
    graph[1, 0, 1] = "<--"
    retained = retained_link_table(
        graph,
        np.zeros_like(pvalues),
        pvalues,
        pvalues,
        ["epsilon_a", "F10.7_OBS"],
        ["epsilon_a"],
        "primary",
        175,
    )
    assert retained.empty


def test_preprocessing_and_cache_identity_preserve_missingness_and_inputs() -> None:
    dates = pd.Series(pd.date_range("2000-01-01", periods=12, freq="D"))
    values = np.arange(12, dtype=float)
    values[4] = np.nan
    assert np.isnan(seasonal_detrended_standardized(values, dates)[4])

    frame = pd.DataFrame({"date": dates, "SABER_CO2_COOLING_139KM": values})
    first = pcmci_fingerprint({"saber_extension": frame}, np.array([175]), 2)
    changed_altitude = pcmci_fingerprint({"saber_extension": frame}, np.array([200]), 2)
    changed_saber = frame.copy()
    changed_saber.loc[0, "SABER_CO2_COOLING_139KM"] = 99.0
    changed_values = pcmci_fingerprint(
        {"saber_extension": changed_saber}, np.array([175]), 2
    )
    assert first != changed_altitude != changed_values

    unavailable = {
        product for product, available, _, _ in AVAILABILITY if not available
    }
    assert {
        "F30",
        "SABER_NO_COOLING",
        "SABER_OH_EMISSION",
        "SABER_O2_EMISSION",
    } <= unavailable
