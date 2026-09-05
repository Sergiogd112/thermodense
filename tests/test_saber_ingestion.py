from datetime import date, datetime, timedelta
from inspect import signature
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from scipy.io import netcdf_file

from thermodense import saber
from thermodense.downloader import saber as downloader_saber


def _cooling_file(
    path: Path, variable: str, values: tuple[float, float], orbit_number: int = 7
) -> None:
    with netcdf_file(path, "w") as ds:
        ds.createDimension("scan", 2)
        ds.createDimension("alt", 3)
        ds.createDimension("one", 1)
        for name, data in {"year": np.array(2002), "day": np.array(25)}.items():
            value = ds.createVariable(name, "i4", ("one",))
            value[:] = [data]
        altitude = ds.createVariable("altitude", "f4", ("alt",))
        altitude[:] = [100, 119, 139]
        orbit = ds.createVariable("orbit", "i4", ("scan",))
        orbit[:] = [orbit_number, orbit_number]
        for name, data in {
            "tplatitude": [[20, 20, 20], [20, 20, 20]],
            "tplongitude": [[357.6] * 3, [357.6] * 3],
            "time": [[0] * 3, [10_799_000] * 3],
            variable: [[values[0]] * 3, [values[1]] * 3],
        }.items():
            value = ds.createVariable(name, "f8", ("scan", "alt"))
            value[:] = data


def _l2a_file(path: Path) -> None:
    with netcdf_file(path, "w") as ds:
        ds.createDimension("scan", 2)
        ds.createDimension("alt", 3)
        day = ds.createVariable("date", "i4", ("scan",))
        day[:] = [2002025, 2002025]
        for name, data in {
            "tpaltitude": [[100, 119, 139]] * 2,
            "tplatitude": [[20] * 3] * 2,
            "tplongitude": [[357.6] * 3] * 2,
            "time": [[0] * 3, [10_799_000] * 3],
            "O2_1delta_ver": [[10] * 3, [20] * 3],
            "OH_16_ver": [[30] * 3, [50] * 3],
            "OH_20_ver": [[40] * 3, [60] * 3],
        }.items():
            value = ds.createVariable(name, "f8", ("scan", "alt"))
            value[:] = data


def _hasdm_inputs(root: Path) -> tuple[Path, Path]:
    source, samples = root / "HASDM_2000_merged.parquet", root / "samples.parquet"
    pl.DataFrame(
        {
            "Latitude (deg)": [
                latitude for latitude in range(-90, 91, 10) for _ in range(24)
            ],
            "Longitude (deg)": list(range(0, 360, 15)) * 19,
            "timestamp": [datetime(2002, 1, 25)] * (19 * 24),
            "Altitude (m)": [175_000.0] * (19 * 24),
        }
    ).write_parquet(source)
    pl.DataFrame(
        {
            "timestamp": [datetime(2002, 1, 25, hour) for hour in range(0, 24, 3)],
            "Longitude (deg)": [5.0] * 8,
        }
    ).write_parquet(samples)
    return source, samples


def _manifest(directory: Path, filenames: list[str]) -> None:
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "entries": [
                    {"path": str(directory / name), "status": "downloaded"}
                    for name in filenames
                ]
            }
        )
    )
    files = []
    for filename in filenames:
        match = saber.CO2_FILE_RE.fullmatch(filename) or saber.NO_FILE_RE.fullmatch(
            filename
        )
        if match:
            files.append(
                {
                    "date": (
                        date(int(match.group(1)), 1, 1)
                        + saber.timedelta(days=int(match.group(2)) - 1)
                    ).isoformat(),
                    "filename": filename,
                }
            )
    if files:
        (directory / "remote_inventory.json").write_text(
            json.dumps(
                {
                    "dataset": "synthetic",
                    "source_url": "https://example.test/saber/",
                    "discovered_at": "2002-01-25T00:00:00+00:00",
                    "files": files,
                    "available_through": files[-1]["date"],
                }
            )
        )


def test_spatial_altitude_binning_and_proxy_conversion(tmp_path: Path) -> None:
    co2, no, l2a = (tmp_path / name for name in ("co2", "no", "l2a"))
    for directory in (co2, no, l2a):
        directory.mkdir()
    _cooling_file(
        co2 / "SABER_CO2_PROFILE_FLUX_2002025_V1.0.nc", "CO2cool", (1, 3), 716
    )
    _cooling_file(
        no / "SABER_NO_PROFILE_FLUX_2002025_V1.1.nc", "NOcool", (5, 9), 716
    )
    _l2a_file(l2a / "SABER_L2A_2002025_00716_02.07.nc")
    _manifest(co2, ["SABER_CO2_PROFILE_FLUX_2002025_V1.0.nc"])
    _manifest(no, ["SABER_NO_PROFILE_FLUX_2002025_V1.1.nc"])
    _manifest(l2a, ["SABER_L2A_2002025_00716_02.07.nc"])
    source, samples = _hasdm_inputs(tmp_path)
    output = tmp_path / "prepared.parquet"

    result = saber.prepare_hasdm_saber_3hour(
        co2_dir=co2,
        no_dir=no,
        l2a_dir=l2a,
        hasdm_samples=samples,
        hasdm_source=source,
        output=output,
        start=date(2002, 1, 25),
        end=datetime(2002, 1, 25),
    )

    assert result.height == 1
    first = result.row(0, named=True)
    assert first["saber_co2cool_100km_w_m3"] == 2.0
    assert first["saber_nocool_119km_w_m3"] == 7.0
    assert first["saber_o2_1delta_ver_139km_w_m3"] == 1.5
    assert first["saber_oh_16_ver_100km_w_m3"] == 4.0
    assert first["saber_co2cool_100km_w_m3_observations"] == 2
    assert output.with_suffix(".provenance.json").exists()


def test_calendar_rows_before_saber_data_preserve_float_schema(tmp_path: Path) -> None:
    co2, no, l2a = (tmp_path / name for name in ("co2", "no", "l2a"))
    for directory in (co2, no, l2a):
        directory.mkdir()
    _cooling_file(
        co2 / "SABER_CO2_PROFILE_FLUX_2002025_V1.0.nc", "CO2cool", (1, 3), 716
    )
    _cooling_file(
        no / "SABER_NO_PROFILE_FLUX_2002025_V1.1.nc", "NOcool", (5, 9), 716
    )
    _l2a_file(l2a / "SABER_L2A_2002025_00716_02.07.nc")
    _manifest(co2, ["SABER_CO2_PROFILE_FLUX_2002025_V1.0.nc"])
    _manifest(no, ["SABER_NO_PROFILE_FLUX_2002025_V1.1.nc"])
    _manifest(l2a, ["SABER_L2A_2002025_00716_02.07.nc"])
    source, samples = _hasdm_inputs(tmp_path)
    internal_gap = datetime(2002, 1, 3)
    timestamps = [
        datetime(2002, 1, 1) + timedelta(hours=3 * slot)
        for slot in range(25 * 8)
        if datetime(2002, 1, 1) + timedelta(hours=3 * slot) != internal_gap
    ]
    pl.DataFrame(
        {
            "timestamp": timestamps,
            "Longitude (deg)": [5.0] * len(timestamps),
        }
    ).write_parquet(samples)

    result = saber.prepare_hasdm_saber_3hour(
        co2_dir=co2,
        no_dir=no,
        l2a_dir=l2a,
        hasdm_samples=samples,
        hasdm_source=source,
        output=tmp_path / "prepared.parquet",
        start=date(2002, 1, 1),
        end=date(2002, 1, 25),
    )

    assert result.schema["timestamp"] == pl.Datetime
    assert result.schema["hasdm_longitude_deg_east"] == pl.Float64
    assert result.schema["saber_co2cool_100km_w_m3"] == pl.Float64
    assert result.schema["saber_co2cool_100km_w_m3_observations"] == pl.Int64
    assert (
        result.filter(pl.col("timestamp") == datetime(2002, 1, 25)).item(
            0, "saber_co2cool_100km_w_m3"
        )
        == 2.0
    )
    assert result.row(-1, named=True)["timestamp"] == datetime(2002, 1, 25, 21)
    missing = result.filter(pl.col("timestamp") == internal_gap).row(0, named=True)
    assert missing["hasdm_longitude_deg_east"] is None
    assert missing["saber_co2cool_100km_w_m3"] is None
    assert missing["saber_co2cool_100km_w_m3_observations"] == 0


def test_calendar_datetime_endpoints_and_longitude_coverage() -> None:
    start = date(2025, 7, 20)

    assert (
        signature(saber.prepare_hasdm_saber_3hour).parameters["end"].default
        == saber.DEFAULT_END
    )
    assert saber._calendar_slots(start, saber.DEFAULT_END) == [datetime(2025, 7, 20)]
    assert saber._calendar_slots(start, datetime(2025, 7, 20)) == [
        datetime(2025, 7, 20)
    ]
    assert saber._calendar_slots(start, start) == [
        datetime(2025, 7, 20, hour) for hour in range(0, 24, 3)
    ]
    assert saber._end_argument("2025-07-20") == start
    assert saber._end_argument("2025-07-20T00:00:00") == saber.DEFAULT_END
    saber._require_longitude_coverage(
        saber._calendar_slots(start, start),
        {
            datetime(2025, 7, 20): 5.0,
            datetime(2025, 7, 20, 6): 5.0,
            datetime(2025, 7, 20, 21): 5.0,
        },
    )
    with pytest.raises(ValueError, match="starts before.*2025-07-20 03:00:00"):
        saber._require_longitude_coverage(
            saber._calendar_slots(start, datetime(2025, 7, 20, 6)),
            {
                datetime(2025, 7, 20, 3): 5.0,
                datetime(2025, 7, 20, 6): 5.0,
            },
        )
    with pytest.raises(ValueError, match="ends after.*2025-07-20 00:00:00"):
        saber._require_longitude_coverage(
            saber._calendar_slots(start, datetime(2025, 7, 20, 3)),
            {datetime(2025, 7, 20): 5.0},
        )


def test_incomplete_daily_coverage_fails(tmp_path: Path) -> None:
    co2, no, l2a = (tmp_path / name for name in ("co2", "no", "l2a"))
    for directory in (co2, no, l2a):
        directory.mkdir()
    _cooling_file(co2 / "SABER_CO2_PROFILE_FLUX_2002025_V1.0.nc", "CO2cool", (1, 3))
    _manifest(co2, ["SABER_CO2_PROFILE_FLUX_2002025_V1.0.nc"])
    _manifest(no, ["SABER_NO_PROFILE_FLUX_2002025_V1.1.nc"])
    source, samples = _hasdm_inputs(tmp_path)
    with pytest.raises(RuntimeError, match="NO cooling"):
        saber.prepare_hasdm_saber_3hour(
            co2_dir=co2,
            no_dir=no,
            l2a_dir=l2a,
            hasdm_samples=samples,
            hasdm_source=source,
            output=tmp_path / "out.parquet",
            start=date(2002, 1, 25),
            end=date(2002, 1, 25),
        )


def test_manifest_inventory_must_reach_requested_end(tmp_path: Path) -> None:
    directory = tmp_path / "co2"
    directory.mkdir()
    filename = "SABER_CO2_PROFILE_FLUX_2002025_V1.0.nc"
    (directory / filename).write_bytes(b"present")
    _manifest(directory, [filename])

    with pytest.raises(RuntimeError, match="Re-run the downloader"):
        saber._coverage(
            directory,
            saber.CO2_FILE_RE,
            date(2002, 1, 25),
            date(2002, 1, 26),
            "CO2 cooling",
        )


def test_old_manifest_without_remote_inventory_fails(tmp_path: Path) -> None:
    directory = tmp_path / "co2"
    directory.mkdir()
    filename = "SABER_CO2_PROFILE_FLUX_2002025_V1.0.nc"
    (directory / filename).write_bytes(b"present")
    (directory / "manifest.json").write_text(
        json.dumps(
            {"entries": [{"path": str(directory / filename), "status": "skipped"}]}
        )
    )
    with pytest.raises(RuntimeError, match="re-run the downloader"):
        saber._coverage(
            directory,
            saber.CO2_FILE_RE,
            date(2002, 1, 25),
            date(2002, 1, 25),
            "CO2 cooling",
        )


def test_altitude_and_longitude_boundaries() -> None:
    assert saber.select_native_altitudes(np.array([99.5, 118.5, 139.5]))[100.0] == (
        0,
        99.5,
    )
    with pytest.raises(ValueError, match="119"):
        saber.select_native_altitudes(np.array([100.0, 118.4, 139.0]))
    assert saber.circular_longitude_delta(np.array([357.5]), 5.0)[0] == pytest.approx(
        7.5
    )
    assert saber._bin(datetime(2002, 1, 1, 2, 59, 59)) == datetime(2002, 1, 1)
    assert saber._bin(datetime(2002, 1, 1, 3)) == datetime(2002, 1, 1, 3)


def test_multiple_hasdm_longitudes_per_timestamp_fail(tmp_path: Path) -> None:
    samples = tmp_path / "samples.parquet"
    pl.DataFrame(
        {"timestamp": [datetime(2002, 1, 1)] * 2, "Longitude (deg)": [0, 15]}
    ).write_parquet(samples)
    with pytest.raises(ValueError, match="exactly one longitude"):
        saber.hasdm_longitudes_by_timestamp(samples)


def test_url_selection_uses_latest_no_and_official_l2a_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        downloader_saber,
        "curl_text",
        lambda url, timeout_s: (
            '<a href="SABER_NO_PROFILE_FLUX_2002001_V1.0.nc">x</a><a href="SABER_NO_PROFILE_FLUX_2002001_V1.1.nc">x</a>'
            if "NO_Cooling" in url
            else '<a href="SABER_CO2_PROFILE_FLUX_2002001_V1.0.nc">x</a><a href="SABER_CO2_PROFILE_FLUX_2002001_V1.2.nc">x</a><a href="SABER_L2A_2019001_00715_02.07.nc">x</a><a href="SABER_L2A_2019001_00716_02.06.nc">x</a><a href="SABER_L2A_2019001_00716_02.07.nc">x</a><a href="SABER_L2A_2019001_00717_02.08.nc">x</a>'
        ),
    )
    assert downloader_saber.list_saber_no_cooling_urls()[0][1].endswith("V1.1.nc")
    assert downloader_saber.list_saber_co2_cooling_urls()[0][1].endswith("V1.2.nc")
    assert downloader_saber.saber_l2a_url_rows(date(2019, 1, 1), {716}) == [
        (
            "SABER_L2A_2019001_00716_02.07.nc",
            "https://data.gats-inc.com/saber/Version2_0/Level2A/2019/001/SABER_L2A_2019001_00716_02.07.nc",
        )
    ]


def test_l2a_url_selection_resolves_cross_day_orbit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cooling_day = date(2022, 4, 13)
    previous_day_url = f"{downloader_saber.L2A_BASE_URL}2022/102/"
    cooling_day_url = f"{downloader_saber.L2A_BASE_URL}2022/103/"
    next_day_url = f"{downloader_saber.L2A_BASE_URL}2022/104/"
    monkeypatch.setattr(
        downloader_saber,
        "curl_text",
        lambda url, timeout_s: (
            '<a href="SABER_L2A_2022102_110342_02.08.nc">x</a>'
            if url == previous_day_url
            else '<a href="SABER_L2A_2022103_110343_02.08.nc">x</a>'
            if url == cooling_day_url
            else ""
            if url == next_day_url
            else pytest.fail(f"Unexpected URL: {url}")
        ),
    )

    assert downloader_saber.saber_l2a_url_rows(cooling_day, {110342}) == [
        (
            "SABER_L2A_2022102_110342_02.08.nc",
            f"{previous_day_url}SABER_L2A_2022102_110342_02.08.nc",
        )
    ]


@pytest.mark.parametrize(
    ("cooling_day", "archive_day", "version"),
    [
        (date(2019, 12, 14), date(2019, 12, 15), "02.08"),
        (date(2019, 12, 15), date(2019, 12, 14), "02.07"),
    ],
)
def test_l2a_url_selection_uses_adjacent_directory_version(
    monkeypatch: pytest.MonkeyPatch,
    cooling_day: date,
    archive_day: date,
    version: str,
) -> None:
    archive_url = (
        f"{downloader_saber.L2A_BASE_URL}{archive_day.year:04d}/"
        f"{archive_day.timetuple().tm_yday:03d}/"
    )
    filename = (
        f"SABER_L2A_{archive_day.year:04d}{archive_day.timetuple().tm_yday:03d}_"
        f"00716_{version}.nc"
    )
    monkeypatch.setattr(
        downloader_saber,
        "curl_text",
        lambda url, timeout_s: f'<a href="{filename}">x</a>'
        if url == archive_url
        else "",
    )

    assert downloader_saber.saber_l2a_url_rows(cooling_day, {716}) == [
        (filename, f"{archive_url}{filename}")
    ]


def test_cooling_downloaders_write_complete_remote_inventories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    co2, no = tmp_path / "co2", tmp_path / "no"
    monkeypatch.setattr(downloader_saber, "DEST_DIR", co2)
    monkeypatch.setattr(downloader_saber, "MANIFEST_PATH", co2 / "manifest.json")
    monkeypatch.setattr(downloader_saber, "NO_DEST_DIR", no)
    monkeypatch.setattr(downloader_saber, "NO_MANIFEST_PATH", no / "manifest.json")
    rows = [
        (date(2002, 1, 25), "SABER_CO2_PROFILE_FLUX_2002025_V1.1.nc", "co2-url"),
        (date(2002, 1, 26), "SABER_CO2_PROFILE_FLUX_2002026_V1.1.nc", "co2-url"),
    ]
    monkeypatch.setattr(downloader_saber, "list_saber_co2_cooling_urls", lambda: rows)
    monkeypatch.setattr(
        downloader_saber,
        "list_saber_no_cooling_urls",
        lambda: [
            (date(2002, 1, 25), "SABER_NO_PROFILE_FLUX_2002025_V1.1.nc", "no-url"),
            (date(2002, 1, 26), "SABER_NO_PROFILE_FLUX_2002026_V1.1.nc", "no-url"),
        ],
    )
    monkeypatch.setattr(
        downloader_saber, "download_parallel", lambda *_args, **_kwargs: ([], None)
    )
    downloader_saber.download_saber_co2_cooling(
        start_date=date(2002, 1, 26), end_date=date(2002, 1, 26)
    )
    downloader_saber.download_saber_no_cooling(
        start_date=date(2002, 1, 26), end_date=date(2002, 1, 26)
    )
    for directory, dataset in (
        (co2, "saber_co2_cooling_profiles"),
        (no, "saber_no_cooling_profiles"),
    ):
        inventory = json.loads((directory / "remote_inventory.json").read_text())
        assert inventory["dataset"] == dataset
        assert inventory["source_url"].startswith("https://")
        assert inventory["available_through"] == "2002-01-26"
        assert [row["date"] for row in inventory["files"]] == [
            "2002-01-25",
            "2002-01-26",
        ]


def test_l2a_selection_unions_finite_cooling_footprints_not_values(
    tmp_path: Path,
) -> None:
    co2, no = tmp_path / "co2", tmp_path / "no"
    co2.mkdir()
    no.mkdir()
    _cooling_file(
        co2 / "SABER_CO2_PROFILE_FLUX_2002025_V1.0.nc", "CO2cool", (np.nan, np.nan), 7
    )
    _cooling_file(
        no / "SABER_NO_PROFILE_FLUX_2002025_V1.1.nc", "NOcool", (np.nan, np.nan), 8
    )
    required = saber.required_l2a_orbits(
        list(co2.glob("*.nc")),
        list(no.glob("*.nc")),
        {datetime(2002, 1, 25): 5.0, datetime(2002, 1, 25, 3): 5.0},
        20.0,
        10.0,
        15.0,
    )
    assert required == {date(2002, 1, 25): {7, 8}}


@pytest.mark.parametrize(
    "filenames",
    [
        ["SABER_L2A_2002025_00716_02.06.nc"],
        [
            "SABER_L2A_2002025_00716_02.07.nc",
            "SABER_L2A_2002025_00716_02.06.nc",
        ],
    ],
)
def test_l2a_manifest_rejects_wrong_versions_and_duplicates(
    tmp_path: Path, filenames: list[str]
) -> None:
    for filename in filenames:
        (tmp_path / filename).write_bytes(b"present")
    _manifest(tmp_path, filenames)
    with pytest.raises(RuntimeError, match="official SABER_L2A_2002025_00716_02.07.nc"):
        saber._required_l2a_files(tmp_path, {date(2002, 1, 25): {716}})


def test_l2a_manifest_rejects_failed_required_entry(tmp_path: Path) -> None:
    filename = "SABER_L2A_2002025_00716_02.07.nc"
    (tmp_path / filename).write_bytes(b"present")
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {"entries": [{"path": str(tmp_path / filename), "status": "failed"}]}
        )
    )
    with pytest.raises(RuntimeError, match="official SABER_L2A_2002025_00716_02.07.nc"):
        saber._required_l2a_files(tmp_path, {date(2002, 1, 25): {716}})


def test_l2a_manifest_accepts_official_cross_day_orbit(tmp_path: Path) -> None:
    cooling_day = date(2022, 4, 13)
    filename = "SABER_L2A_2022102_110342_02.08.nc"
    (tmp_path / filename).write_bytes(b"present")
    _manifest(tmp_path, [filename])

    assert saber._required_l2a_files(tmp_path, {cooling_day: {110342}}) == {
        (cooling_day, 110342): tmp_path / filename
    }


@pytest.mark.parametrize(
    ("cooling_day", "archive_day", "version"),
    [
        (date(2019, 12, 14), date(2019, 12, 15), "02.08"),
        (date(2019, 12, 15), date(2019, 12, 14), "02.07"),
    ],
)
def test_l2a_manifest_accepts_adjacent_directory_version(
    tmp_path: Path,
    cooling_day: date,
    archive_day: date,
    version: str,
) -> None:
    filename = (
        f"SABER_L2A_{archive_day.year:04d}{archive_day.timetuple().tm_yday:03d}_"
        f"00716_{version}.nc"
    )
    (tmp_path / filename).write_bytes(b"present")
    _manifest(tmp_path, [filename])

    assert saber._required_l2a_files(tmp_path, {cooling_day: {716}}) == {
        (cooling_day, 716): tmp_path / filename
    }
