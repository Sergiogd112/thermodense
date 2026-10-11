import json
from datetime import date, time
from pathlib import Path

import polars as pl
import pytest

from thermodense.decoding import TUDELFT_SCHEMAS, decode_vires_single
from thermodense.downloader import vires
from thermodense.downloader.counter import Counters
from thermodense.downloader.manifest import ManifestEntry

PARAMETERS = [
    {"name": "Timestamp", "type": "isotime", "units": "UTC", "length": 24, "fill": None},
    {"name": "Latitude_GD", "type": "double", "units": "deg", "fill": None},
    {"name": "Longitude_GD", "type": "double", "units": "deg", "fill": None},
    {"name": "Height_GD", "type": "double", "units": "m", "fill": None},
    {"name": "local_solar_time", "type": "double", "units": "h", "fill": None},
    {"name": "density", "type": "double", "units": "kg/m3", "fill": None},
    {"name": "density_orbitmean", "type": "double", "units": "kg/m3", "fill": None},
    {"name": "validity_flag", "type": "integer", "units": None, "fill": None},
    {"name": "validity_flag_orbitmean", "type": "integer", "units": None, "fill": None},
]
# Real CH_OPER_DNS_ACC_2_ rows (2003-01-01), including the orbit-mean sentinel.
ROWS = [
    "2003-01-01T00:00:00.000Z,42.949843242256506,-113.84026853032262,410124.01296452526,"
    "16.357240049248137,2.5868027570591155e-12,9.99e+32,0,1",
    "2003-01-01T00:59:50.000Z,9.143737140025623,54.12883299310372,408069.64462576713,"
    "4.552072571023765,1.268042094034767e-12,1.8782e-12,0,0",
]


def _write_hapi_csv(path: Path, parameters: list[dict], rows: list[str]) -> None:
    header = json.dumps({"HAPI": "3.0", "x_dataset": "CH_OPER_DNS_ACC_2_", "parameters": parameters}, indent=2)
    lines = [f"#{line}" for line in header.splitlines()] + rows
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def test_month_helpers_roll_over_years_and_name_files_like_tudelft() -> None:
    assert vires.month_starts(date(2002, 11, 15), date(2003, 1, 3)) == [
        date(2002, 11, 1),
        date(2002, 12, 1),
        date(2003, 1, 1),
    ]
    url = vires.month_url("CH_OPER_DNS_ACC_2_", date(2002, 12, 1))
    assert "start=2002-12-01T00%3A00%3A00Z" in url
    assert "stop=2003-01-01T00%3A00%3A00Z" in url
    assert "include=header" in url and "format=csv" in url
    path = vires.month_path("GR_OPER_DNS2ACC_2_", date(2005, 3, 1), Path("root"))
    assert path == Path("root/grace/GB_DNS_ACC_2005_03_vires.csv.gz")


def test_default_collections_use_swarm_pod_like_existing_tudelft_files() -> None:
    assert "SW_OPER_DNSAPOD_2_" in vires.DEFAULT_COLLECTIONS
    assert "SW_OPER_DNSAACC_2_" not in vires.DEFAULT_COLLECTIONS
    assert "CH_OPER_DNS_ACC_2_" in vires.DEFAULT_COLLECTIONS


def test_download_requests_only_covered_months_without_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        vires,
        "collection_info",
        lambda _c: {"startDate": "2000-07-29T00:00:00Z", "stopDate": "2000-09-04T23:59:50Z"},
    )
    seen: dict[str, object] = {}

    def fake_parallel(downloads, base_dir, **kwargs):
        seen["downloads"], seen["kwargs"] = downloads, kwargs
        return [], Counters()

    monkeypatch.setattr(vires, "download_parallel", fake_parallel)
    vires.download_vires(["CH_OPER_DNS_ACC_2_"], start_ym=(2000, 8))

    paths = [path.name for _url, path in seen["downloads"]]
    assert paths == ["CH_DNS_ACC_2000_08_vires.csv", "CH_DNS_ACC_2000_09_vires.csv"]
    assert seen["kwargs"]["resume"] is False


def test_download_gzips_new_months_and_skips_existing_ones(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        vires,
        "collection_info",
        lambda _c: {"startDate": "2003-01-01T00:00:00Z", "stopDate": "2003-02-28T23:59:50Z"},
    )
    existing = vires.month_path("CH_OPER_DNS_ACC_2_", date(2003, 1, 1))
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"already here")
    stale = vires.month_path("CH_OPER_DNS_ACC_2_", date(2003, 2, 1)).with_suffix("")
    stale.write_text("partial")

    def fake_parallel(downloads, base_dir, **kwargs):
        (url, path), = downloads
        assert not path.exists()  # the partial download was discarded
        _write_hapi_csv(path, PARAMETERS, ROWS)
        entry = ManifestEntry(str(path.relative_to(base_dir)), url, "downloaded", "now")
        return [entry], Counters(downloaded=1)

    monkeypatch.setattr(vires, "download_parallel", fake_parallel)
    counters = vires.download_vires(["CH_OPER_DNS_ACC_2_"])

    february = vires.month_path("CH_OPER_DNS_ACC_2_", date(2003, 2, 1))
    assert counters.downloaded == 1 and counters.skipped_existing == 1
    assert february.exists() and not stale.exists()
    assert existing.read_bytes() == b"already here"
    output = tmp_path / "feb.parquet"
    assert decode_vires_single("champ", str(february), str(output), str(tmp_path / "m.csv"))
    assert pl.read_parquet(output).height == 2


def test_download_rejects_unknown_collection(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="Unknown VirES collection"):
        vires.download_vires(["GO_OPER_DNS_ACC_2_"])


def test_decode_vires_matches_tudelft_schema_with_explicit_differences(tmp_path: Path) -> None:
    source = tmp_path / "vires" / "champ" / "CH_DNS_ACC_2003_01_vires.csv"
    _write_hapi_csv(source, PARAMETERS, ROWS)
    output = tmp_path / "decoded" / "CH_DNS_ACC_2003_01_vires.parquet"

    result = decode_vires_single("champ", str(source), str(output), str(tmp_path / "m.csv"))

    assert result is not None and result[1] == "CH"
    df = pl.read_parquet(output)
    assert df.schema == pl.Schema(TUDELFT_SCHEMAS["CHAMP"])
    assert df["Date yyyy-mm-dd"].to_list() == [date(2003, 1, 1)] * 2
    assert df["Time hh:mm:ss.sss"].to_list()[1] == time(0, 59, 50)
    assert set(df["Time System"].to_list()) == {"UTC"}
    assert df["Argument of Latitude (deg)"].null_count() == 2
    assert df["Latitude (deg)"].to_list()[1] == pytest.approx(9.143737140025623)
    assert df["Longitude (deg)"].to_list()[1] == pytest.approx(54.12883299310372)
    assert df["Density (kg/m^3)"].to_list()[1] == pytest.approx(1.268042094034767e-12)
    assert df["Density Mean (kg/m^3)"].to_list() == [None, pytest.approx(1.8782e-12)]
    assert df["Anomalus Density Mean (kg/m^3)"].to_list() == [1.0, 0.0]


def test_decode_vires_leaves_missing_swarm_acc_columns_null(tmp_path: Path) -> None:
    source = tmp_path / "SWA_DNS_ACC_2014_02_vires.csv"
    _write_hapi_csv(source, PARAMETERS[:6], [row.rsplit(",", 3)[0] for row in ROWS])
    output = tmp_path / "SWA_DNS_ACC_2014_02_vires.parquet"

    assert decode_vires_single("swarm", str(source), str(output), str(tmp_path / "m.csv"))
    df = pl.read_parquet(output)
    assert df["Density Mean (kg/m^3)"].null_count() == 2
    assert df["Anomalus Density (kg/m^3)"].null_count() == 2


def test_decode_vires_skips_months_without_rows(tmp_path: Path) -> None:
    source = tmp_path / "CH_DNS_ACC_2010_10_vires.csv"
    _write_hapi_csv(source, PARAMETERS, [])
    output = tmp_path / "out.parquet"

    assert decode_vires_single("champ", str(source), str(output), str(tmp_path / "m.csv")) is None
    assert not output.exists()


def test_interrupted_compression_leaves_no_final_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    csv_path = tmp_path / "CH_DNS_ACC_2003_01_vires.csv"
    csv_path.write_text("a,b\n1,2\n")
    entry = ManifestEntry(path=str(csv_path), url="u", status="downloaded", timestamp="t", size_bytes=8)

    def interrupted(source, target):
        target.write(b"partial")
        raise KeyboardInterrupt

    monkeypatch.setattr(vires.shutil, "copyfileobj", interrupted)
    with pytest.raises(KeyboardInterrupt):
        vires._compress(entry, csv_path)

    gz_path = csv_path.with_name(csv_path.name + ".gz")
    assert not gz_path.exists()  # a later run must not skip this month
    assert not list(tmp_path.glob("*.partial"))
    assert csv_path.exists()  # the downloaded CSV is kept until compression succeeds


def test_interrupted_batch_keeps_completed_months_tracked_and_rerun_reconciles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from thermodense.downloader.manifest import Manifest

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        vires,
        "collection_info",
        lambda _c: {"startDate": "2003-01-01T00:00:00Z", "stopDate": "2003-02-28T23:59:50Z"},
    )

    def fake_parallel(downloads, base_dir, **kwargs):
        entries = []
        for url, path in downloads:
            _write_hapi_csv(path, PARAMETERS, ROWS)
            entries.append(ManifestEntry(str(path.relative_to(base_dir)), url, "downloaded", "now"))
        return entries, Counters(downloaded=len(entries))

    real_compress = vires._compress
    calls = {"n": 0}

    def compress_then_interrupt(entry, path):
        calls["n"] += 1
        if calls["n"] == 2:
            raise KeyboardInterrupt
        return real_compress(entry, path)

    monkeypatch.setattr(vires, "download_parallel", fake_parallel)
    monkeypatch.setattr(vires, "_compress", compress_then_interrupt)
    with pytest.raises(KeyboardInterrupt):
        vires.download_vires(["CH_OPER_DNS_ACC_2_"])

    january = vires.month_path("CH_OPER_DNS_ACC_2_", date(2003, 1, 1))
    tracked = {e.path for e in Manifest.load(vires.MANIFEST_PATH).entries}
    assert january.exists() and str(january.relative_to(vires.REF_ROOT)) in tracked

    # A rerun of an archive that is present but untracked records it instead of leaving it invisible.
    vires.MANIFEST_PATH.unlink()
    monkeypatch.setattr(vires, "_compress", real_compress)
    vires.download_vires(["CH_OPER_DNS_ACC_2_"])
    entries = {e.path: e for e in Manifest.load(vires.MANIFEST_PATH).entries}
    jan_entry = entries[str(january.relative_to(vires.REF_ROOT))]
    assert jan_entry.status == "skipped" and "without a manifest entry" in jan_entry.error
