from pathlib import Path

from thermodense.downloader import space_weather
from thermodense.downloader.counter import Counters
from thermodense.downloader.space_weather import (
    CSV_COLUMNS,
    SPACE_WEATHER_CSV_PATH,
    parse_celestrak_space_weather,
    prepare_space_weather_csv,
)


_FIELD_WIDTHS = (4, 3, 3, 5, 3, *([3] * 8), 4, *([4] * 8), 4, 4, 2, 4, 6, 2, *([6] * 5))


def _record(*, adjusted: str, observed: str, quality: str = "0", monthly: bool = False) -> str:
    fields = [
        "2024", "01", "02", "2500", "3", *map(str, range(1, 9)), "36",
        *map(str, range(1, 9)), "4", "0.4", "2", "123", adjusted, quality,
        "140.0", "130.0", observed, "150.0", "145.0",
    ]
    if monthly:
        fields[1:5] = ["03", "01", "2502", "7"]
        fields[5:25] = [""] * 20
        fields[25] = "100"
    return "".join(value.rjust(width) for value, width in zip(fields, _FIELD_WIDTHS, strict=True))


def test_parser_preserves_schema_and_maps_celestrak_flux_columns() -> None:
    rows = parse_celestrak_space_weather(
        [
            "BEGIN OBSERVED",
            _record(adjusted="135.0", observed="145.0"),
            "END OBSERVED",
            "BEGIN DAILY_PREDICTED",
            _record(adjusted="136.0", observed="146.0", quality=""),
            "END DAILY_PREDICTED",
            "BEGIN MONTHLY_PREDICTED",
            _record(adjusted="137.0", observed="147.0", quality="", monthly=True),
            "END MONTHLY_PREDICTED",
        ]
    )

    assert [row["F10.7_DATA_TYPE"] for row in rows] == ["OBS", "PRD", "PRM"]
    assert rows[1]["F10.7_ADJ"] == "136.0"
    assert rows[1]["KP_SUM"] == "36"
    assert rows[2]["KP1"] == ""
    assert rows[2]["AP_AVG"] == ""
    assert rows[2]["ISN"] == "100"
    assert rows[0] == {
        "DATE": "2024-01-02",
        "BSRN": "2500",
        "ND": "3",
        "KP1": "1",
        "KP2": "2",
        "KP3": "3",
        "KP4": "4",
        "KP5": "5",
        "KP6": "6",
        "KP7": "7",
        "KP8": "8",
        "KP_SUM": "36",
        "AP1": "1",
        "AP2": "2",
        "AP3": "3",
        "AP4": "4",
        "AP5": "5",
        "AP6": "6",
        "AP7": "7",
        "AP8": "8",
        "AP_AVG": "4",
        "CP": "0.4",
        "C9": "2",
        "ISN": "123",
        "F10.7_OBS": "145.0",
        "F10.7_ADJ": "135.0",
        "F10.7_DATA_TYPE": "OBS",
        "F10.7_OBS_CENTER81": "150.0",
        "F10.7_OBS_LAST81": "145.0",
        "F10.7_ADJ_CENTER81": "140.0",
        "F10.7_ADJ_LAST81": "130.0",
    }


def test_preparation_writes_compatibility_csv_to_requested_path(tmp_path: Path) -> None:
    source = tmp_path / "SW-All.txt"
    destination = tmp_path / "space_weather" / "SW-All.csv"
    source.write_text(
        "BEGIN OBSERVED\n" + _record(adjusted="135.0", observed="145.0"),
        encoding="utf-8",
    )

    assert prepare_space_weather_csv(source, destination) == 1
    assert destination.read_text(encoding="utf-8").splitlines() == [
        ",".join(CSV_COLUMNS),
        "2024-01-02,2500,3,1,2,3,4,5,6,7,8,36,1,2,3,4,5,6,7,8,4,0.4,2,123,145.0,135.0,OBS,150.0,145.0,140.0,130.0",
    ]
    assert SPACE_WEATHER_CSV_PATH == Path("data/original/space_weather/SW-All.csv")


def test_downloader_prepares_csv_after_downloading_text(
    tmp_path: Path, monkeypatch
) -> None:
    destination_dir = tmp_path / "space_weather"
    destination_dir.mkdir()
    (destination_dir / "SW-All.txt").write_text(
        "BEGIN OBSERVED\n" + _record(adjusted="135.0", observed="145.0"), encoding="utf-8"
    )
    monkeypatch.setattr(space_weather, "DEST_DIR", destination_dir)
    monkeypatch.setattr(space_weather, "REF_ROOT", tmp_path)
    monkeypatch.setattr(space_weather, "MANIFEST_PATH", destination_dir / "manifest.json")
    monkeypatch.setattr(
        space_weather,
        "download_parallel",
        lambda *args, **kwargs: ([], Counters(skipped_existing=4)),
    )

    space_weather.download_space_weather()

    assert (destination_dir / "SW-All.csv").exists()
