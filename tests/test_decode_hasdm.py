import zipfile
from datetime import datetime
from pathlib import Path

import polars as pl

from thermodense.decoding import decode_hasdm_single


def _write_hasdm_zip(path: Path, member_name: str, content: str) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(member_name, content)


def test_decode_hasdm_single_supports_plain_text_legacy_files(tmp_path: Path) -> None:
    source_path = tmp_path / "hasdm_2010_01"
    source_path.write_text(
        "\n".join(
            [
                ":Data_Source:SET HASDM density database, 2010",
                "#YYYYMMDDhhmm JulianDay HTM LAT LON LST RHO",
                "201001010000 21916 175 -90 0.83 0 4.124e-10",
                "201001010100 21916.04 175 -90 15.83 1 4.000e-10",
            ]
        )
        + "\n"
    )
    output_path = tmp_path / "decoded" / "hasdm_2010_01.parquet"
    manifest_path = tmp_path / "tmp_decode_manifest.csv"

    result = decode_hasdm_single(str(source_path), str(output_path), str(manifest_path))

    assert result is not None
    df = pl.read_parquet(output_path)
    assert df["timestamp"].to_list()[0] == datetime(2010, 1, 1, 0, 0)
    assert df["Altitude (m)"].to_list()[0] == 175_000.0


def test_decode_hasdm_single_supports_null_julian_day_and_hour_only_timestamps(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "hasdm_2023_11"
    _write_hasdm_zip(
        source_path,
        "20231101000000_20231130235959_HASDM.txt",
        "\n".join(
            [
                ":Data_Source:SET HASDM density database, 2023",
                "#YYYYMMDDhhmm JulianDay HTM LAT LON LST RHO",
                "2023110100 null 175 -90 10.91 1 7.1204e-10",
                "2023110101 null 225 -60 25.91 2 1.7675e-10",
            ]
        )
        + "\n",
    )
    output_path = tmp_path / "decoded" / "hasdm_2023_11.parquet"
    manifest_path = tmp_path / "tmp_decode_manifest.csv"

    result = decode_hasdm_single(str(source_path), str(output_path), str(manifest_path))

    assert result is not None
    df = pl.read_parquet(output_path)
    assert df["JulianDay"].null_count() == 2
    assert df["timestamp"].to_list() == [
        datetime(2023, 11, 1, 0, 0),
        datetime(2023, 11, 1, 1, 0),
    ]


def test_decode_hasdm_single_skips_header_only_archives(tmp_path: Path) -> None:
    source_path = tmp_path / "hasdm_2025_08"
    _write_hasdm_zip(
        source_path,
        "20250801000000_20250831235959_HASDM.txt",
        "\n".join(
            [
                ":Data_Source:SET HASDM density database, 2025",
                "#YYYYMMDDhhmm JulianDay HTM LAT LON LST RHO",
            ]
        )
        + "\n",
    )
    output_path = tmp_path / "decoded" / "hasdm_2025_08.parquet"
    manifest_path = tmp_path / "tmp_decode_manifest.csv"

    result = decode_hasdm_single(str(source_path), str(output_path), str(manifest_path))

    assert result is None
    assert not output_path.exists()


def test_decode_hasdm_single_skips_damaged_zip_archives(tmp_path: Path) -> None:
    source_path = tmp_path / "hasdm_2024_05"
    source_path.write_bytes(b"PK\x03\x04not-a-real-zip")
    output_path = tmp_path / "decoded" / "hasdm_2024_05.parquet"
    manifest_path = tmp_path / "tmp_decode_manifest.csv"

    result = decode_hasdm_single(str(source_path), str(output_path), str(manifest_path))

    assert result is None
    assert not output_path.exists()
