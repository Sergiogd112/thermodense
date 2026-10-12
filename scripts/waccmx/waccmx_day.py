"""Open one stored WACCM-X regional day: verifies the provider CRC, returns an h5py.File.

    from waccmx_day import open_day
    f = open_day("2014-06-15")   # needs h5py, e.g. `uv run --no-project --with h5py --with numpy`
"""
import io, json, lzma, zlib
from pathlib import Path

# Default: the repository's gitignored store written by fetch_archive.py.
ROOT = Path(__file__).resolve().parents[2] / "data/original/waccm_x_regional"


def open_day(day, root=ROOT):
    import h5py
    for year_dir in root.glob("WACCM-X_%s-*_16-25_200-210" % day[:4]):
        name = "WACCM-X_%s_16-25_200-210.nc" % day
        entry = next((m for m in json.loads((year_dir / "manifest.json").read_text()) if m["name"] == name), None)
        if entry:
            data = lzma.decompress((year_dir / (name + ".xz")).read_bytes())
            if "%08x" % zlib.crc32(data) != entry["crc32"]:
                raise ValueError("CRC mismatch for " + name)
            return h5py.File(io.BytesIO(data), "r")
    raise FileNotFoundError("no stored WACCM-X member for " + day)
