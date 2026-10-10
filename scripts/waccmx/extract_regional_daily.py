"""Extract compact daily series from the stored UKSSDC regional WACCM-X archive.

One call processes one stored year directory (``WACCM-X_<Y>-..._16-25_200-210``)
under ``data/original/waccm_x_regional`` and writes ``<year>.npz`` + ``<year>.json``:

- daily means of the forcing series (co2vmr, f107, ap, ...);
- daily-mean profiles on the 126 model levels for the central column
  (19.894737°N, 205°E) and the 25-cell latitude-weighted mean;
- daily means at fixed geopotential heights, interpolated per hour and cell
  before averaging (log-linear for DEN); NaN if any hour lies above the column top;
- per-day run tags (base date, date written) to keep restart segments apart.

Each member's provider CRC-32 is re-checked. Needs h5py (not a project dependency).
"""
import argparse
import io
import json
import lzma
from pathlib import Path
import zlib

import h5py
import numpy as np

PROFILE_VARS = ["CO2", "T", "DEN", "Z3", "O", "O2", "N2_vmr", "He", "NO"]
HEIGHT_VARS = ["CO2", "T", "DEN", "O", "O2", "N2_vmr", "He"]
HEIGHTS_KM = [20, 40, 60, 70, 80, 90, 100, 110, 120, 140, 160, 200, 250, 300, 350, 400, 450, 500]
DRIVERS = ["co2vmr", "ch4vmr", "n2ovmr", "f107", "f107a", "f107p", "kp", "ap", "sol_tsi"]
CENTRE = (2, 2)  # 19.894737°N, 205°E in the 5 x 5 subset
VERSION = "waccmx-regional-daily-v1"


def profile(ds):
    a = ds[()]
    if a.shape[:2] == (126, 24):  # DEN is stored (lev, time, lat, lon)
        a = a.transpose(1, 0, 2, 3)
    if a.shape != (24, 126, 5, 5):
        raise ValueError("unexpected shape %s for %s" % (a.shape, ds.name))
    return a.astype(np.float64)


def at_heights(z_km, values, log=False):
    """Interpolate (24,126,5,5) profiles to HEIGHTS_KM; NaN outside a column's range."""
    out = np.full((24, len(HEIGHTS_KM), 5, 5), np.nan)
    v = np.log(values) if log else values
    for t in range(24):
        for i in range(5):
            for j in range(5):
                z, y = z_km[t, ::-1, i, j], v[t, ::-1, i, j]  # stored top-down; make ascending
                if np.any(np.diff(z) <= 0):
                    raise ValueError("non-monotonic geopotential height")
                out[t, :, i, j] = np.interp(HEIGHTS_KM, z, y, left=np.nan, right=np.nan)
    return np.exp(out) if log else out


def reduce(a, weights):
    """(24, n, 5, 5) hourly -> (n, 2): daily mean of central cell and lat-weighted 25-cell mean."""
    daily = a.mean(axis=0)  # NaN propagates: any missing hour invalidates the day
    centre = daily[:, CENTRE[0], CENTRE[1]]
    w = np.broadcast_to(weights[:, None], (5, 5))
    region = (daily * w).sum(axis=(1, 2)) / w.sum()
    return np.stack([centre, region], axis=-1)


def extract(year_dir, out_dir):
    members = json.loads((year_dir / "manifest.json").read_text())
    n = len(members)
    dates, base, written = [], [], []
    drivers = np.full((n, len(DRIVERS)), np.nan)
    prof = np.full((n, len(PROFILE_VARS), 126, 2), np.nan, np.float32)
    fixed = np.full((n, len(HEIGHT_VARS), len(HEIGHTS_KM), 2), np.nan, np.float32)
    ps = np.full((n, 2), np.nan)
    static = None
    for k, m in enumerate(members):
        data = lzma.decompress((year_dir / (m["name"] + ".xz")).read_bytes())
        if "%08x" % zlib.crc32(data) != m["crc32"]:
            raise ValueError("CRC mismatch " + m["name"])
        with h5py.File(io.BytesIO(data), "r") as f:
            if f["time"].shape != (24,):
                raise ValueError("expected 24 hourly records in " + m["name"])
            day = str(int(f["date"][0]))
            dates.append("%s-%s-%s" % (day[:4], day[4:6], day[6:]))
            base.append(int(f["nbdate"][()]))
            written.append(bytes(f["date_written"][0]).decode().strip())
            gw = f["gw"][()].astype(np.float64)
            if static is None:
                static = {name: f[name][()].tolist() for name in ("lev", "hyam", "hybm", "lat", "lon")}
                static["P0"] = float(f["P0"][()])
            drivers[k] = [float(np.mean(f[name][()])) for name in DRIVERS]
            fields = {name: profile(f[name]) for name in set(PROFILE_VARS) | set(HEIGHT_VARS)}
            for i, name in enumerate(PROFILE_VARS):
                prof[k, i] = reduce(fields[name], gw)
            z_km = fields["Z3"] / 1000.0
            for i, name in enumerate(HEIGHT_VARS):
                fixed[k, i] = reduce(at_heights(z_km, fields[name], log=(name == "DEN")), gw)
            p = f["PS"][()].astype(np.float64)[:, None]  # (24, 1, 5, 5)
            ps[k] = reduce(p, gw)[0]
    year = dates[0][:4]
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / (year + ".npz"), dates=np.array(dates), base_date=np.array(base),
                        date_written=np.array(written), drivers=drivers, profiles=prof,
                        fixed_heights=fixed, ps=ps)
    (out_dir / (year + ".json")).write_text(json.dumps({
        "version": VERSION, "source_year_dir": year_dir.name, "days": n,
        "first": dates[0], "last": dates[-1], "profile_vars": PROFILE_VARS, "height_vars": HEIGHT_VARS,
        "heights_km_geopotential": HEIGHTS_KM, "drivers": DRIVERS, "columns": ["central", "region_lat_weighted"],
        "static": static, "base_dates": sorted(set(base)),
        "notes": "daily means of 24 hourly records; fixed heights interpolated per hour/cell, NaN above column top"},
        indent=1) + "\n")
    return n


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=Path("data/original/waccm_x_regional"))
    p.add_argument("--out", type=Path, default=Path("data/derived/waccmx_regional_daily_v1"))
    p.add_argument("--index", type=int, required=True, help="1-based index into sorted year directories")
    a = p.parse_args()
    years = sorted(d for d in a.source.glob("WACCM-X_*_16-25_200-210") if (d / "COMPLETE.json").exists())
    print(extract(years[a.index - 1], a.out))


if __name__ == "__main__":
    main()
