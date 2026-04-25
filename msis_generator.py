import gc
import numpy as np
import polars as pl
from time import sleep
from pymsis import msis
from pymsis.utils import get_f107_ap
from multiprocessing import Pool, cpu_count


MODELS = ["0", "2.0", "2.1"]


def compute_msis_density(args):
    if type(args) == str:
        file = args
        n = 0
    else:
        file, n = args
    df = pl.read_parquet(file)
    timestamps = df["timestamp"].to_numpy()
    lats = df["Latitude (deg)"].to_numpy()
    lons = df["Longitude (deg)"].to_numpy()
    alts = (df["Altitude (m)"] / 1000.0).to_numpy()
    f107, f107a, aps, f107_type = get_f107_ap(timestamps)
    f107_ct = np.ones(np.shape(f107)) * 150.0
    f107a_ct = np.ones(np.shape(f107a)) * 150.0
    ap_ct = np.zeros(np.shape(aps))
    densities = []
    for version in MODELS:
        density = msis.run(
            timestamps,
            lons,
            lats,
            alts,
            f107,
            f107a,
            aps,
            version=version,
        )[:, 0]
        densities.append(pl.Series(f"msis_density_{version}", density))
        density_ct = msis.run(
            timestamps,
            lons,
            lats,
            alts,
            f107_ct,
            f107a_ct,
            ap_ct,
            version=version,
        )[:, 0]
        densities.append(pl.Series(f"msis_density_ct_{version}", density_ct))
    gc.collect()
    df = df.with_columns(
        *densities,
        pl.Series("f107", f107),
        pl.Series("ap", aps[:, 0]),
        pl.Series("qflag", f107_type),
    )
    df.write_parquet(file.replace(".parquet", "_msis.parquet"))
    del df
    del densities
    del f107, f107a, aps, f107_type
    del timestamps, lats, lons, alts
    gc.collect()
    return file


def compute_msis_density_single_worker(args):
    timestamps, lats, lons, alts, f107, f107a, aps, version, mission_code = args
    print(f"Computing MSIS density for {mission_code} with version {version}")
    return (
        msis.run(
            timestamps,
            lons,
            lats,
            alts,
            f107,
            f107a,
            aps,
            version=version,
        )[:, 0],
        print(
            f"Finished computing MSIS density for {mission_code} with version {version}"
        ),
    )[0]


def compute_msis_density_parallel(file):
    if isinstance(file, tuple):
        file, n = file
        print(f"Processing {file}")
    df = pl.read_parquet(file)
    mission_code = file.split("/")[-1].split("_")[0]
    timestamps = df["timestamp"].to_numpy()
    lats = df["Latitude (deg)"].to_numpy()
    lons = df["Longitude (deg)"].to_numpy()
    alts = (df["Altitude (m)"] / 1000.0).to_numpy()
    f107, f107a, aps, f107_type = get_f107_ap(timestamps)
    f107_ct = np.ones(np.shape(f107)) * 150.0
    f107a_ct = np.ones(np.shape(f107a)) * 150.0
    ap_ct = np.zeros(np.shape(aps))
    args = [
        (timestamps, lats, lons, alts, f107, f107a, aps, version, mission_code)
        for version in MODELS
    ] + [
        (timestamps, lats, lons, alts, f107_ct, f107a_ct, ap_ct, version, mission_code)
        for version in MODELS
    ]
    with Pool(processes=cpu_count() - 1) as pool:
        densities = pool.map(compute_msis_density_single_worker, args)
    densities = [
        pl.Series(f"msis_density_{version}", density)
        for density, version in zip(densities[:3], MODELS)
    ] + [
        pl.Series(f"msis_density_ct_{version}", density)
        for density, version in zip(densities[3:], MODELS)
    ]
    gc.collect()
    df = df.with_columns(
        *densities,
        pl.Series("f107", f107),
        pl.Series("ap", aps[:, 0]),
        pl.Series("qflag", f107_type),
    )
    df.write_parquet(file.replace(".parquet", "_msis.parquet"))
    del df
    del densities
    del args
    del f107, f107a, aps, f107_type
    del timestamps, lats, lons, alts
    gc.collect()
    return file
