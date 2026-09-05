#!/usr/bin/env python3
"""Execute unmodified local SET JB2006/JB2008 sources on an exact HASDM frame.

The provider files and indices deliberately remain under ignored ``data/external``.
This tracked wrapper only extracts the same unchanged subroutines used by the
local daily runner and supplies a small, generated Fortran *main program*.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import pandas as pd

from data.external import run_maunaloa_jb_baselines as legacy

ALTITUDES = range(175, 826, 25)
HERE = legacy.HERE

FORTRAN = r"""      PROGRAM HASDM_JB_EXACT
      USE, INTRINSIC :: IEEE_ARITHMETIC
      IMPLICIT REAL*8(A-H,O-Z)
      IMPLICIT INTEGER*4(I-N)
      CHARACTER*256 ARG,FLUXFILE,DTCFILE,APFILE,OUTFILE,REQUEST
      DIMENSION SUN(2),SAT(3),GEO(3),TEMP6(2),TEMP8(2)
      PARAMETER (PI=3.1415927D0,DG2RD=PI/180.D0,
     * AELL=6378.137D0,E2=6.6943799901413165D-3)
      CALL GET_COMMAND_ARGUMENT(1,REQUEST)
      CALL GET_COMMAND_ARGUMENT(2,FLUXFILE)
      CALL GET_COMMAND_ARGUMENT(3,DTCFILE)
      CALL GET_COMMAND_ARGUMENT(4,APFILE)
      CALL GET_COMMAND_ARGUMENT(5,OUTFILE)
      OPEN(10,FILE=REQUEST,STATUS='OLD')
      OPEN(11,FILE=OUTFILE,STATUS='REPLACE')
      WRITE(11,'(A)') 'year,doy,hour,latitude,longitude,altitude_km,density'
   10 CONTINUE
      READ(10,*,END=99) IYEAR,IDOY,IHR,GLAT,GLON
      D1950=DAY1950(IYEAR,IDOY)+DBLE(IHR)/24.D0
      AMJD=D1950+33281.D0
      CALL SUNPOS(AMJD,SOLRAS,SOLDEC)
      SUN(1)=SOLRAS
      SUN(2)=SOLDEC
C JB2006 retains provider lags: F10/S10 one day, M10 five days, Ap 0.279 day.
      CALL SOLFSMY(D1950-1.D0,F10,F10B,S10,S10B,XMXX,XMXXB,
     * XYXX,XYXXB,FLUXFILE)
#ifdef JB06
      CALL SOLFSMY(D1950-5.D0,XFXX,XFXXB,XSXX,XSXXB,X6,X6B,
     * XYXX,XYXXB,FLUXFILE)
      CALL SOLFLUX(D1950-0.279D0,XFAP,XFAPB,AP,AP3,APFILE)
      GEO(1)=F10
      GEO(2)=F10B
      GEO(3)=AP3
#else
C JB2008 retains provider lags: F10/S10 one day, M10 two, Y10 five, DTC now.
      CALL SOLFSMY(D1950-2.D0,XFXX,XFXXB,XSXX,XSXXB,X8,X8B,
     * XYXX,XYXXB,FLUXFILE)
      CALL SOLFSMY(D1950-5.D0,XFXX,XFXXB,XSXX,XSXXB,XMXX,XMXXB,
     * Y8,Y8B,FLUXFILE)
      CALL DTCVAL(D1950,IDTC,DTCFILE,'dtc_gap_unused.txt')
      DSTDTC=IDTC
#endif
      GEODLAT=GLAT*DG2RD
      DO 20 IALT=175,825,25
      SAT(1)=DMOD(THETA(D1950)+GLON*DG2RD+2.D0*PI,2.D0*PI)
      SAT(3)=DBLE(IALT)
      RN=AELL/DSQRT(1.D0-E2*DSIN(GEODLAT)**2)
      XP=(RN+SAT(3))*DCOS(GEODLAT)
      ZP=(RN*(1.D0-E2)+SAT(3))*DSIN(GEODLAT)
      SAT(2)=DATAN2(ZP,XP)
#ifdef JB06
      CALL JB2006(AMJD,SUN,SAT,GEO,S10,S10B,X6,X6B,TEMP6,RHO)
#else
      CALL JB2008(AMJD,SUN,SAT,F10,F10B,S10,S10B,X8,X8B,Y8,Y8B,
     * DSTDTC,TEMP8,RHO)
#endif
      IF (.NOT.IEEE_IS_FINITE(RHO).OR.RHO.LE.0.D0) STOP 2
      WRITE(11,100) IYEAR,IDOY,IHR,GLAT,GLON,IALT,RHO
  100 FORMAT(I4,',',I3.3,',',I2.2,',',F12.6,',',F12.6,',',I4,',',ES24.16)
   20 CONTINUE
      GO TO 10
   99 CONTINUE
      CLOSE(10)
      CLOSE(11)
      END
      DOUBLE PRECISION FUNCTION DAY1950(IYEAR,IDOY)
      IMPLICIT REAL*8(A-H,O-Z)
      IMPLICIT INTEGER*4(I-N)
      IYR=IYEAR
      IF(IYR.GT.1900) IYR=(IYR-2000)+100
      IF(IYR.LT.50) IYR=IYR+100
      IYY=(IYR-50)*365+((IYR-1)/4-12)
      DAY1950=DBLE(IYY+IDOY)
      RETURN
      END
"""


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def unique_requests(frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse 27 identical-altitude geometry rows to one point per timestamp."""
    required = {"timestamp", "altitude_km", "Latitude (deg)", "Longitude (deg)"}
    if missing := required - set(frame):
        raise ValueError(f"request lacks {sorted(missing)}")
    grouped = frame.groupby("timestamp", sort=True)
    if (grouped["Longitude (deg)"].nunique() != 1).any() or (
        grouped["Latitude (deg)"].nunique() != 1
    ).any():
        raise ValueError(
            "exact HASDM frame has altitude-varying geometry; cannot use 27-altitude expansion"
        )
    if not grouped.altitude_km.nunique().eq(27).all():
        raise ValueError(
            "every retained timestamp must contain all 27 requested altitudes"
        )
    return grouped[["Latitude (deg)", "Longitude (deg)"]].first().reset_index()


def build() -> dict[str, str]:
    hashes = legacy.extract()
    main = HERE / "hasdm_jb_exact.for"
    main.write_text(FORTRAN)
    compiler = [
        "gfortran",
        "-cpp",
        "-std=legacy",
        "-ffixed-form",
        "-ffixed-line-length-none",
        "-O3",
    ]
    subprocess.run(
        [
            *compiler,
            "-DJB06",
            "-o",
            "jb2006_exact",
            "hasdm_jb_exact.for",
            "jb2006_model.for",
            "solar_geometry.f",
            "solflux.f",
        ],
        cwd=HERE,
        check=True,
    )
    # JB2008's extracted utilities already contain SOLFSMY, SUNPOS, and THETA.
    # Adding solar_geometry.f here would define those provider routines twice.
    subprocess.run(
        [
            *compiler,
            "-DJB08",
            "-o",
            "jb2008_exact",
            "hasdm_jb_exact.for",
            "jb2008_model.f",
            "jb2008_utilities.f",
            "solflux.f",
        ],
        cwd=HERE,
        check=True,
    )
    return {**hashes, "hasdm_jb_exact.for": sha(main)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("request", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    began = time.monotonic()
    frame = pd.read_parquet(args.request)
    points = unique_requests(frame)
    HERE.mkdir(parents=True, exist_ok=True)
    request = HERE / "hasdm_exact_points.txt"
    with request.open("w") as handle:
        for point in points.itertuples(index=False):
            stamp = pd.Timestamp(point.timestamp)
            handle.write(
                f"{stamp.year} {stamp.dayofyear} {stamp.hour} {point._1:.6f} {point._2:.6f}\n"
            )
    hashes = build()
    ap_file, ap_metadata = legacy.make_celestrak_ap_file()
    outputs = {}
    for model in ("jb2006", "jb2008"):
        output = HERE / f"hasdm_{model}_exact.csv"
        subprocess.run(
            [
                str(HERE / f"{model}_exact"),
                str(request),
                str(legacy.IDX / "SOLFSMY.TXT"),
                str(legacy.IDX / "DTCFILE.TXT"),
                str(ap_file),
                str(output),
            ],
            cwd=HERE,
            check=True,
        )
        outputs[model] = pd.read_csv(output)
    keys = ["year", "doy", "hour", "latitude", "longitude", "altitude_km"]
    joined = outputs["jb2006"].merge(
        outputs["jb2008"],
        on=keys,
        validate="one_to_one",
        suffixes=("_jb2006", "_jb2008"),
    )
    joined["timestamp"] = pd.to_datetime(
        joined.year.astype(str) + joined.doy.astype(str).str.zfill(3), format="%Y%j"
    ) + pd.to_timedelta(joined.hour, unit="h")
    result = joined.rename(
        columns={
            "latitude": "Latitude (deg)",
            "longitude": "Longitude (deg)",
            "density_jb2006": "jb2006_density",
            "density_jb2008": "jb2008_density",
        }
    )[
        [
            "timestamp",
            "altitude_km",
            "Latitude (deg)",
            "Longitude (deg)",
            "jb2006_density",
            "jb2008_density",
        ]
    ]
    if (
        len(result) != len(frame)
        or result.duplicated(
            ["timestamp", "altitude_km", "Latitude (deg)", "Longitude (deg)"]
        ).any()
    ):
        raise ValueError("provider expansion did not reproduce every exact HASDM key")
    if not (result[["jb2006_density", "jb2008_density"]] > 0).all().all():
        raise ValueError("provider returned non-positive density")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_parquet(result, args.output)
    provenance = {
        "rows": len(result),
        "unique_timestamp_points": len(points),
        "runtime_seconds": time.monotonic() - began,
        "compiler": subprocess.check_output(
            ["gfortran", "--version"], text=True
        ).splitlines()[0],
        "source_index_hashes": {
            "JB2006_Test.fixed.for": sha(legacy.JB06),
            "JB2008.f": sha(legacy.JB08 / "JB2008.f"),
            "JB08DRVAUTO.f": sha(legacy.JB08 / "JB08DRVAUTO.f"),
            "DTCMAKEDR_AUTO.f": sha(legacy.JB08 / "DTCMAKEDR_AUTO.f"),
            "SOLFSMY.TXT": sha(legacy.IDX / "SOLFSMY.TXT"),
            "DTCFILE.TXT": sha(legacy.IDX / "DTCFILE.TXT"),
            "SW-All.csv": sha(legacy.SPACE_WEATHER),
            **hashes,
        },
        "ap_input": ap_metadata,
        "driver_lags": {
            "jb2006": "F10/S10 1 day; M10 5 days; Ap 0.279 day",
            "jb2008": "F10/S10 1 day; M10 2 days; Y10 5 days; DTC at timestamp",
        },
    }
    args.output.with_suffix(args.output.suffix + ".provider.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "rows": len(result),
                "timestamps": len(points),
                "seconds": provenance["runtime_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
