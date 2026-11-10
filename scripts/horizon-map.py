#!/usr/bin/env python3
"""
Run the whole photo -> horizon pipeline in one go:
  horizon-calibrate.py (diagnostic) -> horizon-refine.py -> horizon-build.py

Usage: horizon-map.py PHOTO_DIR [--outdir horizon_out] [--declination DEG] [--drop A.HEIC,B.HEIC] [--skip-calibrate]

Writes calib.json, refined.json, horizon.csv, horizon_stellarium.txt, mosaic.png and
mask_*.jpg into --outdir. Needs only Python 3 and uv; each step declares its own deps.
"""
import argparse, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run(script, *args):
    cmd = [str(HERE / script), *map(str, args)]
    print(f"\n$ {' '.join(cmd)}", flush=True)
    if subprocess.run(cmd).returncode:
        sys.exit(f"{script} failed")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("photos")
    ap.add_argument("--outdir", default="horizon_out")
    ap.add_argument("--declination", type=float, help="magnetic declination, deg east (default: World Magnetic Model)")
    ap.add_argument("--drop", default="", help="comma-separated filenames to exclude")
    ap.add_argument("--skip-calibrate", action="store_true", help="skip the Sun-check diagnostic")
    a = ap.parse_args()
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    decl = ["--declination", a.declination] if a.declination is not None else []
    if not a.skip_calibrate:
        run("horizon-calibrate.py", a.photos, *decl, "--out", out / "calib.json")
    run("horizon-refine.py", a.photos, *decl, *(["--drop", a.drop] if a.drop else []), "--out", out / "refined.json")
    run("horizon-build.py", a.photos, out / "refined.json", "--outdir", out)
    print(f"\nDone. Check {out}/mosaic.png before trusting {out}/horizon.csv.")


if __name__ == "__main__":
    main()
