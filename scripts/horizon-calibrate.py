#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pillow", "pillow-heif", "astropy", "scipy"]
# ///
"""
Step 1-2 of backyard horizon mapping: per-frame camera orientation from iPhone
EXIF (GPSImgDirection + Apple AccelerationVector), pixel -> az/alt mapping, and
Sun-based azimuth/altitude calibration.

Usage: horizon-calibrate.py <photo_dir> --declination DEG [--out calib.json]
"""
import argparse, json, subprocess, sys
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
from PIL import Image
import pillow_heif
from scipy import ndimage
from astropy.coordinates import EarthLocation, AltAz, get_sun
from astropy.time import Time
import astropy.units as u

pillow_heif.register_heif_opener()

# displayed (up, right) axes expressed in device coords (x right, y up, z out of screen)
# when device is portrait: up=+y right=+x. Derived from EXIF orientation + gravity.
ORI_AXES = {
    "Rotate 90 CW": ((0, 1, 0), (1, 0, 0)),
    "Horizontal (normal)": ((1, 0, 0), (0, -1, 0)),
    "Rotate 180": ((0, -1, 0), (-1, 0, 0)),
    "Rotate 270 CW": ((-1, 0, 0), (0, 1, 0)),
}


def read_meta(d):
    out = subprocess.run(
        ["exiftool", "-json", "-n", "-r", "-ext", "heic", "-ext", "jpg", str(d)],
        capture_output=True, text=True, check=True).stdout
    meta = {}
    for m in json.loads(out):
        meta[Path(m["SourceFile"]).name] = m
    # orientation as text (not -n)
    out = subprocess.run(
        ["exiftool", "-json", "-Orientation", "-r", "-ext", "heic", "-ext", "jpg", str(d)],
        capture_output=True, text=True, check=True).stdout
    for m in json.loads(out):
        meta[Path(m["SourceFile"]).name]["OrientationText"] = m.get("Orientation")
    return meta


def camera_basis(m, decl):
    """Return (E, N, U, F, R, Up) as device-frame unit vectors, or None if no gravity."""
    acc = m.get("AccelerationVector")
    if acc is None:
        return None
    g = np.array([float(x) for x in str(acc).split()])
    Uw = -g / np.linalg.norm(g)  # world up in device coords
    F = np.array([0, 0, -1.0])
    Fh = F - (F @ Uw) * Uw
    if np.linalg.norm(Fh) < 1e-3:
        return None
    h = Fh / np.linalg.norm(Fh)
    H = np.radians(m["GPSImgDirection"] + decl)  # true azimuth of camera axis
    w = np.cross(Uw, h)
    N = np.cos(H) * h + np.sin(H) * w
    E = np.sin(H) * h - np.cos(H) * w
    up, right = ORI_AXES[m["OrientationText"]]
    return E, N, Uw, F, np.array(right, float), np.array(up, float)


def pix_to_azalt(px, py, W, Hh, fpx, basis):
    E, N, U, F, R, Up = basis
    d = (F[None, :] * fpx + R[None, :] * (px - W / 2)[:, None] + Up[None, :] * (Hh / 2 - py)[:, None])
    e, n, up = d @ E, d @ N, d @ U
    az = np.degrees(np.arctan2(e, n)) % 360
    alt = np.degrees(np.arctan2(up, np.hypot(e, n)))
    return az, alt


def sun_azalt(m, loc):
    ts = f"{m['GPSDateStamp'].replace(':', '-')} {m['GPSTimeStamp']}"
    t = Time(ts, scale="utc")
    s = get_sun(t).transform_to(AltAz(obstime=t, location=loc))
    return s.az.deg, s.alt.deg


def find_sun_blob(img_small):
    a = np.asarray(img_small.convert("RGB")).astype(float)
    mn = a.min(axis=2)
    mask = mn >= 254
    mask = ndimage.binary_opening(mask, iterations=2)
    lab, n = ndimage.label(mask)
    if n == 0:
        return None
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    k = int(np.argmax(sizes)) + 1
    cy, cx = ndimage.center_of_mass(mask, lab, k)
    return cx, cy, int(sizes[k - 1])


def ang_diff(a, b):
    return (a - b + 180) % 360 - 180


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--declination", type=float, required=True, help="magnetic declination at your site, deg east (e.g. from NOAA calculator)")
    ap.add_argument("--out", default="calib.json")
    ap.add_argument("--min-blob", type=int, default=8, help="min saturated pixels (at 1000px scale)")
    a = ap.parse_args()

    meta = read_meta(a.dir)
    first = next(iter(meta.values()))
    loc = EarthLocation(lat=first["GPSLatitude"] * u.deg, lon=first["GPSLongitude"] * u.deg, height=first.get("GPSAltitude", 100) * u.m)

    rows = []
    for name in sorted(meta):
        m = meta[name]
        if "AccelerationVector" not in m or m.get("AccelerationVector") in (None, "-"):
            print(f"{name}: no accel vector, skipped")
            continue
        img = Image.open(Path(a.dir) / name)
        W0, H0 = img.size
        scale = 1000 / max(W0, H0)
        small = img.resize((int(W0 * scale), int(H0 * scale)))
        fov = float(m["FOV"])
        fpx_small = (max(small.size) / 2) / np.tan(np.radians(fov / 2))
        basis = camera_basis(m, a.declination)
        if basis is None:
            continue
        cen_az, cen_alt = pix_to_azalt(np.array([small.size[0] / 2]), np.array([small.size[1] / 2]), *small.size[:1], small.size[1], fpx_small, basis)
        sun_az, sun_alt = sun_azalt(m, loc)
        row = dict(file=name, w=W0, h=H0, fov=fov, heading_mag=m["GPSImgDirection"],
                   cen_az=float(cen_az[0]), cen_alt=float(cen_alt[0]), sun_az=float(sun_az), sun_alt=float(sun_alt))
        blob = find_sun_blob(small)
        if blob and blob[2] >= a.min_blob:
            bx, by, bs = blob
            az, alt = pix_to_azalt(np.array([bx]), np.array([by]), small.size[0], small.size[1], fpx_small, basis)
            row.update(blob_px=int(bs), blob_xy=[float(bx), float(by)], meas_az=float(az[0]), meas_alt=float(alt[0]),
                       d_az=float(ang_diff(az[0], sun_az)), d_alt=float(alt[0] - sun_alt))
        rows.append(row)
        extra = f" SUN blob={row['blob_px']}px dAz={row['d_az']:+.1f} dAlt={row['d_alt']:+.1f}" if "d_az" in row else ""
        print(f"{name} {W0}x{H0} fov={fov:.0f} head={row['heading_mag']:.0f} centre az={cen_az[0]:.0f} alt={cen_alt[0]:.0f}{extra}")

    print(f"\nSun true az/alt at capture: {rows[0]['sun_az']:.1f} / {rows[0]['sun_alt']:.1f}")
    d = [r for r in rows if "d_az" in r]
    if d:
        da = np.array([r["d_az"] for r in d]); dl = np.array([r["d_alt"] for r in d])
        print(f"{len(d)} frames with Sun blob: dAz median {np.median(da):+.1f} (std {da.std():.1f}), dAlt median {np.median(dl):+.1f} (std {dl.std():.1f})")
    Path(a.out).write_text(json.dumps(dict(declination=a.declination, frames=rows), indent=1))


if __name__ == "__main__":
    main()
