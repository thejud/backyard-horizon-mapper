#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "astropy", "astroquery"]
# ///
"""
Which targets are imageable tonight given the backyard horizon profile?
Usage: horizon-visibility.py [--date YYYY-MM-DD] [--horizon horizon.csv] [--min-alt 20] TARGET [TARGET...]
TARGET is a name resolvable by astropy/Sesame (e.g. "IC 434", "M45") or "name=ra_deg,dec_deg".
Reports hour-by-hour alt/az, windows above max(horizon, min-alt) for the ground profile and the
conservative canopy profile, the Moon, and a no-overlap one-target-at-a-time schedule.
"""
import argparse, warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np
from astropy.coordinates import EarthLocation, AltAz, SkyCoord, get_sun, get_body
from astropy.time import Time
import astropy.units as u

import os


def load_horizon(p):
    rows = [l.strip().split(",") for l in Path(p).read_text().splitlines()[1:]]
    az = np.array([float(r[0]) for r in rows]); g = np.array([float(r[1]) for r in rows]); c = np.array([float(r[2]) for r in rows])
    return az, g, c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("targets", nargs="+"); ap.add_argument("--date", default=None)
    ap.add_argument("--horizon", default="horizon.csv")
    ap.add_argument("--min-alt", type=float, default=20.0); ap.add_argument("--tz", type=float, default=float(os.environ.get("OBSERVER_TZ_OFFSET", "nan")), help="UTC offset hours (or env OBSERVER_TZ_OFFSET)")
    ap.add_argument("--step-min", type=int, default=10)
    ap.add_argument("--lat", type=float, default=float(os.environ.get("OBSERVER_LAT", "nan")))
    ap.add_argument("--lon", type=float, default=float(os.environ.get("OBSERVER_LON", "nan")))
    ap.add_argument("--elev", type=float, default=float(os.environ.get("OBSERVER_ELEV", "0")))
    a = ap.parse_args()
    if any(np.isnan(v) for v in (a.lat, a.lon, a.tz)):
        ap.error("set --lat/--lon/--tz (or OBSERVER_LAT, OBSERVER_LON, OBSERVER_TZ_OFFSET)")
    LAT, LON, ELEV = a.lat, a.lon, a.elev
    az_t, hg, hc = load_horizon(a.horizon)
    loc = EarthLocation(lat=LAT * u.deg, lon=LON * u.deg, height=ELEV * u.m)
    from datetime import datetime, timedelta, timezone
    d0 = datetime.strptime(a.date, "%Y-%m-%d") if a.date else datetime.now()
    # evening local noon -> next local noon
    t0 = Time(datetime(d0.year, d0.month, d0.day, 12) - timedelta(hours=a.tz))
    times = t0 + np.arange(0, 24 * 60, a.step_min) * u.min
    frame = AltAz(obstime=times, location=loc)
    sun = get_sun(times).transform_to(frame); moon = get_body("moon", times, loc).transform_to(frame)
    dark = sun.alt.deg < -18
    ilum = 0.5 * (1 - np.cos(np.radians((get_body("moon", times[len(times)//2], loc).separation(get_sun(times[len(times)//2])).deg))))
    print(f"Date: evening of {d0:%Y-%m-%d} (UTC{a.tz:+.0f}); astronomical dark {np.sum(dark) * a.step_min / 60:.1f} h; Moon illumination ~{ilum*100:.0f}%")
    loc_t = lambda t: (t.to_datetime() + timedelta(hours=a.tz)).strftime("%H:%M")
    dk = np.where(dark)[0]
    if len(dk): print(f"  dark from {loc_t(times[dk[0]])} to {loc_t(times[dk[-1]])}")
    res = {}
    for spec in a.targets:
        if "=" in spec:
            nm, rd = spec.split("=", 1); ra, de = [float(x) for x in rd.split(",")]; c = SkyCoord(ra * u.deg, de * u.deg)
        else:
            nm = spec; c = SkyCoord.from_name(spec)
        aa = c.transform_to(frame)
        alt, az = aa.alt.deg, aa.az.deg
        gmin = np.maximum(np.interp(az, np.r_[az_t, 360], np.r_[hg, hg[0]]), a.min_alt)
        cmin = np.maximum(np.interp(az, np.r_[az_t, 360], np.r_[hc, hc[0]]), a.min_alt)
        okg = dark & (alt > gmin); okc = dark & (alt > cmin)
        moon_sep = c.separation(moon).deg
        res[nm] = (okg, okc, alt, az, moon_sep)
        print(f"\n== {nm}  (RA {c.ra.deg:.2f} Dec {c.dec.deg:+.2f})")
        hrs = lambda m: m.sum() * a.step_min / 60
        print(f"  dark & above sky>{a.min_alt:.0f}deg only:    {hrs(dark & (alt > a.min_alt)):.1f} h")
        print(f"  dark & clear of ground profile:   {hrs(okg):.1f} h")
        print(f"  dark & clear of canopy profile:   {hrs(okc):.1f} h")
        print("  time  alt  az  | ground canopy | moon sep")
        for k in range(0, len(times), 60 // a.step_min):
            if dark[k] and alt[k] > -5:
                print(f"  {loc_t(times[k])} {alt[k]:5.1f} {az[k]:5.0f} |   {'Y' if okg[k] else '-'}      {'Y' if okc[k] else '-'}   | {moon_sep[k]:4.0f}")
    # schedule: one target at a time, scarcest window gets priority
    if len(a.targets) > 1:
        print("\n== One-at-a-time schedule (canopy-clear windows; scarcest target first, then higher altitude)")
        names = list(res); scarce = {n: res[n][1].sum() for n in names}; cur = None; start = None; tot = {n: 0 for n in names}
        for k in range(len(times)):
            if not dark[k]: continue
            cands = [n for n in names if res[n][1][k]]
            pick = min(cands, key=lambda n: (scarce[n], -res[n][2][k])) if cands else None
            if pick: tot[pick] += a.step_min / 60
            if pick != cur:
                if cur is not None: print(f"  {loc_t(start)}-{loc_t(times[k])}  {cur}")
                elif start is not None and cur is None and k > 0: print(f"  {loc_t(start)}-{loc_t(times[k])}  (nothing clear)")
                cur = pick; start = times[k]
        print(f"  {loc_t(start)}-{loc_t(times[dk[-1]])}  {cur or '(nothing clear)'}")
        print("  totals:", ", ".join(f"{n} {h:.1f} h" for n, h in tot.items()))
        # overlap
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                ov = (res[names[i]][1] & res[names[j]][1]).sum() * a.step_min / 60
                print(f"  windows overlap ({names[i]} vs {names[j]}): {ov:.1f} h")

main()
