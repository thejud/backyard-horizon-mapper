#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "astropy", "astroquery"]
# ///
"""
Pick Seestar targets for a night, honouring the backyard horizon profile.

  pick-targets.py                     # tonight, top 15 targets + suggested schedule
  pick-targets.py --date 2026-11-12 --top 25 --scope s30
  pick-targets.py --filter LP         # only narrowband (LP) targets
  pick-targets.py --name "IC 434" --name M45 --schedule   # restrict to named targets

Ranking: hours that are astronomically dark AND above max(horizon(az), --min-alt)
on the conservative canopy profile (ground profile also shown), with Moon penalty.
Schedule: one target at a time (dynamic programming, switching penalty).
Catalog coordinates are resolved once via Sesame and cached in lib/targets_catalog.json.
"""
import argparse, json, warnings
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
from astropy.coordinates import EarthLocation, AltAz, SkyCoord, get_sun, get_body
from astropy.time import Time
import astropy.units as u

warnings.filterwarnings("ignore")
import os
HERE = Path(__file__).resolve().parent
CAT = HERE.parent / "targets_catalog.json"
HORIZON = "horizon.csv"

# name, type, filter (LP = dual-band/narrowband; IRCUT = broadband), approx size (arcmin)
SEED = [
    ("M1", "SNR", "LP", 6), ("M8", "emission", "LP", 90), ("M13", "globular", "IRCUT", 20), ("M16", "emission", "LP", 35),
    ("M17", "emission", "LP", 40), ("M20", "emission+refl", "LP", 28), ("M27", "planetary", "LP", 8), ("M31", "galaxy", "IRCUT", 180),
    ("M32", "galaxy", "IRCUT", 8), ("M33", "galaxy", "IRCUT", 70), ("M42", "emission", "LP", 65), ("M43", "emission", "LP", 20),
    ("M45", "reflection", "IRCUT", 110), ("M51", "galaxy", "IRCUT", 11), ("M57", "planetary", "LP", 4), ("M63", "galaxy", "IRCUT", 12),
    ("M64", "galaxy", "IRCUT", 10), ("M65", "galaxy", "IRCUT", 10), ("M66", "galaxy", "IRCUT", 9), ("M74", "galaxy", "IRCUT", 10),
    ("M81", "galaxy", "IRCUT", 27), ("M82", "galaxy", "IRCUT", 11), ("M87", "galaxy", "IRCUT", 8), ("M97", "planetary", "LP", 3),
    ("M101", "galaxy", "IRCUT", 29), ("M104", "galaxy", "IRCUT", 9), ("M106", "galaxy", "IRCUT", 18), ("M3", "globular", "IRCUT", 18),
    ("M5", "globular", "IRCUT", 23), ("M15", "globular", "IRCUT", 18), ("M92", "globular", "IRCUT", 14), ("M2", "globular", "IRCUT", 16),
    ("M35", "cluster", "IRCUT", 28), ("M36", "cluster", "IRCUT", 12), ("M37", "cluster", "IRCUT", 24), ("M38", "cluster", "IRCUT", 21),
    ("M44", "cluster", "IRCUT", 95), ("M52", "cluster", "IRCUT", 13), ("M103", "cluster", "IRCUT", 6),
    ("NGC 7000", "emission", "LP", 120), ("IC 5070", "emission", "LP", 60), ("NGC 6992", "SNR", "LP", 60), ("NGC 6960", "SNR", "LP", 70),
    ("NGC 6888", "emission", "LP", 20), ("IC 1396", "emission", "LP", 170), ("NGC 7635", "emission", "LP", 15), ("NGC 281", "emission", "LP", 35),
    ("IC 1805", "emission", "LP", 60), ("IC 1848", "emission", "LP", 60), ("IC 405", "emission", "LP", 30), ("IC 410", "emission", "LP", 40),
    ("IC 443", "SNR", "LP", 50), ("IC 434", "emission+dark", "LP", 60), ("NGC 2024", "emission", "LP", 30), ("NGC 2237", "emission", "LP", 80),
    ("NGC 2264", "emission", "LP", 60), ("NGC 2174", "emission", "LP", 40), ("NGC 2175", "emission", "LP", 40), ("IC 2118", "reflection", "IRCUT", 180),
    ("NGC 7293", "planetary", "LP", 15), ("NGC 253", "galaxy", "IRCUT", 27), ("NGC 891", "galaxy", "IRCUT", 13), ("NGC 4631", "galaxy", "IRCUT", 15),
    ("NGC 5139", "globular", "IRCUT", 36), ("NGC 2392", "planetary", "LP", 1), ("IC 342", "galaxy", "IRCUT", 20), ("NGC 7331", "galaxy", "IRCUT", 10),
    ("NGC 6946", "galaxy", "IRCUT", 11), ("NGC 2403", "galaxy", "IRCUT", 22), ("NGC 4565", "galaxy", "IRCUT", 16), ("NGC 4725", "galaxy", "IRCUT", 11),
    ("NGC 1499", "emission", "LP", 160), ("NGC 4244", "galaxy", "IRCUT", 16), ("M110", "galaxy", "IRCUT", 20), ("IC 63", "emission", "LP", 10),
    ("NGC 7822", "emission", "LP", 60), ("NGC 1333", "reflection", "IRCUT", 6), ("IC 1318", "emission", "LP", 120), ("NGC 6820", "emission", "LP", 40),
]


def load_catalog():
    if CAT.exists():
        return json.loads(CAT.read_text())
    print("Building catalog (one-time Sesame lookup)...")
    out = []
    for n, t, f, s in SEED:
        try:
            c = SkyCoord.from_name(n)
            out.append(dict(name=n, type=t, filter=f, size=s, ra=float(c.ra.deg), dec=float(c.dec.deg)))
        except Exception as e:
            print("  skip", n, e)
    CAT.parent.mkdir(exist_ok=True); CAT.write_text(json.dumps(out, indent=1))
    return out


def load_horizon(p):
    rows = [l.strip().split(",") for l in Path(p).read_text().splitlines()[1:]]
    az = np.array([float(r[0]) for r in rows]); g = np.array([float(r[1]) for r in rows]); c = np.array([float(r[2]) for r in rows])
    return np.r_[az, 360], np.r_[g, g[0]], np.r_[c, c[0]]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date"); ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--min-alt", type=float, default=20); ap.add_argument("--max-alt", type=float, default=80, help="alt-az mounts struggle near zenith"); ap.add_argument("--horizon", default=str(HORIZON))
    ap.add_argument("--filter", choices=["LP", "IRCUT"]); ap.add_argument("--scope", choices=["s30", "s50"], default=None,
                    help="s30 = wide field (targets up to ~200'), s50 = ~76'x43' field; hides targets that are far too large/small")
    ap.add_argument("--name", action="append", help="restrict to these names (repeatable)")
    ap.add_argument("--step-min", type=int, default=10); ap.add_argument("--no-schedule", action="store_true")
    ap.add_argument("--switch-min", type=int, default=30, help="penalty (minutes of imaging) for changing target")
    ap.add_argument("--max-targets", type=int, default=3, help="max targets in the schedule")
    ap.add_argument("--lat", type=float, default=float(os.environ.get("OBSERVER_LAT", "nan")))
    ap.add_argument("--lon", type=float, default=float(os.environ.get("OBSERVER_LON", "nan")))
    ap.add_argument("--elev", type=float, default=float(os.environ.get("OBSERVER_ELEV", "0")))
    ap.add_argument("--tz", default=os.environ.get("OBSERVER_TZ"), help="IANA zone, e.g. America/Denver (or env OBSERVER_TZ)")
    a = ap.parse_args()
    if np.isnan(a.lat) or np.isnan(a.lon) or not a.tz:
        ap.error("set --lat/--lon/--tz (or OBSERVER_LAT, OBSERVER_LON, OBSERVER_TZ)")
    LAT, LON, ELEV, TZ = a.lat, a.lon, a.elev, ZoneInfo(a.tz)

    now = datetime.now(TZ)
    d = datetime.strptime(a.date, "%Y-%m-%d") if a.date else (now - timedelta(days=1 if now.hour < 12 else 0)).replace(tzinfo=None)
    start_local = datetime(d.year, d.month, d.day, 12, tzinfo=TZ)
    t0 = Time(start_local.astimezone(ZoneInfo("UTC")).replace(tzinfo=None))
    times = t0 + np.arange(0, 24 * 60, a.step_min) * u.min
    loc = EarthLocation(lat=LAT * u.deg, lon=LON * u.deg, height=ELEV * u.m)
    frame = AltAz(obstime=times, location=loc)
    sun = get_sun(times).transform_to(frame); moon = get_body("moon", times, loc).transform_to(frame)
    dark = sun.alt.deg < -18
    mid = times[len(times) // 2]
    elong = get_body("moon", mid, loc).separation(get_sun(mid)).deg
    illum = 0.5 * (1 - np.cos(np.radians(elong)))
    ltime = lambda t: t.to_datetime(timezone=ZoneInfo("UTC")).astimezone(TZ).strftime("%H:%M")
    dk = np.where(dark)[0]
    print(f"Night of {d:%Y-%m-%d}: dark {ltime(times[dk[0]])}-{ltime(times[dk[-1]])} ({dark.sum() * a.step_min / 60:.1f} h), Moon {illum*100:.0f}% lit")

    az_t, hg, hc = load_horizon(a.horizon)
    cat = load_catalog()
    if a.name:
        want = {n.lower().replace(" ", "") for n in a.name}
        cat = [c for c in cat if c["name"].lower().replace(" ", "") in want]
    if a.filter: cat = [c for c in cat if c["filter"] == a.filter]
    if a.scope == "s50": cat = [c for c in cat if c["size"] <= 120]
    stepf = a.step_min / 60
    rows = []
    for c in cat:
        sc = SkyCoord(c["ra"] * u.deg, c["dec"] * u.deg)
        aa = sc.transform_to(frame); alt, az = aa.alt.deg, aa.az.deg
        gmin = np.maximum(np.interp(az, az_t, hg), a.min_alt); cmin = np.maximum(np.interp(az, az_t, hc), a.min_alt)
        okc = dark & (alt > cmin) & (alt < a.max_alt); okg = dark & (alt > gmin) & (alt < a.max_alt)
        sep = sc.separation(moon).deg
        # Moon penalty: only while moon is up; narrowband tolerates more
        moon_up = moon.alt.deg > 0
        bad_sep = 60 if c["filter"] == "IRCUT" else 35
        moon_bad = moon_up & (sep < bad_sep) & (illum > 0.3)
        eff = okc & ~moon_bad
        if okg.sum() == 0: continue
        rows.append(dict(c=c, okc=okc, okg=okg, eff=eff, alt=alt, az=az, sep=sep,
                         hc_=okc.sum() * stepf, hg_=okg.sum() * stepf, he=eff.sum() * stepf,
                         peak=alt[okc].max() if okc.any() else (alt[okg].max() if okg.any() else 0)))
    rows.sort(key=lambda r: (-r["he"], -r["peak"]))
    rows = [r for r in rows if r["he"] > 0][:a.top]
    print(f"\n{'target':<10}{'type':<14}{'filt':<6}{'size':>5}  {'clear h':>7}{'ground h':>9}{'peak':>6}  window (canopy-clear)       moon sep")
    for r in rows:
        k = np.where(r["eff"])[0]
        win = f"{ltime(times[k[0]])}-{ltime(times[k[-1]])}"
        print(f"{r['c']['name']:<10}{r['c']['type']:<14}{r['c']['filter']:<6}{r['c']['size']:>4}'  {r['he']:>7.1f}{r['hg_']:>9.1f}{r['peak']:>5.0f}°  {win:<26}{r['sep'][k].min():>6.0f}°")
    if a.no_schedule or not rows: return

    # schedule DP over top candidates
    cands = rows[:max(a.max_targets * 4, 8)] if not a.name else rows
    n = len(cands); T = len(times); sw = a.switch_min / a.step_min
    mean_he = np.mean([c['he'] for c in cands]); wgt = [mean_he / c['he'] for c in cands]  # scarcer window => more valuable per slice
    best = np.full((T, n + 1), -1e9); back = np.zeros((T, n + 1), int)
    for t in range(T):
        if not dark[t]:
            continue
        for j in range(n + 1):
            gain = 0 if j == n else ((1 + 0.002 * cands[j]["alt"][t]) * wgt[j] if cands[j]["eff"][t] else -5)
            if t == 0 or best[t - 1].max() < -1e8:
                best[t, j] = gain; back[t, j] = j; continue
            opts = best[t - 1] - np.where(np.arange(n + 1) == j, 0, sw)
            k = int(np.argmax(opts)); best[t, j] = opts[k] + gain; back[t, j] = k
    t_end = dk[-1]; j = int(np.argmax(best[t_end])); path = [0] * T
    for t in range(t_end, dk[0] - 1, -1):
        path[t] = j; j = back[t, j]
    segs = []; cur = None
    for t in range(dk[0], t_end + 1):
        j = path[t]
        j = j if (j < n and cands[j]["eff"][t]) else n
        if cur is None or j != cur[0]: cur = [j, t, t]; segs.append(cur)
        else: cur[2] = t
    print("\nSuggested schedule (one target at a time):")
    used = []
    for j, s, e in segs:
        nm = "(idle / nothing clear)" if j == n else f"{cands[j]['c']['name']}  [{cands[j]['c']['filter']}]"
        print(f"  {ltime(times[s])}-{ltime(times[min(e + 1, T - 1)])}  {(e - s + 1) * stepf:4.1f} h  {nm}")
        if j < n: used.append(cands[j]["c"]["filter"])
    if len(set(used)) > 1: print("  note: schedule mixes LP and IRCUT; keep them in separate stacks.")


main()
