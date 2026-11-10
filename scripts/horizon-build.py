#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pillow", "pillow-heif", "astropy", "scipy", "opencv-python-headless"]
# ///
"""
Step 3: sky/obstruction segmentation per frame -> az/alt horizon profile.
Usage: horizon-build.py <photo_dir> <refined.json> [--outdir DIR]
Outputs: horizon.csv (az_deg,min_alt_deg), horizon_stellarium.txt, mosaic.png, mosaic_sky.png, and sample masks.
"""
import argparse, importlib.util, json
from pathlib import Path
import numpy as np, cv2
from PIL import Image
import pillow_heif
from scipy import ndimage

pillow_heif.register_heif_opener()
here = Path(__file__).parent
def _load(name, fn):
    sp = importlib.util.spec_from_file_location(name, here / fn); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m
hc = _load("hc", "horizon-calibrate.py")
hr = _load("hr", "horizon-refine.py")
WORK = 1600


def sky_mask(rgb):
    a = rgb.astype(np.int16)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    blue = (b - r > 25) & (b > 100)
    mean = a.mean(axis=2); spread = rgb.max(axis=2).astype(int) - rgb.min(axis=2)
    white = ((mean > 175) & (b >= r - 8) & (spread < 45)) | (rgb.min(axis=2) > 235)  # bright cloud / glare
    cand = blue | white
    cand = ndimage.binary_opening(cand, iterations=2)
    # keep only components that contain solidly blue sky or are cloud touching it
    seeds = (b - r > 60) & (b > 140)
    lab, n = ndimage.label(cand)
    if n == 0:
        return cand
    keep = np.unique(lab[seeds & cand]); keep = keep[keep > 0]
    m = np.isin(lab, keep)
    m = ndimage.binary_closing(m, iterations=3)
    m = ndimage.binary_fill_holes(m) if False else m
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir"); ap.add_argument("refined")
    ap.add_argument("--outdir", default="horizon_out"); ap.add_argument("--declination", type=float)
    ap.add_argument("--step", type=int, default=3); ap.add_argument("--edge", type=float, default=0.9)
    ap.add_argument("--thresh", type=float, default=0.5); ap.add_argument("--margin", type=float, default=2.0)
    a = ap.parse_args()
    out = Path(a.outdir); out.mkdir(exist_ok=True, parents=True)
    ref = json.loads(Path(a.refined).read_text()); decl = a.declination if a.declination is not None else ref["declination"]
    kf = ref.get("kf", [1, 1])
    meta = hc.read_meta(a.dir)
    AZ, ALT = 360, 130  # 1deg cells; alt -20..110 (cap at 90 later)
    sky = np.zeros((ALT, AZ)); tot = np.zeros((ALT, AZ)); col = np.zeros((ALT, AZ, 3))
    for n, info in sorted(ref["frames"].items()):
        m = dict(meta[n]); m["GPSImgDirection"] += info["yaw_corr"]
        b = hc.camera_basis(m, decl)
        from PIL import ImageOps
        img = Image.open(Path(a.dir) / n).convert("RGB"); s = WORK / max(img.size)
        img = img.resize((round(img.size[0] * s), round(img.size[1] * s)))
        W, Hh = img.size; rgb = np.asarray(img)
        fpx = (W if W >= Hh else Hh) / 2 / np.tan(np.radians(float(m["FOV"]) / 2)) * (kf[0] if float(m["FOV"]) > 90 else kf[1])
        sm = sky_mask(rgb)
        if n in sorted(ref["frames"])[::max(1, len(ref["frames"]) // 4)][:4]:  # a few diagnostic overlays (these show your property!)
            ov = rgb.copy(); ov[sm] = (0.5 * ov[sm] + 0.5 * np.array([255, 0, 255])).astype(np.uint8)
            Image.fromarray(ov).resize((W // 2, Hh // 2)).save(out / f"mask_{n[:8]}.jpg")
        ys, xs = np.mgrid[0:Hh:a.step, 0:W:a.step]
        ex = (np.abs(xs - W / 2) / (W / 2) < a.edge) & (np.abs(ys - Hh / 2) / (Hh / 2) < a.edge)
        xs, ys = xs[ex], ys[ex]
        az, alt = hc.pix_to_azalt(xs.astype(float), ys.astype(float), W, Hh, fpx, b)
        ai = (az.astype(int)) % AZ; li = np.floor(alt).astype(int) + 20
        ok = (li >= 0) & (li < ALT)
        sk = sm[ys, xs]
        np.add.at(tot, (li[ok], ai[ok]), 1); np.add.at(sky, (li[ok], ai[ok]), sk[ok])
        np.add.at(col, (li[ok], ai[ok]), rgb[ys, xs][ok])
    frac = np.where(tot > 0, sky / np.maximum(tot, 1), np.nan)
    # horizon per azimuth: highest altitude cell (<=89) with covered & frac<thresh, contiguous up to the first well-covered sky
    hor = np.full(AZ, np.nan); cov = np.zeros(AZ); canopy = np.zeros(AZ)
    for i in range(AZ):
        c = frac[:, i]
        alts = np.arange(ALT) - 20
        valid = (~np.isnan(c)) & (alts >= 0) & (alts <= 89)
        if valid.sum() < 5: continue
        blocked = valid & (c < a.thresh)
        # walk up from the ground; stop at the first sky gap >= 5 deg once obstructed (ignores glare speckle and floating canopy)
        h = 0; gap = 0; started = False
        for k_, al in enumerate(alts):
            if not valid[k_]: continue
            if blocked[k_]:
                if not started and al > 25: break
                started = True; h = al + 1; gap = 0
            elif started:
                gap += 1
                if gap >= 5: break
        hor[i] = h
        # canopy profile: top of the highest thick (>=4 deg, gaps <=2 merged) obstructed run
        top = h; run0 = None; last = None
        runs = []
        for k_, al in enumerate(alts):
            if not valid[k_]: continue
            if blocked[k_]:
                if run0 is None: run0 = al
                elif al - last > 3: runs.append((run0, last)); run0 = al
                last = al
        if run0 is not None: runs.append((run0, last))
        for r0, r1 in runs:
            if r1 - r0 >= 3: top = max(top, r1 + 1)
        canopy[i] = top
        cov[i] = valid.sum()
    # fill gaps by interpolation, smooth, add margin
    good = ~np.isnan(hor)
    azs = np.arange(AZ) + 0.5
    horf = np.interp(azs, azs[good], hor[good], period=360) if good.any() else hor
    k = 3; horf = np.convolve(np.r_[horf[-k:], horf, horf[:k]], np.ones(2 * k + 1) / (2 * k + 1), "valid")
    hmax = np.maximum(horf, ndimage.maximum_filter1d(horf, 5, mode="wrap") * 0 + horf)  # keep
    final = np.clip(hmax + a.margin, 0, 90)
    cf = np.convolve(np.r_[canopy[-k:], canopy, canopy[:k]], np.ones(2 * k + 1) / (2 * k + 1), "valid")
    canopy_f = np.clip(np.maximum(cf, horf) + a.margin, 0, 90)
    with open(out / "horizon.csv", "w") as f:
        f.write("az_deg,min_alt_deg,canopy_alt_deg,coverage_cells\n")
        for az0 in range(0, 360, 5):
            f.write(f"{az0},{np.interp(az0 + .5, azs, final, period=360):.1f},{np.interp(az0 + .5, azs, canopy_f, period=360):.1f},{int(cov[az0:az0+5].mean())}\n")
    with open(out / "horizon_stellarium.txt", "w") as f:
        for az0 in range(0, 360, 5):
            f.write(f"{az0} {np.interp(az0 + .5, azs, final, period=360):.1f}\n")
    # mosaic render: az x alt (-10..80)
    cm = np.where(tot[..., None] > 0, col / np.maximum(tot[..., None], 1), 40).astype(np.uint8)
    sub = cm[10:100][::-1]
    big = cv2.resize(sub, (1440, 360), interpolation=cv2.INTER_NEAREST)
    for i, h in enumerate(final):
        y = int((100 - (h + 20 - 0)) * 4) if False else int((80 - h) * 4)
        cv2.circle(big, (int(i * 4 + 2), y), 2, (255, 0, 255), -1)
    for i, h in enumerate(canopy_f):
        cv2.circle(big, (int(i * 4 + 2), int((80 - h) * 4)), 1, (255, 255, 0), -1)
    for ang in range(0, 360, 30):
        cv2.putText(big, f"{ang}", (ang * 4 + 2, 352), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    for ang in (0, 20, 40, 60):
        cv2.putText(big, f"{ang}d", (2, int((80 - ang) * 4) - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
    Image.fromarray(big).save(out / "mosaic.png")
    print("coverage az bins with data:", int(good.sum()), "of 360")
    print("canopy every 15deg:", " ".join(f"{az0}:{np.interp(az0 + .5, azs, canopy_f, period=360):.0f}" for az0 in range(0, 360, 15)))
    print("horizon (az:alt) every 15deg:", " ".join(f"{az0}:{np.interp(az0 + .5, azs, final, period=360):.0f}" for az0 in range(0, 360, 15)))

main()
