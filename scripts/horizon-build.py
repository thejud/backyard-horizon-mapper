#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pillow", "pillow-heif", "astropy", "scipy", "opencv-python-headless", "pygeomag"]
# ///
"""
Sky/obstruction segmentation per frame -> azimuth/altitude horizon profile.

Usage: horizon-build.py <photo_dir> <refined.json> [--outdir DIR]

Outputs (in --outdir):
  horizon.csv             az_deg,min_alt_deg,canopy_alt_deg,coverage_cells (every 5 deg, true north)
  horizon_stellarium.txt  two-column "az alt" (ground profile)
  mosaic.png              az x alt render with both profiles drawn (magenta = ground, yellow = canopy)
  mask_*.jpg              a few frames with the detected sky tinted magenta

mosaic.png and mask_*.jpg are pictures of your property; don't publish them.
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
    sp = importlib.util.spec_from_file_location(name, here / fn)
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    return m


hc = _load("hc", "horizon-calibrate.py")

WORK = 1600          # frames are resized so the long side is this many pixels
N_AZ = 360           # 1 deg azimuth cells
ALT_MIN = -20        # altitude grid covers ALT_MIN .. ALT_MIN + N_ALT - 1 deg
N_ALT = 130
GAP_STOP = 5         # ground profile: a clear-sky gap this tall ends the obstruction
GROUND_START_MAX = 25  # ground profile: obstruction must start below this altitude
CANOPY_MIN_THICK = 4   # canopy profile: obstructed run must be at least this thick (deg)
CANOPY_MERGE_GAP = 3   # canopy profile: runs separated by <= this many deg are merged
SMOOTH_HALF = 3      # moving-average half width (deg) applied to both profiles


def sky_mask(rgb):
    """Boolean mask of sky pixels: blue sky, plus bright low-saturation pixels
    (cloud, glare) that are connected to solidly blue sky."""
    a = rgb.astype(np.int16)
    r, b = a[..., 0], a[..., 2]
    blue = (b - r > 25) & (b > 100)
    mean = a.mean(axis=2)
    spread = rgb.max(axis=2).astype(int) - rgb.min(axis=2)
    white = ((mean > 175) & (b >= r - 8) & (spread < 45)) | (rgb.min(axis=2) > 235)
    cand = ndimage.binary_opening(blue | white, iterations=2)
    seeds = (b - r > 60) & (b > 140)
    lab, n = ndimage.label(cand)
    if n == 0:
        return cand
    keep = np.unique(lab[seeds & cand])
    keep = keep[keep > 0]
    return ndimage.binary_closing(np.isin(lab, keep), iterations=3)


def ground_height(alts, valid, blocked):
    """Walk up from the ground while cells are obstructed; stop at a GAP_STOP-deg
    sky gap. Ignores glare speckle and canopy floating over open sky."""
    h, gap, started = 0, 0, False
    for k, al in enumerate(alts):
        if not valid[k]:
            continue
        if blocked[k]:
            if not started and al > GROUND_START_MAX:
                break
            started, h, gap = True, al + 1, 0
        elif started:
            gap += 1
            if gap >= GAP_STOP:
                break
    return h


def canopy_height(alts, valid, blocked, ground):
    """Top of the highest obstructed run at least CANOPY_MIN_THICK deg thick
    (counts palm crowns and branches over open sky); never below `ground`."""
    runs, run0, last = [], None, None
    for k, al in enumerate(alts):
        if not (valid[k] and blocked[k]):
            continue
        if run0 is None:
            run0 = al
        elif al - last > CANOPY_MERGE_GAP:
            runs.append((run0, last))
            run0 = al
        last = al
    if run0 is not None:
        runs.append((run0, last))
    top = ground
    for r0, r1 in runs:
        if r1 - r0 >= CANOPY_MIN_THICK - 1:
            top = max(top, r1 + 1)
    return top


def profiles(frac, thresh):
    """frac: (N_ALT, N_AZ) sky fraction per cell, NaN where no data.
    Returns raw ground, canopy (both NaN/0 where too little data) and coverage per azimuth."""
    alts = np.arange(N_ALT) + ALT_MIN
    ground = np.full(N_AZ, np.nan); canopy = np.zeros(N_AZ); cov = np.zeros(N_AZ)
    for i in range(N_AZ):
        c = frac[:, i]
        valid = ~np.isnan(c) & (alts >= 0) & (alts <= 89)
        if valid.sum() < 5:
            continue
        blocked = valid & (c < thresh)
        ground[i] = ground_height(alts, valid, blocked)
        canopy[i] = canopy_height(alts, valid, blocked, ground[i])
        cov[i] = valid.sum()
    return ground, canopy, cov


def smooth_circular(x, k=SMOOTH_HALF):
    return np.convolve(np.r_[x[-k:], x, x[:k]], np.ones(2 * k + 1) / (2 * k + 1), "valid")


def finalize(ground, canopy, margin):
    """Fill missing azimuths, smooth, add safety margin. Returns (azs, ground, canopy)."""
    good = ~np.isnan(ground)
    azs = np.arange(N_AZ) + 0.5
    g = np.interp(azs, azs[good], ground[good], period=360) if good.any() else ground
    g = smooth_circular(g)
    c = smooth_circular(canopy)
    return azs, np.clip(g + margin, 0, 90), np.clip(np.maximum(c, g) + margin, 0, 90)


def accumulate(photo_dir, ref, decl, step, edge, outdir):
    """Project every frame's sky mask onto the az/alt grid."""
    kf = ref.get("kf", [1, 1])  # focal scale: [ultrawide, main]
    meta = hc.read_meta(photo_dir)
    sky = np.zeros((N_ALT, N_AZ)); tot = np.zeros((N_ALT, N_AZ)); col = np.zeros((N_ALT, N_AZ, 3))
    names = sorted(ref["frames"])
    diag = set(names[::max(1, len(names) // 4)][:4])  # a few mask overlays for checking
    for n in names:
        m = dict(meta[n])
        m["GPSImgDirection"] += ref["frames"][n]["yaw_corr"]
        basis = hc.camera_basis(m, decl)
        img = Image.open(Path(photo_dir) / n).convert("RGB")
        s = WORK / max(img.size)
        img = img.resize((round(img.size[0] * s), round(img.size[1] * s)))
        W, H = img.size
        rgb = np.asarray(img)
        fov = float(m["FOV"])
        fpx = max(W, H) / 2 / np.tan(np.radians(fov / 2)) * (kf[0] if fov > 90 else kf[1])
        sm = sky_mask(rgb)
        if n in diag:
            ov = rgb.copy()
            ov[sm] = (0.5 * ov[sm] + 0.5 * np.array([255, 0, 255])).astype(np.uint8)
            Image.fromarray(ov).resize((W // 2, H // 2)).save(outdir / f"mask_{Path(n).stem}.jpg")
        ys, xs = np.mgrid[0:H:step, 0:W:step]
        inner = (np.abs(xs - W / 2) / (W / 2) < edge) & (np.abs(ys - H / 2) / (H / 2) < edge)  # skip lens edges
        xs, ys = xs[inner], ys[inner]
        az, alt = hc.pix_to_azalt(xs.astype(float), ys.astype(float), W, H, fpx, basis)
        ai = az.astype(int) % N_AZ
        li = np.floor(alt).astype(int) - ALT_MIN
        ok = (li >= 0) & (li < N_ALT)
        np.add.at(tot, (li[ok], ai[ok]), 1)
        np.add.at(sky, (li[ok], ai[ok]), sm[ys, xs][ok])
        np.add.at(col, (li[ok], ai[ok]), rgb[ys, xs][ok])
    return sky, tot, col


def render_mosaic(col, tot, ground, canopy, path):
    """az (0-360) x alt (-10..80) colour mosaic, 4 px per degree, with both profiles."""
    cm = np.where(tot[..., None] > 0, col / np.maximum(tot[..., None], 1), 40).astype(np.uint8)
    sub = cm[-10 - ALT_MIN:80 - ALT_MIN][::-1]
    big = cv2.resize(sub, (N_AZ * 4, 90 * 4), interpolation=cv2.INTER_NEAREST)
    for i, h in enumerate(ground):
        cv2.circle(big, (i * 4 + 2, int((80 - h) * 4)), 2, (255, 0, 255), -1)
    for i, h in enumerate(canopy):
        cv2.circle(big, (i * 4 + 2, int((80 - h) * 4)), 1, (255, 255, 0), -1)
    for ang in range(0, 360, 30):
        cv2.putText(big, f"{ang}", (ang * 4 + 2, 352), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    for ang in (0, 20, 40, 60):
        cv2.putText(big, f"{ang}d", (2, int((80 - ang) * 4) - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
    Image.fromarray(big).save(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", help="photo directory")
    ap.add_argument("refined", help="refined.json from horizon-refine.py")
    ap.add_argument("--outdir", default="horizon_out")
    ap.add_argument("--declination", type=float, help="default: value stored in refined.json")
    ap.add_argument("--step", type=int, default=3, help="sample every Nth pixel")
    ap.add_argument("--edge", type=float, default=0.9, help="use only the central fraction of each frame")
    ap.add_argument("--thresh", type=float, default=0.5, help="cell counts as obstructed if sky fraction < this")
    ap.add_argument("--margin", type=float, default=2.0, help="safety margin added to both profiles (deg)")
    a = ap.parse_args()
    out = Path(a.outdir); out.mkdir(exist_ok=True, parents=True)
    ref = json.loads(Path(a.refined).read_text())
    decl = a.declination if a.declination is not None else ref["declination"]

    sky, tot, col = accumulate(a.dir, ref, decl, a.step, a.edge, out)
    frac = np.where(tot > 0, sky / np.maximum(tot, 1), np.nan)
    ground_raw, canopy_raw, cov = profiles(frac, a.thresh)
    azs, ground, canopy = finalize(ground_raw, canopy_raw, a.margin)

    at = lambda prof, az0: np.interp(az0 + .5, azs, prof, period=360)
    with open(out / "horizon.csv", "w") as f:
        f.write("az_deg,min_alt_deg,canopy_alt_deg,coverage_cells\n")
        for az0 in range(0, 360, 5):
            f.write(f"{az0},{at(ground, az0):.1f},{at(canopy, az0):.1f},{int(cov[az0:az0 + 5].mean())}\n")
    with open(out / "horizon_stellarium.txt", "w") as f:
        for az0 in range(0, 360, 5):
            f.write(f"{az0} {at(ground, az0):.1f}\n")
    render_mosaic(col, tot, ground, canopy, out / "mosaic.png")

    print("coverage az bins with data:", int((~np.isnan(ground_raw)).sum()), "of 360")
    print("canopy every 15deg:", " ".join(f"{az0}:{at(canopy, az0):.0f}" for az0 in range(0, 360, 15)))
    print("horizon (az:alt) every 15deg:", " ".join(f"{az0}:{at(ground, az0):.0f}" for az0 in range(0, 360, 15)))
    print(f"wrote {out}/horizon.csv, horizon_stellarium.txt, mosaic.png, mask_*.jpg")


if __name__ == "__main__":
    main()
