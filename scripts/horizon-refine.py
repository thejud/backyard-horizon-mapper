#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pillow", "pillow-heif", "astropy", "scipy", "opencv-python-headless"]
# ///
"""
Refine per-frame compass headings by feature-matching overlapping frames and
solving a yaw correction per frame (robust least squares), anchored to the Sun.
Usage: horizon-refine.py <photo_dir> [--out refined.json]
"""
import argparse, importlib.util, json, sys
from pathlib import Path
import numpy as np, cv2
from PIL import Image
import pillow_heif
from scipy.optimize import least_squares

pillow_heif.register_heif_opener()
spec = importlib.util.spec_from_file_location("hc", Path(__file__).with_name("horizon-calibrate.py"))
hc = importlib.util.module_from_spec(spec); spec.loader.exec_module(hc)

WORK = 1600


def load(path):
    img = Image.open(path).convert("RGB")
    s = WORK / max(img.size)
    return img.resize((round(img.size[0] * s), round(img.size[1] * s)))


def rays(pts, size, fpx, basis):
    E, N, U, F, R, Up = basis
    W, Hh = size
    d = F[None] * fpx + R[None] * (pts[:, 0] - W / 2)[:, None] + Up[None] * (Hh / 2 - pts[:, 1])[:, None]
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return np.stack([d @ E, d @ N, d @ U], axis=1)  # ENU unit vectors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir"); ap.add_argument("--declination", type=float, required=True, help="magnetic declination, deg east")
    ap.add_argument("--out", default="refined.json"); ap.add_argument("--max-alt", type=float, default=40); ap.add_argument("--drop", default="", help="comma-separated filenames to exclude (bad headings, finger over lens)"); ap.add_argument("--prior", type=float, default=6.0)
    a = ap.parse_args()
    meta = hc.read_meta(a.dir)
    names = [n for n in sorted(meta) if meta[n].get("AccelerationVector") not in (None, "-") and n.lower().endswith(".heic")]
    # keep frames whose optical axis is below max_alt (azimuth is ill-defined near zenith); drop listed frames
    def centre_alt(n):
        m = meta[n]; b = hc.camera_basis(m, a.declination)
        return float(np.degrees(np.arcsin(rays(np.array([[1.0, 1.0]]), (2, 2), 1e-6, b)[0][2])))
    names = [n for n in names if centre_alt(n) < a.max_alt and n not in [d for d in a.drop.split(",") if d]]
    print("using", len(names), "frames")
    sift = cv2.SIFT_create(nfeatures=5000)
    fr = {}
    for n in names:
        m = meta[n]; img = load(Path(a.dir) / n)
        gray = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY)
        kp, des = sift.detectAndCompute(gray, None)
        fr[n] = dict(m=m, size=img.size, fpx=(max(img.size) / 2) / np.tan(np.radians(float(m["FOV"]) / 2)),
                     kp=np.array([k.pt for k in kp], np.float32), des=des, img=img)
        print(n, len(kp), "kp", flush=True)

    idx = {n: i for i, n in enumerate(names)}
    bf = cv2.BFMatcher()
    matches = []  # (i, j, pts_i, pts_j)
    for ia, na in enumerate(names):
        ba = hc.camera_basis(fr[na]["m"], a.declination)
        fa = -ba[3] @ np.array([0, 0, 0]) if False else None
        for nb in names[ia + 1:]:
            bb = hc.camera_basis(fr[nb]["m"], a.declination)
            # angular separation of optical axes
            fwa = rays(np.array([[fr[na]["size"][0] / 2, fr[na]["size"][1] / 2]]), fr[na]["size"], fr[na]["fpx"], ba)[0]
            fwb = rays(np.array([[fr[nb]["size"][0] / 2, fr[nb]["size"][1] / 2]]), fr[nb]["size"], fr[nb]["fpx"], bb)[0]
            if fwa @ fwb < np.cos(np.radians(75)):
                continue
            raw = bf.knnMatch(fr[na]["des"], fr[nb]["des"], k=2)
            good = [m1 for m1, m2 in (r for r in raw if len(r) == 2) if m1.distance < 0.75 * m2.distance]
            if len(good) < 30:
                continue
            pa = fr[na]["kp"][[g.queryIdx for g in good]]; pb = fr[nb]["kp"][[g.trainIdx for g in good]]
            H, mask = cv2.findHomography(pa, pb, cv2.RANSAC, 4.0)
            if H is None or mask.sum() < 30:
                continue
            mk = mask.ravel().astype(bool)
            matches.append((idx[na], idx[nb], pa[mk], pb[mk]))
            print(f"  {na}-{nb}: {mk.sum()} inliers", flush=True)

    print(f"{len(matches)} matched pairs")
    connected = sorted({i for m in matches for i in m[:2]})
    print("frames in match graph:", len(connected), "of", len(names))

    # Sun anchors
    from astropy.coordinates import EarthLocation
    import astropy.units as u
    first = meta[names[0]]
    loc = EarthLocation(lat=first["GPSLatitude"] * u.deg, lon=first["GPSLongitude"] * u.deg, height=100 * u.m)
    sun = {}
    for n in names:
        sa, sl = hc.sun_azalt(fr[n]["m"], loc)
        sun[n] = (sa, sl)
    sun_vec = lambda sa, sl: np.array([np.sin(np.radians(sa)) * np.cos(np.radians(sl)), np.cos(np.radians(sa)) * np.cos(np.radians(sl)), np.sin(np.radians(sl))])

    # initial sun anchors: blob in frame, accept if within 25deg of predicted az (reject glare/mismatch)
    anchors = []
    for n in names:
        small = fr[n]["img"]
        b = hc.find_sun_blob(small.resize((round(small.size[0] * 1000 / max(small.size)), round(small.size[1] * 1000 / max(small.size)))))
        if not b or b[2] < 200: continue
        sc = max(small.size) / 1000
        anchors.append((idx[n], np.array([[b[0] * sc, b[1] * sc]], float)))
    print("sun candidate frames:", [names[i] for i, _ in anchors])

    nF = len(names)
    ultra = np.array([fr[n]['fpx'] and float(fr[n]['m']['FOV']) > 90 for n in names])

    def basis_for(i, d):
        m = dict(fr[names[i]]["m"]); m["GPSImgDirection"] = m["GPSImgDirection"] + d
        return hc.camera_basis(m, a.declination)

    def resid(x, sunframes):
        d = x[:nF]; r = []
        cache = {}
        kf = x[nF:nF + 2]
        def R(i, pts):
            return rays(pts, fr[names[i]]["size"], fr[names[i]]["fpx"] * (kf[0] if ultra[i] else kf[1]), basis_for(i, d[i]))
        for i, j, pa, pb in matches:
            sub = slice(None, None, max(1, len(pa) // 40))
            r.append((R(i, pa[sub]) - R(j, pb[sub])).ravel() * 20)
        r.append(d / a.prior); r.append((kf - 1) / 0.1)
        for i, pt in sunframes:
            sa, sl = sun[names[i]]
            r.append((R(i, pt)[0] - sun_vec(sa, sl)) * 300)
        return np.concatenate(r)

    # pass 1: no sun; pass 2: add sun anchors that agree (< 15 deg) with pass-1 solution
    x0 = np.concatenate([np.zeros(nF), [1.0, 1.0]])
    sol = least_squares(resid, x0, args=([],), loss="soft_l1", f_scale=0.5)
    keep = []
    for i, pt in anchors:
        kf = sol.x[nF:nF + 2]
        r = rays(pt, fr[names[i]]["size"], fr[names[i]]["fpx"] * (kf[0] if ultra[i] else kf[1]), basis_for(i, sol.x[i]))[0]
        sa, sl = sun[names[i]]
        err = np.degrees(np.arccos(np.clip(r @ sun_vec(sa, sl), -1, 1)))
        print(f"  sun cand {names[i]}: angular err {err:.1f} deg after matching-only solve")
        if err < 20: keep.append((i, pt))
    sol = least_squares(resid, sol.x, args=(keep,), loss="soft_l1", f_scale=0.5)
    print("anchors used:", [names[i] for i, _ in keep])

    print('focal scale ultra/main:', sol.x[nF:nF+2])
    out = {}
    for n in names:
        i = idx[n]
        out[n] = dict(yaw_corr=float(sol.x[i]), heading_true=float((fr[n]["m"]["GPSImgDirection"] + a.declination + sol.x[i]) % 360), in_graph=i in connected)
    print("\nyaw corrections (deg):")
    print(" ".join(f"{n[4:8]}:{out[n]['yaw_corr']:+.1f}" for n in names))
    # sun residuals after refinement
    errs = []
    for i, pt in anchors:
        kf = sol.x[nF:nF + 2]
        r = rays(pt, fr[names[i]]["size"], fr[names[i]]["fpx"] * (kf[0] if ultra[i] else kf[1]), basis_for(i, sol.x[i]))[0]
        sa, sl = sun[names[i]]
        e = np.degrees(np.arccos(np.clip(r @ sun_vec(sa, sl), -1, 1)))
        errs.append((names[i], e))
    print("Sun angular error after refinement:", ", ".join(f"{n[4:8]}={e:.1f}" for n, e in errs))
    Path(a.out).write_text(json.dumps(dict(declination=a.declination, kf=list(sol.x[nF:nF+2]), frames=out), indent=1))

if __name__ == "__main__":
    main()
