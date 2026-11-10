# Backyard horizon mapper

Build an **azimuth → minimum usable altitude** table for a backyard (trees, houses, fences) from a
handful of iPhone photos, then use it to decide which astrophotography targets you can actually image
on a given night. Written for smart-telescope users (Seestar S50 / S30 Pro) but nothing here is
telescope-specific except the filter hints in the target picker.

> **iPhone only.** The tilt/roll comes from Apple's `AccelerationVector` maker-note tag, which Android
> phones don't write. Developed and tested with an iPhone 16 Pro (main + ultrawide lenses); other recent
> iPhones that record the tag should work. Expect a few frames per set to lack it; they are skipped.

All image processing runs locally (OpenCV, SciPy, astropy); no cloud services or ML models. The only
network calls are astropy/Sesame name lookups for targets.

No special app is needed. The phone already records, in each HEIC photo's metadata:

| Tag | Used for |
|---|---|
| `GPSImgDirection` (+ ref) | compass heading of the camera |
| Apple `AccelerationVector` | gravity direction → tilt and roll, so altitude is known |
| `FOV` / 35 mm-equivalent focal length | pinhole camera model |
| GPS position, UTC date/time | Sun position for calibration |

## Pipeline

```
                ┌─► horizon-calibrate ─► calib.json   (optional diagnostic: how good is the raw compass?)
photos ─────────┤
                └─► horizon-refine ─► refined.json ─► horizon-build ─► horizon.csv
                    (feature-match overlap,            (sky/obstruction    (az, min_alt,
                     fix per-frame yaw, Sun anchor)     segmentation)        canopy_alt)
                                                                               │
                                       pick-targets / horizon-visibility ◄─────┘
```

Each script reads the photos directly; `calib.json` is not consumed by later steps. To run the three
photo steps in one go:

```bash
./scripts/horizon-map.py PHOTO_DIR --outdir out     # → out/horizon.csv, out/mosaic.png, ...
```

### 1. Capture (≈3 minutes)

- Stand where the telescope sits. Hold the phone still for each shot, no selfie-style arm-swinging.
- Take overlapping photos in a full ring (about 12 shots per ring, ~30° apart) at a **low tilt** (horizon
  level) and again tilted **up ~30°** so the skyline and overhanging branches are both covered.
  Using the ultrawide and main lenses both is fine.
- Do it on a clear-ish **daytime** with a blue sky; the segmentation is colour-based.
- Get the **Sun** in at least 3–4 frames (it is the absolute azimuth reference). Avoid shots with a finger
  or the lens hood in view, and avoid panning during exposure.
- Export as original HEIC (metadata intact). Screenshots, "Live" edits and re-saved JPEGs lose the tags.

### 2. Calibrate (`horizon-calibrate.py`)

```bash
./scripts/horizon-calibrate.py PHOTO_DIR --out calib.json
```

Builds each camera's orientation: world-up from gravity, azimuth from heading plus **magnetic
declination for your site**, pinhole camera from the FOV. Declination is computed from the photos' GPS
position and date with the World Magnetic Model; pass `--declination DEG` (east positive, e.g. from the
NOAA calculator) to override. If a photo's `GPSImgDirectionRef` is already true north, no declination
is added.
Predicts where the Sun should be (astropy) and compares it to the saturated blob in frames where
it appears. Typical result: altitude good to 1–2°; raw compass heading good to about ±6°, with a few
outliers (frames shot mid-swing).

### 3. Refine headings (`horizon-refine.py`)

```bash
./scripts/horizon-refine.py PHOTO_DIR --out refined.json [--drop IMG_0012.HEIC,IMG_0013.HEIC]
```

SIFT feature matching between overlapping frames (RANSAC homographies), then a robust least-squares
solve for one yaw correction per frame and a focal scale per lens, anchored to the Sun frames.
Frames aimed above 40° altitude are dropped for azimuth purposes (heading is ill-defined near the
zenith), as are frames you list with `--drop`. Expect Sun error ≈ 2° afterwards and corrections of
roughly ±10° per frame.

### 4. Build the horizon (`horizon-build.py`)

```bash
./scripts/horizon-build.py PHOTO_DIR refined.json --outdir out
```

Each frame is segmented into sky (blue sky plus bright cloud/glare connected to solid blue) and
obstruction, projected onto 1° azimuth × 1° altitude cells, and reduced to two profiles:

- `min_alt_deg` — **ground-connected** skyline: walk up from the ground while cells are obstructed,
  stop at a ≥5° sky gap. Houses, fences, trunks.
- `canopy_alt_deg` — **conservative**: top of the highest obstructed run ≥4° thick. Also counts
  palm crowns or branches over open sky.

Both include a 2° margin. Outputs: `horizon.csv`, `horizon_stellarium.txt` (two-column `az alt`, usable
in Stellarium/NINA-style horizon files), and a `mosaic.png` diagnostic (this image shows your yard — see
[Privacy](#privacy)).

`horizon.csv` format (azimuth is true north, clockwise, every 5°):

```
az_deg,min_alt_deg,canopy_alt_deg,coverage_cells
```

`coverage_cells` is how many 1° altitude cells (0–89°) had photo data at that azimuth; low values mean
the profile there is interpolated. `example/horizon.csv` is a **synthetic** profile in this format (not
a real yard), for trying the target tools without taking photos.

### 5. Use it

```bash
export OBSERVER_LAT=40.0 OBSERVER_LON=-105.0 OBSERVER_TZ=America/Denver   # your site (both tools)

# rank a ~80-object catalog for a night + suggest a one-target-at-a-time schedule
./scripts/pick-targets.py --horizon horizon.csv --date 2026-11-12
./scripts/pick-targets.py --horizon horizon.csv --filter LP --top 20 --scope s50
./scripts/pick-targets.py --horizon horizon.csv --name "IC 434" --name M45

# check any object (name via Sesame, or name=ra_deg,dec_deg), hour by hour
./scripts/horizon-visibility.py --horizon horizon.csv --date 2026-11-12 M45 "NGC 2359"

# try it without photos
./scripts/pick-targets.py --horizon example/horizon.csv
```

`--date` is the evening's date; without it, "tonight" (or last night, before local noon). Site can also
be given as `--lat/--lon/--tz` (IANA zone name, so daylight-saving changes are handled).

Example output (`example/horizon.csv`, the Denver-ish site above):

```
$ ./scripts/pick-targets.py --horizon example/horizon.csv --date 2026-11-12 --top 5
Night of 2026-11-12: dark 18:20-05:00 (10.8 h), Moon 14% lit

target    type          filt   size  clear h ground h  peak  window (canopy-clear)       moon sep
M45       reflection    IRCUT  110'     10.3     10.3   74°  18:50-05:00                   10°
NGC 1333  reflection    IRCUT    6'     10.0     10.0   80°  18:20-05:00                   14°
M33       galaxy        IRCUT   70'      9.2      9.2   80°  18:20-04:00                   25°
NGC 1499  emission      LP     160'      9.0      9.0   79°  18:30-05:00                   22°
M74       galaxy        IRCUT   10'      9.0      9.0   66°  18:20-03:10                   23°

Suggested schedule (one target at a time):
  18:20-01:30   7.2 h  M74  [IRCUT]
  01:30-05:10   3.7 h  NGC 1499  [LP]
  note: schedule mixes LP and IRCUT; keep them in separate stacks.
```

*clear h* uses the canopy profile minus Moon-affected time; *ground h* uses the ground profile.

- *Clear hours* = astronomically dark **and** above `max(horizon(az), --min-alt 20)` **and** below
  `--max-alt 80` (alt-az mounts struggle near zenith). Canopy profile is used for ranking, ground profile
  is shown alongside.
- Moon penalty applies only while the Moon is up, >30% lit and close (60° broadband, 35° narrowband).
- Schedule: dynamic programming over the top candidates, scarcest window weighted highest, 30-minute
  switching penalty. It suggests **LP** (dual-band) for emission nebulae, supernova remnants and
  planetaries, and **IRCUT** (broadband) for galaxies, clusters and reflection nebulae.
- `--scope s50` hides targets larger than 120′; `--scope s30` hides those larger than 200′ or smaller than 3′.
- Catalog coordinates come from `targets_catalog.json` (resolved once via Sesame). To add targets, edit
  `SEED` in `pick-targets.py` and delete the JSON so it is rebuilt; or use `horizon-visibility.py`, which
  takes any name.

## Install

```bash
brew install exiftool   # or your package manager
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Scripts are `uv` single-file scripts (dependencies declared inline); just run them. The first run of
each downloads its dependencies (OpenCV, astropy, …) into uv's cache.

Tested on macOS.

## Accuracy and limits

- Absolute azimuth is anchored only by the Sun frames, so trust it to a few degrees away from the
  Sun's azimuth. Altitude comes from gravity and is good to ~1–2°.
- It's a snapshot: leaf-on vs leaf-off, new fences or shade structures aren't tracked. Redo after changes.
- Colour-based segmentation: sunlit pale walls vs bright cloud can confuse it; glare near the Sun speckles
  the canopy line. Check `mosaic.png` and the mask overlays before trusting the numbers.
- Frames above 40° altitude don't contribute azimuth, so the overhead canopy is the least certain region.
- A single horizon number per azimuth can't represent floating canopy with clear sky beneath; that's why
  there are two columns.
- Not yet implemented: weighting by integration time you already have on a target; per-scope field-of-view
  matching beyond a simple size cutoff.

## Privacy

Everything this tool reads and writes can identify where you live:

- Photos contain **GPS coordinates**, timestamps and sometimes people.
- `mosaic.png` and any mask overlays are pictures of your property and neighbours'.
- `horizon.csv`, combined with a location or a photo, can fingerprint a specific yard.

Do not commit photos or outputs. `.gitignore` excludes them. The scripts take your latitude, longitude and
time zone from CLI flags or environment variables, never from source.

## License

BSD 3-Clause. See [LICENSE](LICENSE).
