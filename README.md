# Backyard horizon mapper

Build an **azimuth → minimum usable altitude** table for a backyard (trees, houses, fences) from a
handful of iPhone photos, then use it to decide which astrophotography targets you can actually image
on a given night. Written for smart-telescope users (Seestar S50 / S30 Pro) but nothing here is
telescope-specific except the filter hints in the target picker.

No special app is needed. The phone already records, in each HEIC photo's metadata:

| Tag | Used for |
|---|---|
| `GPSImgDirection` (+ ref) | compass heading of the camera |
| Apple `AccelerationVector` | gravity direction → tilt and roll, so altitude is known |
| `FOV` / 35 mm-equivalent focal length | pinhole camera model |
| GPS position, UTC date/time | Sun position for calibration |

## Pipeline

```
photos ──► horizon-calibrate ──► horizon-refine ──► horizon-build ──► horizon.csv
          (orientation + Sun      (feature-match        (sky/obstruction     (az, min_alt,
           check)                  overlap, fix yaw)     segmentation)         canopy_alt)
                                                                              │
                                      pick-targets / horizon-visibility ◄─────┘
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
./scripts/horizon-calibrate.py PHOTO_DIR --declination 12.9 --out calib.json
```

Builds each camera's orientation: world-up from gravity, azimuth from heading plus **magnetic
declination for your site** (use the NOAA calculator; east is positive), pinhole camera from the FOV.
Predicts where the Sun should be (astropy) and compares it to the saturated blob in frames where
it appears. Typical result: altitude good to 1–2°; raw compass heading good to about ±6°, with a few
outliers (frames shot mid-swing).

### 3. Refine headings (`horizon-refine.py`)

```bash
./scripts/horizon-refine.py PHOTO_DIR --declination 12.9 --out refined.json
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

`horizon.csv` format (azimuth is true north, clockwise, every 5°; see `example/horizon.csv`):

```
az_deg,min_alt_deg,canopy_alt_deg,coverage_cells
```

### 5. Use it

```bash
export OBSERVER_LAT=40.0 OBSERVER_LON=-105.0 OBSERVER_TZ=America/Denver   # your site

# rank a ~80-object catalog for a night + suggest a one-target-at-a-time schedule
./scripts/pick-targets.py --horizon horizon.csv --date 2026-11-12
./scripts/pick-targets.py --horizon horizon.csv --filter LP --top 20
./scripts/pick-targets.py --horizon horizon.csv --name "IC 434" --name M45

# check specific objects (name via Sesame, or name=ra_deg,dec_deg)
./scripts/horizon-visibility.py --lat 40 --lon -105 --tz -7 --horizon horizon.csv --date 2026-11-12 M45
```

- *Clear hours* = astronomically dark **and** above `max(horizon(az), --min-alt 20)` **and** below
  `--max-alt 80` (alt-az mounts struggle near zenith). Canopy profile is used for ranking, ground profile
  is shown alongside.
- Moon penalty applies only while the Moon is up, >30% lit and close (60° broadband, 35° narrowband).
- Schedule: dynamic programming over the top candidates, scarcest window weighted highest, 30-minute
  switching penalty. It suggests **LP** (dual-band) for emission nebulae, supernova remnants and
  planetaries, and **IRCUT** (broadband) for galaxies, clusters and reflection nebulae.
- Catalog coordinates come from `targets_catalog.json` (resolved once via Sesame).

## Install

```bash
brew install exiftool   # or your package manager
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Scripts are `uv` single-file scripts (dependencies declared inline); just run them.

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
