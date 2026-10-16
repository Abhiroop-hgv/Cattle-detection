# bloat-detection: flank-geometry bloat (ruminal tympany) screening

No public bloat image dataset exists, so v1 does not train a model. It segments the cow,
measures the left and right flank width and the flank-to-hip ratio from a consistent rear
(or top-down) view, compares each reading with that cow's OWN normal for the same time of
day, and alerts on a sustained or rising change.

| File | What it does |
|---|---|
| `bloat/geometry.py` | Cow outline (mask) -> asymmetry and flank-to-hip ratio, with "can't assess" checks |
| `bloat/baseline.py` | Per-cow normal range by time of day (median and MAD) |
| `bloat/monitor.py` | Averages a short burst of frames into one reading, then ok / watch / urgent / unknown |
| `bloat/synthetic.py` | Fake cow outlines with adjustable swelling, for tests only |
| `scripts/sweep_detectability.py` | Simulation: how big a swelling stands out from noise |
| `scripts/measure_images.py` | Run on real photos (needs YOLO segmentation weights) |
| `tests/test_bloat.py` | 13 tests for geometry, baseline and monitor logic |

## Run

```bash
cd bloat-detection
pip install numpy pytest
python3 -m pytest -q tests
python3 scripts/sweep_detectability.py --jitter 1.5

# real photos (also needs: pip install ultralytics opencv-python-headless)
python3 scripts/measure_images.py photos/ --out readings.csv --save-masks overlays/
```

## Alert rules (in `bloat/monitor.py`)
- A reading is the median of a burst of frames; fewer than 3 usable frames means "unknown", never an alert.
- "watch": the reading is above the watch threshold for this cow's own baseline.
- "urgent": two strongly abnormal readings in a row, or three flagged readings that keep rising.
- Abnormal readings are not added to the cow's baseline.

## What this does NOT prove
- Thresholds are guesses and the measurement bands are placeholders. Nothing was validated on real bloat.
- The simulation uses fake outlines. It shows the code behaves and how noise limits sensitivity, not how real bloat looks.
- Normal fullness after feeding, pregnancy, posture and bad outlines can all change flank width.
- It flags a possible problem for a human to check. It never diagnoses. Bloat is an emergency, so a person should call a vet.
- Next evidence to gather: several days of normal readings from the same cow (real false-alarm rate), and a few vet-confirmed bloat cases.
