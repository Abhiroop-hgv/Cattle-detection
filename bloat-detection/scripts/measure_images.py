"""Measure flank width / asymmetry on REAL rear-view cow photos (run in Colab or on a laptop).

NOT TESTED against real photos in the build workspace: it needs the segmentation model
weights. Expect to debug the first run.

Setup (from the bloat-detection folder):
    pip install ultralytics opencv-python-headless
    python scripts/measure_images.py photos/ --out readings.csv --save-masks overlays/

Folder tip: name files so they sort by time, e.g. cow1_2026-10-14_0600.jpg. Take every
photo from the same spot and angle (behind the cow, camera level, cow centred).
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bloat.geometry import Config, measure

COCO_COW = 19  # class id for "cow" in COCO-trained YOLO models


def segment_cow(model, image):
    """Return a boolean mask of the largest cow in the image, or None."""
    result = model(image, classes=[COCO_COW], verbose=False)[0]
    if result.masks is None or len(result.masks.xy) == 0:
        return None
    polygons = [p for p in result.masks.xy if len(p) >= 3]
    if not polygons:
        return None
    biggest = max(polygons, key=lambda p: cv2.contourArea(p.astype(np.float32)))
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [biggest.astype(np.int32)], 1)
    return mask.astype(bool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--out", default="readings.csv")
    ap.add_argument("--weights", default="yolov8n-seg.pt")
    ap.add_argument("--cow-left-is-image-right", action="store_true",
                    help="use when the camera sees the cow from the front or top-down flipped")
    ap.add_argument("--save-masks", default="", help="folder to save mask overlays for checking")
    args = ap.parse_args()

    from ultralytics import YOLO  # imported late so the rest of the project needs no GPU libs
    model = YOLO(args.weights)
    cfg = Config(left_is_image_left=not args.cow_left_is_image_right)
    overlay_dir = Path(args.save_masks) if args.save_masks else None
    if overlay_dir:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(p for p in Path(args.folder).iterdir()
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "valid", "reason", "left_px", "right_px", "hip_width_px",
                    "asymmetry", "flank_to_hip"])
        for p in paths:
            img = cv2.imread(str(p))
            mask = segment_cow(model, img) if img is not None else None
            if mask is None:
                w.writerow([p.name, False, "no cow found", "", "", "", "", ""])
                continue
            m = measure(mask, cfg)
            w.writerow([p.name, m.valid, m.reason, f"{m.left:.1f}", f"{m.right:.1f}",
                        f"{m.hip_width:.1f}", f"{m.asymmetry:.4f}", f"{m.flank_to_hip:.4f}"])
            if overlay_dir:
                shown = img.copy()
                shown[mask] = (0.5 * shown[mask] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)
                cv2.imwrite(str(overlay_dir / p.name), shown)
    print(f"wrote {args.out} for {len(paths)} images")


if __name__ == "__main__":
    main()
