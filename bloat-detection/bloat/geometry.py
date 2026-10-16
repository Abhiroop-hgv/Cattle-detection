"""Measure flank width and left-right asymmetry from a cow outline (binary mask).

Assumes a REAR view (or top-down view) with the cow's body axis roughly vertical
in the image, like the stage drawings used to describe bloat. Standing behind a
cow and looking the way she faces, her left side is on the image's left.

The band positions below are PLACEHOLDERS. Calibrate them on real photos taken
from your camera position before trusting any number.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Config:
    flank_band: tuple = (0.25, 0.50)   # fraction of cow height, measured from the top
    hip_band: tuple = (0.55, 0.65)     # reference region that bloat does not change
    left_is_image_left: bool = True    # True for a rear view
    min_rows: int = 8                  # rows needed in each band
    border_margin: int = 2             # px: cow touching the edge means she is cut off
    min_height_frac: float = 0.25      # cow must cover at least this share of image height
    percentile: float = 90.0           # robust "widest part" of the flank band


@dataclass(frozen=True)
class Measurement:
    valid: bool
    reason: str = ""
    left: float = 0.0          # left flank half-width (px) from the body midline
    right: float = 0.0
    hip_width: float = 0.0
    asymmetry: float = 0.0     # (left - right) / (left + right); positive = left bigger
    flank_to_hip: float = 0.0  # (left + right) / hip_width; camera distance cancels out
    height_px: int = 0


def _invalid(reason: str) -> Measurement:
    return Measurement(valid=False, reason=reason)


def _row_extents(mask: np.ndarray):
    rows = np.flatnonzero(mask.any(axis=1))
    sub = mask[rows]
    width = mask.shape[1]
    xmin = sub.argmax(axis=1)
    xmax = width - 1 - sub[:, ::-1].argmax(axis=1)
    return rows, xmin, xmax


def measure(mask: np.ndarray, cfg: Config = Config()) -> Measurement:
    """Return flank measurements, or an invalid Measurement saying why not."""
    if mask.ndim != 2:
        raise ValueError("mask must be 2-D")
    mask = mask.astype(bool)
    h, w = mask.shape
    if not mask.any():
        return _invalid("empty mask")

    rows, xmin, xmax = _row_extents(mask)
    top, bottom = int(rows[0]), int(rows[-1])
    height = bottom - top + 1
    if height < cfg.min_height_frac * h:
        return _invalid("cow too small in frame")
    m = cfg.border_margin
    if top <= m or bottom >= h - 1 - m or xmin.min() <= m or xmax.max() >= w - 1 - m:
        return _invalid("cow cut off by image edge")

    rel = (rows - top) / height
    flank = (rel >= cfg.flank_band[0]) & (rel < cfg.flank_band[1])
    hip = (rel >= cfg.hip_band[0]) & (rel < cfg.hip_band[1])
    if flank.sum() < cfg.min_rows or hip.sum() < cfg.min_rows:
        return _invalid("not enough rows in measurement bands")

    midline = float(np.median((xmin[hip] + xmax[hip]) / 2.0))
    to_image_left = midline - xmin[flank]
    to_image_right = xmax[flank] - midline
    if cfg.left_is_image_left:
        left_ext, right_ext = to_image_left, to_image_right
    else:
        left_ext, right_ext = to_image_right, to_image_left

    left = float(np.percentile(left_ext, cfg.percentile))
    right = float(np.percentile(right_ext, cfg.percentile))
    hip_width = float(np.median(xmax[hip] - xmin[hip]))
    if left + right <= 0 or hip_width <= 0:
        return _invalid("degenerate widths")

    return Measurement(
        valid=True,
        left=left,
        right=right,
        hip_width=hip_width,
        asymmetry=(left - right) / (left + right),
        flank_to_hip=(left + right) / hip_width,
        height_px=height,
    )
