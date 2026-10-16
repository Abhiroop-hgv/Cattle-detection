"""Fake rear-view cow outlines for testing the measuring code.

This only proves the CODE works and shows how big a swelling must be to stand out
from measurement noise. It says nothing about real bloat appearance.
"""
import numpy as np

FLANK_BAND = (0.25, 0.50)


def _bump(f: float) -> float:
    lo, hi = FLANK_BAND
    if f < lo or f >= hi:
        return 0.0
    return float(np.sin(np.pi * (f - lo) / (hi - lo)))


def synthetic_cow(canvas=(480, 360), scale=1.0, swell_left=0.0, swell_right=0.0,
                  jitter_px=0.0, offset=(0, 0), rng=None) -> np.ndarray:
    """Draw a cow outline. swell_* is the PEAK widening of that flank (0.10 = 10%)."""
    rng = rng or np.random.default_rng(0)
    h, w = canvas
    mask = np.zeros((h, w), dtype=bool)
    cow_h = int(0.9 * h * scale)
    top = (h - cow_h) // 2 + offset[1]
    cx = w // 2 + offset[0]
    body_half = 0.22 * w * scale
    for i in range(cow_h):
        f = i / cow_h
        y = top + i
        if not 0 <= y < h:
            continue
        if f < 0.12:                      # head
            hl = hr = 0.06 * w * scale
            segs = [(cx - hl, cx + hr)]
        elif f < 0.65:                    # body (ellipse)
            u = (f - 0.385) / 0.265
            half = body_half * np.sqrt(max(0.0, 1 - u * u)) + 2
            hl = half * (1 + swell_left * _bump(f))
            hr = half * (1 + swell_right * _bump(f))
            segs = [(cx - hl, cx + hr)]
        else:                             # two legs
            off, lw = 0.12 * w * scale, 0.05 * w * scale
            segs = [(cx - off - lw, cx - off + lw), (cx + off - lw, cx + off + lw)]
        if jitter_px:
            segs = [(a + rng.normal(0, jitter_px), b + rng.normal(0, jitter_px)) for a, b in segs]
        for a, b in segs:
            x0, x1 = int(round(max(0, a))), int(round(min(w - 1, b)))
            if x1 >= x0:
                mask[y, x0:x1 + 1] = True
    return mask
