"""How big must a (simulated) left-flank swelling be before the system notices it?

SIMULATION ONLY. Fake outlines show whether the code and thresholds behave sensibly
and how much measurement noise matters. They do NOT show how real bloat looks.

Usage: python3 scripts/sweep_detectability.py [--jitter 1.5] [--tests 120]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bloat.baseline import CowBaseline
from bloat.geometry import measure
from bloat.monitor import MonitorConfig, summarize_burst
from bloat.synthetic import synthetic_cow

HOUR = 18.0
FRAMES_PER_BURST = 5


def random_burst(rng, jitter, swell_left=0.0):
    """A few frames of the same cow, with random distance and position per burst."""
    scale = rng.uniform(0.85, 1.1)
    off = (int(rng.integers(-15, 16)), int(rng.integers(-10, 11)))
    frames = [measure(synthetic_cow(scale=scale, swell_left=swell_left, jitter_px=jitter,
                                    offset=off, rng=rng)) for _ in range(FRAMES_PER_BURST)]
    return summarize_burst(frames)


def flagged(base, burst, z_watch):
    zs = [base.zscore(HOUR, k, burst.metrics[k]) for k in ("asymmetry", "flank_to_hip")]
    return max(zs) >= z_watch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jitter", type=float, default=1.5, help="outline noise in pixels")
    ap.add_argument("--baseline-bursts", type=int, default=40)
    ap.add_argument("--tests", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    base = CowBaseline()
    for _ in range(args.baseline_bursts):
        b = random_burst(rng, args.jitter)
        if b.valid:
            base.add(HOUR, b.metrics)

    z_watch = MonitorConfig().z_watch
    print(f"outline noise {args.jitter}px, baseline of {args.baseline_bursts} bursts, "
          f"watch threshold z>={z_watch}")
    print("peak left-flank swelling | flagged (of assessable readings) | assessable")
    for s in (0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20):
        bursts = [random_burst(rng, args.jitter, s) for _ in range(args.tests)]
        ok = [b for b in bursts if b.valid]
        hits = sum(flagged(base, b, z_watch) for b in ok)
        label = "  <- false-alarm rate" if s == 0 else ""
        print(f"{s*100:5.0f}%                   | {hits / max(len(ok), 1):6.1%}"
              f"                          | {len(ok) / args.tests:5.0%}{label}")


if __name__ == "__main__":
    main()
