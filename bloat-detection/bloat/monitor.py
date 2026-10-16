"""Turn flank measurements into ok / watch / urgent / unknown for one cow.

All thresholds are STARTING GUESSES. They have not been validated on real bloat
cases. This tool flags a possible problem for a human to check; it never diagnoses.
"""
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

from bloat.geometry import Measurement
from bloat.baseline import CowBaseline

OK, WATCH, URGENT, UNKNOWN = "ok", "watch", "urgent", "unknown"
METRICS = ("asymmetry", "flank_to_hip")


@dataclass
class Burst:
    valid: bool
    metrics: dict = field(default_factory=dict)
    n_valid: int = 0
    reason: str = ""


def summarize_burst(measurements: List[Measurement], min_valid: int = 3) -> Burst:
    """Combine a few frames taken seconds apart into ONE reading (median per metric)."""
    good = [m for m in measurements if m.valid]
    if len(good) < min_valid:
        return Burst(False, n_valid=len(good), reason=f"only {len(good)} usable frames")
    return Burst(True, {k: float(np.median([getattr(m, k) for m in good])) for k in METRICS},
                 n_valid=len(good))


@dataclass
class MonitorConfig:
    z_watch: float = 3.5
    z_urgent: float = 6.0
    abs_asym_watch: float = 0.06    # used only while a cow has no baseline yet
    abs_asym_urgent: float = 0.12
    confirm: int = 3                # consecutive flagged readings needed for the "rising" rule


@dataclass
class Assessment:
    level: str
    reasons: List[str]
    score: Optional[float] = None


class BloatMonitor:
    def __init__(self, baseline: CowBaseline, cfg: MonitorConfig = MonitorConfig(), learn: bool = True):
        self.baseline = baseline
        self.cfg = cfg
        self.learn = learn
        self._history: List[float] = []   # recent scores (z-scores, or asymmetry w/o baseline)

    def _score(self, hour: float, metrics: dict):
        zs = [self.baseline.zscore(hour, k, metrics[k]) for k in METRICS]
        zs = [z for z in zs if z is not None]
        if zs:
            return max(zs), True
        return metrics["asymmetry"], False

    def update(self, hour: float, burst: Burst) -> Assessment:
        if not burst.valid:
            return Assessment(UNKNOWN, [f"cannot assess: {burst.reason}"])
        c = self.cfg
        score, has_baseline = self._score(hour, burst.metrics)
        watch_at, urgent_at = (c.z_watch, c.z_urgent) if has_baseline else (c.abs_asym_watch, c.abs_asym_urgent)
        self._history.append(score)
        self._history = self._history[-c.confirm:]
        reasons, level = [], OK

        flagged = score >= watch_at
        h = self._history
        strong_twice = len(h) >= 2 and h[-1] >= urgent_at and h[-2] >= urgent_at
        rising = (len(h) >= c.confirm and all(x >= watch_at for x in h)
                  and all(b > a for a, b in zip(h, h[1:])))

        if strong_twice:
            level = URGENT
            reasons.append("flank far outside normal on two readings in a row")
        elif rising:
            level = URGENT
            reasons.append(f"flagged {c.confirm} times in a row and rising each time")
        elif flagged:
            level = WATCH
            reasons.append("flank wider or more uneven than normal")
        if not has_baseline and level != OK:
            reasons.append("no baseline for this cow yet, using fixed limits")

        if self.learn and level == OK:
            self.baseline.add(hour, burst.metrics)
        return Assessment(level, reasons, score)
