"""Per-cow baseline: what is NORMAL for this cow at this time of day.

Uses median and MAD (robust to the odd bad frame). Buckets by hour of day because a
cow's left side is normally fuller after feeding.
"""
import numpy as np


class CowBaseline:
    def __init__(self, bucket_hours: int = 3, min_samples: int = 5, min_sigma: float = 0.01):
        self.bucket_hours = bucket_hours
        self.min_samples = min_samples
        self.min_sigma = min_sigma
        self._data = {}  # bucket -> metric -> list[float]

    def _bucket(self, hour: float) -> int:
        return int((hour % 24) // self.bucket_hours)

    def add(self, hour: float, metrics: dict) -> None:
        b = self._data.setdefault(self._bucket(hour), {})
        for name, value in metrics.items():
            b.setdefault(name, []).append(float(value))

    def _samples(self, hour: float, name: str):
        own = self._data.get(self._bucket(hour), {}).get(name, [])
        if len(own) >= self.min_samples:
            return own
        pooled = [v for bucket in self._data.values() for v in bucket.get(name, [])]
        return pooled if len(pooled) >= 2 * self.min_samples else None

    def ready(self, hour: float, name: str) -> bool:
        return self._samples(hour, name) is not None

    def zscore(self, hour: float, name: str, value: float):
        """Robust z-score, or None when this cow has too little history."""
        s = self._samples(hour, name)
        if s is None:
            return None
        arr = np.asarray(s)
        med = float(np.median(arr))
        sigma = max(1.4826 * float(np.median(np.abs(arr - med))), self.min_sigma)
        return (float(value) - med) / sigma
