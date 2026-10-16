import numpy as np
from PIL import Image


def sharpness(img: Image.Image) -> float:
    """Variance of a Laplacian filter; low = blurry."""
    g = np.asarray(img.convert("L"), dtype=np.float32)
    lap = (
        -4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
    )
    return float(lap.var())


def check(img: Image.Image, min_side: int, min_sharpness: float) -> str | None:
    """Return None if OK, else a rejection reason."""
    if min(img.size) < min_side:
        return "too_small"
    if sharpness(img) < min_sharpness:
        return "blurry"
    return None
