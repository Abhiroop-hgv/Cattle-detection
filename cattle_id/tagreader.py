"""Reads the printed number on an ear tag (OCR) and votes over several frames of one animal."""
import re
from collections import Counter

import numpy as np
from PIL import Image


class TagReader:
    """OCR on a tag crop. Tries the crop upright and rotated, because tags hang at any angle."""

    def __init__(self, digits: int | None = None, min_digits: int = 3, max_digits: int = 6,
                 min_conf: float = 0.5, ocr=None):
        """digits: exact number length on your farm's tags (e.g. 4); rejects reads of any other
        length, which is where most single-frame OCR mistakes (dropped/added digit) land."""
        if ocr is None:
            from rapidocr_onnxruntime import RapidOCR
            ocr = RapidOCR()
        self.ocr = ocr
        self.min_digits, self.max_digits = (digits, digits) if digits else (min_digits, max_digits)
        self.min_conf = min_conf

    def _prep(self, crop: Image.Image) -> np.ndarray:
        w, h = crop.size
        s = max(1.0, 96 / min(w, h))  # OCR needs digits at least ~30 px tall
        crop = crop.convert("RGB").resize((int(w * s), int(h * s)), Image.LANCZOS)
        pad = 12
        out = Image.new("RGB", (crop.width + 2 * pad, crop.height + 2 * pad), (255, 255, 255))
        out.paste(crop, (pad, pad))
        return np.asarray(out)[:, :, ::-1]  # BGR

    def read(self, crop: Image.Image) -> tuple[str, float] | None:
        """Best digit string and its confidence, or None if nothing plausible is found."""
        best = None
        for rot in (0, 90, 270, 180):
            img = self._prep(crop.rotate(rot, expand=True))
            res, _ = self.ocr(img)
            for _, text, conf in res or []:
                digits = re.sub(r"\D", "", text)
                conf = float(conf)
                if (self.min_digits <= len(digits) <= self.max_digits and conf >= self.min_conf
                        and (best is None or conf > best[1])):
                    best = (digits, conf)
        return best


def vote(readings: list[tuple[str, float] | None], min_votes: int = 2) -> str | None:
    """Majority vote over per-frame reads of the same tag: a number is accepted only when at
    least `min_votes` frames agree on it."""
    c = Counter()
    for r in readings:
        if r:
            c[r[0]] += r[1]
    counts = Counter(r[0] for r in readings if r)
    if not c:
        return None
    num, _ = c.most_common(1)[0]
    return num if counts[num] >= min_votes else None
