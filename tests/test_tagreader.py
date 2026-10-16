"""Tag-number voting and OCR filtering with a fake OCR engine (no models needed)."""
from PIL import Image

from cattle_id.tagreader import TagReader, vote


class FakeOCR:
    def __init__(self, text, conf=0.9):
        self.out = [[None, text, conf]]

    def __call__(self, img):
        return self.out, None


def crop():
    return Image.new("RGB", (40, 60), "white")


def test_reads_digits_and_ignores_other_text():
    assert TagReader(ocr=FakeOCR("No. 6213"))._prep(crop()) is not None
    assert TagReader(ocr=FakeOCR("No. 6213")).read(crop())[0] == "6213"
    assert TagReader(ocr=FakeOCR("abc")).read(crop()) is None


def test_exact_length_rejects_dropped_or_extra_digit():
    r = lambda t: TagReader(digits=4, ocr=FakeOCR(t)).read(crop())
    assert r("5097")[0] == "5097" and r("15097") is None and r("529") is None


def test_low_confidence_rejected():
    assert TagReader(ocr=FakeOCR("6213", 0.3)).read(crop()) is None


def test_vote_needs_agreement():
    assert vote([("5804", .9), ("5804", .8), ("5604", .95)]) == "5804"
    assert vote([("5804", .9), ("5604", .9)]) is None   # a single read is not trusted
    assert vote([None, None]) is None and vote([]) is None
