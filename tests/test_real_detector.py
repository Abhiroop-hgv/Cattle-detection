"""Real detector + real embedder on real cattle photos (skipped if models or photos are missing).

Photos come from the merged training set built in C:/Users/Admin/cattle_data (validation split), so
this checks the wiring, not generalisation to your cameras.
"""
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("timm")
pytest.importorskip("ultralytics")
ROOT = Path(__file__).resolve().parents[1]
DET, EMB = ROOT / "checkpoints" / "face_tag_detector.pt", ROOT / "checkpoints" / "face_embedder.pt"
PHOTOS = Path("C:/Users/Admin/cattle_data/small/images/val")
pytestmark = pytest.mark.skipif(not (DET.exists() and EMB.exists() and PHOTOS.exists()),
                                reason="models or sample photos not present")

from cattle_id.config import Config  # noqa: E402
from cattle_id.detector import FaceTagDetector  # noqa: E402
from cattle_id.index import NumpyIndex  # noqa: E402
from cattle_id.registry import JsonRegistry  # noqa: E402
from cattle_id.service import CattleID  # noqa: E402


@pytest.fixture(scope="module")
def cfg():
    return Config(checkpoint=str(EMB), face_detector_weights=str(DET), min_sharpness=0.0)


@pytest.fixture(scope="module")
def det(cfg):
    return FaceTagDetector(cfg)


def photos(prefix, n):
    return [Image.open(p).convert("RGB") for p in sorted(PHOTOS.glob(prefix + "*.jpg"))[:n]]


def test_finds_faces_in_real_photos(det):
    ims = photos("lupora_", 20)
    hit = sum(len(det.faces(i)) >= 1 for i in ims)
    assert hit >= 18, f"face found in only {hit}/20 photos"


def test_non_cow_images_have_no_face(det):
    rng = np.random.default_rng(0)
    for arr in (np.zeros((480, 640, 3)), np.full((480, 640, 3), 255),
                rng.integers(0, 255, (480, 640, 3))):
        assert det.faces(Image.fromarray(arr.astype(np.uint8))) == []


def test_pipeline_rejects_non_cow_after_registering(cfg, det, tmp_path):
    svc = CattleID(cfg, index=NumpyIndex(str(tmp_path / "i.npz")),
                   registry=JsonRegistry(str(tmp_path / "r.json")), detector=det)
    reg = svc.register_frames(photos("lupora_", 6), {"name": "test cow"})
    assert reg["status"] in ("registered", "rejected"), reg
    blank = Image.fromarray(np.full((480, 640, 3), 255, np.uint8))
    res = svc.identify_frames([blank])
    assert res["status"] == "rejected" and res["reason"] == "no_cattle_face"
