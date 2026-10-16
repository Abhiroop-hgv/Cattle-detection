"""Frame-level entry points with a fake detector and fake embedder (no models needed)."""
from types import SimpleNamespace

import numpy as np
from PIL import Image

from cattle_id.config import Config
from cattle_id.index import NumpyIndex
from cattle_id.registry import JsonRegistry
from cattle_id.service import CattleID
from test_service import FakeEmbedder, noisy


class FakeDetector:
    """Frame width decides what is in it: 100 = no cow face, 200 = one face, 300 = two faces."""
    def faces(self, image):
        n = {100: 0, 200: 1, 300: 2}[image.size[0]]
        det = SimpleNamespace(box=(0, 0, 10, 10), conf=0.9)
        return [(image.resize((128, 128)), det) for _ in range(n)]


def frames(color, n, width, seed=0):
    return [im.resize((width, 128)) for im in noisy(color, n, seed)]


def make(tmp_path):
    cfg = Config(min_sharpness=0.0, match_threshold=0.999)
    return CattleID(cfg, FakeEmbedder(), NumpyIndex(str(tmp_path / "i.npz")),
                    JsonRegistry(str(tmp_path / "r.json")), detector=FakeDetector())


RED, GREEN = (220, 20, 20), (20, 220, 20)


def test_register_and_identify_from_frames(tmp_path):
    s = make(tmp_path)
    r = s.register_frames(frames(RED, 4, 200), {"name": "Gauri"})
    assert r["status"] == "registered"
    f = s.identify_frames(frames(RED, 2, 200, seed=1))
    assert f["status"] == "found" and f["details"]["name"] == "Gauri"


def test_no_face_is_rejected_not_matched(tmp_path):
    s = make(tmp_path)
    s.register_frames(frames(RED, 4, 200), {"name": "Gauri"})
    r = s.identify_frames(frames(RED, 3, 100))  # same colour as a registered cow, but no face
    assert r["status"] == "rejected" and r["reason"] == "no_cattle_face"
    assert s.register_frames(frames(GREEN, 4, 100), {})["status"] == "rejected"


def test_multiple_faces_frames_skipped(tmp_path):
    s = make(tmp_path)
    mixed = frames(RED, 4, 200) + frames(RED, 2, 300, seed=3)
    r = s.register_frames(mixed, {"name": "Gauri"})
    assert r["status"] == "registered" and r["skipped_frames"].count("multiple_faces") == 2


def test_identify_frame_returns_one_result_per_face(tmp_path):
    s = make(tmp_path)
    s.register_frames(frames(RED, 4, 200), {"name": "Gauri"})
    res = s.identify_frame(frames(RED, 1, 300)[0])
    assert len(res) == 2 and all(r["status"] == "found" and "box" in r for r in res)
    assert s.identify_frame(frames(RED, 1, 100)[0]) == []
