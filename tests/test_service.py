"""Logic tests with a fake embedder (no model download, no GPU)."""
import numpy as np
from PIL import Image

from cattle_id.config import Config
from cattle_id.index import NumpyIndex
from cattle_id.registry import JsonRegistry
from cattle_id.service import CattleID


class FakeEmbedder:
    """Embeds an image as a unit vector derived from its mean colour -> 'cow identity = colour'."""
    def embed(self, images):
        out = []
        for im in images:
            r, g, b = np.asarray(im.convert("RGB"), dtype=np.float32).mean((0, 1))
            v = np.array([r, g, b, 1.0], dtype=np.float32)
            out.append(v / np.linalg.norm(v))
        return np.stack(out)


def noisy(color, n, seed=0):
    rng = np.random.default_rng(seed)
    imgs = []
    for _ in range(n):
        a = np.clip(np.array(color) + rng.normal(0, 25, (128, 128, 3)), 0, 255).astype(np.uint8)
        imgs.append(Image.fromarray(a))
    return imgs


def make(tmp_path):
    cfg = Config(min_sharpness=0.0, match_threshold=0.999)
    return CattleID(cfg, FakeEmbedder(), NumpyIndex(str(tmp_path / "i.npz")),
                    JsonRegistry(str(tmp_path / "r.json")))


def test_register_identify_duplicate_notfound(tmp_path):
    sys = make(tmp_path)
    red, green = (220, 20, 20), (20, 220, 20)

    r = sys.register(noisy(red, 4), {"name": "Gauri", "height_cm": 130})
    assert r["status"] == "registered"

    f = sys.identify(noisy(red, 2, seed=1))
    assert f["status"] == "found" and f["cattle_id"] == r["cattle_id"]
    assert f["details"]["name"] == "Gauri"

    assert sys.register(noisy(red, 3, seed=2), {"name": "x"})["status"] == "duplicate"

    nf = sys.identify(noisy(green, 2))
    assert nf["status"] == "not_found" and "register" in nf["message"]


def test_rejects_too_few_images(tmp_path):
    sys = make(tmp_path)
    assert sys.register(noisy((200, 0, 0), 1), {})["status"] == "rejected"


def test_persistence(tmp_path):
    a = make(tmp_path)
    cid = a.register(noisy((220, 20, 20), 4), {"name": "Gauri"})["cattle_id"]
    b = make(tmp_path)  # fresh instance reads saved files
    assert b.identify(noisy((220, 20, 20), 2, seed=5))["cattle_id"] == cid
