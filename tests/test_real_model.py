"""Smoke test with the real face weights (skipped if timm or the checkpoint is missing).

Checks that the checkpoint loads into Embedder with the default Config and that register() /
identify() work end to end with it. It cannot check recognition accuracy: that needs real cow photos
(see the Results section of the README).
"""
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("timm")
CKPT = Path(__file__).resolve().parents[1] / "checkpoints" / "face_embedder.pt"
pytestmark = pytest.mark.skipif(not CKPT.exists(), reason="face_embedder.pt not present")

from cattle_id.config import Config  # noqa: E402
from cattle_id.embedder import Embedder  # noqa: E402
from cattle_id.index import NumpyIndex  # noqa: E402
from cattle_id.registry import JsonRegistry  # noqa: E402
from cattle_id.service import CattleID  # noqa: E402


def texture(seed, n=4):
    """n slightly different views of one random textured image (stand-in for one animal)."""
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, (16, 16, 3)).astype(np.uint8)
    big = np.asarray(Image.fromarray(base).resize((256, 256), Image.BICUBIC), dtype=np.float32)
    return [Image.fromarray(np.clip(big + rng.normal(0, 6, big.shape), 0, 255).astype(np.uint8))
            for _ in range(n)]


@pytest.fixture(scope="module")
def cfg():
    return Config(checkpoint=str(CKPT), min_sharpness=0.0)


def test_embeddings_shape_and_norm(cfg):
    e = Embedder(cfg, device="cpu").embed(texture(0, 2))
    assert e.shape == (2, 256)
    assert np.allclose(np.linalg.norm(e, axis=1), 1.0, atol=1e-4)


def test_register_identify_end_to_end(cfg, tmp_path):
    svc = CattleID(cfg, Embedder(cfg, device="cpu"), NumpyIndex(str(tmp_path / "i.npz")),
                   JsonRegistry(str(tmp_path / "r.json")))
    a = svc.register(texture(1), {"name": "Gauri", "height_cm": 130})
    assert a["status"] == "registered"

    found = svc.identify(texture(1, 2))
    assert found["status"] == "found" and found["cattle_id"] == a["cattle_id"]
    assert found["details"]["name"] == "Gauri"

    assert svc.register(texture(1, 3), {"name": "again"})["status"] == "duplicate"

    # Unrelated textures all score ~0.93-0.95 against each other with these weights (the model only
    # separates cow faces from other cow faces), so "not found" cannot be tested with fake images
    # against a populated index. A cow-face check must run BEFORE identify()/register().
    empty = CattleID(cfg, svc.embedder, NumpyIndex(), JsonRegistry(str(tmp_path / "r2.json")))
    other = empty.identify(texture(99, 2))
    assert other["status"] == "not_found"
    assert other["message"] == "Cattle not found. Please go and register."
