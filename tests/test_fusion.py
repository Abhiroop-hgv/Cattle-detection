"""register(tag_number=...) and face + ear-tag fusion, with fake detector / OCR / embedder."""
from types import SimpleNamespace

from PIL import Image

from cattle_id.config import Config
from cattle_id.index import NumpyIndex
from cattle_id.registry import JsonRegistry
from cattle_id.service import CattleID
from test_service import FakeEmbedder, noisy


class FakeDetector:
    """Frame width: 100 = no face, 200 = one face. A frame whose .info has 'tag' shows that tag."""
    def faces(self, image):
        det = SimpleNamespace(box=(0, 0, 10, 10), conf=0.9)
        return [(image.resize((128, 128)), det)] if image.size[0] == 200 else []

    def tags(self, image, near=None, near_frac=0.6):
        if not image.info.get("tag"):
            return []
        crop = Image.new("RGB", (30, 30))
        crop.info["tag"] = image.info["tag"]
        return [(crop, SimpleNamespace(box=(0, 0, 5, 5), conf=0.9))]


class FakeReader:
    def read(self, crop):
        return (crop.info["tag"], 0.9)


RED, GREEN, BLUE = (220, 20, 20), (20, 220, 20), (20, 20, 220)


def frames(color, n, width=200, tag=None, seed=0):
    out = [im.resize((width, 128)) for im in noisy(color, n, seed)]
    for im in out:
        im.info["tag"] = tag
    return out


def make(tmp_path):
    cfg = Config(min_sharpness=0.0, match_threshold=0.999)
    return CattleID(cfg, FakeEmbedder(), NumpyIndex(str(tmp_path / "i.npz")),
                    JsonRegistry(str(tmp_path / "r.json")), detector=FakeDetector(),
                    tag_reader=FakeReader())


def two_cows(tmp_path):
    s = make(tmp_path)
    a = s.register_frames(frames(RED, 4), {"name": "Gauri"}, tag_number="1111")["cattle_id"]
    b = s.register_frames(frames(GREEN, 4, seed=5), {"name": "Nandi"}, tag_number="2222")["cattle_id"]
    return s, a, b


def test_register_stores_tag_and_rejects_reused_tag(tmp_path):
    s, a, _ = two_cows(tmp_path)
    assert s.registry.get(a)["tag_number"] == "1111"
    r = s.register_frames(frames(BLUE, 4, seed=7), {"name": "x"}, tag_number=" 1111 ")
    assert r["status"] == "duplicate" and r["reason"] == "tag_already_registered"
    assert r["existing_id"] == a


def test_face_and_tag_agree(tmp_path):
    s, a, _ = two_cows(tmp_path)
    r = s.identify_frames(frames(RED, 3, tag="1111", seed=1))
    assert r["status"] == "found" and r["cattle_id"] == a and r["evidence"] == "face+tag"


def test_face_and_tag_disagree_is_a_conflict(tmp_path):
    s, a, b = two_cows(tmp_path)
    r = s.identify_frames(frames(RED, 3, tag="2222", seed=1))
    assert r["status"] == "conflict"
    assert r["face_candidate"]["cattle_id"] == a and r["tag_candidate"]["cattle_id"] == b


def test_unregistered_tag_read_does_not_override_face(tmp_path):
    s, a, _ = two_cows(tmp_path)
    r = s.identify_frames(frames(RED, 3, tag="9999", seed=1))
    assert r["status"] == "found" and r["cattle_id"] == a
    assert r["evidence"] == "face" and r["tag_status"] == "tag_read_not_registered"


def test_tag_rescues_unrecognised_face(tmp_path):
    s, a, _ = two_cows(tmp_path)
    r = s.identify_frames(frames(BLUE, 3, tag="1111", seed=1))  # face looks like nobody
    assert r["status"] == "found" and r["cattle_id"] == a and r["evidence"] == "tag"


def test_tag_alone_without_a_face(tmp_path):
    s, a, _ = two_cows(tmp_path)
    r = s.identify_frames(frames(BLUE, 3, width=100, tag="1111", seed=1))
    assert r["status"] == "found" and r["cattle_id"] == a and r["evidence"] == "tag"
    nothing = s.identify_frames(frames(BLUE, 3, width=100, seed=1))
    assert nothing["status"] == "rejected" and nothing["reason"] == "no_cattle_face"


def test_single_frame_tag_read_is_not_trusted(tmp_path):
    s, a, _ = two_cows(tmp_path)
    one = frames(RED, 2, seed=1)
    one[0].info["tag"] = "2222"  # only one frame shows (a possibly wrong) tag
    r = s.identify_frames(one)
    assert r["status"] == "found" and r["cattle_id"] == a and r["evidence"] == "face"


def test_not_found_when_neither_matches(tmp_path):
    s, _, _ = two_cows(tmp_path)
    r = s.identify_frames(frames(BLUE, 3, tag="5555", seed=1))
    assert r["status"] == "not_found" and "register" in r["message"]


def test_register_warns_when_photos_show_a_different_tag(tmp_path):
    s = make(tmp_path)
    r = s.register_frames(frames(RED, 4, tag="1111"), {"name": "Gauri"}, tag_number="1112")
    assert r["status"] == "registered" and r["tag_check"]["read_from_photos"] == "1111"
