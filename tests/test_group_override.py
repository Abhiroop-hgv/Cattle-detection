"""'Register anyway' override and group-photo identification (fake detector / OCR / embedder)."""
from types import SimpleNamespace

from PIL import Image

from cattle_id.config import Config
from cattle_id.index import NumpyIndex
from cattle_id.registry import JsonRegistry
from cattle_id.service import CattleID
from test_service import FakeEmbedder, noisy


class GroupDetector:
    """frame.info['faces'] = [(face box, colour)], frame.info['tags'] = [(tag box, number)]."""
    def faces(self, frame):
        return [(noisy(c, 1)[0].resize((128, 128)), SimpleNamespace(box=b, conf=0.9))
                for b, c in frame.info.get("faces", [])]

    def tags(self, frame, near=None, near_frac=0.6):
        out = []
        for b, n in frame.info.get("tags", []):
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            if near:
                mx, my = (near[2] - near[0]) * near_frac, (near[3] - near[1]) * near_frac
                if not (near[0] - mx <= cx <= near[2] + mx and near[1] - my <= cy <= near[3] + my):
                    continue
            crop = Image.new("RGB", (30, 30))
            crop.info["tag"] = n
            out.append((crop, SimpleNamespace(box=b, conf=0.9)))
        return out


class FakeReader:
    def read(self, crop):
        return (crop.info["tag"], 0.9)


RED, GREEN, BLUE = (220, 20, 20), (20, 220, 20), (20, 20, 220)
F1, F2 = (50, 50, 110, 110), (400, 50, 460, 110)


def make(tmp_path):
    cfg = Config(min_sharpness=0.0, match_threshold=0.999)
    return CattleID(cfg, FakeEmbedder(), NumpyIndex(str(tmp_path / "i.npz")),
                    JsonRegistry(str(tmp_path / "r.json")), detector=GroupDetector(),
                    tag_reader=FakeReader())


def group(faces, tags=()):
    f = Image.new("RGB", (600, 300))
    f.info.update(faces=list(faces), tags=list(tags))
    return f


def test_duplicate_is_rejected_then_can_be_overridden(tmp_path):
    s = make(tmp_path)
    a = s.register(noisy(RED, 4, seed=1), {"name": "A"})["cattle_id"]
    dup = s.register(noisy(RED, 4, seed=2), {"name": "B"})
    assert dup["status"] == "duplicate" and dup["existing_id"] == a
    assert len(s.registry.data) == 1                       # nothing stored on rejection
    ok = s.register(noisy(RED, 4, seed=2), {"name": "B"}, force=True)
    assert ok["status"] == "registered" and ok["overrode_duplicate_of"] == a
    assert s.registry.get(ok["cattle_id"])["confirmed_different_from"] == a


def test_override_never_bypasses_a_reused_tag(tmp_path):
    s = make(tmp_path)
    s.register(noisy(RED, 4), {"name": "A"}, tag_number="1111")
    r = s.register(noisy(GREEN, 4), {"name": "B"}, tag_number="1111", force=True)
    assert r["status"] == "duplicate" and r["reason"] == "tag_already_registered"


def test_group_photo_one_result_per_cow(tmp_path):
    s = make(tmp_path)
    a = s.register(noisy(RED, 4), {"name": "A"}, tag_number="1111")["cattle_id"]
    b = s.register(noisy(GREEN, 4), {"name": "B"}, tag_number="2222")["cattle_id"]
    res = s.identify_frame(group([(F1, RED), (F2, GREEN), ((250, 50, 310, 110), BLUE)]))
    assert [r["status"] for r in res] == ["found", "found", "not_found"]
    assert res[0]["cattle_id"] == a and res[1]["cattle_id"] == b and res[1]["box"] == F2


def test_group_photo_tag_goes_to_the_nearest_face(tmp_path):
    s = make(tmp_path)
    s.register(noisy(RED, 4), {"name": "A"}, tag_number="1111")
    s.register(noisy(GREEN, 4), {"name": "B"}, tag_number="2222")
    tags = [((60, 120, 90, 150), "1111"), ((410, 120, 440, 150), "2222")]   # under each face
    res = s.identify_frame(group([(F1, RED), (F2, GREEN)], tags), use_tags=True)
    assert [r["evidence"] for r in res] == ["face+tag", "face+tag"]
    swapped = [((60, 120, 90, 150), "2222")]                                # A's tag reads as B's
    res = s.identify_frame(group([(F1, RED), (F2, GREEN)], swapped), use_tags=True)
    assert res[0]["status"] == "conflict" and res[1]["evidence"] == "face"
