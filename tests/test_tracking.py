"""IdentityTracker with scripted fake frames: no models needed.

A fake frame carries its own script in .info: 'tracks' = [(track_id, body box)], 'faces' =
[(face box, colour)], 'tags' = [(tag box, number)].
"""
from types import SimpleNamespace

from PIL import Image

from cattle_id.config import Config
from cattle_id.index import NumpyIndex
from cattle_id.registry import JsonRegistry
from cattle_id.service import CattleID
from cattle_id.tracking import IdentityTracker, Track, iou
from test_service import FakeEmbedder, noisy


class FakeBodies:
    def update(self, frame):
        return [Track(i, b) for i, b in frame.info["tracks"]]


class FakeDetector:
    def faces(self, frame):
        return [(noisy(c, 1)[0].resize((128, 128)), SimpleNamespace(box=b, conf=0.9))
                for b, c in frame.info.get("faces", [])]

    def tags(self, frame, near=None, near_frac=0.6):
        out = []
        for b, n in frame.info.get("tags", []):
            crop = Image.new("RGB", (30, 30))
            crop.info["tag"] = n
            out.append((crop, SimpleNamespace(box=b, conf=0.9)))
        return out


class FakeReader:
    def read(self, crop):
        return (crop.info["tag"], 0.9)


RED, GREEN, BLUE = (220, 20, 20), (20, 220, 20), (20, 20, 220)
A_BODY, B_BODY = (0, 0, 300, 300), (400, 0, 700, 300)
A_FACE, B_FACE = (100, 20, 160, 80), (500, 20, 560, 80)


def frame(tracks, faces=(), tags=()):
    f = Image.new("RGB", (800, 400))
    f.info.update(tracks=list(tracks), faces=list(faces), tags=list(tags))
    return f


def setup(tmp_path):
    cfg = Config(min_sharpness=0.0, match_threshold=0.999, track_min_crops=2, tag_min_votes=2)
    svc = CattleID(cfg, FakeEmbedder(), NumpyIndex(str(tmp_path / "i.npz")),
                   JsonRegistry(str(tmp_path / "r.json")), detector=FakeDetector(),
                   tag_reader=FakeReader())
    cows = {}
    for name, colour, tag in (("A", RED, "1111"), ("B", GREEN, "2222")):
        cows[name] = svc.register(noisy(colour, 4, seed=3), {"name": name}, tag_number=tag)["cattle_id"]
    return svc, IdentityTracker(svc, FakeBodies(), "cam1"), cows


def run(tr, frames, t0=0.0):
    out = None
    for i, f in enumerate(frames):
        out = tr.update(f, t0 + i)
    return out


def test_identity_needs_evidence_then_agreement(tmp_path):
    svc, tr, cows = setup(tmp_path)
    first = tr.update(frame([(1, A_BODY)]), 0)[0]
    assert first["status"] == "unidentified"
    with_face = lambda: frame([(1, A_BODY)], faces=[(A_FACE, RED)])
    r = run(tr, [with_face() for _ in range(2)], 1)[0]        # one attempt: not trusted yet
    assert r["status"] == "unidentified"
    r = run(tr, [with_face() for _ in range(2)], 3)[0]        # second attempt agrees
    assert r["status"] == "identified" and r["cattle_id"] == cows["A"] and r["evidence"] == "face"
    # identity survives frames where no face is visible
    assert tr.update(frame([(1, A_BODY)]), 10)[0]["cattle_id"] == cows["A"]


def test_face_plus_tag_locks_at_once(tmp_path):
    svc, tr, cows = setup(tmp_path)
    f = lambda: frame([(1, A_BODY)], faces=[(A_FACE, RED)], tags=[((110, 90, 130, 110), "1111")])
    r = run(tr, [f(), f()])[0]
    assert r["status"] == "identified" and r["evidence"] == "face+tag" and r["tag_number"] == "1111"


def test_two_cows_get_their_own_faces(tmp_path):
    svc, tr, cows = setup(tmp_path)
    f = lambda: frame([(1, A_BODY), (2, B_BODY)], faces=[(A_FACE, RED), (B_FACE, GREEN)])
    recs = {r["track_id"]: r for r in run(tr, [f() for _ in range(4)])}
    assert recs[1]["cattle_id"] == cows["A"] and recs[2]["cattle_id"] == cows["B"]


def test_face_outside_every_body_is_ignored(tmp_path):
    svc, tr, cows = setup(tmp_path)
    f = lambda: frame([(1, A_BODY)], faces=[((600, 20, 660, 80), RED)])
    assert run(tr, [f() for _ in range(6)])[0]["status"] == "unidentified"


def test_conflict_between_face_and_tag(tmp_path):
    svc, tr, cows = setup(tmp_path)
    f = lambda: frame([(1, A_BODY)], faces=[(A_FACE, RED)], tags=[((110, 90, 130, 110), "2222")])
    r = run(tr, [f(), f()])[0]
    assert r["status"] == "conflict"
    assert r["face_candidate"]["cattle_id"] == cows["A"] and r["tag_candidate"]["cattle_id"] == cows["B"]


def test_unregistered_cow(tmp_path):
    svc, tr, _ = setup(tmp_path)
    f = lambda: frame([(1, A_BODY)], faces=[(A_FACE, BLUE)])
    r = run(tr, [f() for _ in range(4)])[0]
    assert r["status"] == "unregistered"


def test_identity_at_finds_the_cow_in_a_box_at_a_time(tmp_path):
    svc, tr, cows = setup(tmp_path)
    f = lambda: frame([(1, A_BODY), (2, B_BODY)], faces=[(A_FACE, RED), (B_FACE, GREEN)])
    run(tr, [f() for _ in range(4)])                           # t = 0..3
    assert tr.identity_at((20, 150, 280, 290), 2.2)["cattle_id"] == cows["A"]   # e.g. a lying cow
    assert tr.identity_at((420, 150, 690, 290), 2.9)["cattle_id"] == cows["B"]
    assert tr.identity_at((20, 150, 280, 290), 60)["status"] == "no_cow_at_box"  # wrong time
    assert tr.identity_at((300, 320, 380, 390), 2)["status"] == "no_cow_at_box"  # empty floor


def test_old_tracks_are_forgotten(tmp_path):
    svc, tr, _ = setup(tmp_path)
    tr.update(frame([(1, A_BODY)]), 0)
    tr.update(frame([(2, B_BODY)]), tr.cfg.track_history_s + 5)
    assert tr.identity_of(1)["status"] == "no_such_track" and 2 in tr.states


def test_iou():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0 and iou((0, 0, 1, 1), (5, 5, 6, 6)) == 0.0
