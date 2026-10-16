"""Carries a cow's identity along its body track, so other models (posture, sound, ...) can ask
"which registered cow is this?" without needing a visible face or tag at that moment.

One IdentityTracker per camera. Each frame: track cow bodies (ByteTrack), find faces and ear tags,
give each to the body track that contains it, and identify a track once it has collected enough
evidence. A track stays "unidentified" until then (and after a track break, until it is identified
again: identity does not jump between tracks or cameras by itself).
"""
import time
from collections import Counter, deque
from dataclasses import dataclass

from .config import Config


@dataclass
class Track:
    track_id: int
    box: tuple[float, float, float, float]  # x0, y0, x1, y1 in frame pixels
    conf: float = 1.0


class UltralyticsBodyTracker:
    """Cow bodies with ByteTrack ids (COCO 'cow' class by default)."""

    def __init__(self, cfg: Config | None = None):
        from ultralytics import YOLO

        self.cfg = cfg or Config()
        self.model = YOLO(self.cfg.detector_weights)

    def update(self, frame) -> list[Track]:
        r = self.model.track(frame.convert("RGB"), persist=True, tracker="bytetrack.yaml",
                             classes=[self.cfg.cow_class_id], conf=self.cfg.detect_conf,
                             verbose=False)[0]
        if r.boxes is None or r.boxes.id is None:
            return []
        return [Track(int(i), tuple(float(v) for v in b), float(c))
                for b, i, c in zip(r.boxes.xyxy.tolist(), r.boxes.id.tolist(), r.boxes.conf.tolist())]


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _owner(point, tracks: list[Track]) -> Track | None:
    """The track whose body box contains `point` (smallest box wins when boxes overlap)."""
    x, y = point
    inside = [t for t in tracks if t.box[0] <= x <= t.box[2] and t.box[1] <= y <= t.box[3]]
    return min(inside, key=lambda t: (t.box[2] - t.box[0]) * (t.box[3] - t.box[1]), default=None)


def _centre(box):
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


class _State:
    def __init__(self, track_id: int, t: float):
        self.track_id, self.first_seen, self.last_seen = track_id, t, t
        self.history: deque = deque()  # (t, box)
        self.faces: list = []          # (det conf, crop): best few since the last attempt
        self.tags: list = []
        self.attempts = 0
        self.votes: Counter = Counter()
        self.not_found = 0
        self.conflict = None
        self.locked = None             # result of a face+tag agreement: final
        self.last_result: dict | None = None


class IdentityTracker:
    def __init__(self, service, body_tracker=None, camera_id: str = "cam", cfg: Config | None = None):
        self.service = service
        self.cfg = cfg or service.cfg
        self.body_tracker = body_tracker
        self.camera_id = camera_id
        self.states: dict[int, _State] = {}

    # ---- per frame -------------------------------------------------------------------------

    def update(self, frame, t: float | None = None) -> list[dict]:
        """Process one frame (PIL image) taken at time t (seconds). Returns one identity record per
        tracked cow in this frame."""
        t = time.time() if t is None else t
        if self.body_tracker is None:
            self.body_tracker = UltralyticsBodyTracker(self.cfg)
        tracks = self.body_tracker.update(frame)
        for tr in tracks:
            st = self.states.setdefault(tr.track_id, _State(tr.track_id, t))
            st.last_seen = t
            st.history.append((t, tr.box))
        self._collect(frame, tracks)
        for tr in tracks:
            self._maybe_identify(self.states[tr.track_id])
        self._prune(t)
        return [self._record(self.states[tr.track_id], tr.box) for tr in tracks]

    def _collect(self, frame, tracks: list[Track]) -> None:
        det = self.service.detector
        for crop, d in det.faces(frame):
            tr = _owner(_centre(d.box), tracks)
            if tr:
                self._keep(self.states[tr.track_id].faces, d.conf, crop)
        if self.cfg.read_tags and hasattr(det, "tags"):
            for crop, d in det.tags(frame):
                tr = _owner(_centre(d.box), tracks)
                if tr:
                    self._keep(self.states[tr.track_id].tags, d.conf, crop)

    def _keep(self, bucket: list, conf: float, crop) -> None:
        bucket.append((conf, crop))
        bucket.sort(key=lambda x: -x[0])
        del bucket[self.cfg.track_max_crops:]

    def _maybe_identify(self, st: _State) -> None:
        if st.locked or st.attempts >= self.cfg.track_max_attempts:
            return
        enough = (len(st.faces) >= self.cfg.track_min_crops
                  or len(st.tags) >= self.cfg.tag_min_votes)
        if not enough:
            return
        res = self.service.identify_evidence([c for _, c in st.faces], [c for _, c in st.tags])
        st.faces, st.tags = [], []  # next attempt needs fresh evidence, so votes stay independent
        st.attempts += 1
        status = res.get("status")
        if status == "found":
            st.last_result = res
            if res.get("evidence") == "face+tag":
                st.locked = res
            else:
                st.votes[res["cattle_id"]] += 1
        elif status == "conflict":
            st.conflict = res
        elif status == "not_found":
            st.not_found += 1

    def _prune(self, now: float) -> None:
        keep = self.cfg.track_history_s
        for tid in [k for k, s in self.states.items() if now - s.last_seen > keep]:
            del self.states[tid]
        for s in self.states.values():
            while s.history and now - s.history[0][0] > keep:
                s.history.popleft()

    # ---- identity records ------------------------------------------------------------------

    def _record(self, st: _State, box) -> dict:
        rec = {"track_id": st.track_id, "camera_id": self.camera_id, "box": tuple(box),
               "first_seen": st.first_seen, "last_seen": st.last_seen}
        if st.locked:
            return {**rec, "status": "identified", "cattle_id": st.locked["cattle_id"],
                    "details": st.locked["details"], "evidence": "face+tag",
                    "tag_number": st.locked.get("tag_number")}
        top = st.votes.most_common(1)
        if top and top[0][1] >= 2:
            cid = top[0][0]
            res = st.last_result if st.last_result and st.last_result["cattle_id"] == cid else None
            return {**rec, "status": "identified", "cattle_id": cid,
                    "details": (res or {}).get("details") or self.service.registry.get(cid),
                    "evidence": (res or {}).get("evidence", "face"),
                    "tag_number": (res or {}).get("tag_number")}
        if st.conflict:
            return {**rec, "status": "conflict", "message": st.conflict["message"],
                    "face_candidate": st.conflict["face_candidate"],
                    "tag_candidate": st.conflict["tag_candidate"]}
        if st.not_found >= 2 and not st.votes:
            return {**rec, "status": "unregistered",
                    "message": "A cow that is not registered (face and tag match nobody)."}
        return {**rec, "status": "unidentified",
                "message": "Unidentified cow (not enough face/tag evidence yet)."}

    def identity_of(self, track_id: int) -> dict:
        """Current identity of a track, or {'status': 'no_such_track'}."""
        st = self.states.get(track_id)
        if not st:
            return {"status": "no_such_track", "track_id": track_id}
        return self._record(st, st.history[-1][1] if st.history else (0, 0, 0, 0))

    def identity_at(self, box, t: float) -> dict:
        """Which cow was in `box` (x0, y0, x1, y1 in this camera's pixels) at time t? For a posture
        or other detection from the SAME camera. The track with the best overlap at that moment
        wins; if none overlaps enough, there was no tracked cow there."""
        best, best_iou = None, self.cfg.track_match_iou
        for st in self.states.values():
            near = min(st.history, key=lambda h: abs(h[0] - t), default=None)
            if near is None or abs(near[0] - t) > self.cfg.track_time_tol_s:
                continue
            o = iou(box, near[1])
            if o >= best_iou:
                best, best_iou = (st, near[1]), o
        if best is None:
            return {"status": "no_cow_at_box", "camera_id": self.camera_id}
        rec = self._record(best[0], best[1])
        rec["overlap"] = round(best_iou, 3)
        return rec
