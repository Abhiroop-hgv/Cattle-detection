"""The two functions the integrated app calls: register() and identify()."""
import re
import uuid

import numpy as np
from PIL import Image

from . import quality
from .config import Config
from .index import NumpyIndex, VectorIndex
from .registry import JsonRegistry, Registry

NOT_FOUND_MSG = "Cattle not found. Please go and register."


class CattleID:
    def __init__(self, cfg: Config | None = None, embedder=None,
                 index: VectorIndex | None = None, registry: Registry | None = None,
                 detector=None, tag_reader=None):
        self.cfg = cfg or Config()
        if embedder is None:
            from .embedder import Embedder
            embedder = Embedder(self.cfg)
        self.embedder = embedder
        self._detector = detector  # built lazily: only needed by the *_frame(s) methods
        self._tag_reader = tag_reader  # built lazily (OCR); cfg.read_tags=False switches it off
        self.index = index or NumpyIndex(self.cfg.index_path)
        self.registry = registry or JsonRegistry(self.cfg.registry_path)

    def _good_embeddings(self, crops: list[Image.Image]) -> tuple[np.ndarray | None, list[str]]:
        keep, reasons = [], []
        for c in crops:
            r = quality.check(c, self.cfg.min_crop_side, self.cfg.min_sharpness)
            (reasons.append(r) if r else keep.append(c))
        return (self.embedder.embed(keep) if keep else None), reasons

    def _match(self, embs: np.ndarray) -> tuple[str, float] | None:
        """Aggregate several views of one animal into one query; return best (id, score)."""
        query = embs.mean(axis=0)
        hits = self.index.search(query)
        return hits[0] if hits else None

    def identify(self, crops: list[Image.Image]) -> dict:
        """crops: one or more images of ONE animal (e.g. a tracked cow across frames)."""
        embs, reasons = self._good_embeddings(crops)
        if embs is None:
            return {"status": "rejected", "reason": reasons[0] if reasons else "no_images"}
        hit = self._match(embs)
        if hit and hit[1] >= self.cfg.match_threshold:
            cid, score = hit
            return {"status": "found", "cattle_id": cid, "score": score,
                    "details": self.registry.get(cid)}
        return {"status": "not_found", "message": NOT_FOUND_MSG,
                "best_score": hit[1] if hit else None}

    # ---- ear tag --------------------------------------------------------------------------

    @staticmethod
    def _norm_tag(tag) -> str | None:
        digits = re.sub(r"\D", "", str(tag)) if tag is not None else ""
        return digits or None

    def _find_by_tag(self, tag: str | None) -> str | None:
        find = getattr(self.registry, "find_by_tag", None)
        return find(tag) if (tag and find) else None

    @property
    def tag_reader(self):
        if self._tag_reader is None:
            from .tagreader import TagReader
            self._tag_reader = TagReader(digits=self.cfg.tag_digits)
        return self._tag_reader

    def _read_tag(self, tag_crops: list[Image.Image], min_votes: int | None = None) -> str | None:
        """Number printed on the tag, if enough frames agree; None if unreadable or no OCR."""
        if not (self.cfg.read_tags and tag_crops):
            return None
        from .tagreader import vote
        try:
            reads = [self.tag_reader.read(c) for c in tag_crops[: self.cfg.tag_max_reads]]
        except ImportError:  # OCR package not installed: behave as if no tag was readable
            return None
        return vote(reads, self.cfg.tag_min_votes if min_votes is None else min_votes)

    # ---- full-frame entry points: find the face (and tag) first, then identify/register ----

    @property
    def detector(self):
        if self._detector is None:
            from .detector import FaceTagDetector
            self._detector = FaceTagDetector(self.cfg)
        return self._detector

    def _analyse(self, frames: list[Image.Image]) -> tuple[list[Image.Image], list[str], list[Image.Image]]:
        """Face crops (one per frame; frames with no or several faces are skipped, because a
        frame with several animals cannot be assigned to ONE cow), skip reasons, and the ear-tag
        crop of each frame (a tag belongs to the face in that frame, or to the only tag if there
        is no face)."""
        crops, skipped, tag_crops = [], [], []
        tags_of = getattr(self.detector, "tags", None) if self.cfg.read_tags else None
        for f in frames:
            faces = self.detector.faces(f)
            if len(faces) > 1:
                skipped.append("multiple_faces")
                continue
            tags = tags_of(f, near=faces[0][1].box if faces else None,
                           near_frac=self.cfg.tag_near_face) if tags_of else []
            if tags and (faces or len(tags) == 1):
                tag_crops.append(tags[0][0])
            if faces:
                crops.append(faces[0][0])
            else:
                skipped.append("no_cattle_face")
        return crops, skipped, tag_crops

    def _fuse(self, face: dict | None, tag: str | None) -> dict:
        """Combine the face match with the tag number read from the same animal.

        face + tag agree -> found (evidence face+tag); face and tag name different cattle ->
        conflict (a person should look); face only / tag only -> found on that evidence alone."""
        tag_cid = self._find_by_tag(tag)
        face_cid = face["cattle_id"] if face and face["status"] == "found" else None
        out = dict(face) if face else {"status": "not_found", "message": NOT_FOUND_MSG,
                                       "best_score": None}
        out["tag_number"] = tag
        if face_cid and tag_cid and face_cid != tag_cid:
            return {"status": "conflict", "tag_number": tag,
                    "message": "Face and ear tag point to different cattle. Please check this animal.",
                    "face_candidate": {"cattle_id": face_cid, "score": face["score"],
                                       "details": face["details"]},
                    "tag_candidate": {"cattle_id": tag_cid, "details": self.registry.get(tag_cid)}}
        if face_cid:
            out["evidence"] = "face+tag" if tag_cid else "face"
            if tag and not tag_cid:
                out["tag_status"] = "tag_read_not_registered"  # often an OCR slip: not a conflict
            return out
        if tag_cid:  # face missing or below threshold, but the tag names a registered animal
            return {"status": "found", "cattle_id": tag_cid, "score": None, "evidence": "tag",
                    "tag_number": tag, "details": self.registry.get(tag_cid),
                    "face_best_score": out.get("best_score"),
                    "note": "Matched by ear tag only; the face did not match."}
        return out

    def identify_frames(self, frames: list[Image.Image]) -> dict:
        """frames: full frames/photos showing ONE animal (e.g. several frames of one cow). Uses the
        face and, when readable, the ear-tag number (see _fuse for the rules)."""
        crops, skipped, tag_crops = self._analyse(frames)
        res = self.identify_evidence(crops, tag_crops)
        if res.get("reason") == "no_evidence":
            return {"status": "rejected", "reason": skipped[0] if skipped else "no_images",
                    "message": "No single cattle face found in the image(s).", "skipped": skipped}
        res["skipped"] = skipped
        return res

    def identify_evidence(self, face_crops: list[Image.Image], tag_crops: list[Image.Image]) -> dict:
        """Face crops and ear-tag crops already known to belong to ONE animal (e.g. collected along
        a tracked cow): the fused identification used by identify_frames() and the tracker."""
        tag = self._read_tag(tag_crops)
        if not face_crops and not self._find_by_tag(tag):
            return {"status": "rejected", "reason": "no_evidence"}
        return self._fuse(self.identify(face_crops) if face_crops else None, tag)

    def identify_frame(self, frame: Image.Image, use_tags: bool = False) -> list[dict]:
        """One photo/CCTV frame with any number of animals: one result per detected face, with its
        box. Single-photo matches are weaker than multi-photo ones; track a cow over frames and use
        identify_frames() when you can. use_tags=True also reads the tag next to each face and
        fuses it (one frame cannot vote, so a single read is used: a read that names a different
        registered cow than the face shows up as a `conflict`, not as a silent change)."""
        out = []
        tags_of = getattr(self.detector, "tags", None) if (use_tags and self.cfg.read_tags) else None
        for crop, d in self.detector.faces(frame):
            res = self.identify([crop])
            if tags_of:
                tags = tags_of(frame, near=d.box, near_frac=self.cfg.tag_near_face)
                tag = self._read_tag([tags[0][0]], min_votes=1) if tags else None
                res = self._fuse(res, tag)
            res["box"] = d.box
            res["face_conf"] = d.conf
            out.append(res)
        return out

    def register_frames(self, frames: list[Image.Image], details: dict,
                        tag_number: str | None = None, force: bool = False) -> dict:
        crops, skipped, tag_crops = self._analyse(frames)
        res = self.register(crops, details, tag_number, force) if crops else {
            "status": "rejected", "reason": "not_enough_good_images", "good_images": 0,
            "needed": self.cfg.min_register_images, "rejections": []}
        res["skipped_frames"] = skipped
        given = self._norm_tag(tag_number)
        if res["status"] == "registered" and given:
            seen = self._read_tag(tag_crops)
            if seen and seen != given:  # warn, do not block: the typed number may be right
                res["tag_check"] = {"typed": given, "read_from_photos": seen,
                                    "message": "The tag number in the photos differs from the one entered."}
        return res

    def register(self, crops: list[Image.Image], details: dict, tag_number: str | None = None,
                 force: bool = False) -> dict:
        """force=True: register even though the face looks like an existing cow (the farmer has
        confirmed they are different animals). A reused tag number is never overridden."""
        tag = self._norm_tag(tag_number)
        owner = self._find_by_tag(tag)
        if owner:
            return {"status": "duplicate", "existing_id": owner, "reason": "tag_already_registered",
                    "details": self.registry.get(owner),
                    "message": "This ear-tag number is already registered to another animal."}
        embs, reasons = self._good_embeddings(crops)
        if embs is None or len(embs) < self.cfg.min_register_images:
            return {"status": "rejected", "reason": "not_enough_good_images",
                    "good_images": 0 if embs is None else len(embs),
                    "needed": self.cfg.min_register_images, "rejections": reasons}
        hit = self._match(embs)
        lookalike = hit[0] if (hit and hit[1] >= self.cfg.match_threshold) else None
        if lookalike and not force:
            return {"status": "duplicate", "existing_id": lookalike, "score": hit[1],
                    "details": self.registry.get(lookalike),
                    "message": "This cattle appears to be already registered."}
        cid = f"COW-{uuid.uuid4().hex[:8].upper()}"
        record = {**details, "tag_number": tag} if tag else dict(details)
        if lookalike:
            record["confirmed_different_from"] = lookalike  # audit trail for the override
        self.index.add(cid, embs)
        self.registry.put(cid, record)
        self.index.save()
        res = {"status": "registered", "cattle_id": cid}
        if lookalike:
            res["overrode_duplicate_of"] = lookalike
        return res
