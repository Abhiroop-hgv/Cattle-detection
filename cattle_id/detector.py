"""Finds cattle faces and ear tags in a frame (YOLOv8, trained on merged Roboflow cattle datasets)."""
from dataclasses import dataclass

from PIL import Image

from .config import Config

FACE, EAR_TAG = "cow_face", "ear_tag"


@dataclass
class Detection:
    label: str
    conf: float
    box: tuple[int, int, int, int]  # x0, y0, x1, y1 in frame pixels


class FaceTagDetector:
    def __init__(self, cfg: Config | None = None):
        from ultralytics import YOLO

        self.cfg = cfg or Config()
        self.model = YOLO(self.cfg.face_detector_weights)

    def detect(self, image: Image.Image, conf: float | None = None) -> list[Detection]:
        r = self.model.predict(image.convert("RGB"), imgsz=self.cfg.face_detector_imgsz,
                               conf=self.cfg.face_conf if conf is None else conf, verbose=False)[0]
        names = r.names
        return [Detection(names[int(c)], float(p), tuple(int(v) for v in b))
                for b, c, p in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist())]

    def faces(self, image: Image.Image) -> list[tuple[Image.Image, Detection]]:
        """Padded face crops, most confident first."""
        w, h = image.size
        pad = self.cfg.face_crop_pad
        out = []
        for d in sorted((d for d in self.detect(image) if d.label == FACE), key=lambda d: -d.conf):
            x0, y0, x1, y1 = d.box
            px, py = (x1 - x0) * pad, (y1 - y0) * pad
            box = (max(0, int(x0 - px)), max(0, int(y0 - py)), min(w, int(x1 + px)), min(h, int(y1 + py)))
            out.append((image.convert("RGB").crop(box), d))
        return out

    def tags(self, image: Image.Image, near: tuple[int, int, int, int] | None = None,
             pad: float = 0.15, near_frac: float = 0.6) -> list[tuple[Image.Image, Detection]]:
        """Ear-tag crops, most confident first. With `near` (a face box), only tags around that face."""
        w, h = image.size
        out = []
        for d in sorted((d for d in self.detect(image) if d.label == EAR_TAG), key=lambda d: -d.conf):
            x0, y0, x1, y1 = d.box
            if near:
                fx0, fy0, fx1, fy1 = near
                mx, my = (fx1 - fx0) * near_frac, (fy1 - fy0) * near_frac
                cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                if not (fx0 - mx <= cx <= fx1 + mx and fy0 - my <= cy <= fy1 + my):
                    continue
            px, py = (x1 - x0) * pad, (y1 - y0) * pad
            box = (max(0, int(x0 - px)), max(0, int(y0 - py)), min(w, int(x1 + px)), min(h, int(y1 + py)))
            out.append((image.convert("RGB").crop(box), d))
        return out
