"""CCTV side: video -> per-cow tracks of crops, ready for CattleID.identify()."""
from collections import defaultdict

from PIL import Image

from .config import Config


def tracks_from_video(path: str, cfg: Config | None = None, stride: int = 3,
                      max_crops_per_track: int = 20) -> dict[int, list[Image.Image]]:
    """Detect + track cows (YOLO + ByteTrack). Returns {track_id: [crops]}.
    stride: use every Nth frame. Crops are evenly capped per track."""
    from ultralytics import YOLO

    cfg = cfg or Config()
    model = YOLO(cfg.detector_weights)
    tracks: dict[int, list[Image.Image]] = defaultdict(list)
    results = model.track(path, stream=True, persist=True, tracker="bytetrack.yaml",
                          classes=[cfg.cow_class_id], conf=cfg.detect_conf,
                          vid_stride=stride, verbose=False)
    for r in results:
        if r.boxes.id is None:
            continue
        frame = Image.fromarray(r.orig_img[..., ::-1])  # BGR -> RGB
        for box, tid in zip(r.boxes.xyxy.int().tolist(), r.boxes.id.int().tolist()):
            tracks[tid].append(frame.crop(tuple(box)))
    return {t: _spread(c, max_crops_per_track) for t, c in tracks.items()}


def _spread(items: list, n: int) -> list:
    if len(items) <= n:
        return items
    step = len(items) / n
    return [items[int(i * step)] for i in range(n)]


def identify_video(path: str, system, cfg: Config | None = None) -> dict[int, dict]:
    """Run identify() on every tracked cow. Returns {track_id: identify-result}."""
    return {tid: system.identify(crops) for tid, crops in tracks_from_video(path, cfg).items()}
