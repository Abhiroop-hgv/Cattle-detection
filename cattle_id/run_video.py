"""Run a video file through the identity tracker and write one JSON line per cow per processed frame.

    python -m cattle_id.run_video clip.mp4 --camera pen-3 --stride 5 --out identities.jsonl

Use it to try the pipeline on your own footage: see how many cows get identified, how many stay
"unidentified", and where the detector misfires. Needs registered cattle (see README).
"""
import argparse
import json
from collections import Counter

from PIL import Image


def process(video: str, tracker, stride: int = 5, max_frames: int | None = None,
            out_path: str | None = None) -> dict:
    import cv2

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    sink = open(out_path, "w") if out_path else None
    last, n, seen = {}, 0, 0
    try:
        i = 0
        while True:
            ok, bgr = cap.read()
            if not ok or (max_frames and seen >= max_frames):
                break
            if i % stride == 0:
                t = i / fps
                frame = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                for rec in tracker.update(frame, t):
                    rec = {**rec, "t": round(t, 3), "frame": i}
                    last[rec["track_id"]] = rec
                    n += 1
                    if sink:
                        sink.write(json.dumps(rec, default=str) + "\n")
                seen += 1
            i += 1
    finally:
        cap.release()
        if sink:
            sink.close()
    by_status = Counter(r["status"] for r in last.values())
    return {"frames_processed": seen, "records": n, "tracks": len(last),
            "tracks_by_final_status": dict(by_status)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--camera", default="cam")
    ap.add_argument("--stride", type=int, default=5, help="process every Nth frame")
    ap.add_argument("--max-frames", type=int, default=None, help="stop after N processed frames")
    ap.add_argument("--out", default=None, help="JSONL file for per-frame identity records")
    a = ap.parse_args()

    from .service import CattleID
    from .tracking import IdentityTracker

    svc = CattleID()
    summary = process(a.video, IdentityTracker(svc, camera_id=a.camera), a.stride, a.max_frames, a.out)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
