"""run_video.process on a tiny synthetic video, with a fake tracker (no models)."""
import json

import cv2
import numpy as np

from cattle_id.run_video import process


class FakeTracker:
    def update(self, frame, t):
        return [{"track_id": 1, "status": "unidentified" if t < 0.3 else "identified", "box": (0, 0, 5, 5)}]


def make_video(path, n=20, fps=10):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 48))
    for i in range(n):
        w.write(np.full((48, 64, 3), i * 10, np.uint8))
    w.release()


def test_process_writes_records_and_summary(tmp_path):
    v, out = tmp_path / "v.mp4", tmp_path / "o.jsonl"
    make_video(v)
    s = process(str(v), FakeTracker(), stride=5, out_path=str(out))
    assert s["frames_processed"] == 4 and s["tracks"] == 1
    assert s["tracks_by_final_status"] == {"identified": 1}
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert [r["frame"] for r in rows] == [0, 5, 10, 15] and rows[0]["t"] == 0.0
