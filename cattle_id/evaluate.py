"""Open-set evaluation on HELD-OUT animals (same split as train.py).

Half of the held-out animals are 'enrolled' (a few images each go in the gallery);
the rest are 'unknown'. We report:
  - top-1 accuracy for enrolled animals
  - false-accept rate for unknown animals at each threshold
and suggest a match_threshold for a target false-accept rate.

  python -m cattle_id.evaluate --data_dir data/crops --checkpoint checkpoints/embedder.pt
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from .config import Config
from .embedder import Embedder
from .train import IMG_EXT, split_identities


def embed_folder(embedder, folder: Path, limit: int = 40):
    files = [f for f in sorted(folder.iterdir()) if f.suffix.lower() in IMG_EXT][:limit]
    return embedder.embed([Image.open(f) for f in files])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--checkpoint", default=None, help="omit to evaluate the pretrained backbone")
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--gallery_size", type=int, default=5)
    ap.add_argument("--target_far", type=float, default=0.01)
    args = ap.parse_args()

    cfg = Config(checkpoint=args.checkpoint)
    emb = Embedder(cfg)
    _, val_ids = split_identities(args.data_dir, args.val_frac)
    known, unknown = val_ids[: len(val_ids) // 2], val_ids[len(val_ids) // 2:]
    print(f"held-out: {len(known)} enrolled animals, {len(unknown)} unknown animals")

    gallery, centroids = {}, []
    queries = []  # (true_id, vec)
    for cid in known:
        e = embed_folder(emb, Path(args.data_dir) / cid)
        gallery[cid] = e[: args.gallery_size].mean(0)
        queries += [(cid, v) for v in e[args.gallery_size:]]
    ids = list(gallery)
    G = np.stack([gallery[i] / np.linalg.norm(gallery[i]) for i in ids])

    correct, known_scores = 0, []
    for true, v in queries:
        s = G @ v
        correct += ids[int(s.argmax())] == true
        known_scores.append(s.max())
    print(f"top-1 accuracy on enrolled animals: {correct / max(1, len(queries)):.3f}")

    unk_scores = []
    for cid in unknown:
        for v in embed_folder(emb, Path(args.data_dir) / cid):
            unk_scores.append((G @ v).max())
    unk_scores = np.array(unk_scores)
    known_scores = np.array(known_scores)

    print("threshold  recall(enrolled)  false-accept(unknown)")
    for t in np.arange(0.5, 0.96, 0.05):
        print(f"  {t:.2f}      {np.mean(known_scores >= t):.3f}             "
              f"{np.mean(unk_scores >= t):.3f}")
    sug = float(np.quantile(unk_scores, 1 - args.target_far)) if len(unk_scores) else None
    print(f"\nSuggested match_threshold for ~{args.target_far:.0%} false-accept: {sug}")


if __name__ == "__main__":
    main()
