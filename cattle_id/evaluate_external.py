"""Evaluate a checkpoint on a DIFFERENT dataset than it was trained on (e.g. Cows2021).

Layout: data_dir/<cow_id>/*.jpg  (file names containing YYYY-MM-DD are sorted by date).
Cows are split into enrolled (gallery = 5 earliest images) and unknown. Queries are groups of
k consecutive later crops, like a short track. Reports results at fixed thresholds.

  python -m cattle_id.evaluate_external --data_dir <folder> --checkpoint checkpoints/embedder.pt
"""
import argparse
import re
from pathlib import Path

import numpy as np
from PIL import Image

from .config import Config
from .embedder import Embedder
from .experiment import l2
from .train import IMG_EXT


def load(emb, data_dir, per_cow, min_images):
    E, labels = [], []
    for d in sorted(p for p in Path(data_dir).iterdir() if p.is_dir()):
        fs = [f for f in d.iterdir() if f.suffix.lower() in IMG_EXT]
        if len(fs) < min_images:
            continue
        key = lambda f: (re.search(r"\d{4}-\d{2}-\d{2}", f.name).group(0) if re.search(r"\d{4}-\d{2}-\d{2}", f.name) else "", f.name)
        fs = sorted(fs, key=key)
        fs = [fs[i] for i in np.unique(np.linspace(0, len(fs) - 1, min(per_cow, len(fs))).astype(int))]
        for i in range(0, len(fs), 32):
            E.append(emb.embed([Image.open(f) for f in fs[i:i + 32]]))
        labels += [d.name] * len(fs)
    return np.concatenate(E), np.array(labels)


def run(E, labels, seed, k, gallery=5, thresholds=(0.75, 0.80, 0.84)):
    cows = sorted(set(labels))
    rng = np.random.default_rng(seed)
    rng.shuffle(cows)
    known, unknown = cows[: len(cows) // 2], cows[len(cows) // 2:]
    ids, G = [], []
    for c in known:
        idx = np.where(labels == c)[0]          # already date-sorted
        ids.append(c)
        G.append(E[idx[:gallery]].mean(0))
    G = l2(np.stack(G))

    def groups(c, start):
        idx = np.where(labels == c)[0][start:]
        return [l2(E[idx[i:i + k]].mean(0)) for i in range(0, len(idx) - k + 1, k)]

    ks, top1 = [], []
    for c in known:
        for q in groups(c, gallery):
            s = G @ q
            ks.append(s.max())
            top1.append(ids[int(s.argmax())] == c)
    us = np.array([(G @ q).max() for c in unknown for q in groups(c, 0)])
    ks = np.array(ks)
    out = {"top1": float(np.mean(top1)), "auc": float(np.mean(ks[:, None] > us[None, ::max(1, len(us) // 800)])),
           "recall@1%FAR(own thr)": float(np.mean(ks >= np.quantile(us, 0.99))),
           "own thr": float(np.quantile(us, 0.99))}
    for t in thresholds:
        out[f"recall@{t}"] = float(np.mean(ks >= t))
        out[f"false-accept@{t}"] = float(np.mean(us >= t))
    return out, len(known), len(unknown), len(ks), len(us)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--per_cow", type=int, default=40)
    ap.add_argument("--min_images", type=int, default=15)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--splits", type=int, default=3)
    args = ap.parse_args()
    emb = Embedder(Config(checkpoint=args.checkpoint))
    E, labels = load(emb, args.data_dir, args.per_cow, args.min_images)
    print(f"{len(E)} images, {len(set(labels))} cows; checkpoint={args.checkpoint}")
    res = [run(E, labels, s, args.k) for s in range(args.splits)]
    print(f"{res[0][1]} enrolled / {res[0][2]} unknown cows per split; {res[0][3]} known and {res[0][4]} unknown queries (k={args.k})")
    for key in res[0][0]:
        print(f"  {key:26s} {np.mean([r[0][key] for r in res]):.3f}")


if __name__ == "__main__":
    main()
