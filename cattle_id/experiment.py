"""Compare ways of turning frozen-backbone features into cattle embeddings, over several
different animal splits (so one lucky/unlucky split can't mislead).

Backbone features are computed once and cached; every method then runs in seconds.

  python -m cattle_id.experiment --data_dir data/crops
"""
import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from .config import Config
from .embedder import EmbeddingNet
from .train import IMG_EXT, ArcFace


def extract(data_dir, cache):
    cache = Path(cache)
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return z["feats"], z["labels"]
    import timm
    cfg = Config()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = EmbeddingNet(cfg.backbone, cfg.embed_dim, img_size=cfg.input_size).to(dev).eval()
    tf = timm.data.create_transform(**{**net.data_config, "input_size": (3, cfg.input_size, cfg.input_size)})
    files, labels = [], []
    for d in sorted(p for p in Path(data_dir).iterdir() if p.is_dir()):
        for f in sorted(d.iterdir()):
            if f.suffix.lower() in IMG_EXT:
                files.append(f)
                labels.append(d.name)
    feats = []
    with torch.no_grad():
        for i in range(0, len(files), 64):
            x = torch.stack([tf(Image.open(f).convert("RGB")) for f in files[i:i + 64]]).to(dev)
            feats.append(net.backbone(x).float().cpu().numpy())
    feats, labels = np.concatenate(feats), np.array(labels)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, feats=feats, labels=labels)
    return feats, labels


def l2(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def train_head(X, y, n_cls, dim, epochs, bias, seed):
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    head = nn.Sequential(nn.Dropout(0.2), nn.Linear(X.shape[1], dim, bias=bias)).to(dev)
    arc = ArcFace(dim, n_cls, s=20.0, m=0.3).to(dev)
    opt = torch.optim.AdamW(list(head.parameters()) + list(arc.parameters()), lr=1e-3, weight_decay=1e-2)
    Xt, yt = torch.tensor(X).to(dev), torch.tensor(y).to(dev)
    for _ in range(epochs):
        head.train()
        for idx in torch.randperm(len(Xt), device=dev).split(128):
            noisy = Xt[idx] + 0.05 * torch.randn_like(Xt[idx]) * Xt[idx].std()
            loss = arc(F.normalize(head(noisy), dim=-1), yt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    head.eval()
    return lambda A: F.normalize(head(torch.tensor(A).to(dev)), dim=-1).detach().cpu().numpy()


def open_set_eval(E, labels, known, unknown, gallery_size, rng):
    ids, G, queries = [], [], []
    for cid in known:
        idx = np.where(labels == cid)[0]
        rng.shuffle(idx)
        ids.append(cid)
        G.append(E[idx[:gallery_size]].mean(0))
        queries += [(cid, E[j]) for j in idx[gallery_size:]]
    G = l2(np.stack(G))
    top1, ks = [], []
    for cid, v in queries:
        s = G @ v
        top1.append(ids[int(s.argmax())] == cid)
        ks.append(s.max())
    us = np.array([(G @ E[j]).max() for cid in unknown for j in np.where(labels == cid)[0]])
    ks = np.array(ks)
    thr = np.quantile(us, 0.99)  # threshold at 1% false-accept on this split
    auc = float(np.mean(ks[:, None] > us[None, ::max(1, len(us) // 500)]))  # P(known > unknown)
    return np.mean(top1), auc, float(np.mean(ks >= thr)), float(np.mean(us)), float(np.mean(ks))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--cache", default="cache/feats.npz")
    ap.add_argument("--splits", type=int, default=6)
    ap.add_argument("--val_frac", type=float, default=0.3)
    ap.add_argument("--gallery_size", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()

    feats, labels = extract(args.data_dir, args.cache)
    animals = sorted(set(labels))
    print(f"{len(feats)} images, {len(animals)} animals, feat dim {feats.shape[1]}")
    results = {}
    for seed in range(args.splits):
        ids = animals[:]
        random.Random(seed).shuffle(ids)
        n_val = int(len(ids) * args.val_frac)
        val, train = ids[:n_val], ids[n_val:]
        known, unknown = val[: n_val // 2], val[n_val // 2:]
        tr = np.isin(labels, train)
        mean = feats[tr].mean(0)
        y = np.array([train.index(l) for l in labels[tr]])
        rng = np.random.default_rng(seed)

        methods = {"raw": l2(feats), "centered": l2(feats - mean)}
        # whitening fitted on TRAIN animals only
        U, S, Vt = np.linalg.svd(feats[tr] - mean, full_matrices=False)
        k = 128
        W = Vt[:k].T / (S[:k] / np.sqrt(len(y)) + 1e-6)
        methods["centered+whiten128"] = l2((feats - mean) @ W)
        for bias in (True, False):
            f = train_head(feats[tr] - mean, y, len(train), 256, args.epochs, bias, seed)
            methods[f"arcface-head(bias={bias})+center"] = f(feats - mean)
        for name, E in methods.items():
            results.setdefault(name, []).append(
                open_set_eval(E, labels, known, unknown, args.gallery_size, np.random.default_rng(seed)))
    print(f"\nmean over {args.splits} animal splits "
          f"(top1 | P(known>unknown) | recall@1%FAR | avg unknown score | avg known score)")
    for name, r in results.items():
        m = np.mean(r, axis=0)
        print(f"{name:34s} {m[0]:.3f} | {m[1]:.3f} | {m[2]:.3f} | {m[3]:.3f} | {m[4]:.3f}")


if __name__ == "__main__":
    main()
