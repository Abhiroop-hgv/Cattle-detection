"""Final training recipe: fine-tune the last N backbone blocks + bias-free head with ArcFace.

1. Re-runs 3 animal splits only to choose match_threshold from held-out unknown-cow scores
   (queries = average of k crops, matching how identify() is used).
2. Trains the final model on ALL animals and saves it.

  python -m cattle_id.final --data_dir data/crops --out checkpoints/embedder.pt
"""
import argparse
import json
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import Config
from .embedder import EmbeddingNet
from .experiment import l2
from .experiment2 import Imgs
from .train import IMG_EXT, ArcFace


def fit(files, labels, train, n_blocks, epochs, seed):
    import timm
    torch.manual_seed(seed)
    cfg, dev = Config(), "cuda"
    net = EmbeddingNet(cfg.backbone, cfg.embed_dim, img_size=cfg.input_size).to(dev)
    for p in net.backbone.parameters():
        p.requires_grad = False
    tuned = list(net.backbone.blocks[-n_blocks:].parameters()) + list(net.backbone.norm.parameters())
    for p in tuned:
        p.requires_grad = True
    base = {**net.data_config, "input_size": (3, cfg.input_size, cfg.input_size)}
    train_tf = timm.data.create_transform(**base, is_training=True, hflip=0.5, vflip=0.5,
                                          color_jitter=0.2, auto_augment=None)
    eval_tf = timm.data.create_transform(**base)
    items = [(f, train.index(l)) for f, l in zip(files, labels) if l in train]
    dl = DataLoader(Imgs(items, train_tf), batch_size=32, shuffle=True, num_workers=2, drop_last=True)
    arc = ArcFace(cfg.embed_dim, len(train), s=20.0, m=0.3).to(dev)
    opt = torch.optim.AdamW([{"params": tuned, "lr": 1e-5},
                             {"params": list(net.head.parameters()) + list(arc.parameters()), "lr": 1e-3}],
                            weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * len(dl))
    scaler = torch.amp.GradScaler()
    for ep in range(epochs):
        net.train()
        tot = 0.0
        for x, y in dl:
            x, y = x.to(dev), y.to(dev)
            with torch.autocast("cuda", dtype=torch.float16):
                emb = net(x)
            loss = arc(emb.float(), y)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item()
        print(f"   epoch {ep + 1}/{epochs} loss {tot / len(dl):.3f}", flush=True)
    return net.eval(), eval_tf


@torch.no_grad()
def embed(net, eval_tf, files):
    out = []
    for x, _ in DataLoader(Imgs([(f, 0) for f in files], eval_tf), batch_size=64, num_workers=2):
        out.append(net(x.to("cuda")).float().cpu().numpy())
    return np.concatenate(out)


def scores(E, labels, known, unknown, k, rng, gallery=5, groups=15):
    ids, G, ks = [], [], []
    queries = []
    for cid in known:
        idx = rng.permutation(np.where(labels == cid)[0])
        ids.append(cid)
        G.append(E[idx[:gallery]].mean(0))
        for _ in range(groups):
            queries.append(l2(E[rng.choice(idx[gallery:], size=k, replace=False)].mean(0)))
    G = l2(np.stack(G))
    ks = [(G @ q).max() for q in queries]
    us = []
    for cid in unknown:
        idx = np.where(labels == cid)[0]
        for _ in range(groups):
            us.append((G @ l2(E[rng.choice(idx, size=k, replace=False)].mean(0))).max())
    return np.array(ks), np.array(us)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", default="checkpoints/embedder.pt")
    ap.add_argument("--drive_dir", default="/content/drive/MyDrive/cattle_id")
    ap.add_argument("--blocks", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--splits", type=int, default=3)
    ap.add_argument("--val_frac", type=float, default=0.3)
    args = ap.parse_args()

    dirs = sorted(p for p in Path(args.data_dir).iterdir() if p.is_dir())
    files = [f for d in dirs for f in sorted(d.iterdir()) if f.suffix.lower() in IMG_EXT]
    labels = np.array([f.parent.name for f in files])
    animals = sorted(set(labels))

    all_k, all_u = [], []
    for seed in range(args.splits):
        ids = animals[:]
        random.Random(seed).shuffle(ids)
        n_val = int(len(ids) * args.val_frac)
        val, train = ids[:n_val], ids[n_val:]
        print(f"threshold split {seed}: {len(train)} train / {len(val)} held-out", flush=True)
        net, tf = fit(files, list(labels), train, args.blocks, args.epochs, seed)
        vm = np.isin(labels, val)
        E = embed(net, tf, [f for f, m in zip(files, vm) if m])
        ks, us = scores(E, labels[vm], val[: n_val // 2], val[n_val // 2:], args.k, np.random.default_rng(seed))
        all_k.append(ks)
        all_u.append(us)
        del net
        torch.cuda.empty_cache()
    ks, us = np.concatenate(all_k), np.concatenate(all_u)
    report = {"k_crops": args.k,
              "threshold_1pct_far": float(np.quantile(us, 0.99)),
              "threshold_5pct_far": float(np.quantile(us, 0.95)),
              "recall_at_1pct_far": float(np.mean(ks >= np.quantile(us, 0.99))),
              "recall_at_5pct_far": float(np.mean(ks >= np.quantile(us, 0.95))),
              "known_score_median": float(np.median(ks)), "unknown_score_median": float(np.median(us)),
              "n_known_queries": int(len(ks)), "n_unknown_queries": int(len(us))}
    print(json.dumps(report, indent=1))

    print(f"final model on all {len(animals)} animals", flush=True)
    net, _ = fit(files, list(labels), animals, args.blocks, args.epochs, 0)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), out)
    out.with_suffix(".json").write_text(json.dumps(report, indent=1))
    if Path(args.drive_dir).is_dir():
        shutil.copy(out, args.drive_dir)
        shutil.copy(out.with_suffix(".json"), args.drive_dir)
        print(f"copied to {args.drive_dir}")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
