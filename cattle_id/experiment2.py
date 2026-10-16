"""Multi-crop query averaging + gentle partial fine-tuning vs frozen features.

  python -m cattle_id.experiment2 --data_dir data/crops
"""
import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .config import Config
from .embedder import EmbeddingNet
from .experiment import extract, l2, train_head
from .train import IMG_EXT, ArcFace


def multi_eval(E, labels, known, unknown, k, rng, gallery=5, groups=15):
    ids, G, kq = [], [], []
    for cid in known:
        idx = rng.permutation(np.where(labels == cid)[0])
        ids.append(cid)
        G.append(E[idx[:gallery]].mean(0))
        rest = idx[gallery:]
        for _ in range(groups):
            sel = rng.choice(rest, size=min(k, len(rest)), replace=False)
            kq.append((cid, l2(E[sel].mean(0))))
    G = l2(np.stack(G))
    top1, ks = [], []
    for cid, v in kq:
        s = G @ v
        top1.append(ids[int(s.argmax())] == cid)
        ks.append(s.max())
    us = []
    for cid in unknown:
        idx = np.where(labels == cid)[0]
        for _ in range(groups):
            sel = rng.choice(idx, size=min(k, len(idx)), replace=False)
            us.append((G @ l2(E[sel].mean(0))).max())
    ks, us = np.array(ks), np.array(us)
    auc = float(np.mean(ks[:, None] > us[None, :]))
    r1 = float(np.mean(ks >= np.quantile(us, 0.99)))
    r5 = float(np.mean(ks >= np.quantile(us, 0.95)))
    return np.mean(top1), auc, r1, r5


class Imgs(Dataset):
    def __init__(self, items, tf):
        self.items, self.tf = items, tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, y = self.items[i]
        return self.tf(Image.open(p).convert("RGB")), y


def finetune_and_embed(files, labels, train, val, n_blocks, epochs, seed):
    import timm
    torch.manual_seed(seed)
    cfg = Config()
    dev = "cuda"
    net = EmbeddingNet(cfg.backbone, cfg.embed_dim, img_size=cfg.input_size).to(dev)
    net.head = nn.Linear(net.backbone.num_features, cfg.embed_dim, bias=False).to(dev)
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
        print(f"   finetune epoch {ep + 1}/{epochs} loss {tot / len(dl):.3f}", flush=True)
    net.eval()
    vitems = [(f, 0) for f, l in zip(files, labels) if l in val]
    vlabels = np.array([l for l in labels if l in val])
    out = []
    with torch.no_grad():
        for x, _ in DataLoader(Imgs(vitems, eval_tf), batch_size=64, num_workers=2):
            out.append(net(x.to(dev)).float().cpu().numpy())
    return np.concatenate(out), vlabels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--cache", default="cache/feats.npz")
    ap.add_argument("--splits", type=int, default=3)
    ap.add_argument("--val_frac", type=float, default=0.3)
    ap.add_argument("--blocks", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=6)
    args = ap.parse_args()

    feats, labels = extract(args.data_dir, args.cache)
    files = [f for d in sorted(p for p in Path(args.data_dir).iterdir() if p.is_dir())
             for f in sorted(d.iterdir()) if f.suffix.lower() in IMG_EXT]
    assert len(files) == len(labels)
    animals = sorted(set(labels))
    ks = (1, 5, 10)
    res = {}
    for seed in range(args.splits):
        ids = animals[:]
        random.Random(seed).shuffle(ids)
        n_val = int(len(ids) * args.val_frac)
        val, train = ids[:n_val], ids[n_val:]
        known, unknown = val[: n_val // 2], val[n_val // 2:]
        print(f"split {seed}: {len(train)} train / {len(val)} held-out animals", flush=True)
        tr = np.isin(labels, train)
        mean = feats[tr].mean(0)
        y = np.array([train.index(l) for l in labels[tr]])
        head = train_head(feats[tr] - mean, y, len(train), 256, 30, False, seed)
        vm = np.isin(labels, val)
        sets = {"raw frozen": (l2(feats[vm]), labels[vm]),
                "frozen + ArcFace head": (head(feats[vm] - mean), labels[vm])}
        E, vl = finetune_and_embed(files, list(labels), train, val, args.blocks, args.epochs, seed)
        sets[f"partial fine-tune (last {args.blocks} blocks)"] = (E, vl)
        for name, (E, lab) in sets.items():
            for k in ks:
                res.setdefault((name, k), []).append(
                    multi_eval(E, lab, known, unknown, k, np.random.default_rng(seed)))
    print(f"\nmean over {args.splits} splits   (top1 | P(known>unknown) | recall@1%FAR | recall@5%FAR)")
    for (name, k), r in res.items():
        m = np.mean(r, axis=0)
        print(f"{name:38s} crops={k:<2d} {m[0]:.3f} | {m[1]:.3f} | {m[2]:.3f} | {m[3]:.3f}")


if __name__ == "__main__":
    main()
