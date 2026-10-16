"""Fine-tune the embedder with ArcFace.

Data layout (cow crops, one folder per animal):  data_dir/<animal_id>/*.jpg
Whole animals are held out for validation, so the metric reflects unseen cattle.

  python -m cattle_id.train --data_dir data/crops --out checkpoints/embedder.pt
  (--data_dir and --out can be gs:// paths if you mount the bucket with gcsfuse)
"""
import argparse
import math
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .config import Config
from .embedder import EmbeddingNet

IMG_EXT = {".jpg", ".jpeg", ".png"}


def split_identities(data_dir: str, val_frac: float, seed: int = 0):
    ids = sorted(p.name for p in Path(data_dir).iterdir() if p.is_dir())
    random.Random(seed).shuffle(ids)
    n_val = max(1, int(len(ids) * val_frac))
    return ids[n_val:], ids[:n_val]


class CowDataset(Dataset):
    def __init__(self, data_dir, identities, transform):
        self.items, self.transform = [], transform
        self.label_of = {cid: i for i, cid in enumerate(identities)}
        for cid in identities:
            for f in (Path(data_dir) / cid).iterdir():
                if f.suffix.lower() in IMG_EXT:
                    self.items.append((f, self.label_of[cid]))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        return self.transform(Image.open(path).convert("RGB")), label


class ArcFace(nn.Module):
    def __init__(self, dim, n_classes, s=30.0, m=0.5):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_classes, dim) * 0.01)
        self.s, self.m = s, m

    def forward(self, emb, labels):
        cos = F.linear(emb, F.normalize(self.W)).clamp(-1 + 1e-7, 1 - 1e-7)
        theta = torch.acos(cos)
        target = torch.cos(theta + self.m)
        onehot = F.one_hot(labels, cos.size(1)).bool()
        return F.cross_entropy(self.s * torch.where(onehot, target, cos), labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", default="checkpoints/embedder.pt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--val_frac", type=float, default=0.2)
    args = ap.parse_args()

    import timm
    cfg = Config()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = EmbeddingNet(cfg.backbone, cfg.embed_dim, img_size=cfg.input_size).to(dev)
    cfg_t = {**net.data_config, "input_size": (3, cfg.input_size, cfg.input_size)}
    train_tf = timm.data.create_transform(**cfg_t, is_training=True, hflip=0.5, vflip=0.5,
                                          color_jitter=0.2, auto_augment=None)

    train_ids, val_ids = split_identities(args.data_dir, args.val_frac)
    print(f"{len(train_ids)} train animals, {len(val_ids)} held-out animals")
    ds = CowDataset(args.data_dir, train_ids, train_tf)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)

    arc = ArcFace(cfg.embed_dim, len(train_ids)).to(dev)
    opt = torch.optim.AdamW(
        [{"params": net.backbone.parameters(), "lr": args.lr},
         {"params": list(net.head.parameters()) + list(arc.parameters()), "lr": args.lr * 10}],
        weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs * len(dl))

    for ep in range(args.epochs):
        net.train()
        total = 0.0
        for x, y in dl:
            x, y = x.to(dev), y.to(dev)
            loss = arc(net(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item()
        print(f"epoch {ep + 1}/{args.epochs}  loss {total / len(dl):.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), args.out)
    print(f"saved {args.out}. Now run: python -m cattle_id.evaluate --data_dir {args.data_dir} "
          f"--checkpoint {args.out}")


if __name__ == "__main__":
    main()
