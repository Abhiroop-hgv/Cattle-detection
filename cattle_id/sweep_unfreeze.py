"""Sweep: how many backbone blocks to unfreeze (and how long to train), same data and test as v2.

Trains on OpenCows2020 + the Cows2021 Train cows already on disk (IDs not in Test), evaluates on
the Cows2021 Test cows. Run after pipeline_v2 has populated /content/c21_train_raw.

  python -u -m cattle_id.sweep_unfreeze
"""
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import Config
from .embedder import Embedder, EmbeddingNet
from .evaluate_external import load, run
from .experiment2 import Imgs
from .pipeline_v2 import DRIVE, RAW, TEST, build, log
from .train import ArcFace

CONFIGS = [  # (blocks to unfreeze, epochs, backbone lr)
    (4, 15, 2e-5),
    (8, 15, 2e-5),
    (12, 15, 1e-5),
]


def fit2(files, labels, train, n_blocks, epochs, lr_bb, seed=0):
    import timm
    torch.manual_seed(seed)
    cfg, dev = Config(), "cuda"
    net = EmbeddingNet(cfg.backbone, cfg.embed_dim, img_size=cfg.input_size).to(dev)
    for p in net.backbone.parameters():
        p.requires_grad = False
    blocks = list(net.backbone.blocks)[-n_blocks:]
    tuned = [p for b in blocks for p in b.parameters()] + list(net.backbone.norm.parameters())
    if n_blocks >= len(net.backbone.blocks):  # everything else too (patch embed, cls/pos tokens)
        tuned += [p for n, p in net.backbone.named_parameters()
                  if not n.startswith("blocks.") and not n.startswith("norm.")]
    for p in tuned:
        p.requires_grad = True
    base = {**net.data_config, "input_size": (3, cfg.input_size, cfg.input_size)}
    train_tf = timm.data.create_transform(**base, is_training=True, hflip=0.5, vflip=0.5,
                                          color_jitter=0.2, auto_augment=None)
    items = [(f, train.index(l)) for f, l in zip(files, labels)]
    dl = DataLoader(Imgs(items, train_tf), batch_size=32, shuffle=True, num_workers=2, drop_last=True)
    arc = ArcFace(cfg.embed_dim, len(train), s=20.0, m=0.3).to(dev)
    opt = torch.optim.AdamW([{"params": tuned, "lr": lr_bb},
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
        log(f"   blocks={n_blocks} epoch {ep + 1}/{epochs} loss {tot / len(dl):.3f}")
    return net.eval()


def main():
    names = [str(p.relative_to(RAW)) for p in RAW.rglob("*.jpg")]
    files, labels = build(names)
    train = sorted(set(labels))
    results, best = {}, (-1, None)
    for n_blocks, epochs, lr in CONFIGS:
        name = f"unfreeze {n_blocks} blocks, {epochs} ep, lr {lr:g}"
        t0 = time.time()
        net = fit2(files, labels, train, n_blocks, epochs, lr)
        ck = f"checkpoints/v3_{n_blocks}blocks.pt"
        torch.save(net.state_dict(), ck)
        del net
        torch.cuda.empty_cache()
        emb = Embedder(Config(checkpoint=ck))
        E, lab = load(emb, TEST, 40, 15)
        del emb
        torch.cuda.empty_cache()
        for k in (1, 5):
            res = [run(E, lab, s, k) for s in range(3)]
            results[f"{name} | k={k}"] = {key: float(np.mean([r[0][key] for r in res])) for key in res[0][0]}
        auc5 = results[f"{name} | k=5"]["auc"]
        if auc5 > best[0]:
            best = (auc5, ck)
        log(f"done {name} in {time.time() - t0:.0f}s: k=5 top1 {results[f'{name} | k=5']['top1']:.3f} auc {auc5:.3f}")
    print(json.dumps(results, indent=1))
    Path("sweep_results.json").write_text(json.dumps(results, indent=1))
    if DRIVE.is_dir():
        shutil.copy("sweep_results.json", DRIVE)
        shutil.copy(best[1], DRIVE)
        log("copied results + best checkpoint", best[1], "to Drive")
    log("ALL DONE")


if __name__ == "__main__":
    main()
