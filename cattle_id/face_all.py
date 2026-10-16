"""Self-contained face-recognition run on the 300-Cattle dataset (works on GPU or CPU).

  python -u face_all.py fetch   # WebDAV download, shrink to max side 640, keep small copies
  python -u face_all.py train   # baseline vs trained on held-out cows, then final model on all cows
"""
import io
import json
import shutil
import sys
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import unquote

import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset

TOKEN = "eMwaHAPK08dCDru"
BASE = "https://cloud.une.edu.au/public.php/webdav/300-Cattle-source/"
SMALL = Path("/content/face300_small")
DRIVE = Path("/content/drive/MyDrive/cattle_id")
BACKBONE = "vit_small_patch14_dinov2.lvd142m"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
IMG_EXT = {".jpg", ".jpeg", ".png"}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ---------------------------------------------------------------- fetch
def session():
    s = requests.Session()
    s.auth = (TOKEN, "")
    return s


def ls(s, url):
    for t in range(6):
        try:
            r = s.request("PROPFIND", url, headers={"Depth": "1"}, timeout=60)
            r.raise_for_status()
            hrefs = [unquote(h.text) for h in ET.fromstring(r.content).iterfind(".//{DAV:}href")]
            return [h.rstrip("/").rsplit("/", 1)[-1] for h in hrefs[1:]]
        except Exception as e:
            log("ls retry", url, e)
            time.sleep(3 + 3 * t)
    return []


def grab(job):
    cow, name = job
    dst = SMALL / cow / (Path(name).stem + ".jpg")
    if dst.exists():
        return 1
    s = session()
    for t in range(6):
        try:
            r = s.get(f"{BASE}{cow}/{name}", timeout=120)
            r.raise_for_status()
            im = ImageOps.exif_transpose(Image.open(io.BytesIO(r.content))).convert("RGB")
            im.thumbnail((640, 640))
            dst.parent.mkdir(parents=True, exist_ok=True)
            im.save(dst, quality=92)
            return 1
        except Exception as e:
            if t == 5:
                log("failed", cow, name, e)
            time.sleep(2 + 2 * t)
    return 0


def fetch():
    s = session()
    cows = ls(s, BASE)
    log(len(cows), "cow folders")
    jobs = [(c, n) for c in cows for n in ls(s, f"{BASE}{c}/") if Path(n).suffix.lower() in IMG_EXT]
    log(len(jobs), "images to fetch")
    done = 0
    with ThreadPoolExecutor(8) as ex:
        for i, ok in enumerate(ex.map(grab, jobs), 1):
            done += ok
            if i % 200 == 0:
                log("fetched", done, "/", len(jobs))
    log("fetched", done, "/", len(jobs))
    if DRIVE.is_dir():
        shutil.make_archive("/content/face300_small", "zip", SMALL)
        shutil.copy("/content/face300_small.zip", DRIVE)
        log("saved face300_small.zip to Drive")
    log("FETCH DONE")


# ---------------------------------------------------------------- model
class ArcFace(nn.Module):
    def __init__(self, dim, n_classes, s=20.0, m=0.3):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_classes, dim) * 0.01)
        self.s, self.m = s, m

    def forward(self, emb, labels):
        cos = F.linear(emb, F.normalize(self.W)).clamp(-1 + 1e-7, 1 - 1e-7)
        target = torch.cos(torch.acos(cos) + self.m)
        onehot = F.one_hot(labels, cos.size(1)).bool()
        return F.cross_entropy(self.s * torch.where(onehot, target, cos), labels)


class Net(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        import timm
        self.backbone = timm.create_model(BACKBONE, pretrained=pretrained, num_classes=0, img_size=224)
        self.head = nn.Linear(self.backbone.num_features, 256, bias=False)
        self.cfg = {**timm.data.resolve_data_config({}, model=self.backbone), "input_size": (3, 224, 224)}

    def forward(self, x):
        return F.normalize(self.head(self.backbone(x)), dim=-1)


class Imgs(Dataset):
    def __init__(self, items, tf):
        self.items, self.tf = items, tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, y = self.items[i]
        return self.tf(Image.open(p).convert("RGB")), y


def fit(files, labels, classes, epochs, n_blocks=2, seed=0):
    import timm
    torch.manual_seed(seed)
    net = Net().to(DEV)
    for p in net.backbone.parameters():
        p.requires_grad = False
    tuned = list(net.backbone.blocks[-n_blocks:].parameters()) + list(net.backbone.norm.parameters())
    for p in tuned:
        p.requires_grad = True
    tf = timm.data.create_transform(**net.cfg, is_training=True, hflip=0.5, vflip=0.0,
                                    color_jitter=0.2, auto_augment=None)
    items = [(f, classes.index(l)) for f, l in zip(files, labels)]
    dl = DataLoader(Imgs(items, tf), batch_size=32, shuffle=True, num_workers=2, drop_last=True)
    arc = ArcFace(256, len(classes)).to(DEV)
    opt = torch.optim.AdamW([{"params": tuned, "lr": 1e-5},
                             {"params": list(net.head.parameters()) + list(arc.parameters()), "lr": 1e-3}],
                            weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * len(dl))
    amp = DEV == "cuda"
    scaler = torch.amp.GradScaler(enabled=amp)
    for ep in range(epochs):
        net.train()
        tot = 0.0
        for x, y in dl:
            x, y = x.to(DEV), y.to(DEV)
            with torch.autocast(DEV, dtype=torch.float16, enabled=amp):
                emb = net(x)
            loss = arc(emb.float(), y)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item()
        log(f"   epoch {ep + 1}/{epochs} loss {tot / len(dl):.3f}")
    return net.eval()


@torch.no_grad()
def embed(net, files):
    import timm
    tf = timm.data.create_transform(**net.cfg)
    out = []
    for x, _ in DataLoader(Imgs([(f, 0) for f in files], tf), batch_size=32, num_workers=2):
        out.append(net(x.to(DEV)).float().cpu().numpy())
    return np.concatenate(out)


# ---------------------------------------------------------------- evaluation
def l2(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def run(E, labels, seed, k, gallery=3, thresholds=(0.75, 0.80, 0.84)):
    cows = sorted(set(labels))
    rng = np.random.default_rng(seed)
    rng.shuffle(cows)
    known, unknown = cows[: len(cows) // 2], cows[len(cows) // 2:]
    ids = list(known)
    G = l2(np.stack([E[np.where(labels == c)[0][:gallery]].mean(0) for c in known]))

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
    out = {"top1": float(np.mean(top1)), "auc": float(np.mean(ks[:, None] > us[None, :])),
           "recall@1%FAR": float(np.mean(ks >= np.quantile(us, 0.99))), "thr@1%FAR": float(np.quantile(us, 0.99)),
           "thr@5%FAR": float(np.quantile(us, 0.95)), "recall@5%FAR": float(np.mean(ks >= np.quantile(us, 0.95)))}
    for t in thresholds:
        out[f"recall@{t}"] = float(np.mean(ks >= t))
        out[f"false-accept@{t}"] = float(np.mean(us >= t))
    return out


def evaluate(name, net, data, test_cows, results):
    files = [f for c in test_cows for f in data[c]]
    lab = np.array([c for c in test_cows for _ in data[c]])
    E = embed(net, files)
    for k in (1, 3):
        res = [run(E, lab, s, k) for s in range(5)]
        r = {key: float(np.mean([x[key] for x in res])) for key in res[0]}
        results[f"{name} | k={k}"] = r
        log(f"{name} k={k}: top1 {r['top1']:.3f} auc {r['auc']:.3f} recall@1%FAR {r['recall@1%FAR']:.3f} "
            f"thr@1%FAR {r['thr@1%FAR']:.3f} thr@5%FAR {r['thr@5%FAR']:.3f}")


def train():
    zp = DRIVE / "face300_small.zip"
    if not SMALL.exists() and zp.exists():
        shutil.unpack_archive(zp, SMALL)
    data = {}
    for f in sorted(SMALL.rglob("*.jpg")):
        data.setdefault(f.parent.name, []).append(f)
    data = {c: v for c, v in data.items() if len(v) >= 4}
    log(len(data), "cows,", sum(len(v) for v in data.values()), "images; per cow",
        min(len(v) for v in data.values()), "to", max(len(v) for v in data.values()), "| device", DEV)
    epochs = 15 if DEV == "cuda" else 6
    cows = sorted(data)
    np.random.default_rng(0).shuffle(cows)
    test_cows, train_cows = sorted(cows[:100]), sorted(cows[100:])
    results = {"device": DEV, "epochs": epochs, "n_cows": len(cows)}
    Path("checkpoints").mkdir(exist_ok=True)

    evaluate("untrained DINOv2", Net().to(DEV).eval(), data, test_cows, results)
    files = [f for c in train_cows for f in data[c]]
    labels = [c for c in train_cows for _ in data[c]]
    log("training on", len(train_cows), "cows,", len(files), "images")
    net = fit(files, labels, train_cows, epochs)
    evaluate("trained on 200 cows", net, data, test_cows, results)
    Path("face_results.json").write_text(json.dumps(results, indent=1))
    print(json.dumps(results, indent=1), flush=True)
    del net

    log("final model on all cows")
    files = [f for c in cows for f in data[c]]
    labels = [c for c in cows for _ in data[c]]
    net = fit(files, labels, cows, epochs)
    torch.save(net.state_dict(), "checkpoints/face_embedder.pt")
    log("saved checkpoints/face_embedder.pt")
    if DRIVE.is_dir():
        shutil.copy("checkpoints/face_embedder.pt", DRIVE)
        shutil.copy("face_results.json", DRIVE)
        log("copied to Drive")
    log("ALL DONE")


if __name__ == "__main__":
    {"fetch": fetch, "train": train}[sys.argv[1]]()
