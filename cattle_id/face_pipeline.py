"""Face model on the 300-Cattle dataset (one folder per animal, about 10 frontal photos each).

Trains on 200 cows, tests on 100 held-out cows (some enrolled, some unknown), compares against the
untrained backbone, then trains the final model on all cows and saves it.

  python -u -m cattle_id.face_pipeline
"""
import json
import shutil
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

from .config import Config
from .embedder import Embedder
from .evaluate_external import run
from .final import fit
from .train import IMG_EXT

ZIP = Path("/content/face300.zip")
RAW = Path("/content/face300")
SMALL = Path("/content/face300_small")
DRIVE = Path("/content/drive/MyDrive/cattle_id")
ROOT_NAME = "300-Cattle-source"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def label_of(f):
    parts = f.relative_to(RAW).parts
    return parts[parts.index(ROOT_NAME) + 1] if ROOT_NAME in parts else f.parent.name


def shrink(job):
    src, dst = job
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        im = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
        im.thumbnail((640, 640))
        im.save(dst, quality=92)
    except Exception as e:
        log("skip", src.name, e)


def prepare():
    if not SMALL.exists():
        if not RAW.exists():
            log("unzipping", ZIP.stat().st_size // 2**20, "MB")
            zipfile.ZipFile(ZIP).extractall(RAW)
        files = [f for f in RAW.rglob("*") if f.is_file() and f.suffix.lower() in IMG_EXT]
        log(len(files), "images found; shrinking to max side 640")
        jobs = [(f, SMALL / label_of(f) / f"{f.parent.name}_{f.stem}.jpg") for f in files]
        with ThreadPoolExecutor(4) as ex:
            list(ex.map(shrink, jobs))
        shutil.rmtree(RAW, ignore_errors=True)
    data = {}
    for f in sorted(SMALL.rglob("*.jpg")):
        data.setdefault(f.parent.name, []).append(f)
    data = {c: v for c, v in data.items() if len(v) >= 4}
    log(len(data), "cows,", sum(len(v) for v in data.values()), "images; images per cow",
        min(len(v) for v in data.values()), "to", max(len(v) for v in data.values()))
    return data


def embed_all(ck, data, cows):
    emb = Embedder(Config(checkpoint=ck))
    E, lab = [], []
    for c in cows:
        fs = data[c]
        for i in range(0, len(fs), 32):
            E.append(emb.embed([Image.open(f).convert("RGB") for f in fs[i:i + 32]]))
        lab += [c] * len(fs)
    del emb
    torch.cuda.empty_cache()
    return np.concatenate(E), np.array(lab)


def evaluate(name, ck, data, test_cows, results):
    E, lab = embed_all(ck, data, test_cows)
    for k in (1, 3):
        res = [run(E, lab, s, k, gallery=3) for s in range(5)]
        results[f"{name} | k={k}"] = {key: float(np.mean([r[0][key] for r in res])) for key in res[0][0]}
        r = results[f"{name} | k={k}"]
        log(f"{name} k={k}: top1 {r['top1']:.3f} auc {r['auc']:.3f} recall@1%FAR {r['recall@1%FAR(own thr)']:.3f} "
            f"thr {r['own thr']:.3f}")


def main():
    data = prepare()
    cows = sorted(data)
    np.random.default_rng(0).shuffle(cows)
    test_cows, train_cows = sorted(cows[:100]), sorted(cows[100:])
    results = {}
    Path("checkpoints").mkdir(exist_ok=True)

    evaluate("untrained DINOv2", None, data, test_cows, results)

    files = [f for c in train_cows for f in data[c]]
    labels = [c for c in train_cows for _ in data[c]]
    log("training on", len(train_cows), "cows,", len(files), "images")
    net, _ = fit(files, labels, train_cows, 2, 15, 0)
    torch.save(net.state_dict(), "checkpoints/face_eval.pt")
    del net
    torch.cuda.empty_cache()
    evaluate("face model (trained on 200 cows)", "checkpoints/face_eval.pt", data, test_cows, results)

    print(json.dumps(results, indent=1))
    Path("face_results.json").write_text(json.dumps(results, indent=1))

    log("training final model on all cows")
    files = [f for c in cows for f in data[c]]
    labels = [c for c in cows for _ in data[c]]
    net, _ = fit(files, labels, sorted(cows), 2, 15, 0)
    torch.save(net.state_dict(), "checkpoints/face_embedder.pt")
    if DRIVE.is_dir():
        shutil.copy("checkpoints/face_embedder.pt", DRIVE)
        shutil.copy("face_results.json", DRIVE)
        log("copied face_embedder.pt + face_results.json to Drive")
    log("ALL DONE")


if __name__ == "__main__":
    main()
