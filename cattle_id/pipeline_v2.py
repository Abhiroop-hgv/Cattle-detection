"""v2: add Cows2021 'Train' cows whose IDs are NOT in Cows2021 'Test', retrain, compare on Test.

Run in Colab after the Cows2021 Test subset is in /content/cows2021_raw:
  python -u -m cattle_id.pipeline_v2
"""
import collections
import glob
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from remotezip import RemoteZip

from .config import Config
from .embedder import Embedder
from .evaluate_external import load, run
from .final import fit

URL = "https://data.bris.ac.uk/datasets/tar/4vnrca7qw1642qlwxjadp87h7.zip"
RAW = Path("/content/c21_train_raw")
TEST = Path(glob.glob("/content/cows2021_raw/*/Sub-levels/Identification/Test")[0])
PER_COW = 40
DRIVE = Path("/content/drive/MyDrive/cattle_id")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def cow_of(name):
    return name.split("/RGB/")[1].split("/")[0]


def fetch():
    test_ids = {p.name for p in TEST.iterdir() if p.is_dir()}
    with RemoteZip(URL) as z:
        infos = [i.filename for i in z.infolist()
                 if not i.is_dir() and "/Identification/Train/" in i.filename
                 and "/RGB/" in i.filename and i.filename.endswith(".jpg")]
    by = collections.defaultdict(list)
    for n in infos:
        by[cow_of(n)].append(n)
    names = []
    for cid, v in sorted(by.items()):
        if cid in test_ids:
            continue
        v = sorted(v)
        idx = np.unique(np.linspace(0, len(v) - 1, min(PER_COW, len(v))).astype(int))
        names += [v[i] for i in idx]
    log(len(names), "files for", len({cow_of(n) for n in names}), "train cows whose IDs are not in Test")

    def work(chunk):
        z = RemoteZip(URL)
        for n in chunk:
            if (RAW / n).exists():
                continue
            for t in range(8):
                try:
                    z.extract(n, str(RAW))
                    break
                except Exception:
                    time.sleep(2 + 2 * t)
                    z = RemoteZip(URL)

    list(ThreadPoolExecutor(8).map(work, [names[k::8] for k in range(8)]))
    log("fetched", sum((RAW / n).exists() for n in names), "/", len(names))
    return names


def build(names):
    files, labels = [], []
    for d in sorted(Path("data/crops").iterdir()):  # OpenCows2020, capped per cow
        fs = sorted(d.glob("*.jpg"))
        for i in np.unique(np.linspace(0, len(fs) - 1, min(60, len(fs))).astype(int)):
            files.append(fs[i])
            labels.append("oc_" + d.name)
    for n in names:
        p = RAW / n
        try:
            Image.open(p).verify()
        except Exception:
            continue
        files.append(p)
        labels.append("c21_" + cow_of(n))
    log(len(files), "training images,", len(set(labels)), "cows")
    return files, labels


def main():
    names = fetch()
    files, labels = build(names)
    net, _ = fit(files, labels, sorted(set(labels)), 2, 8, 0)
    Path("checkpoints").mkdir(exist_ok=True)
    torch.save(net.state_dict(), "checkpoints/embedder_v2.pt")
    del net
    torch.cuda.empty_cache()
    log("saved checkpoints/embedder_v2.pt")

    results = {}
    for name, ck in [("untrained DINOv2", None), ("v1: OpenCows2020 only", "checkpoints/embedder.pt"),
                     ("v2: + Cows2021 train cows", "checkpoints/embedder_v2.pt")]:
        emb = Embedder(Config(checkpoint=ck))
        E, lab = load(emb, TEST, 40, 15)
        del emb
        for k in (1, 5):
            res = [run(E, lab, s, k) for s in range(3)]
            results[f"{name} | k={k}"] = {key: float(np.mean([r[0][key] for r in res])) for key in res[0][0]}
            results[f"{name} | k={k}"]["n_cows"] = len(set(lab))
        log("evaluated", name)
    print(json.dumps(results, indent=1))
    Path("pipeline_v2_results.json").write_text(json.dumps(results, indent=1))
    if DRIVE.is_dir():
        import shutil
        shutil.copy("checkpoints/embedder_v2.pt", DRIVE)
        shutil.copy("pipeline_v2_results.json", DRIVE)
        log("copied checkpoint + results to Drive")
    log("ALL DONE")


if __name__ == "__main__":
    main()
