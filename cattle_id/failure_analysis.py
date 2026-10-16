"""Why does re-identification fail on Cows2021 Test? Image-level and cow-level diagnostics.

  python -u -m cattle_id.failure_analysis --checkpoint checkpoints/embedder_v2.pt
"""
import argparse
import collections
import re

import numpy as np
from PIL import Image

from .config import Config
from .embedder import Embedder
from .pipeline_v2 import TEST
from .quality import sharpness
from .train import IMG_EXT


def date_of(f):
    m = re.search(r"\d{4}-\d{2}-\d{2}", f.name)
    return m.group(0) if m else ""


def l2(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def collect(emb, per_cow=40, min_images=15):
    rows, E = [], []
    for d in sorted(p for p in TEST.iterdir() if p.is_dir()):
        fs = [f for f in d.iterdir() if f.suffix.lower() in IMG_EXT]
        if len(fs) < min_images:
            continue
        fs = sorted(fs, key=lambda f: (date_of(f), f.name))
        fs = [fs[i] for i in np.unique(np.linspace(0, len(fs) - 1, min(per_cow, len(fs))).astype(int))]
        for i in range(0, len(fs), 32):
            chunk = fs[i:i + 32]
            ims = [Image.open(f).convert("RGB") for f in chunk]
            E.append(emb.embed(ims))
            rows += [(d.name, date_of(f), f, min(im.size), sharpness(im)) for f, im in zip(chunk, ims)]
    return rows, np.concatenate(E)


def gallery_eval(E, cows, mode, k=5, gallery=5, seeds=3):
    out = []
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        cs = sorted(set(cows))
        rng.shuffle(cs)
        known, unk = cs[: len(cs) // 2], cs[len(cs) // 2:]
        ids, G, kq = [], [], []
        for c in known:
            idx = np.where(cows == c)[0]
            g = idx[:gallery] if mode == "earliest" else idx[np.unique(np.linspace(0, len(idx) - 1, gallery).astype(int))]
            rest = np.array([i for i in idx if i not in set(g)])
            ids.append(c)
            G.append(E[g].mean(0))
            kq += [(c, l2(E[rest[i:i + k]].mean(0))) for i in range(0, len(rest) - k + 1, k)]
        G = l2(np.stack(G))
        ks = np.array([(G @ q).max() for _, q in kq])
        top1 = np.mean([ids[int((G @ q).argmax())] == c for c, q in kq])
        us = np.array([(G @ l2(E[np.where(cows == c)[0][i:i + k]].mean(0))).max()
                       for c in unk for i in range(0, len(np.where(cows == c)[0]) - k + 1, k)])
        out.append((top1, np.mean(ks[:, None] > us[None, :])))
    return np.mean(out, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/embedder_v2.pt")
    args = ap.parse_args()
    rows, E = collect(Embedder(Config(checkpoint=args.checkpoint)))
    cows = np.array([r[0] for r in rows])
    dates = np.array([r[1] for r in rows])
    ms = np.array([r[3] for r in rows])
    sh = np.array([r[4] for r in rows])
    n = len(E)
    print(f"{n} images, {len(set(cows))} cows, {len(set(dates))} dates")
    S = E @ E.T
    same_cow = cows[:, None] == cows[None, :]
    same_date = dates[:, None] == dates[None, :]
    eye = np.eye(n, dtype=bool)

    def nn(excl):
        S2 = np.where(excl, -9.0, S)
        j = S2.argmax(1)
        return j, cows[j] == cows

    j_any, ok_any = nn(eye)
    j_x, ok_x = nn(eye | (same_cow & same_date))
    print("\n[1] single-image nearest neighbour accuracy (all other images are the gallery)")
    print(f"   neighbour may be from the same day : {ok_any.mean():.3f}")
    print(f"   same-cow-same-day neighbours removed: {ok_x.mean():.3f}   <- the day-to-day change")

    print("\n[2] similarity of pairs (cosine)")
    off = ~eye
    print(f"   same cow, same day   : {S[same_cow & same_date & off].mean():.3f}")
    print(f"   same cow, other days : {S[same_cow & ~same_date].mean():.3f}")
    print(f"   different cows       : {S[~same_cow].mean():.3f}   (mean of best different-cow match per image: "
          f"{np.where(same_cow, -9, S).max(1).mean():.3f})")

    print("\n[3] cross-day accuracy by crop size (shorter side, px)")
    for lo, hi in [(0, 60), (60, 100), (100, 200), (200, 9999)]:
        m = (ms >= lo) & (ms < hi)
        print(f"   {lo:>3}-{hi:<4} n={m.sum():5d}  acc={ok_x[m].mean():.3f}" if m.sum() else f"   {lo}-{hi} none")
    print("   by sharpness quartile:")
    qs = np.quantile(sh, [0, .25, .5, .75, 1])
    for a, b in zip(qs[:-1], qs[1:]):
        m = (sh >= a) & (sh <= b)
        print(f"   {a:7.0f}-{b:<7.0f} n={m.sum():5d}  acc={ok_x[m].mean():.3f}")

    per_cow = {c: ok_x[cows == c].mean() for c in set(cows)}
    v = np.array(list(per_cow.values()))
    print(f"\n[4] per-cow cross-day accuracy: <20%: {(v < .2).sum()} cows, 20-60%: {((v >= .2) & (v < .6)).sum()}, "
          f">=60%: {(v >= .6).sum()} (of {len(v)})")
    wrong = collections.Counter((cows[i], cows[j_x[i]]) for i in range(n) if not ok_x[i])
    print("   most frequent confusions (true -> predicted): " + ", ".join(f"{a}->{b} x{c}" for (a, b), c in wrong.most_common(6)))
    partner = collections.Counter(cows[j_x[i]] for i in range(n) if not ok_x[i])
    print("   cows that attract the most wrong matches   : " + ", ".join(f"{c} x{k}" for c, k in partner.most_common(6)))

    print("\n[5] how the cow is enrolled (5 crops per query, unknown-cow rejection AUC)")
    for mode in ("earliest", "spread"):
        t1, auc = gallery_eval(E, cows, mode)
        print(f"   gallery = {'5 earliest consecutive crops' if mode == 'earliest' else '5 crops spread over the cow time span':42s} top1 {t1:.3f}  auc {auc:.3f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(0)
    bad = np.where(~ok_x)[0]
    pick = rng.choice(bad, size=min(8, len(bad)), replace=False)
    fig, ax = plt.subplots(len(pick), 3, figsize=(6.5, 2.0 * len(pick)))
    for r, i in enumerate(pick):
        true_best = np.where(~eye[i] & same_cow[i] & ~same_date[i])[0]
        t = true_best[S[i, true_best].argmax()] if len(true_best) else i
        for c, (idx, title) in enumerate([(i, f"query cow {cows[i]}"),
                                          (j_x[i], f"WRONG match {cows[j_x[i]]} {S[i, j_x[i]]:.2f}"),
                                          (t, f"closest true {S[i, t]:.2f}")]):
            ax[r, c].imshow(Image.open(rows[idx][2]).convert("RGB"))
            ax[r, c].set_title(title, fontsize=7)
            ax[r, c].axis("off")
    plt.tight_layout()
    plt.savefig("failures.png", dpi=110)
    print("\nsaved failures.png")


if __name__ == "__main__":
    main()
