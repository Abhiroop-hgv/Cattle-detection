"""Convert OpenCows2020 (identification part) into cattle_id's layout.

Source layout (per the dataset ReadMe):
    <root>/identification/images/{train,test}/<001..046>/*.jpg
Output layout:
    <out>/<animal_id>/*.jpg

train/ and test/ are merged: both contain the same 46 cows, and cattle_id.train /
cattle_id.evaluate re-split BY ANIMAL (whole cows held out), which is what open-set
identification needs.

  python scripts/prep_opencows2020.py --src /path/to/unzipped --out data/crops
"""
import argparse
import shutil
from pathlib import Path

EXT = {".jpg", ".jpeg", ".png"}


def find_images_root(src: Path) -> Path:
    for p in src.rglob("images"):
        if (p / "train").is_dir() and (p / "test").is_dir():
            return p
    raise SystemExit(f"No identification/images/{{train,test}} folder found under {src}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="unzipped OpenCows2020 folder")
    ap.add_argument("--out", default="data/crops")
    args = ap.parse_args()

    root = find_images_root(Path(args.src))
    out = Path(args.out)
    n_img = 0
    for split in ("train", "test"):
        for cow_dir in sorted(p for p in (root / split).iterdir() if p.is_dir()):
            dest = out / f"cow_{cow_dir.name}"
            dest.mkdir(parents=True, exist_ok=True)
            for f in cow_dir.iterdir():
                if f.suffix.lower() in EXT:
                    shutil.copy2(f, dest / f"{split}_{f.name}")
                    n_img += 1
    n_cows = sum(1 for p in out.iterdir() if p.is_dir())
    print(f"Wrote {n_img} images for {n_cows} animals to {out}")


if __name__ == "__main__":
    main()
