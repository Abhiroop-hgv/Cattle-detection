"""Fetch the 300-Cattle face dataset one file at a time over WebDAV, shrink each image, and keep
only the small copies (a few hundred MB instead of ~15 GB). Resumable; also zips the result to Drive.

  python -u -m cattle_id.face_fetch
"""
import io
import shutil
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import unquote

import requests
from PIL import Image, ImageOps

from .train import IMG_EXT

TOKEN = "eMwaHAPK08dCDru"
BASE = "https://cloud.une.edu.au/public.php/webdav/300-Cattle-source/"
SMALL = Path("/content/face300_small")
DRIVE = Path("/content/drive/MyDrive/cattle_id")
NS = {"d": "DAV:"}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def session():
    s = requests.Session()
    s.auth = (TOKEN, "")
    return s


def ls(s, url):
    for t in range(6):
        try:
            r = s.request("PROPFIND", url, headers={"Depth": "1"}, timeout=60)
            r.raise_for_status()
            hrefs = [unquote(h.text) for h in ET.fromstring(r.content).iterfind(".//d:href", NS)]
            return [h.rsplit("/", 2)[-2] if h.endswith("/") else h.rsplit("/", 1)[-1] for h in hrefs[1:]]
        except Exception as e:
            log("ls retry", url, e)
            time.sleep(3 + 3 * t)
    return []


def grab(job):
    cow, name = job
    dst = SMALL / cow / (Path(name).stem + ".jpg")
    if dst.exists():
        return True
    s = session()
    for t in range(6):
        try:
            r = s.get(f"{BASE}{cow}/{name}", timeout=120)
            r.raise_for_status()
            im = ImageOps.exif_transpose(Image.open(io.BytesIO(r.content))).convert("RGB")
            im.thumbnail((640, 640))
            dst.parent.mkdir(parents=True, exist_ok=True)
            im.save(dst, quality=92)
            return True
        except Exception as e:
            if t == 5:
                log("failed", cow, name, e)
            time.sleep(2 + 2 * t)
    return False


def main():
    s = session()
    cows = [c for c in ls(s, BASE)]
    log(len(cows), "cow folders")
    jobs = []
    for c in cows:
        for n in ls(s, f"{BASE}{c}/"):
            if Path(n).suffix.lower() in IMG_EXT:
                jobs.append((c, n))
    log(len(jobs), "images to fetch")
    done = 0
    with ThreadPoolExecutor(8) as ex:
        for ok in ex.map(grab, jobs):
            done += ok
            if done % 200 == 0:
                log("fetched", done, "/", len(jobs))
    log("fetched", done, "/", len(jobs))
    if DRIVE.is_dir():
        shutil.make_archive("/content/face300_small", "zip", SMALL)
        shutil.copy("/content/face300_small.zip", DRIVE)
        log("saved face300_small.zip to Drive")
    log("FETCH DONE")


if __name__ == "__main__":
    main()
