# cattle_id: cattle re-identification module

Identifies individual cattle from photos of the FACE (front view, head and face filling the image).
Built to be plugged into a larger app later. Pass full photos/frames to the `*_frames()` calls
(a YOLO detector finds the face first) or ready-made face crops to `register()` / `identify()`.
Ear-tag READING (OCR) is not built yet; the detector already finds tags (`FaceTagDetector.detect`).

## Integration contract

```python
from cattle_id import CattleID, Config
system = CattleID(Config())  # defaults to checkpoints/face_embedder.pt

# Registration service: crops = 3+ images of ONE animal
system.register(crops, {"name": "Gauri", "height_cm": 130, ...})
# -> {"status": "registered", "cattle_id": "COW-1A2B3C4D"}
# -> {"status": "duplicate", "existing_id": ..., "details": {...}, "message": ...}
# -> {"status": "rejected", "reason": "not_enough_good_images", ...}

# Identification service: crops = 1+ images of ONE animal (e.g. one tracked cow)
system.identify(crops)
# -> {"status": "found", "cattle_id": ..., "score": 0.91, "details": {name, height_cm, ...}}
# -> {"status": "not_found", "message": "Cattle not found. Please go and register."}
# -> {"status": "rejected", "reason": "blurry" | "too_small" | "no_images"}

# Same two services starting from FULL frames/photos (detector finds the face first)
system.register_frames(frames, details)   # frames with no face / several faces are skipped
system.identify_frames(frames)            # frames of ONE animal -> same results as identify()
# -> {"status": "rejected", "reason": "no_cattle_face"} when no cattle face is found, so a person,
#    an empty frame or a cow's back is NOT matched to a registered cow
system.identify_frame(frame)              # one CCTV frame, many animals -> [{box, status, ...}, ...]

# CCTV video -> {track_id: identify-result}
from cattle_id.video import identify_video
identify_video("barn.mp4", system)
```

Registration and identification are separate calls. In the integrated app, expose them as two
services; give the identification one read-only access to storage.

## Swappable parts (all behind tiny interfaces)

| Part | Now | Later |
|---|---|---|
| `index.py` `NumpyIndex` | local file | Vertex AI Vector Search |
| `registry.py` `JsonRegistry` | local JSON | `FirestoreRegistry` (included) |
| `detector.py` face + ear-tag detector | YOLOv8s on merged Roboflow sets (`checkpoints/face_tag_detector.pt`) | fine-tuned on your barn footage |
| `video.py` whole-cow tracker | YOLOv8 pretrained (COCO "cow") | fine-tuned on your barn footage |

## Workflow

1. **Get data** into `data/crops/<animal_id>/*.jpg` (cropped cows, 20+ images per animal, many
   animals). Public overhead datasets (e.g. OpenCows2020, Cows2021) are a starting point; check
   licenses and write a small prep script to crop them into this layout.
2. **Baseline** with no training: `python -m cattle_id.evaluate --data_dir data/crops`
3. **Train** (Colab GPU is enough): `python -m cattle_id.train --data_dir data/crops`
4. **Evaluate** on held-out animals: `python -m cattle_id.evaluate --data_dir data/crops --checkpoint checkpoints/embedder.pt`
   and copy the suggested threshold into `Config.match_threshold`.
5. Move to Vertex AI custom training later; `train.py` is a plain script, so it wraps directly.

Test logic without models: `python -m pytest tests`

## Results (face model, 300-Cattle dataset)

DINOv2 ViT-S, last 2 blocks fine-tuned, bias-free 256-d head, ArcFace, 15 epochs (T4 GPU).
Trained on 200 cows, tested on 100 unseen cows (half enrolled with 3 photos, half unknown), 5 splits.

| | 1 photo | average of 3 photos |
|---|---|---|
| right cow (top-1) | 93.0% | 96.4% |
| enrolled cows recognised at 1% false-accept | 76.6% | 83.7% |
| untrained base model, top-1 | 78.5% | 82.8% |

Thresholds for 1% false-accept: 0.68 (1 photo) / 0.70 (3 photos); 5%: 0.60 / 0.62. Default
`match_threshold` is 0.69. The shipped weights (`checkpoints/face_embedder.pt`) are the same recipe
trained on all 300 cows, so they cannot be tested on those cows.
These are controlled close-ups, probably one photo session per cow: real CCTV will score lower.
Re-tune the threshold on your own camera footage before relying on it.

## Face / ear-tag detector

YOLOv8s, 416 px, 40 epochs, trained on a merge of CC BY 4.0 Roboflow Universe sets: Lupora "Cattle
Face Detection" (face), Beef Cattle Face Detection "Cattle Biometric Identification" (tag; face box
built from its muzzle + head-whorl boxes), Graphic Era Hill University "cattle feature extraction"
(tag). Credit the authors if you redistribute. Validation (1,459 images from the same sources):
cow_face recall 98.8% / precision 99.4% / mAP50 99.5%; ear_tag recall 90.5% / precision 93.5% /
mAP50 95.9%. Optimistic: same farms and animals as training, no empty-scene or person images.

## Duplicate override and group photos

`register(..., force=True)` / `register_frames(..., force=True)` registers a cow even though its face
matches an existing one (the farmer confirmed they are different animals, e.g. look-alikes). The
new record gets `confirmed_different_from` for the audit trail and the result has
`overrode_duplicate_of`. A tag number already in use is never overridden.
`identify_frame(frame, use_tags=True)` identifies every face in one photo/frame (each with its
`box`) and fuses the tag found next to each face; a single read is used (one frame cannot vote), so
a tag that names a different cow than the face shows up as `conflict`.

## Ear-tag number and fusion with the face match

`register(crops, details, tag_number=None)` / `register_frames(frames, details, tag_number=None)`
store the tag number (digits only) in `details["tag_number"]`. A tag number already registered to
another animal is rejected as `duplicate` with `reason: "tag_already_registered"`. If the photos show
a different number than the one typed, registration still succeeds but carries a `tag_check` warning.
Registries need `find_by_tag(tag)` (implemented for `JsonRegistry` and `FirestoreRegistry`).

`identify_frames(frames)` reads the tag (RapidOCR on the detector's ear-tag crops near the face;
a number is trusted only when `tag_min_votes` frames agree) and fuses it with the face match:

| face | tag read | result |
|---|---|---|
| matches cow A | registered to A | `found`, `evidence: "face+tag"` |
| matches cow A | registered to B | `conflict` (both candidates returned: a person should look) |
| matches cow A | not registered / unreadable | `found`, `evidence: "face"` (`tag_status` set if a number was read) |
| no match / no face | registered to B | `found`, `evidence: "tag"` (face did not match) |
| no match | none | `not_found` ("Cattle not found. Please go and register.") |

Set `tag_digits` in `Config` to your tags' exact length (e.g. 4): reads of any other length are
dropped, which removes most OCR slips. Measured on 40 real, sharp tag crops (beef cattle, printed
4-digit numbers; numbers read by eye): 21 read, 14 right, 2 wrong, 5 unjudged, 19 unreadable. Expect
less on CCTV footage; ~3 s per crop on CPU, so read a few good frames per tracked cow, not every frame.
Untested on dairy tags (different colours/formats) and on real footage.

## Tracking: which cow is this, in this camera, at this time

For the health models (posture, sound, ...): `IdentityTracker(service, camera_id="pen-3")`, one per
camera. Call `update(frame, t)` on every frame (PIL image, `t` in seconds). It tracks cow bodies
(ByteTrack, COCO `cow` class), gives each detected face / ear tag to the body track that contains
it, and identifies a track once it has `track_min_crops` face crops (or enough tag crops):

- face + tag agree -> identified at once (`evidence: "face+tag"`);
- otherwise two independent attempts must name the same cow (`evidence: "face"`);
- face and tag name different cows -> `status: "conflict"`; face and tag match nobody twice ->
  `status: "unregistered"`; not enough evidence yet -> `status: "unidentified"`.

`update()` returns one record per tracked cow (`track_id, box, status, cattle_id, details, ...`).
`identity_at(box, t)` answers "which cow was in this box at time t" for a detection made on the SAME
camera (best box overlap at that moment; `status: "no_cow_at_box"` if none): this is what a posture
model calls with the box of a lying cow. `identity_of(track_id)` looks up a track directly.

Try it on a video file (every 5th frame, one JSON line per cow per frame, summary at the end):

    python -m cattle_id.run_video clip.mp4 --camera pen-3 --stride 5 --out identities.jsonl

Limits: identity lives on a track; when a track breaks (long occlusion, cow leaves and returns) the
new track is unidentified until it collects evidence again. Cameras are independent: an overhead
posture camera cannot see faces, so it needs the identification camera's view to overlap it (or a
zone/calibration mapping, not built). Sound cannot be attributed to a cow without microphones near
it. Tested with scripted fake frames (logic) and a real-weights smoke run on still frames;
NOT yet tested on continuous video from a real barn.

## Demo app (hackathon)

    streamlit run demo_app.py

Tabs: **Register** (name, height, tag number, 3+ photos; a face duplicate can be overridden with
"register anyway"), **Group photo** (every cow in one photo boxed and named), **Identify** (upload a photo: found /
"Cattle not found. Please go and register." / conflict), **Health alert** (pick a simulated alert,
upload the camera frame: "Lying down >3 h - Gauri (tag 5804). Please check this cow."), and the list
of registered cattle. Data is kept in `demo_store/` (the Reset button deletes it). The alerts are
simulated; identification is the real pipeline. Demo it on photos of cows that face the camera.

## Known limits

- The face detector was trained on mostly frontal faces: on a side-on feed-barrier view (YouTube
  clip, 7 frames) it found no faces, while the ear-tag class fired on some real tags and on
  ceiling lights/hay. Place identification cameras head-on (gate, walkway exit, milking entry).

- Solid-colored animals have little visual signal; expect more confusion in mixed herds.
  Consider an ear tag/RFID fallback for those.
- The embedding model only separates cow faces from other cow faces: a non-cow image scores ~0.93
  against registered cows. The `*_frames()` calls guard against this with the detector; calling
  `identify()` / `register()` directly with crops does not. The detector's false-positive rate on
  real empty scenes / people is untested (needs your camera footage).
- `match_threshold` comes from held-out close-up photos. Tune it per camera setup.
- Re-tune after changing camera, lighting, or model.
