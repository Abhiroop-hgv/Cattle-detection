from dataclasses import dataclass


@dataclass
class Config:
    # Detector (COCO class 19 = cow)
    detector_weights: str = "checkpoints/yolov8m.pt"
    cow_class_id: int = 19
    detect_conf: float = 0.4

    # Face / ear-tag detector (classes: cow_face, ear_tag), used by CattleID.*_frames()
    face_detector_weights: str = "checkpoints/face_tag_detector.pt"
    face_conf: float = 0.5
    face_detector_imgsz: int = 640  # trained at 416; larger helps small faces in CCTV frames
    face_crop_pad: float = 0.1

    # Ear-tag reading (OCR) and fusion with the face match, used by identify_frames()
    read_tags: bool = True
    tag_digits: int | None = None  # exact length of your farm's tag numbers (e.g. 4); None = 3-6
    tag_max_reads: int = 5         # OCR is ~3 s/crop on CPU: read at most this many tag crops
    tag_min_votes: int = 2         # frames that must agree on a number before it is trusted
    tag_near_face: float = 0.6     # a tag counts as this animal's if it is within this fraction
                                   # of the face size around the face box

    # Tracking (IdentityTracker): carry a cow's identity along its body track
    track_min_crops: int = 3        # face crops (or tag crops = tag_min_votes) before an attempt
    track_max_crops: int = 8        # best crops kept per track between attempts
    track_max_attempts: int = 4     # give up (stay "unidentified") after this many attempts
    track_history_s: float = 600.0  # keep a track's box history this long, for identity_at()
    track_match_iou: float = 0.3    # identity_at(): minimum overlap with the queried box
    track_time_tol_s: float = 1.0   # identity_at(): how far from t a stored box may be

    # Embedder
    backbone: str = "vit_small_patch14_dinov2.lvd142m"  # timm name; swap for resnet50 etc.
    embed_dim: int = 256
    input_size: int = 224
    # Face model (300-Cattle, 15 epochs, trained on all 300 cows). None = untrained backbone only.
    checkpoint: str | None = "checkpoints/face_embedder.pt"

    # Quality gate
    min_crop_side: int = 48  # OpenCows2020 crops have median shorter side ~100px, 5th pct ~44px
    min_sharpness: float = 40.0  # variance of Laplacian

    # Matching. Cosine similarity of the average of the query photos.
    # Face model, 100 held-out cows, 3-photo gallery, 3-photo queries: 0.70 = 1% false-accept
    # (0.62 = 5%); 1 photo per query: 0.68 / 0.60. Controlled close-ups, so this is optimistic:
    # RE-TUNE on your own camera footage before trusting it.
    match_threshold: float = 0.69
    min_register_images: int = 3

    # Storage
    index_path: str = "store/index.npz"
    registry_path: str = "store/registry.json"
