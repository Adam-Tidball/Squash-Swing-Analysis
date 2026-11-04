"""Every tunable, with its reason. Snapshotted into each run's ``run.json``.

The RFP's rule is that no threshold is invented silently. Values here are
either plumbing (paths, formats), safety limits, or starting points that say
where they came from and what will replace them. None is a technique
threshold: those come from coach-labelled reps in M4.

Override per run from the CLI where a flag exists. Otherwise edit here; the
snapshot records what each run used.
"""

from __future__ import annotations

from pathlib import Path

SQUASH_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = SQUASH_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"        # renditions and pose responses, reused across runs
OUTPUT_DIR = DATA_DIR / "output"      # one folder per run
RUN_STAMP_FORMAT = "%Y%m%d-%H%M%S"

# ── Capture (FR-1, FR-2) ─────────────────────────────────────────────────────
# The capture protocol asks for 240 fps. Warn below 200 rather than 240: a
# native file measures 239.98 and the 120 fps mode is a different setup, not a
# rounding error.
MIN_CAPTURE_FPS = 200.0
# Clips shorter than this cannot hold a swing window (FR-5: 1.0 s + 0.75 s).
MIN_CLIP_SECONDS = 1.75

# ── Renditions ───────────────────────────────────────────────────────────────
# Upstream's choice: ViTPose caps the long edge at 2048 px internally, so more
# is upload time for nothing. 1080 is upstream's tested value.
INFERENCE_HEIGHT = 1080
# Visually lossless for H.264. Upstream uses 23 for its uploads; 18 because a
# 240 fps window is short and wrist detail is what the fine pass is for.
CONVERT_CRF = 18
# Bytes hashed from each end of a source file for its cache identity, plus its
# size. Hashing a whole 2 GB slo-mo file on every run is slow; a re-export or
# a trim changes the ends. Renditions (small) are hashed in full.
SOURCE_HASH_BYTES = 4 * 1024 * 1024

# ── Coarse pass (FR-4) ───────────────────────────────────────────────────────
# The RFP default. PROPOSAL.md recommends 15 pending M0's price per frame. The
# gateway does not subsample (`video_fps` is detector cadence), so this rate is
# produced locally with ffmpeg before upload.
COARSE_FPS = 30.0
# Frames per gateway request. 900 is the gateway's own default cap when
# `video_max_frames` is unset; staying at it avoids relying on an
# unverified larger limit. M0 measures the real limit.
CHUNK_FRAMES = 900

# ── Fine pass (FR-5) ─────────────────────────────────────────────────────────
WINDOW_BEFORE_S = 1.0    # RFP default
WINDOW_AFTER_S = 0.75    # RFP default

# ── Pose ─────────────────────────────────────────────────────────────────────
POSE_MODEL = "usyd-community/vitpose-plus-large"
GATEWAY_BASE_URL = "https://gateway.vlm.run/v1/openai"
REQUEST_TIMEOUT = 1800.0  # upstream: video pose takes minutes, not seconds
PRECISION = 4             # upstream: decimal places on normalized coordinates
PERSON_IOU = 0.6          # upstream: above this, two boxes are the same person

# Local stand-in (MediaPipe Pose Landmarker). Not part of the RFP's stack;
# see DECISIONS.md. The model file is downloaded by hand and kept out of git.
MEDIAPIPE_MODEL = DATA_DIR / "models" / "pose_landmarker_heavy.task"

# ── Input checks after pose (FR-2, FR-6) ─────────────────────────────────────
# Starting points for warnings, not verdicts. Each says what it guards.
# A box this close to the frame edge (normalized) counts as touching it.
EDGE_MARGIN = 0.01
# Warn when the player's box touches an edge in more than this share of
# posed frames: an occasional lunge to the edge is fine, a cropped player is not.
CLIPPED_WARN_SHARE = 0.05
# Warn when a second distinct person appears in more than this share of
# frames (solo-practice scope: one visible player).
MULTI_PERSON_WARN_SHARE = 0.05
# Warn when nobody is detected in more than this share of frames.
NOBODY_WARN_SHARE = 0.10


def snapshot() -> dict:
    """Every setting above, as plain JSON values, for ``run.json``."""
    out = {}
    for name, value in globals().items():
        if name.isupper():
            out[name] = str(value) if isinstance(value, Path) else value
    return out
