# Copied from vision-demos running/src/skeleton.py (Apache-2.0,
# https://github.com/jeremyipark/vision-demos). Modified for squash:
#   - kept only KPT_NAMES; drawing arrives with rendering in M3

"""COCO-17 keypoint order, as the pose model returns it."""

KPT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
INDEX = {name: i for i, name in enumerate(KPT_NAMES)}
