"""MediaPipe Pose Landmarker, locally, in the gateway's response shape.

A stand-in, not the RFP's model: no key, no cost, runs on a laptop CPU. On
the two M0 test clips it returned useful per-keypoint visibility (see
``m0/findings``). Whether it can replace ViTPose is open question 2.

BlazePose's 33 landmarks map onto COCO-17; ``visibility`` becomes
``kpts_scores``. Landmarks outside the frame become ``(0, 0)``, the gateway's
"not visible" sentinel.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import config as cfg
from ..upstream.skeleton import KPT_NAMES

# BlazePose landmark index for each COCO-17 joint, in KPT_NAMES order.
BLAZE_FOR_COCO = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]


def landmarks_to_item(frame_id: int, track: int, lms) -> dict:
    """One BlazePose result as a gateway person record."""
    pts = [lms[b] for b in BLAZE_FOR_COCO]
    xs = np.clip([p.x for p in lms], 0, 1)
    ys = np.clip([p.y for p in lms], 0, 1)
    x0, y0, x1, y1 = float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())
    return {
        "frame_id": frame_id, "track_id": track,
        "bbox_xywh": [round(x0, 4), round(y0, 4), round(x1 - x0, 4), round(y1 - y0, 4)],
        "kpts_xy": [[round(p.x, 4), round(p.y, 4)] if 0 <= p.x <= 1 and 0 <= p.y <= 1
                    else [0, 0] for p in pts],
        "kpts_scores": [round(float(p.visibility), 4) for p in pts],
    }


class MediaPipePose:
    name = "mediapipe"
    billable = False

    def __init__(self, model: Path = cfg.MEDIAPIPE_MODEL, *, num_poses: int = 2):
        self.model = Path(model)
        self.num_poses = num_poses    # 2, so a second person is seen and can be warned about

    def describe(self) -> dict:
        try:
            import mediapipe
            version = mediapipe.__version__
        except ImportError:
            version = None
        return {"backend": self.name, "model": self.model.name, "mediapipe": version,
                "num_poses": self.num_poses}

    def estimate(self, video: Path, *, n_frames: int) -> tuple[dict, dict | None]:
        if not self.model.is_file():
            raise FileNotFoundError(
                f"MediaPipe model not found at {self.model}. Download "
                "pose_landmarker_heavy.task (see README) into squash/data/models/.")
        import cv2
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(self.model)),
            running_mode=vision.RunningMode.VIDEO, num_poses=self.num_poses)
        cap = cv2.VideoCapture(str(video))
        frames, items = [], []
        try:
            with vision.PoseLandmarker.create_from_options(options) as landmarker:
                i, last_ms = 0, -1
                while True:
                    ok, bgr = cap.read()
                    if not ok:
                        break
                    ms = max(int(round(cap.get(cv2.CAP_PROP_POS_MSEC))), last_ms + 1)
                    last_ms = ms
                    image = mp.Image(image_format=mp.ImageFormat.SRGB,
                                     data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                    result = landmarker.detect_for_video(image, ms)
                    frames.append({"frame_id": i})
                    items += [landmarks_to_item(i, t, lms)
                              for t, lms in enumerate(result.pose_landmarks)]
                    i += 1
        finally:
            cap.release()
        payload = {"content": {"object": "vid.pose.kpts", "kpts_labels": KPT_NAMES,
                               "frames": frames, "items": items}}
        return payload, {"frames_posed": len(frames), "billed": 0}
