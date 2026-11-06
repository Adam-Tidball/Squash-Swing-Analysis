"""M0 stand-in: local pose with MediaPipe Pose Landmarker, in the gateway's shape.

Not ViTPose. This exists so the M0 quality and key-frame tools can run with
no gateway, no key and no cost. Its numbers describe MediaPipe on this clip,
and only hint at what ViTPose would do. It also answers part of open question 2
(is local inference viable?).

BlazePose returns 33 landmarks. They are mapped onto the 17 COCO joints the rest
of the project uses, and each landmark's ``visibility`` is written as
``kpts_scores``. Output matches the gateway's ``vid.pose.kpts`` envelope, so
``keypoint_quality`` reads it unchanged.

    python -m m0.local_pose clip.mov --model pose_landmarker_heavy.task \\
        --start 1.5 --duration 3.5 --out poses.json

Get the model (Apache-2.0) from
https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task
and keep it out of git.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from swing.backends.mediapipe_local import landmarks_to_item
from swing.upstream.skeleton import KPT_NAMES


def run(clip: Path, model: Path, *, start: float = 0.0, duration: float | None = None
        ) -> tuple[dict, dict]:
    """Pose every frame of [start, start+duration). Returns (payload, info)."""
    import cv2
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model)),
        running_mode=vision.RunningMode.VIDEO, num_poses=2,
        min_pose_detection_confidence=0.5, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(str(clip))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    end = None if duration is None else start + duration

    # Frames are chosen by their decoded timestamp and numbered in decode
    # order, never by seeking to a container frame number: files with hidden
    # or duplicated packets make the two disagree.
    frames, items = [], []
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        i, last_ms = -1, -1
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            i += 1
            t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if t < start:
                continue
            if end is not None and t >= end:
                break
            image = mp.Image(image_format=mp.ImageFormat.SRGB,
                             data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            ms = max(int(round(t * 1000)), last_ms + 1)   # VIDEO mode: strictly increasing
            last_ms = ms
            result = landmarker.detect_for_video(image, ms)
            frames.append({"frame_id": i, "time": round(t, 5)})
            items += [landmarks_to_item(i, track, lms)
                      for track, lms in enumerate(result.pose_landmarks)]
    cap.release()

    payload = {"content": {"object": "vid.pose.kpts", "kpts_labels": KPT_NAMES,
                           "frames": frames, "items": items},
               "backend": {"name": "mediapipe-pose-landmarker", "model": model.name}}
    info = {"fps": fps, "width": width, "height": height, "start": start,
            "duration": duration, "n_frames": len(frames),
            "first_frame": frames[0]["frame_id"] if frames else None}
    return payload, info


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Local MediaPipe pose, gateway-shaped output.")
    ap.add_argument("clip", type=Path)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--duration", type=float)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    payload, info = run(args.clip, args.model, start=args.start, duration=args.duration)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload))
    args.out.with_suffix(".info.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info))
    return 0


if __name__ == "__main__":
    sys.exit(main())
