"""M0: can a vision model read the racket from key frames?

Keypoints stop at the wrist, so racket preparation, racket face and the
racket half of follow-through (RFP checks) depend on a vision read. This asks
fixed-choice questions of three frames per swing through TypeSafe System One
on the VLM Run Gateway, the same pattern as ``deadlift/src/classify.py``:
the answer is constrained to the labels and comes back as label
probabilities, never free text.

Key frames come from the M0 run's poses, not from a guess:

* contact   -- peak hitting-wrist speed (the RFP's contact proxy)
* backswing -- highest hitting wrist in the 0.6 s before contact
* finish    -- highest hitting wrist in the 0.6 s after contact

Each frame is cropped to the player's box, padded generously because the
racket reaches well past the body, and saved as a JPEG beside the answers so
every read can be checked by eye.

    python -m m0.keyframe_probe ../running/data/output/<run-id> --dry-run
    python -m m0.keyframe_probe ../running/data/output/<run-id>

``--dry-run`` writes the crops and the exact questions, and calls nothing.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import sys
import time
from pathlib import Path

import numpy as np

from . import keypoint_quality as kq

READ_MODEL = "google/gemma-4-26b-a4b-it"     # the model deadlift settled on
TYPESAFE_BASE_URL = "https://gateway.vlm.run/typesafe"
IMAGE_LONG_EDGE = 1024
IMAGE_DETAIL = "high"
JPEG_QUALITY = 90
CROP_PAD = 0.6          # share of the person box added on every side; the racket is long
SEARCH_SECONDS = 0.6    # backswing / finish search either side of contact

STATE = "A side view of a squash player hitting a drive during solo practice."

# Every question carries an explicit "cannot tell" label. The RFP's rule is
# that low confidence becomes Can't assess, never a guess, and a question
# with no way to abstain forces one.
QUESTIONS = {
    "racket_visible": {
        "frames": ("backswing", "contact", "finish"),
        "instructions": "Is the player's racket clearly visible in this image?",
        "labels": {
            "visible": "The whole racket, head and shaft, can be seen clearly.",
            "partly": "Part of the racket is visible, or it is blurred or cut off.",
            "not_visible": "The racket cannot be seen.",
        },
    },
    "racket_head_height": {
        "frames": ("backswing",),
        "instructions": "Where is the racket head relative to the player's hitting shoulder?",
        "labels": {
            "above_shoulder": "The racket head is clearly above the hitting shoulder.",
            "level": "The racket head is about level with the shoulder.",
            "below_shoulder": "The racket head is below the shoulder.",
            "cannot_tell": "The racket head is not visible or too blurred to judge.",
        },
    },
    "racket_face": {
        "frames": ("contact",),
        "instructions": "Look at the racket strings at this moment. How is the face angled?",
        "labels": {
            "open": "The strings face slightly upward (an open face).",
            "square": "The strings face straight ahead toward the front wall.",
            "closed": "The strings face downward toward the floor (a closed face).",
            "cannot_tell": "The strings are not visible or too blurred to judge.",
        },
    },
    "finish_direction": {
        "frames": ("finish",),
        "instructions": "Where has the racket finished after the swing?",
        "labels": {
            "front_wall": "The racket has followed through toward the front wall, "
                          "finishing up in front of the body.",
            "across_body": "The racket has wrapped around across the body.",
            "short": "The swing stopped early with little follow-through.",
            "cannot_tell": "The racket is not visible or too blurred to judge.",
        },
    },
}


def key_frames(xy: np.ndarray, index: np.ndarray, *, fps: float, hand: str) -> dict[str, int]:
    """Frame indices (source numbering) of backswing, contact and finish."""
    unit = kq.torso_length(xy)
    speed = kq.wrist_speed(xy, fps, hand, unit)
    c = int(np.nanargmax(speed))
    reach = round(SEARCH_SECONDS * fps)
    wy = xy[:, kq.J[f"{hand}_wrist"], 1]       # image y: smaller is higher

    def highest(lo: int, hi: int) -> int:
        seg = wy[lo:hi]
        if not np.isfinite(seg).any():
            return c
        return lo + int(np.nanargmin(seg))

    return {
        "backswing": int(index[highest(max(0, c - reach), c)]),
        "contact": int(index[c]),
        "finish": int(index[highest(c + 1, min(len(wy), c + reach + 1))]),
    }


def crop_frame(video: Path, frame_no: int, box_xywh, out: Path) -> str:
    """Decode one frame, crop to the padded box, save a JPEG, return a data URL."""
    import cv2

    cap = cv2.VideoCapture(str(video))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok:
        raise RuntimeError(f"could not read frame {frame_no} of {video}")
    h, w = frame.shape[:2]
    x, y, bw, bh = box_xywh
    x0, y0 = max(0.0, x - CROP_PAD * bw), max(0.0, y - CROP_PAD * bh)
    x1, y1 = min(1.0, x + bw + CROP_PAD * bw), min(1.0, y + bh + CROP_PAD * bh)
    img = frame[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]
    scale = IMAGE_LONG_EDGE / max(img.shape[:2])
    if scale < 1:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


async def ask(images: dict[str, str], api_key: str) -> dict:
    from typesafe_sdk import AsyncTypeSafeClient, Choice, RetryPolicy, TypeSafeError

    results = {}
    async with AsyncTypeSafeClient(api_key=api_key, base_url=TYPESAFE_BASE_URL,
                                   model=READ_MODEL, timeout=60.0,
                                   retry=RetryPolicy(max_retries=2)) as client:
        for key, url in images.items():
            wanted = {q: Choice(instructions=spec["instructions"], criteria=spec["labels"])
                      for q, spec in QUESTIONS.items() if key in spec["frames"]}
            t0 = time.perf_counter()
            try:
                r = await client.system_one(STATE, wanted, extra_body={"content": [
                    {"type": "image_url", "image_url": {"url": url, "detail": IMAGE_DETAIL}}]})
            except TypeSafeError as exc:
                results[key] = {"error": str(exc).splitlines()[0][:200]}
                continue
            results[key] = {
                "latency_s": round(time.perf_counter() - t0, 3),
                "usage": json.loads(r.raw_http_response.text).get("usage", {}),
                "answers": {q: {"choice": a.choice,
                                "probabilities": dict(a.probabilities),
                                "confidence": a.confidence}
                            for q, a in r.answers.items()},
            }
    return results


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Racket reads on key frames (M0).")
    ap.add_argument("run_dir", type=Path, help="an M0 run folder (has m0_timing.json)")
    ap.add_argument("--dry-run", action="store_true", help="save crops and questions only")
    args = ap.parse_args(argv)

    timing = json.loads((args.run_dir / "m0_timing.json").read_text())
    quality = json.loads((args.run_dir / "m0_quality.json").read_text())
    meta, window = timing["meta"], timing["window"]
    video = Path(meta["cut"])
    width, height = window["width"], window["height"]
    if abs(window["rotation"]) in (90, 270):
        width, height = height, width

    frames = kq.frames_from_payload(json.loads((args.run_dir / "poses.json").read_text()))
    persons, _ = kq.lock_player(frames)
    xy, _, _ = kq.keypoint_arrays(persons, width=width, height=height)
    index = np.array([f["index"] for f in frames])
    keys = key_frames(xy, index, fps=quality["fps"], hand=meta["hand"])

    images = {}
    by_frame = {f["index"]: p for f, p in zip(frames, persons)}
    for name, frame_no in keys.items():
        person = by_frame.get(frame_no)
        if person is None:
            print(f"{name}: no player at frame {frame_no}, skipped")
            continue
        images[name] = crop_frame(video, frame_no, person["bbox_xywh"],
                                  args.run_dir / "keyframes" / f"{name}_{frame_no}.jpg")

    out = {"model": READ_MODEL, "state": STATE, "questions": QUESTIONS,
           "key_frames": keys, "dry_run": args.dry_run}
    if not args.dry_run:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "running"))
        from src.env import load_api_key
        api_key, _ = load_api_key(Path(__file__).resolve().parents[2] / "running")
        out["reads"] = asyncio.run(ask(images, api_key))

    (args.run_dir / "m0_keyframes.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({"key_frames": keys, "reads": out.get("reads", "dry run")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
