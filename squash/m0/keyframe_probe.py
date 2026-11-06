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
Answered reads are cached in the run folder, keyed on the image and the exact
questions, so a re-run does not pay for them again.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

from . import keypoint_quality as kq

SQUASH_DIR = Path(__file__).resolve().parents[1]

READ_MODEL = "google/gemma-4-26b-a4b-it"     # the model deadlift settled on
TYPESAFE_BASE_URL = "https://gateway.vlm.run/typesafe"
IMAGE_LONG_EDGE = 1024
IMAGE_DETAIL = "high"
JPEG_QUALITY = 90
CROP_PAD = 0.6          # share of the person box added on every side; the racket is long
SEARCH_SECONDS = 0.6    # backswing / finish search either side of contact

# The view the clip was filmed from goes into the description the model reads.
STATE_TEMPLATE = "A {view} view of a squash player hitting a drive during solo practice."
VIEW_WORDS = {"side": "side", "back": "behind-the-player"}


def state_for(view: str) -> str:
    return STATE_TEMPLATE.format(view=VIEW_WORDS.get(view, view))

# Every question has a way to abstain: "cannot_tell", or "not_visible" for
# the visibility question itself. The RFP's rule is
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


def key_frames(xy: np.ndarray, scores: np.ndarray | None, t: np.ndarray,
               index: np.ndarray, *, hand: str, min_score: float | None = None
               ) -> dict[str, int]:
    """Frame indices of backswing, contact and finish.

    Contact is ``keypoint_quality.contact``: the same rule, and the same gate,
    as the quality report. Backswing and finish are the highest *seen* wrist
    within ``SEARCH_SECONDS`` of real time before and after it.
    """
    c, _, _ = kq.contact(xy, scores, t, hand=hand, min_score=min_score)
    wy = kq.gate(xy, scores, min_score)[:, kq.J[f"{hand}_wrist"], 1]   # image y: smaller is higher

    def highest(lo: int, hi: int) -> int:
        seg = wy[lo:hi]
        if not np.isfinite(seg).any():
            return c
        return lo + int(np.nanargmin(seg))

    lo = int(np.searchsorted(t, t[c] - SEARCH_SECONDS - 1e-9))
    hi = int(np.searchsorted(t, t[c] + SEARCH_SECONDS + 1e-9, side="right"))
    return {
        "backswing": int(index[highest(lo, c)]),
        "contact": int(index[c]),
        "finish": int(index[highest(c + 1, hi)]),
    }


def read_frames(video: Path, wanted: set[int]) -> dict[int, np.ndarray]:
    """Decode sequentially and keep the wanted frames, numbered in decode order.

    Never seeks by frame number: in files with hidden or repeated packets the
    container's frame numbers and the decoded frames disagree.
    """
    import cv2

    frames: dict[int, np.ndarray] = {}
    cap = cv2.VideoCapture(str(video))
    try:
        i = 0
        while len(frames) < len(wanted):
            ok, frame = cap.read()
            if not ok:
                break
            if i in wanted:
                frames[i] = frame
            i += 1
    finally:
        cap.release()
    missing = wanted - frames.keys()
    if missing:
        raise RuntimeError(f"{video} has no frame(s) {sorted(missing)}")
    return frames


def crop_frame(frame: np.ndarray, box_xywh, out: Path) -> tuple[str, bytes]:
    """Crop to the padded box, save a JPEG, return ``(data URL, jpeg bytes)``."""
    import cv2

    h, w = frame.shape[:2]
    x, y, bw, bh = box_xywh
    x0, y0 = max(0.0, x - CROP_PAD * bw), max(0.0, y - CROP_PAD * bh)
    x1, y1 = min(1.0, x + bw + CROP_PAD * bw), min(1.0, y + bh + CROP_PAD * bh)
    img = frame[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]
    scale = IMAGE_LONG_EDGE / max(img.shape[:2])
    if scale < 1:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    data = buf.tobytes()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    return "data:image/jpeg;base64," + base64.b64encode(data).decode(), data


def read_key(name: str, jpeg: bytes, state: str) -> str:
    """Cache key for one read: the image, the model, and the exact questions asked."""
    spec = {q: QUESTIONS[q] for q in QUESTIONS if name in QUESTIONS[q]["frames"]}
    blob = json.dumps({"model": READ_MODEL, "state": state, "questions": spec,
                       "detail": IMAGE_DETAIL}, sort_keys=True).encode()
    return hashlib.sha256(blob + jpeg).hexdigest()[:16]


async def ask(images: dict[str, str], api_key: str, state: str) -> dict:
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
                r = await client.system_one(state, wanted, extra_body={"content": [
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
    ap.add_argument("--min-score", type=float,
                    help="default: the gate the quality report used")
    args = ap.parse_args(argv)

    timing = json.loads((args.run_dir / "m0_timing.json").read_text())
    quality = json.loads((args.run_dir / "m0_quality.json").read_text())
    meta, window = timing["meta"], timing["window"]
    min_score = args.min_score if args.min_score is not None else quality.get("min_score")
    state = state_for(meta.get("view", "side"))
    video = Path(meta["cut"])
    width, height = window["width"], window["height"]
    if abs(window["rotation"]) in (90, 270):
        width, height = height, width

    frames = kq.load_frames(args.run_dir / "poses.json")
    persons, _ = kq.lock_player(frames)
    xy, scores, _ = kq.keypoint_arrays(persons, width=width, height=height)
    t = kq.frame_times(frames, quality.get("fps"))
    index = np.array([f["index"] for f in frames])
    keys = key_frames(xy, scores, t, index, hand=meta["hand"], min_score=min_score)
    if keys["contact"] != quality["peak"]["frame"]:
        print(f"note: contact frame {keys['contact']} differs from the quality report's "
              f"{quality['peak']['frame']} (different --min-score?)")

    by_frame = {f["index"]: p for f, p in zip(frames, persons, strict=True)}
    wanted = {n for n in keys.values() if by_frame.get(n) is not None}
    decoded = read_frames(video, wanted)
    images, jpegs = {}, {}
    for name, frame_no in keys.items():
        person = by_frame.get(frame_no)
        if person is None:
            print(f"{name}: no player at frame {frame_no}, skipped")
            continue
        images[name], jpegs[name] = crop_frame(
            decoded[frame_no], person["bbox_xywh"],
            args.run_dir / "keyframes" / f"{name}_{frame_no}.jpg")

    out = {"model": READ_MODEL, "state": state, "questions": QUESTIONS, "min_score": min_score,
           "key_frames": keys, "dry_run": args.dry_run}
    if not args.dry_run:
        # Reads are billed: reuse any answered for the same image and questions.
        cache_file = args.run_dir / "m0_reads_cache.json"
        cache = json.loads(cache_file.read_text()) if cache_file.is_file() else {}
        keyed = {name: read_key(name, jpegs[name], state) for name in images}
        todo = {name: url for name, url in images.items() if keyed[name] not in cache}
        if todo:
            from swing.upstream.env import load_api_key
            api_key, _ = load_api_key(SQUASH_DIR)
            fresh = asyncio.run(ask(todo, api_key, state))
            cache.update({keyed[n]: r for n, r in fresh.items() if "error" not in r})
            cache_file.write_text(json.dumps(cache, indent=2))
        else:
            fresh = {}
        out["reads"] = {name: fresh.get(name) or {**cache[keyed[name]], "cached": True}
                        for name in images}

    (args.run_dir / "m0_keyframes.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({"key_frames": keys, "reads": out.get("reads", "dry run")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
