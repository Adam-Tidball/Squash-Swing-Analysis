"""M0: run the existing `running` pose pipeline on one swing window.

Cuts a short window out of a 240 fps clip (every frame kept), refuses to call
the gateway if that would bill more frames than ``--max-billed-frames``, then
drives ``running/main.py`` unchanged except for settings: gait analysis and
the gait panel off, ``VIDEO_MAX_FRAMES`` set explicitly to the window's frame
count. Afterwards it writes ``m0_timing.json``, ``m0_quality.json`` and
``m0_summary.md`` into the run's own output folder.

    cd squash
    python -m m0.run_pose_m0 ~/clips/fh_drive.MOV --start 12.4 --duration 1.75 \\
        --stroke forehand --hand right --max-billed-frames 500

``--fake-gateway`` swaps the gateway call for a synthetic swinging stick
figure, so the whole path can be exercised offline with no key and no cost.
Nothing it writes is evidence about real footage.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

from . import keypoint_quality, timing_probe

SQUASH_DIR = Path(__file__).resolve().parents[1]
RUNNING_DIR = SQUASH_DIR.parent / "running"


def cut_window(src: Path, dst: Path, *, start: float, duration: float) -> None:
    """Re-encode [start, start+duration) keeping every frame and the source rate.

    Re-encoded rather than stream-copied, because a copy can only start on a
    keyframe and would silently widen the window (and the bill).
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-ss", f"{start}", "-i", str(src),
        "-t", f"{duration}", "-map", "0:v:0", "-c:v", "libx264", "-preset", "fast",
        "-crf", "18", "-pix_fmt", "yuv420p", "-an", str(dst),
    ], check=True)


def fake_payload(n_frames: int, fps: float, hand: str, *, aspect: float = 16 / 9) -> dict:
    """A synthetic side-on player whose hitting arm swings once, peaking mid-window.

    ``aspect`` is the frame's width / height. Coordinates are normalized per
    axis, so the arm's arc is narrowed by it to stay a circle in pixels; an arc
    that is circular in normalized units is an ellipse on screen, and its speed
    peak moves off the middle of the swing.
    """
    rng = np.random.default_rng(0)
    t = np.arange(n_frames) / fps
    peak = t[len(t) // 2]
    phase = np.tanh((t - peak) * 12)            # -1 backswing .. +1 follow-through
    items = []
    for i in range(n_frames):
        k = {
            "nose": (0.50, 0.20), "left_eye": (0.51, 0.19), "right_eye": (0.49, 0.19),
            "left_ear": (0.52, 0.20), "right_ear": (0.48, 0.20),
            "left_shoulder": (0.53, 0.30), "right_shoulder": (0.47, 0.30),
            "left_hip": (0.52, 0.52), "right_hip": (0.48, 0.52),
            "left_knee": (0.56, 0.68), "right_knee": (0.44, 0.68),
            "left_ankle": (0.58, 0.85), "right_ankle": (0.42, 0.85),
        }
        sx, sy = k[f"{hand}_shoulder"]
        # High behind (0.8 pi), straight down at contact (1.5 pi), high in front (2.2 pi).
        ang = np.pi * (0.8 + 0.7 * (phase[i] + 1))
        k[f"{hand}_elbow"] = (sx + 0.08 * np.cos(ang) / aspect, sy - 0.08 * np.sin(ang))
        k[f"{hand}_wrist"] = (sx + 0.16 * np.cos(ang) / aspect, sy - 0.16 * np.sin(ang))
        other = "left" if hand == "right" else "right"
        ox, oy = k[f"{other}_shoulder"]
        k[f"{other}_elbow"], k[f"{other}_wrist"] = (ox + 0.03, oy + 0.10), (ox + 0.06, oy + 0.18)
        pts = [[round(float(x + rng.normal(0, 0.0015)), 4), round(float(y + rng.normal(0, 0.0015)), 4)]
               for x, y in (k[n] for n in keypoint_quality.KPT_NAMES)]
        items.append({"frame_id": i, "track_id": 0, "bbox_xywh": [0.36, 0.15, 0.30, 0.74],
                      "kpts_xy": pts})
    return {"content": {"object": "vid.pose.kpts", "kpts_labels": keypoint_quality.KPT_NAMES,
                        "frames": [{"frame_id": i} for i in range(n_frames)], "items": items}}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="M0 pose run on one swing window.")
    ap.add_argument("clip", type=Path)
    ap.add_argument("--start", type=float, required=True, help="window start, media seconds")
    ap.add_argument("--duration", type=float, default=1.75, help="window length, media seconds")
    ap.add_argument("--stroke", choices=("forehand", "backhand"), required=True)
    ap.add_argument("--hand", choices=("right", "left"), default="right")
    ap.add_argument("--view", default="side")
    ap.add_argument("--capture-fps", type=float, help="declare 240 for a retimed export")
    ap.add_argument("--max-billed-frames", type=int, default=500)
    ap.add_argument("--fake-gateway", action="store_true",
                    help="offline smoke test: synthetic poses, no key, no cost")
    args = ap.parse_args(argv)

    timing = timing_probe.probe(args.clip, capture_fps=args.capture_fps)
    print(f"timing: {timing.verdict} ({timing.measured_fps} fps measured)")
    for note in timing.notes:
        print(f"  - {note}")
    if timing.verdict == "unknown":
        print("refusing: timing is ambiguous; pass --capture-fps or use the original file")
        return 2

    cut = RUNNING_DIR / "data" / "input" / (
        f"m0_{args.clip.stem}_{args.stroke}_{args.start:.2f}s.mp4")
    cut_window(args.clip, cut, start=args.start, duration=args.duration)
    window = timing_probe.probe(cut, capture_fps=args.capture_fps)
    n = window.n_frames
    print(f"window: {n} frames, {window.media_seconds:.3f} media s -> {cut}")
    if n > args.max_billed_frames:
        print(f"refusing: {n} frames exceeds --max-billed-frames {args.max_billed_frames}")
        return 3

    sys.path.insert(0, str(RUNNING_DIR))
    os.chdir(RUNNING_DIR)
    import config as cfg            # running/config.py
    import main as running_main     # running/main.py
    from src import pose

    cfg.INPUT_VIDEO = cut
    cfg.TRIM_SECONDS = None
    cfg.EVERY_FRAME = True
    cfg.VIDEO_MAX_FRAMES = n        # explicit, never the 900-frame default
    cfg.ANALYZE_GAIT = False
    cfg.SIDE_PANEL = False
    cfg.HIGHLIGHT_FEET = False
    cfg.FLASH_ON_CONTACT = False
    cfg.TRAIL_ON_ANKLES = False
    cfg.SAVE_CLEARANCE_PLOT = cfg.SAVE_STEPS_PLOT = cfg.SAVE_KNEE_PLOT = False
    cfg.MAX_PERSONS = 1

    if args.fake_gateway:
        os.environ.setdefault("VLMRUN_API_KEY", "fake-offline-key")
        cfg.REUSE_POSES = False
        pose.request_poses = lambda client, **kw: (
            fake_payload(n, timing.measured_fps or 240.0, args.hand,
                         aspect=window.width / window.height if window.height else 16 / 9),
            {"note": "fake gateway, nothing billed"})

    before = set(cfg.OUTPUT_DIR.glob("*")) if cfg.OUTPUT_DIR.exists() else set()
    code = running_main.main([])
    if code:
        return code
    run_dir = max(set(cfg.OUTPUT_DIR.glob("*")) - before, key=lambda p: p.stat().st_mtime)

    fps = window.measured_fps or 240.0
    if timing.real_time_scale and timing.real_time_scale != 1.0:
        fps = fps / timing.real_time_scale      # frames per *real* second
    width, height = window.width, window.height
    if abs(window.rotation) in (90, 270):
        width, height = height, width

    frames = keypoint_quality.frames_from_payload(json.loads((run_dir / "poses.json").read_text()))
    persons, crowded = keypoint_quality.lock_player(frames)
    xy, scores, field = keypoint_quality.keypoint_arrays(persons, width=width, height=height)
    quality = keypoint_quality.analyze(xy, scores, fps=fps, hand=args.hand,
                                       frame_index=np.array([f["index"] for f in frames]))
    quality.update(score_field=field, frames_with_multiple_people=crowded,
                   frames_with_nobody=sum(p is None for p in persons))

    meta = {"source": str(args.clip), "cut": str(cut),
            "start": args.start, "duration": args.duration,
            "stroke": args.stroke, "hand": args.hand, "view": args.view,
            "billed_frames_requested": n, "fake_gateway": args.fake_gateway}
    (run_dir / "m0_timing.json").write_text(json.dumps(
        {"source": asdict(timing), "window": asdict(window), "meta": meta}, indent=2))
    (run_dir / "m0_quality.json").write_text(json.dumps(quality, indent=2))
    summary = "\n".join([
        f"# M0 run: {args.stroke}, {args.view} view, {args.hand}-handed",
        "",
        "**Synthetic poses — not evidence.**" if args.fake_gateway else "",
        f"Timing: {timing.verdict}; {fps:.1f} real fps; window {n} frames billed.",
        "",
        keypoint_quality.summary_markdown(quality),
    ])
    (run_dir / "m0_summary.md").write_text(summary + "\n")
    print(f"\nM0 outputs in {run_dir}")
    print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
