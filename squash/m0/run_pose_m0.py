"""M0: run the existing `running` pose pipeline on one swing window.

Cuts a short window out of a 240 fps clip (every frame kept, real-time
timing from ``swing.ingest``), refuses to call the gateway if that would bill
more frames than ``--max-billed-frames``, then drives ``running/main.py``
unchanged except for settings, in its own process (``_running_driver``):
gait analysis and the gait panel off, detector stride 1, and exactly the
window's frame count requested. The driver refuses if running's own frame
count disagrees with the count that was budget-checked. Afterwards this
writes ``m0_timing.json``, ``m0_quality.json`` and ``m0_summary.md`` into
the run's own output folder.

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
import shutil
import subprocess
import sys
import uuid
from dataclasses import asdict
from pathlib import Path

from swing.ingest import IngestError, make_rendition, open_source
from swing.timing import probe

from . import keypoint_quality

SQUASH_DIR = Path(__file__).resolve().parents[1]
RUNNING_DIR = SQUASH_DIR.parent / "running"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="M0 pose run on one swing window.")
    ap.add_argument("clip", type=Path)
    ap.add_argument("--start", type=float, required=True, help="window start, real seconds")
    ap.add_argument("--duration", type=float, default=1.75, help="window length, real seconds")
    ap.add_argument("--stroke", choices=("forehand", "backhand"), required=True)
    ap.add_argument("--hand", choices=("right", "left"), default="right")
    ap.add_argument("--view", choices=("side", "back"), default="side")
    ap.add_argument("--capture-fps", type=float, help="declare 240 for a retimed export")
    ap.add_argument("--max-billed-frames", type=int, default=500)
    ap.add_argument("--min-score", type=float,
                    help="treat keypoints scored below this as unseen (quality and key frames)")
    ap.add_argument("--fake-gateway", action="store_true",
                    help="offline smoke test: synthetic poses, no key, no cost")
    args = ap.parse_args(argv)

    try:
        source = open_source(args.clip, capture_fps=args.capture_fps)
        window = make_rendition(source, start=args.start, duration=args.duration,
                                fps=None, height=None)
    except IngestError as exc:
        print(f"refusing: {exc}")
        return 2
    timing = source.timing
    print(f"timing: {timing.verdict} ({timing.measured_fps} fps measured)")
    for note in timing.notes:
        print(f"  - {note}")

    cut = RUNNING_DIR / "data" / "input" / (
        f"m0_{args.clip.stem}_{args.stroke}_{args.start:.2f}s.mp4")
    cut.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(window.path, cut)
    n = window.n_frames
    print(f"window: {n} frames, {window.real_times[0]:.3f}–{window.real_times[-1]:.3f} "
          f"real s -> {cut}")
    if n > args.max_billed_frames:
        print(f"refusing: {n} frames exceeds --max-billed-frames {args.max_billed_frames}")
        return 3

    cut_probe = probe(cut)
    spec = {"cut": str(cut), "n_frames": n, "media_fps": cut_probe.measured_fps,
            "token": uuid.uuid4().hex[:8],
            "fake": {"hand": args.hand,
                     "swing_at": n / 2 / cut_probe.measured_fps,
                     "aspect": window.width / window.height} if args.fake_gateway else None}
    env = {**os.environ,
           "PYTHONPATH": os.pathsep.join(filter(None, [str(SQUASH_DIR),
                                                       os.environ.get("PYTHONPATH")]))}
    proc = subprocess.run([sys.executable, "-m", "m0._running_driver", json.dumps(spec)],
                          cwd=RUNNING_DIR, env=env, capture_output=True, text=True)
    sys.stdout.write(proc.stdout)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr[-3000:])
        return proc.returncode
    run_dir = Path(proc.stdout.strip().splitlines()[-1].removeprefix("RUN_DIR="))

    # Real times per posed frame come from the window itself, so a retimed
    # source is measured in real seconds.
    frames = keypoint_quality.load_frames(run_dir / "poses.json")
    for f in frames:
        f["time"] = float(window.real_times[f["index"]])
    quality = keypoint_quality.report_for(frames, width=window.width, height=window.height,
                                          hand=args.hand, fps=source.real_fps,
                                          min_score=args.min_score)

    meta = {"source": str(args.clip), "cut": str(cut), "start": args.start,
            "duration": args.duration, "stroke": args.stroke, "hand": args.hand,
            "view": args.view, "min_score": args.min_score,
            "billed_frames_requested": n, "fake_gateway": args.fake_gateway}
    window_info = {**window.describe(), "rotation": 0,   # ffmpeg applied any rotation
                   "measured_fps": cut_probe.measured_fps}
    (run_dir / "m0_timing.json").write_text(json.dumps(
        {"source": asdict(timing), "window": window_info, "meta": meta}, indent=2))
    (run_dir / "m0_quality.json").write_text(json.dumps(quality, indent=2))
    summary = "\n".join([
        f"# M0 run: {args.stroke}, {args.view} view, {args.hand}-handed",
        "",
        "**Synthetic poses — not evidence.**" if args.fake_gateway else "",
        f"Timing: {timing.verdict}; {source.real_fps:.1f} real fps; window {n} frames billed.",
        "",
        keypoint_quality.summary_markdown(quality),
    ])
    (run_dir / "m0_summary.md").write_text(summary + "\n")
    print(f"\nM0 outputs in {run_dir}")
    print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
