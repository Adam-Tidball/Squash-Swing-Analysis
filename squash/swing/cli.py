"""The ``swing`` command.

    swing probe clip.mov                       # is the clock real time?
    swing analyze clip.mov --stroke forehand --view side --hand right \\
        --backend vlmrun --max-billed-frames 20000

M1 scope: ``analyze`` runs ingest, input checks and the coarse pose pass, and
writes a run folder. Swing detection (M2), metrics and the report (M3), the
rubric (M4) and baselines, ``compare`` and ``history`` (M5) build on it.

Exit codes: 0 done, 1 input refused, 2 not built yet, 3 frame budget.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from . import backends, checks, config as cfg
from .ingest import Capture, IngestError, coarse_renditions, open_source
from .run import new_run_dir, write_run

LEVEL_MARK = {"ok": "  ok ", "info": " info", "warn": " WARN", "error": "ERROR"}


def _print_checks(items) -> None:
    for c in items:
        print(f"  [{LEVEL_MARK[c.level]}] {c.id}: {c.message}")


def cmd_probe(args) -> int:
    try:
        source = open_source(args.clip, capture_fps=args.capture_fps)
    except IngestError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(asdict(source.timing), indent=2))
    return 0


def cmd_analyze(args) -> int:
    capture = Capture(stroke=args.stroke, view=args.view, hand=args.hand,
                      capture_fps=args.capture_fps)
    try:
        source = open_source(args.clip, capture_fps=args.capture_fps)
    except IngestError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1

    kwargs = {}
    if args.backend == "mediapipe" and args.mediapipe_model:
        kwargs["model"] = args.mediapipe_model
    if args.backend == "fake":
        kwargs["hand"] = args.hand
    backend = backends.make(args.backend, **kwargs)
    if backend.billable and args.max_billed_frames is None:
        print(f"refused: the {args.backend} backend bills per frame; pass "
              "--max-billed-frames so a run cannot overspend.", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir) if args.output_dir else cfg.OUTPUT_DIR
    cache_dir = Path(args.cache_dir) if args.cache_dir else cfg.CACHE_DIR
    run_dir = new_run_dir(source, output_dir=output_dir)
    print(f"run {run_dir.name}")
    print(f"source {source.path.name}: {source.timing.verdict}, "
          f"{source.real_fps:.1f} real fps, {source.real_seconds:.2f} s")

    pre = checks.before_pose(source, capture)
    _print_checks(pre)
    if checks.worst(pre) == "error":
        write_run(run_dir, source=source, capture=capture, status="refused", checks=pre)
        print(f"refused by input checks; see {run_dir / 'checks.json'}")
        return 1

    renditions = coarse_renditions(source, fps=args.coarse_fps, start=args.start,
                                   duration=args.trim, cache_dir=cache_dir)
    n = sum(r.n_frames for r in renditions)
    print(f"coarse pass: {len(renditions)} chunk(s), {n} frames at {args.coarse_fps:g} fps "
          f"({'billed' if backend.billable else 'local, not billed'})")

    budget = backends.FrameBudget(args.max_billed_frames)
    runner = backends.CachedPose(backend, budget=budget, cache_dir=cache_dir,
                                 reuse=not args.no_reuse)
    try:
        track = runner.run(renditions)
    except backends.BudgetExceeded as exc:
        write_run(run_dir, source=source, capture=capture, status="budget_exceeded",
                  checks=pre, backend=backend.describe(), budget_limit=args.max_billed_frames,
                  extra={"error": str(exc), "billed_before_stop": budget.billed})
        print(f"stopped: {exc}", file=sys.stderr)
        return 3

    post = checks.after_pose(track)
    _print_checks(post)
    status = "pose_done"
    write_run(run_dir, source=source, capture=capture, status=status, checks=pre + post,
              backend=backend.describe(), track=track, budget_limit=args.max_billed_frames)
    cached = sum(c.cached for c in track.chunks)
    print(f"posed {len(track.persons)} frames; billed {track.billed}; "
          f"{cached}/{len(track.chunks)} chunk(s) from cache")
    print(f"wrote {run_dir}")
    print("next: swing detection arrives in M2")
    return 0


def cmd_not_yet(milestone: str):
    def run(args) -> int:
        print(f"'swing {args.command}' is not built yet; it arrives in {milestone}.",
              file=sys.stderr)
        return 2
    return run


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="swing", description="Squash swing analysis.")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("probe", help="read a clip's clock and refuse ambiguous timing")
    p.add_argument("clip", type=Path)
    p.add_argument("--capture-fps", type=float, help="declare 240 for a retimed export")
    p.set_defaults(func=cmd_probe)

    a = sub.add_parser("analyze", help="ingest, check and pose a practice clip")
    a.add_argument("clip", type=Path)
    a.add_argument("--stroke", required=True, choices=("forehand", "backhand"))
    a.add_argument("--view", required=True, choices=("side", "back"))
    a.add_argument("--hand", required=True, choices=("right", "left"))
    a.add_argument("--capture-fps", type=float, help="declare 240 for a retimed export")
    a.add_argument("--backend", default="vlmrun", choices=("vlmrun", "mediapipe", "fake"))
    a.add_argument("--max-billed-frames", type=int,
                   help="stop before billing more than this (required for vlmrun)")
    a.add_argument("--coarse-fps", type=float, default=cfg.COARSE_FPS,
                   help=f"coarse-pass rate in real fps (default {cfg.COARSE_FPS:g})")
    a.add_argument("--start", type=float, default=0.0, help="real seconds to skip")
    a.add_argument("--trim", type=float, help="analyze only this many real seconds")
    a.add_argument("--no-reuse", action="store_true", help="ignore cached poses")
    a.add_argument("--mediapipe-model", type=Path, help="path to pose_landmarker_heavy.task")
    a.add_argument("--output-dir", help=argparse.SUPPRESS)
    a.add_argument("--cache-dir", help=argparse.SUPPRESS)
    a.set_defaults(func=cmd_analyze)

    for name, ms in (("baseline", "M5"), ("compare", "M5"), ("history", "M5")):
        s = sub.add_parser(name, help=f"arrives in {ms}")
        s.add_argument("rest", nargs="*")
        s.set_defaults(func=cmd_not_yet(ms))
    return ap


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
