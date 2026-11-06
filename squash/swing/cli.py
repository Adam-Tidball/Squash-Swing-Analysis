"""The ``swing`` command.

    swing probe clip.mov                       # is the clock real time?
    swing analyze clip.mov --stroke forehand --view side --hand right \\
        --backend vlmrun --max-billed-frames 20000

M1 scope: ``analyze`` runs ingest, input checks and the coarse pose pass, and
writes a run folder. Swing detection (M2), metrics and the report (M3), the
rubric (M4) and baselines, ``compare`` and ``history`` (M5) build on it.

Exit codes: 0 done; 1 refused (bad input, or no poses); 2 usage error or not
built yet; 3 stopped by the frame budget; 4 failed (setup, gateway or model
error). Once a run folder exists, every exit writes its ``run.json``.
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

DONE, REFUSED, USAGE, BUDGET, FAILED = 0, 1, 2, 3, 4
LEVEL_MARK = {"ok": "  ok ", "info": " info", "warn": " WARN", "error": "ERROR"}


def _print_checks(items) -> None:
    for c in items:
        print(f"  [{LEVEL_MARK[c.level]}] {c.id}: {c.message}")


def cmd_probe(args) -> int:
    try:
        source = open_source(args.clip, capture_fps=args.capture_fps)
    except IngestError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return REFUSED
    print(json.dumps(asdict(source.timing), indent=2))
    return DONE


def _request(args, backend_kwargs: dict) -> dict:
    """What was asked for, recorded in every run.json."""
    return {
        "argv": sys.argv[1:],
        "clip": str(args.clip),
        "backend": args.backend,
        "backend_kwargs": {k: str(v) for k, v in backend_kwargs.items()},
        "coarse_fps": args.coarse_fps,
        "start": args.start,
        "trim": args.trim,
        "reuse": not args.no_reuse,
        "chunk_frames": cfg.CHUNK_FRAMES,
        "max_billed_frames": args.max_billed_frames,
    }


def _invalid_args(args) -> str | None:
    if args.start < 0:
        return f"--start must be 0 or more (got {args.start})"
    if args.trim is not None and args.trim <= 0:
        return f"--trim must be above 0 (got {args.trim})"
    if args.coarse_fps <= 0:
        return f"--coarse-fps must be above 0 (got {args.coarse_fps})"
    if args.max_billed_frames is not None and args.max_billed_frames < 0:
        return f"--max-billed-frames must be 0 or more (got {args.max_billed_frames})"
    return None


def cmd_analyze(args) -> int:
    problem = _invalid_args(args)
    if problem:
        print(f"refused: {problem}", file=sys.stderr)
        return REFUSED
    capture = Capture(stroke=args.stroke, view=args.view, hand=args.hand,
                      capture_fps=args.capture_fps)
    try:
        source = open_source(args.clip, capture_fps=args.capture_fps)
    except IngestError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return REFUSED
    if args.start >= source.real_seconds:
        print(f"refused: --start {args.start:g} s is past the end of the clip "
              f"({source.real_seconds:.2f} s)", file=sys.stderr)
        return REFUSED

    kwargs: dict = {}
    if args.backend == "mediapipe" and args.mediapipe_model:
        kwargs["model"] = args.mediapipe_model
    if args.backend == "fake":
        kwargs["hand"] = args.hand
    backend = backends.make(args.backend, **kwargs)
    coarse_fps = min(args.coarse_fps, source.real_fps)
    span = (min(source.real_seconds, args.start + args.trim) if args.trim else
            source.real_seconds) - args.start
    if backend.billable and args.max_billed_frames is None:
        print(f"refused: the {args.backend} backend bills per frame; pass "
              "--max-billed-frames so a run cannot overspend. This span needs about "
              f"{int(span * coarse_fps)} frames for the coarse pass.", file=sys.stderr)
        return REFUSED
    try:
        backend.check_ready()           # key, model: fail before any work
    except Exception as exc:            # noqa: BLE001 (any setup failure ends here)
        print(f"failed: {exc}", file=sys.stderr)
        return FAILED

    output_dir = args.output_dir or cfg.OUTPUT_DIR
    cache_dir = args.cache_dir or cfg.CACHE_DIR
    request = _request(args, kwargs)
    overrides = {"COARSE_FPS": coarse_fps, "CACHE_DIR": cache_dir, "OUTPUT_DIR": output_dir}
    run_dir = new_run_dir(source.path, output_dir=output_dir)
    print(f"run {run_dir.name}")
    print(f"source {source.path.name}: {source.timing.verdict}, "
          f"{source.real_fps:.1f} real fps, {source.real_seconds:.2f} s")

    def finish(status: str, code: int, items, **kw) -> int:
        write_run(run_dir, source=source, capture=capture, status=status, checks=items,
                  request=request, config_overrides=overrides, backend=backend.describe(),
                  budget_limit=args.max_billed_frames, **kw)
        return code

    pre = checks.before_pose(source, capture, start=args.start, duration=args.trim,
                             coarse_fps=args.coarse_fps)
    _print_checks(pre)
    if checks.worst(pre) == "error":
        print(f"refused by input checks; see {run_dir / 'checks.json'}")
        return finish("refused", REFUSED, pre)

    budget = backends.FrameBudget(args.max_billed_frames)
    runner = backends.CachedPose(backend, budget=budget, cache_dir=cache_dir,
                                 reuse=not args.no_reuse)
    try:
        renditions = coarse_renditions(source, fps=coarse_fps, start=args.start,
                                       duration=args.trim, cache_dir=cache_dir)
        n = sum(r.n_frames for r in renditions)
        print(f"coarse pass: {len(renditions)} chunk(s), {n} frames at {coarse_fps:g} fps "
              f"({'billed' if backend.billable else 'local, not billed'})")
        track = runner.run(renditions)
    except backends.BudgetExceeded as exc:
        print(f"stopped: {exc}", file=sys.stderr)
        return finish("budget_exceeded", BUDGET, pre, extra=_partial(exc, budget, runner))
    except IngestError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return finish("refused", REFUSED, pre, extra=_partial(exc, budget, runner))
    except Exception as exc:            # noqa: BLE001 (gateway, model, join: recorded)
        print(f"failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        print(f"recorded in {run_dir / 'run.json'}", file=sys.stderr)
        return finish("failed", FAILED, pre, extra=_partial(exc, budget, runner))

    post = checks.after_pose(track)
    _print_checks(post)
    cached = sum(c.cached for c in track.chunks)
    print(f"posed {len(track.persons)} frames; billed {track.billed}; "
          f"{cached}/{len(track.chunks)} chunk(s) from cache")
    if checks.worst(post) == "error":
        print(f"no usable poses; see {run_dir / 'checks.json'}")
        return finish("no_poses", REFUSED, pre + post, track=track)
    code = finish("pose_done", DONE, pre + post, track=track)
    print(f"wrote {run_dir}")
    print("next: swing detection arrives in M2")
    return code


def _partial(exc: Exception, budget, runner) -> dict:
    """What a stopped run had already done and spent."""
    return {"error": f"{type(exc).__name__}: {exc}",
            "billed_before_stop": budget.billed,
            "chunks_done": [{**c.rendition.describe(), "cached": c.cached, "billed": c.billed}
                            for c in runner.done]}


def cmd_not_yet(milestone: str):
    def run(args) -> int:
        print(f"'swing {args.command}' is not built yet; it arrives in {milestone}.",
              file=sys.stderr)
        return USAGE
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
    a.add_argument("--output-dir", type=Path,
                   help=f"where run folders go (default {cfg.OUTPUT_DIR})")
    a.add_argument("--cache-dir", type=Path,
                   help=f"renditions and pose cache (default {cfg.CACHE_DIR})")
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
