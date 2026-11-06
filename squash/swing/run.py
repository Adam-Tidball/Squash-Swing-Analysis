"""One timestamped folder per run, with everything needed to reproduce it.

``run.json`` snapshots the config, the capture setup, the source's timing
verdict, every rendition, the backend and the billed frames, so a run can be
explained and repeated without the console log. ``poses.json`` holds the raw
model responses. Footage never goes here; the source is referenced by path
and content identity.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from . import config as cfg
from .backends.base import PoseTrack
from .checks import Check
from .ingest import Capture, Source


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(cfg.SQUASH_DIR), *args],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def code_version() -> dict:
    """The commit, and whether the working tree had uncommitted changes."""
    status = _git("status", "--porcelain", "--", ".")
    return {"commit": _git("rev-parse", "--short", "HEAD"),
            "dirty": None if status is None else bool(status)}


def new_run_dir(clip: Path, *, output_dir: Path = cfg.OUTPUT_DIR) -> Path:
    stamp = datetime.now().strftime(cfg.RUN_STAMP_FORMAT)
    base = Path(output_dir) / f"{stamp}_{Path(clip).stem}"
    path, n = base, 1
    while path.exists():
        n += 1
        path = base.with_name(f"{base.name}-{n}")
    path.mkdir(parents=True)
    return path


def write_run(run_dir: Path, *, source: Source | None, capture: Capture, status: str,
              checks: list[Check], request: dict | None = None,
              config_overrides: dict | None = None, backend: dict | None = None,
              track: PoseTrack | None = None, budget_limit: int | None = None,
              extra: dict | None = None) -> dict:
    """Write ``run.json`` (and ``checks.json``, and ``poses.json`` when posed).

    Called on every exit path, including failures, so a run folder always
    explains itself.
    """
    record = {
        "run_id": run_dir.name,
        "created": datetime.now().isoformat(timespec="seconds"),
        "status": status,
        "code_version": code_version(),
        "request": request,
        "capture": asdict(capture),
        "source": None if source is None else {
            "path": str(source.path), "identity": source.identity,
            "real_fps": round(source.real_fps, 4),
            "real_seconds": round(source.real_seconds, 4),
            "timing": asdict(source.timing)},
        "backend": backend,
        "pose": None if track is None else {
            "frames_posed": len(track.persons),
            "billed_frames": track.billed,
            "budget_limit": budget_limit,
            "all_cached": track.all_cached,
            "renditions": [c.rendition.describe() for c in track.chunks],
        },
        "checks": [c.as_dict() for c in checks],
        "config": cfg.snapshot(**(config_overrides or {})),
        **(extra or {}),
    }
    (run_dir / "run.json").write_text(json.dumps(record, indent=2))
    (run_dir / "checks.json").write_text(json.dumps([c.as_dict() for c in checks], indent=2))
    if track is not None:
        (run_dir / "poses.json").write_text(json.dumps({
            "times": [round(float(t), 6) for t in track.times],
            "chunks": track.payloads()}))
    return record
