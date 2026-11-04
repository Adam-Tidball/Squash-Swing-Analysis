"""The pose interface (FR-3), its cache, and the billed-frame budget.

A backend turns one rendition into the gateway's ``vid.pose.kpts`` payload.
Nothing else in the project calls a model: swing detection, metrics and
reporting read a ``PoseTrack``, so they test offline.

Caching is here rather than in each backend, keyed on the rendition's bytes
and everything that shapes the request (``describe()``). A cache hit bills
nothing (AC-5). The budget is checked **before** a billable call, so
``--max-billed-frames`` stops a run instead of reporting the overspend.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Protocol

import numpy as np

from .. import config as cfg
from ..ingest import Rendition
from ..upstream import pose as upstream_pose

# Bumped when the cached record's shape changes, so old entries miss cleanly.
CACHE_CONTRACT = "swing-pose-v1"


class BudgetExceeded(RuntimeError):
    pass


class PoseBackend(Protocol):
    name: str
    billable: bool               # does a call cost money?

    def describe(self) -> dict:
        """Everything that shapes the result: model, parameters, versions."""

    def estimate(self, video: Path, *, n_frames: int) -> tuple[dict, dict | None]:
        """Pose every frame of ``video``. Returns ``(payload, usage)``."""


@dataclass
class FrameBudget:
    """Billed frames this run, against an optional limit."""

    limit: int | None = None
    billed: int = 0

    def charge(self, n: int) -> None:
        if self.limit is not None and self.billed + n > self.limit:
            raise BudgetExceeded(
                f"this call would bill {n} frames, bringing the run to "
                f"{self.billed + n}, over --max-billed-frames {self.limit}. "
                "Nothing was sent. Raise the limit or analyze less of the clip (--trim).")
        self.billed += n


@dataclass
class PoseChunk:
    rendition: Rendition
    payload: dict
    usage: dict | None
    billed: int
    cached: bool
    seconds: float


@dataclass
class PoseTrack:
    """Poses on a real-time axis: one entry per posed frame, all chunks joined."""

    times: np.ndarray                       # real seconds, increasing
    persons: list[list[dict]]               # every distinct person per frame
    width: int                              # rendition pixels, for aspect ratio
    height: int
    chunks: list[PoseChunk] = field(default_factory=list)

    @property
    def billed(self) -> int:
        return sum(c.billed for c in self.chunks)

    @property
    def all_cached(self) -> bool:
        return all(c.cached for c in self.chunks)

    def payloads(self) -> list[dict]:
        return [{"rendition": c.rendition.describe(), "usage": c.usage,
                 "billed": c.billed, "cached": c.cached, "payload": c.payload}
                for c in self.chunks]


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class CachedPose:
    """Runs a backend over renditions, through the cache and the budget."""

    def __init__(self, backend: PoseBackend, *, budget: FrameBudget | None = None,
                 cache_dir: Path = cfg.CACHE_DIR, reuse: bool = True):
        self.backend = backend
        self.budget = budget or FrameBudget()
        self.cache_dir = Path(cache_dir) / "poses"
        self.reuse = reuse

    def cache_path(self, rendition: Rendition) -> Path:
        key = json.dumps({"video": file_digest(rendition.path),
                          "backend": self.backend.describe(),
                          "contract": CACHE_CONTRACT}, sort_keys=True)
        return self.cache_dir / f"poses.{hashlib.sha256(key.encode()).hexdigest()[:16]}.json"

    def run_one(self, rendition: Rendition) -> PoseChunk:
        path = self.cache_path(rendition)
        if self.reuse and path.is_file():
            blob = json.loads(path.read_text())
            return PoseChunk(rendition, blob["payload"], blob.get("usage"), 0, True, 0.0)

        n = rendition.n_frames
        if self.backend.billable:
            self.budget.charge(n)        # raises before anything is sent
        t0 = time.perf_counter()
        payload, usage = self.backend.estimate(rendition.path, n_frames=n)
        seconds = time.perf_counter() - t0
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "created": datetime.now().isoformat(timespec="seconds"),
            "backend": self.backend.describe(), "usage": usage, "payload": payload}))
        return PoseChunk(rendition, payload, usage, n if self.backend.billable else 0,
                         False, seconds)

    def run(self, renditions: list[Rendition]) -> PoseTrack:
        chunks = [self.run_one(r) for r in renditions]
        return join(chunks)


def join(chunks: list[PoseChunk]) -> PoseTrack:
    """Put every chunk's frames on the real-time axis, in order."""
    times: list[float] = []
    persons: list[list[dict]] = []
    for chunk in chunks:
        frames, _ = upstream_pose.unwrap(chunk.payload)
        upstream_pose.dedupe(frames, max_persons=None, iou=cfg.PERSON_IOU)
        rt = chunk.rendition.real_times
        for frame in frames:
            i = frame["index"]
            if i >= rt.size:
                raise RuntimeError(
                    f"pose response names frame {i}, but {chunk.rendition.path.name} "
                    f"decodes to {rt.size} frames; the model and ffprobe disagree "
                    "about this file.")
            times.append(float(rt[i]))
            persons.append(frame["persons"])
    order = np.argsort(times, kind="stable")
    first = chunks[0].rendition if chunks else None
    return PoseTrack(times=np.array(times)[order],
                     persons=[persons[i] for i in order],
                     width=first.width if first else 0,
                     height=first.height if first else 0,
                     chunks=chunks)
