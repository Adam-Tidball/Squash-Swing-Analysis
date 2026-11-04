"""Slo-mo-safe ingest: sources, capture setup, and cached renditions (FR-1).

Everything downstream works in **real seconds**. A source is probed once
(``swing.timing``) and gets a real-time scale: 1.0 for a native 240 fps
original, ``file_fps / capture_fps`` for a retimed export declared with
``--capture-fps``. Ambiguous files are refused.

A *rendition* is what the pose model sees: one stretch of the source,
optionally resampled, scaled to the inference height and re-encoded to H.264.
The coarse pass (FR-4) is a low-rate rendition of the whole clip, split into
chunks. The fine pass (FR-5) uses full-rate windows. Each rendition carries the
real time of every frame it holds, read back from the encoded file, so frame
index is never used as a clock.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import config as cfg
from .timing import TimingReport, ffprobe_json, frame_times, probe

ACCEPTED_VERDICTS = ("native", "retimed", "normal-speed")


class IngestError(RuntimeError):
    """The clip cannot be analyzed as given; the message says what to do."""


@dataclass(frozen=True)
class Capture:
    """How the clip was filmed. Only matching setups are ever compared (FR-2)."""

    stroke: str                 # forehand | backhand
    view: str                   # side | back
    hand: str                   # right | left
    capture_fps: float | None = None   # declared, for retimed exports only

    def key(self) -> tuple[str, str, str]:
        return (self.view, self.hand, self.stroke)

    def mismatch(self, other: "Capture") -> list[str]:
        """Fields that differ; empty means the two may be compared or trended."""
        return [name for name in ("view", "hand", "stroke")
                if getattr(self, name) != getattr(other, name)]


@dataclass
class Source:
    path: Path
    timing: TimingReport
    identity: str               # content-derived cache key

    @property
    def scale(self) -> float:
        """Real seconds per media second."""
        return self.timing.real_time_scale or 1.0

    @property
    def media_fps(self) -> float:
        return self.timing.measured_fps or self.timing.average_fps or 0.0

    @property
    def real_fps(self) -> float:
        return self.media_fps / self.scale if self.scale else 0.0

    @property
    def real_seconds(self) -> float:
        return self.timing.media_seconds * self.scale


@dataclass
class Rendition:
    """One encoded stretch of the source, with each frame's real time."""

    path: Path
    source_identity: str
    start: float                # real seconds into the source
    duration: float | None      # real seconds, None = to the end
    target_fps: float | None    # real-time sampling rate, None = every frame
    width: int
    height: int
    real_times: np.ndarray = field(repr=False)

    @property
    def n_frames(self) -> int:
        return int(self.real_times.size)

    def describe(self) -> dict:
        d = asdict(self)
        d["path"] = str(self.path)
        d["real_times"] = None
        d["n_frames"] = self.n_frames
        if self.n_frames:
            d["first_time"] = round(float(self.real_times[0]), 5)
            d["last_time"] = round(float(self.real_times[-1]), 5)
        return d


# ── sources ──────────────────────────────────────────────────────────────────
def source_identity(path: Path, *, edge_bytes: int = cfg.SOURCE_HASH_BYTES) -> str:
    """Size plus a hash of both ends: stable across renames and touches, and
    changed by any re-export or trim."""
    size = path.stat().st_size
    h = hashlib.sha256(str(size).encode())
    with path.open("rb") as f:
        h.update(f.read(edge_bytes))
        if size > edge_bytes:
            f.seek(max(edge_bytes, size - edge_bytes))
            h.update(f.read(edge_bytes))
    return h.hexdigest()[:16]


def open_source(path: Path, *, capture_fps: float | None = None) -> Source:
    """Probe a clip and decide its real-time clock, or refuse it."""
    path = Path(path)
    if not path.is_file():
        raise IngestError(f"no such clip: {path}")
    timing = probe(path, capture_fps=capture_fps)
    if timing.verdict not in ACCEPTED_VERDICTS:
        raise IngestError(
            f"{path.name}: timing is '{timing.verdict}'. "
            + " ".join(timing.notes)
            + " For slo-mo, the original file is best (Photos: Export Unmodified Original).")
    if not timing.n_frames:
        raise IngestError(f"{path.name}: no decodable video frames")
    return Source(path=path, timing=timing, identity=source_identity(path))


# ── renditions ───────────────────────────────────────────────────────────────
def _key(source: Source, **params) -> str:
    blob = json.dumps({"source": source.identity, **params}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def make_rendition(source: Source, *, start: float = 0.0, duration: float | None = None,
                   fps: float | None = None, height: int | None = cfg.INFERENCE_HEIGHT,
                   crf: int = cfg.CONVERT_CRF, cache_dir: Path = cfg.CACHE_DIR) -> Rendition:
    """Encode [start, start+duration) of the source (real seconds), cached.

    ``fps`` is a *real-time* rate: 30 means 30 frames per real second, which
    for a retimed file is fewer frames per media second. Seeking is frame
    exact (decode, not stream copy), so a window never silently widens.
    """
    scale = source.scale
    media_start = start / scale
    media_duration = None if duration is None else duration / scale
    media_fps = None if fps is None else fps * scale

    key = _key(source, start=round(start, 6), duration=duration, fps=fps,
               height=height, crf=crf)
    out = Path(cache_dir) / "renditions" / f"{source.path.stem}.{key}.mp4"
    meta = out.with_suffix(".json")

    if not (out.is_file() and meta.is_file()):
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".partial.mp4")
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{media_start:.6f}",
               "-i", str(source.path)]
        if media_duration is not None:
            cmd += ["-t", f"{media_duration:.6f}"]
        filters = []
        if media_fps is not None:
            filters.append(f"fps={media_fps:.6f}")
        if height:
            # Only ever downscales; ffmpeg applies container rotation first.
            filters.append(f"scale=-2:'min({height},ih)'")
        if filters:
            cmd += ["-vf", ",".join(filters)]
        cmd += ["-map", "0:v:0", "-c:v", "libx264", "-preset", "fast", "-crf", str(crf),
                "-pix_fmt", "yuv420p", "-an", str(tmp)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            tmp.unlink(missing_ok=True)
            raise IngestError(f"ffmpeg failed:\n{proc.stderr[-2000:]}")
        tmp.replace(out)

        # Real time of every frame, from the encoded file itself. -ss before
        # -i resets timestamps to zero at the seek point.
        media_t = frame_times(out)
        stream = next(s for s in ffprobe_json(out)["streams"] if s["codec_type"] == "video")
        meta.write_text(json.dumps({
            "real_times": [round(float((media_start + t) * scale), 6) for t in media_t],
            "width": int(stream["width"]), "height": int(stream["height"]),
        }))

    m = json.loads(meta.read_text())
    return Rendition(path=out, source_identity=source.identity, start=start,
                     duration=duration, target_fps=fps, width=m["width"],
                     height=m["height"], real_times=np.array(m["real_times"]))


def coarse_renditions(source: Source, *, fps: float = cfg.COARSE_FPS,
                      chunk_frames: int = cfg.CHUNK_FRAMES, start: float = 0.0,
                      duration: float | None = None,
                      cache_dir: Path = cfg.CACHE_DIR) -> list[Rendition]:
    """The whole clip (or [start, start+duration)) at ``fps``, in request-sized chunks.

    Chunks tile the span end to end. A frame belongs to the chunk whose start
    it is at or after, so no frame is posed (or billed) twice.
    """
    end = source.real_seconds if duration is None else min(source.real_seconds,
                                                           start + duration)
    chunk_s = chunk_frames / fps
    out, t = [], start
    while t < end - 1e-9:
        length = min(chunk_s, end - t)
        out.append(make_rendition(source, start=t, duration=length, fps=fps,
                                  cache_dir=cache_dir))
        t += length
    return out


def window(source: Source, centre: float, *, before: float = cfg.WINDOW_BEFORE_S,
           after: float = cfg.WINDOW_AFTER_S, cache_dir: Path = cfg.CACHE_DIR) -> Rendition:
    """A full-rate window around ``centre`` (real seconds), clipped to the clip (FR-5)."""
    start = max(0.0, centre - before)
    end = min(source.real_seconds, centre + after)
    return make_rendition(source, start=start, duration=end - start, fps=None,
                          cache_dir=cache_dir)
