"""Slo-mo-safe ingest: sources, capture setup, and cached renditions (FR-1).

Everything downstream works in **real seconds**. A source is probed once
(``swing.timing``) and gets a real-time scale: 1.0 for a native 240 fps
original, ``file_fps / capture_fps`` for a retimed export declared with
``--capture-fps``. Files whose clock is ambiguous are refused. One case cannot
be detected: a retimed export whose Apple metadata was stripped looks like
ordinary normal-speed video. It is accepted with a warning (``checks``), and
only ``--capture-fps`` or the original file can fix its timing.

A *rendition* is what the pose model sees: one stretch of the source,
optionally subsampled, scaled to the inference height and re-encoded to
H.264. Subsampling **selects** source frames (the first frame at or after
each slot of the target rate) and keeps their own timestamps, so each
rendition frame is a real source frame and its real time is that frame's
time. Nothing is duplicated, and frame index is never used as a clock.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import config as cfg
from .timing import ProbeError, TimingReport, frame_times, probe_with_times, video_stream

ACCEPTED_VERDICTS = ("native", "retimed", "normal-speed")

# Bumped when how a rendition is made or labelled changes, so stale cached
# renditions miss instead of being reused with the old frames or times.
RENDITION_CONTRACT = "rendition-v2"

# How far before a requested time a frame may read and still count as at it:
# ffprobe's 6-decimal rounding, with margin, and far below one frame.
SNAP_TOLERANCE = 1e-5

# Slot boundaries are compared after adding this fraction of a slot, so a
# frame that lands exactly on a boundary is not lost to float rounding.
SLOT_EPSILON = 1e-3


class IngestError(RuntimeError):
    """The clip cannot be analyzed as given; the message says what to do."""


@dataclass(frozen=True)
class Capture:
    """How the clip was filmed. Only matching setups are ever compared (FR-2)."""

    stroke: str                 # forehand | backhand
    view: str                   # side | back
    hand: str                   # right | left
    capture_fps: float | None = None   # declared, for retimed or normal-speed files

    def mismatch(self, other: Capture) -> list[str]:
        """Fields that differ; empty means the two may be compared or trended."""
        return [name for name in ("view", "hand", "stroke")
                if getattr(self, name) != getattr(other, name)]


@dataclass
class Source:
    path: Path
    timing: TimingReport
    identity: str               # content-derived cache key
    media_times: np.ndarray = field(repr=False)   # decoded frame times, from 0

    @property
    def scale(self) -> float:
        """Real seconds per media second."""
        return self.timing.real_time_scale or 1.0

    @property
    def media_fps(self) -> float:
        return self.timing.measured_fps or self.timing.average_fps or 0.0

    @property
    def real_fps(self) -> float:
        return self.media_fps / self.scale

    @property
    def real_seconds(self) -> float:
        return self.timing.media_seconds * self.scale

    def snap(self, media_t: float) -> float:
        """The first frame time at or after ``media_t`` (media seconds).

        An input seek lands on that frame and calls it time zero, so labelling
        it with the requested time instead would be off by up to one frame.
        """
        # ffprobe prints times to 6 decimals, so a frame exactly at media_t can
        # read up to 5e-7 s early; 1e-5 s absorbs that and is far below a frame.
        i = int(np.searchsorted(self.media_times, media_t - SNAP_TOLERANCE))
        return float(self.media_times[min(i, self.media_times.size - 1)])


@dataclass
class Rendition:
    """One encoded stretch of the source, with each frame's real time."""

    path: Path
    source_identity: str
    start: float                # real seconds into the source, snapped to a frame
    duration: float | None      # real seconds, None = to the end
    target_fps: float | None    # requested real-time sampling rate, None = every frame
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
            d["first_time"] = round(float(self.real_times[0]), 6)
            d["last_time"] = round(float(self.real_times[-1]), 6)
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
    try:
        timing, pts = probe_with_times(path, capture_fps=capture_fps)
    except ProbeError as exc:
        raise IngestError(str(exc)) from exc
    if not timing.n_frames:
        raise IngestError(f"{path.name}: no decodable video frames; is this a video file?")
    if timing.verdict not in ACCEPTED_VERDICTS:
        raise IngestError(
            f"{path.name}: timing is '{timing.verdict}'. " + " ".join(timing.notes)
            + " For slo-mo, the original file is best (Photos: Export Unmodified Original).")
    return Source(path=path, timing=timing, identity=source_identity(path),
                  media_times=pts - pts[0])


# ── renditions ───────────────────────────────────────────────────────────────
def _key(source: Source, **params) -> str:
    blob = json.dumps({"source": source.identity, "contract": RENDITION_CONTRACT, **params},
                      sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def _select_filter(media_fps: float) -> str:
    """Keep the first frame at or after each slot of ``media_fps``.

    Unlike ffmpeg's ``fps`` filter it never duplicates a frame (a target above
    the source rate keeps every frame), and with ``-fps_mode vfr`` each kept
    frame keeps its own timestamp.
    """
    r, e = f"{media_fps:.9f}", SLOT_EPSILON
    return (f"select='isnan(prev_selected_t)+gt(floor(t*{r}+{e})\\,"
            f"floor(prev_selected_t*{r}+{e}))'")


def make_rendition(source: Source, *, start: float = 0.0, duration: float | None = None,
                   fps: float | None = None, height: int | None = cfg.INFERENCE_HEIGHT,
                   crf: int = cfg.CONVERT_CRF, cache_dir: Path = cfg.CACHE_DIR) -> Rendition:
    """Encode [start, start+duration) of the source (real seconds), cached.

    ``fps`` is a *real-time* rate: 30 means 30 frames per real second, which
    for a retimed file is fewer frames per media second. Seeking decodes
    rather than stream-copies, so a window never silently widens.
    """
    if start < 0 or (duration is not None and duration <= 0):
        raise IngestError(f"invalid span: start {start}, duration {duration}")
    if fps is not None and fps <= 0:
        raise IngestError(f"invalid frame rate {fps}")
    scale = source.scale
    media_start = source.snap(start / scale)
    media_end = None if duration is None else (start + duration) / scale
    media_duration = None if media_end is None else media_end - media_start
    if media_duration is not None and media_duration <= 0:
        raise IngestError(f"{source.path.name}: [{start:.4f} s, +{duration} s) holds no frames")
    media_fps = None if fps is None else fps * scale

    # Keyed on exactly what ffmpeg is given, plus the scale baked into the
    # labels: the same real-time request means different frames under a
    # different --capture-fps.
    key = _key(source, media_start=round(media_start, 6),
               media_duration=None if media_duration is None else round(media_duration, 6),
               media_fps=None if media_fps is None else round(media_fps, 6),
               scale=round(scale, 9), height=height, crf=crf)
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
            filters.append(_select_filter(media_fps))
        if height:
            # Only ever downscales; ffmpeg applies container rotation first.
            filters.append(f"scale=-2:'min({height},ih)'")
        if filters:
            cmd += ["-vf", ",".join(filters)]
        cmd += ["-fps_mode", "vfr", "-map", "0:v:0", "-c:v", "libx264", "-preset", "fast",
                "-crf", str(crf), "-pix_fmt", "yuv420p", "-an", str(tmp)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            tmp.unlink(missing_ok=True)
            raise IngestError(f"ffmpeg failed:\n{proc.stderr[-2000:]}")

        stream = video_stream(tmp)
        media_t = frame_times(tmp) if stream else np.array([])
        if not media_t.size:
            tmp.unlink(missing_ok=True)
            raise IngestError(f"{source.path.name}: [{start:.4f} s, +{duration} s) holds no frames")
        tmp.replace(out)
        # -ss before -i makes the seek frame time zero; it was snapped to a real
        # frame, so media_start + t is that frame's own source time.
        meta.write_text(json.dumps({
            "scale": scale,
            "real_times": [round(float((media_start + t) * scale), 6) for t in media_t],
            "width": int(stream["width"]), "height": int(stream["height"]),
        }))

    m = json.loads(meta.read_text())
    return Rendition(path=out, source_identity=source.identity, start=media_start * scale,
                     duration=duration, target_fps=fps, width=m["width"],
                     height=m["height"], real_times=np.array(m["real_times"]))


def coarse_renditions(source: Source, *, fps: float = cfg.COARSE_FPS,
                      chunk_frames: int = cfg.CHUNK_FRAMES, start: float = 0.0,
                      duration: float | None = None,
                      cache_dir: Path = cfg.CACHE_DIR) -> list[Rendition]:
    """The clip (or [start, start+duration)) at ``fps``, in request-sized chunks.

    Chunks are cut on the output frame grid, never in arbitrary seconds, so
    every chunk holds whole slots and a sub-frame tail cannot produce an empty
    encode. A rate above the source's own is capped at the source rate.
    """
    if fps <= 0:
        raise IngestError(f"invalid coarse rate {fps}")
    fps = min(fps, source.real_fps)
    end = source.real_seconds if duration is None else min(source.real_seconds,
                                                           start + duration)
    n_slots = int(np.floor((end - start) * fps + 1e-6))
    if n_slots <= 0:
        raise IngestError(f"{source.path.name}: nothing to analyze between "
                          f"{start:.3f} s and {end:.3f} s")
    return [make_rendition(source, start=start + k / fps,
                           duration=min(chunk_frames, n_slots - k) / fps,
                           fps=fps, cache_dir=cache_dir)
            for k in range(0, n_slots, chunk_frames)]


def window(source: Source, centre: float, *, before: float = cfg.WINDOW_BEFORE_S,
           after: float = cfg.WINDOW_AFTER_S, cache_dir: Path = cfg.CACHE_DIR) -> Rendition:
    """A full-rate window around ``centre`` (real seconds), clipped to the clip (FR-5)."""
    if not 0 <= centre <= source.real_seconds:
        raise IngestError(f"window centre {centre:.3f} s is outside the clip "
                          f"(0–{source.real_seconds:.3f} s)")
    start = max(0.0, centre - before)
    end = min(source.real_seconds, centre + after)
    return make_rendition(source, start=start, duration=end - start, fps=None,
                          cache_dir=cache_dir)
