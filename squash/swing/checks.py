"""Input checks (FR-2): warn on poor input, refuse what cannot be analyzed.

Two stages. ``before_pose`` needs only the probe, so it runs before anything
is billed; an ``error`` there stops the run. ``after_pose`` reads the poses
for problems no probe can see: nobody in frame, a second person, a cropped
player. Those are warnings: the run continues and says what it saw.

Every check returns a value and the limit it was held to, so ``checks.json``
shows why it fired.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from . import config as cfg
from .backends.base import PoseTrack
from .geometry import area, touches_edge
from .ingest import Capture, Source


@dataclass
class Check:
    id: str
    level: str                 # ok | info | warn | error
    message: str
    value: object = None
    limit: object = None

    def as_dict(self) -> dict:
        return asdict(self)


def before_pose(source: Source, capture: Capture, *, start: float = 0.0,
                duration: float | None = None, coarse_fps: float | None = None) -> list[Check]:
    """Checks that need only the probe. ``start``/``duration`` are the span
    being analyzed (real seconds), which is what has to hold a swing."""
    t = source.timing
    out = [Check("timing", "ok" if t.verdict == "native" else "warn",
                 f"clock: {t.verdict}; real time = media time x {source.scale:g}",
                 t.verdict)]
    if t.verdict == "retimed":
        out[-1].message += (". Retimed exports may keep normal-speed ramps at each end; "
                            "trim them or use the original.")
    if t.verdict == "normal-speed":
        out[-1].message += ". Not slo-mo: swing timing is coarse at this rate."

    fps = source.real_fps
    out.append(Check("frame_rate", "ok" if fps >= cfg.MIN_CAPTURE_FPS else "warn",
                     f"{fps:.1f} real frames per second"
                     + ("" if fps >= cfg.MIN_CAPTURE_FPS else
                        "; the capture protocol asks for 240 "
                        f"(warn below {cfg.MIN_CAPTURE_FPS:g})"),
                     round(fps, 3), cfg.MIN_CAPTURE_FPS))

    end = source.real_seconds if duration is None else min(source.real_seconds,
                                                           start + duration)
    seconds = max(0.0, end - start)
    whole = start == 0 and duration is None
    what = "real seconds" if whole else f"real seconds analyzed (from {start:g} s)"
    out.append(Check("duration", "ok" if seconds >= cfg.MIN_CLIP_SECONDS else "error",
                     f"{seconds:.2f} {what}"
                     + ("" if seconds >= cfg.MIN_CLIP_SECONDS else
                        f"; too short to hold one swing window ({cfg.MIN_CLIP_SECONDS:g} s)"),
                     round(seconds, 3), cfg.MIN_CLIP_SECONDS))
    if coarse_fps is not None and coarse_fps > fps:
        out.append(Check("coarse_rate", "info",
                         f"coarse rate {coarse_fps:g} fps is above the clip's {fps:.1f}; "
                         "every frame is used instead (frames are never duplicated)",
                         round(fps, 3), coarse_fps))

    if t.irregular_intervals:
        out.append(Check("frame_intervals", "warn",
                         f"{t.irregular_intervals} irregular or repeated frame timestamps; "
                         "timing uses each frame's own timestamp",
                         t.irregular_intervals))
    out.append(Check("audio", "ok" if t.audio else "info",
                     "audio present" if t.audio else
                     "no audio: ball-impact timing unavailable for this clip",
                     bool(t.audio)))
    if capture.stroke not in ("forehand", "backhand"):
        out.append(Check("stroke", "warn", f"stroke '{capture.stroke}': v1 rubrics cover "
                         "forehand and backhand drives only"))
    return out


def after_pose(track: PoseTrack) -> list[Check]:
    n = len(track.persons)
    if not n:
        return [Check("poses", "error", "the pose model returned no frames", 0)]

    # A person without a box cannot be placed; such frames count as empty.
    boxed = [[p for p in ps if p.get("bbox_xywh")] for ps in track.persons]
    nobody = sum(1 for ps in boxed if not ps)
    # join() already merged duplicate boxes (IoU > PERSON_IOU), so every
    # remaining box is a distinct person.
    crowded = sum(1 for ps in boxed if len(ps) > 1)
    # The player for this check is the largest box: the one-player lock (FR-6)
    # arrives with swing detection, and a cropped player is usually the
    # largest person in frame anyway.
    largest = [max(ps, key=lambda p: area(p["bbox_xywh"])) for ps in boxed if ps]
    clipped = sum(1 for p in largest if touches_edge(p["bbox_xywh"], cfg.EDGE_MARGIN))

    def share_check(id_, count, denom, limit, what):
        share = count / denom if denom else 0.0
        return Check(id_, "warn" if share > limit else "ok",
                     f"{what} in {count} of {denom} frames ({share:.0%})",
                     round(share, 4), limit)

    return [
        share_check("nobody", nobody, n, cfg.NOBODY_WARN_SHARE, "nobody detected"),
        share_check("multiple_people", crowded, n, cfg.MULTI_PERSON_WARN_SHARE,
                    "more than one person"),
        share_check("clipped_body", clipped, len(largest), cfg.CLIPPED_WARN_SHARE,
                    "player touching the frame edge"),
    ]


def compatible(a: Capture, b: Capture) -> Check:
    """Only matching setups are compared or trended (capture protocol)."""
    diff = a.mismatch(b)
    if diff:
        return Check("setup_match", "error",
                     "setups differ in " + ", ".join(diff) + "; these runs are never compared",
                     diff)
    return Check("setup_match", "ok", "setups match", [])


def worst(checks: list[Check]) -> str:
    order = ("ok", "info", "warn", "error")
    return max((c.level for c in checks), key=order.index, default="ok")
