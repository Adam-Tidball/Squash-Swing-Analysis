"""M0: how good are the keypoints during a squash swing?

Reads one pose response and reports, for the hitting arm above all:

* visibility -- how often each joint was seen (returned, and scored at or
  above ``--min-score`` when gating); ``(0, 0)`` is the "not visible" sentinel
* confidence -- per-keypoint scores, *if the response carries any*. Upstream
  never reads a score field, so whether one exists is itself an M0 finding:
  FR-7 and every metric's confidence depend on it.
* jitter -- median frame-to-frame acceleration, in torso lengths, which is
  what turns into noise on wrist speed and contact timing
* gaps -- the longest stretch without the hitting wrist, in ms of real time
* left/right swaps -- frames where the two wrists look exchanged
* the hitting wrist's speed peak, the contact proxy every later step leans on

Everything runs on each frame's **real time**, never on frame index: a
response can skip frames, and screen recordings have irregular timing. Times
come from the input when it has them (``local_pose`` output, a ``swing`` run's
``poses.json``), otherwise from frame index and ``--fps``.

Lengths use the RFP's unit (median hip-mid to shoulder-mid distance) after x
and y are put in the same pixels, as ``running/src/gait.py`` does.

    python -m m0.keypoint_quality poses.json --fps 240 --width 1920 --height 1080 --hand right
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from swing import config as swing_cfg
from swing.geometry import area, iou
from swing.upstream.skeleton import INDEX as J
from swing.upstream.skeleton import KPT_NAMES

BODY = [n for n in KPT_NAMES if not n.endswith(("eye", "ear")) and n != "nose"]

# Field names a per-keypoint score might come back under. The first one found
# on any person record is used and reported.
SCORE_FIELDS = ("kpts_scores", "kpts_score", "kpts_conf", "keypoint_scores", "scores")

# Two wrists exchanged between frames: the crossed pairing has to beat the
# straight one by this factor, and the wrists must be this far apart (torso
# lengths), so two hands together on the grip do not count as a swap.
SWAP_GAIN = 0.5
SWAP_MIN_SEPARATION = 0.25


class QualityError(ValueError):
    """The input cannot support this measurement; the message says why."""


# ── reading the response ─────────────────────────────────────────────────────
def frames_from_payload(payload: dict) -> list[dict]:
    """One gateway payload as ``[{"index", "time" or None, "persons"}]``.

    ``frames`` is the authority on which frames exist (as in upstream's
    ``unwrap``); a frame with nobody in it still appears.
    """
    content = payload.get("content") or {}
    listed = {int(f["frame_id"]): f.get("time") for f in content.get("frames") or []}
    by_index: dict[int, list[dict]] = {i: [] for i in listed}
    for person in content.get("items") or []:
        if person.get("frame_id") is not None:
            by_index.setdefault(int(person["frame_id"]), []).append(person)
    return [{"index": i, "time": listed.get(i), "persons": by_index[i]} for i in sorted(by_index)]


def load_frames(path: Path) -> list[dict]:
    """Frames from any pose file this project writes.

    Accepts a raw gateway payload, an upstream or ``swing`` pose cache entry
    (``{"payload": ...}``), or a ``swing`` run's ``poses.json`` (chunks plus
    the real time of every frame).
    """
    blob = json.loads(Path(path).read_text())
    if "chunks" in blob and "times" in blob:              # a swing run
        frames: list[dict] = []
        for chunk in blob["chunks"]:
            frames += frames_from_payload(chunk["payload"])
        if len(frames) != len(blob["times"]):
            raise QualityError(f"{path}: {len(frames)} frames but {len(blob['times'])} times")
        for k, (frame, t) in enumerate(zip(frames, blob["times"], strict=True)):
            frame["index"], frame["time"] = k, t          # one index across chunks
        return frames
    if "payload" in blob and "content" not in blob:       # a cache entry
        blob = blob["payload"]
    return frames_from_payload(blob)


def frame_times(frames: list[dict], fps: float | None) -> np.ndarray:
    """Each frame's real time: recorded times if every frame has one, else index / fps."""
    if frames and all(f.get("time") is not None for f in frames):
        return np.array([float(f["time"]) for f in frames])
    if not fps:
        raise QualityError("these poses carry no per-frame times; pass --fps")
    return np.array([f["index"] / fps for f in frames], dtype=float)


def lock_player(frames: list[dict]) -> tuple[list[dict | None], int]:
    """One person per frame, by box continuity, falling back to the largest box.

    Returns the chosen record per frame and how many frames had more than one
    distinct person (IoU below ``swing.config.PERSON_IOU`` with each other),
    which is FR-2's "multiple people" warning.
    """
    chosen: list[dict | None] = []
    crowded = 0
    previous = None
    for frame in frames:
        persons = [p for p in frame["persons"] if p.get("bbox_xywh")]
        if not persons:
            chosen.append(None)
            continue
        distinct: list[dict] = []
        for p in persons:
            if all(iou(p["bbox_xywh"], d["bbox_xywh"]) < swing_cfg.PERSON_IOU for d in distinct):
                distinct.append(p)
        crowded += len(distinct) > 1
        best = max(persons, key=lambda p: (
            iou(p["bbox_xywh"], previous) if previous is not None else 0.0,
            area(p["bbox_xywh"])))
        chosen.append(best)
        previous = best["bbox_xywh"]
    return chosen, crowded


def keypoint_arrays(persons: list[dict | None], *, width: int, height: int
                    ) -> tuple[np.ndarray, np.ndarray | None, str | None]:
    """``(xy[N,17,2] in pixels with NaN for unseen, scores[N,17] or None, field)``."""
    n = len(persons)
    xy = np.full((n, 17, 2), np.nan)
    field = next((f for p in persons if p for f in SCORE_FIELDS if f in p), None)
    scores = np.full((n, 17), np.nan) if field else None
    for i, p in enumerate(persons):
        if not p:
            continue
        for j, pt in enumerate((p.get("kpts_xy") or [])[:17]):
            if tuple(pt) != (0, 0):
                xy[i, j] = (pt[0] * width, pt[1] * height)
        if field and p.get(field) is not None:
            vals = np.asarray(p[field], dtype=float).reshape(-1)[:17]
            scores[i, :len(vals)] = vals
    return xy, scores, field


# ── measurements (pure, tested on synthetic keypoints) ───────────────────────
def torso_length(xy: np.ndarray) -> float:
    shoulders = (xy[:, J["left_shoulder"]] + xy[:, J["right_shoulder"]]) / 2
    hips = (xy[:, J["left_hip"]] + xy[:, J["right_hip"]]) / 2
    d = np.linalg.norm(shoulders - hips, axis=1)
    d = d[np.isfinite(d)]
    if not d.size:
        raise QualityError("no frame has both shoulders and both hips; cannot set a unit")
    return float(np.median(d))


def _fill(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Fill NaN gaps by linear interpolation in time."""
    ok = np.isfinite(x)
    if ok.all() or not ok.any():
        return x
    out = x.copy()
    out[~ok] = np.interp(t[~ok], t[ok], x[ok])
    return out


def _smooth3(x: np.ndarray) -> np.ndarray:
    """Median-3 then mean-3, edge-padded: the upstream gait.py defaults."""
    if len(x) < 3:
        return x
    p = np.pad(x, 1, mode="edge")
    med = np.array([np.median(p[i:i + 3]) for i in range(len(x))])
    p = np.pad(med, 1, mode="edge")
    return np.convolve(p, np.ones(3) / 3, mode="valid")


def wrist_speed(xy: np.ndarray, t: np.ndarray, hand: str, unit: float) -> np.ndarray:
    """Hitting-wrist speed in torso lengths per real second, gaps filled, lightly smoothed.

    The track is resampled onto an even grid at the typical frame interval
    before smoothing, so the 3-sample filters span three real frame intervals
    even where frames are missing; smoothing across a gap in sample space
    would invent a jump between two adjacent frames.
    """
    w = xy[:, J[f"{hand}_wrist"]] / unit
    x, y = _fill(w[:, 0], t), _fill(w[:, 1], t)
    if t.size < 3:
        return np.zeros(t.size)
    step = float(np.median(np.diff(t)))
    grid = np.arange(t[0], t[-1] + step / 2, step)
    gx, gy = _smooth3(np.interp(grid, t, x)), _smooth3(np.interp(grid, t, y))
    speed = np.hypot(np.gradient(gx, step), np.gradient(gy, step))
    return np.interp(t, grid, speed)


def longest_gap(seen: np.ndarray, t: np.ndarray) -> float:
    """Longest stretch of real time without the joint, in seconds.

    Measured between consecutive frames where it *was* seen, less one typical
    frame interval, so frames missing from the response count as well as
    frames returned without the joint.
    """
    if t.size < 2:
        return 0.0
    typical = float(np.median(np.diff(t)))
    ts = t[seen]
    if not ts.size:
        return float(t[-1] - t[0] + typical)
    edges = [ts[0] - t[0], t[-1] - ts[-1]]
    inner = list(np.diff(ts) - typical) if ts.size > 1 else []
    return float(max(0.0, *edges, *inner))


def jitter(xy: np.ndarray, unit: float) -> np.ndarray:
    """Per joint: median |second difference| in torso lengths per frame^2 (sample-based)."""
    acc = np.linalg.norm(np.diff(xy, n=2, axis=0), axis=2) / unit
    out = np.full(xy.shape[1], np.nan)
    for j in range(xy.shape[1]):
        col = acc[:, j]
        col = col[np.isfinite(col)]
        if col.size:
            out[j] = float(np.median(col))
    return out


def wrist_swaps(xy: np.ndarray, unit: float) -> list[int]:
    """Frames where left and right wrists look exchanged from the previous frame."""
    lw, rw = xy[:, J["left_wrist"]] / unit, xy[:, J["right_wrist"]] / unit
    hits = []
    for k in range(1, len(xy)):
        pts = (lw[k - 1], rw[k - 1], lw[k], rw[k])
        if not all(np.isfinite(p).all() for p in pts):
            continue
        if np.linalg.norm(lw[k] - rw[k]) < SWAP_MIN_SEPARATION:
            continue
        straight = np.linalg.norm(lw[k] - lw[k - 1]) + np.linalg.norm(rw[k] - rw[k - 1])
        crossed = np.linalg.norm(lw[k] - rw[k - 1]) + np.linalg.norm(rw[k] - lw[k - 1])
        if crossed < SWAP_GAIN * straight:
            hits.append(k)
    return hits


def gate(xy: np.ndarray, scores: np.ndarray | None, min_score: float | None) -> np.ndarray:
    """Drop keypoints whose score is under *min_score* (a copy; NaN = not seen).

    Without this, a joint the model has lost still has coordinates, usually
    snapped onto another body part, and its jumps read as the fastest motion
    in the clip.
    """
    if scores is None or min_score is None:
        return xy
    out = xy.copy()
    out[scores < min_score] = np.nan
    return out


def contact(xy: np.ndarray, scores: np.ndarray | None, t: np.ndarray, *, hand: str,
            min_score: float | None = None) -> tuple[int, np.ndarray, float]:
    """The contact proxy: peak hitting-wrist speed, only where the wrist was seen.

    Returns ``(row of the peak, speed per row with NaN where unseen, unit)``.
    The single definition used by the quality report and the key-frame probe.
    """
    unit = torso_length(xy)
    gated = gate(xy, scores, min_score)
    seen = np.isfinite(gated[:, J[f"{hand}_wrist"], 0])
    if seen.sum() < 2:
        raise QualityError(
            f"the {hand} wrist is seen in {int(seen.sum())} frame(s)"
            + (f" at --min-score {min_score}" if min_score is not None else "")
            + "; check --hand, or lower --min-score")
    speed = wrist_speed(gated, t, hand, unit)
    speed[~seen] = np.nan
    return int(np.nanargmax(speed)), speed, unit


def analyze(xy: np.ndarray, scores: np.ndarray | None, *, hand: str,
            times: np.ndarray | None = None, fps: float | None = None,
            frame_index: np.ndarray | None = None, before: float = 1.0,
            after: float = 0.75, min_score: float | None = None) -> dict:
    """Quality report for the whole clip and for the window around the speed peak.

    ``times`` are real seconds per row; without them, ``frame_index / fps``.
    Visibility, gaps and the speed peak use keypoints gated at ``min_score``
    (a lost joint keeps coordinates, so ungated it would count as seen).
    Jitter uses the raw keypoints. The window spans ``before``/``after``
    real seconds around the peak.
    """
    n = len(xy)
    frame_index = np.arange(n) if frame_index is None else np.asarray(frame_index)
    if times is None:
        if not fps:
            raise QualityError("pass times or fps")
        times = frame_index / fps
    t = np.asarray(times, dtype=float)
    peak, speed, unit = contact(xy, scores, t, hand=hand, min_score=min_score)
    gated = gate(xy, scores, min_score)
    lo = int(np.searchsorted(t, t[peak] - before - 1e-9))
    hi = int(np.searchsorted(t, t[peak] + after + 1e-9, side="right"))

    def block(sl: slice) -> dict:
        part = xy[sl]
        vis = np.isfinite(gated[sl][..., 0])
        jit = jitter(part, unit)
        hit = J[f"{hand}_wrist"]
        out = {
            "frames": int(part.shape[0]),
            "visible": {name: round(float(vis[:, J[name]].mean()), 3) for name in BODY},
            "jitter_torso_per_frame2": {name: (None if np.isnan(jit[J[name]])
                                               else round(float(jit[J[name]]), 5))
                                        for name in BODY},
            "hitting_wrist_longest_gap_ms": round(longest_gap(vis[:, hit], t[sl]) * 1000, 1),
            "wrist_swap_frames": [int(frame_index[sl][k]) for k in wrist_swaps(part, unit)],
        }
        if scores is not None:
            sc = scores[sl]
            out["score_median"] = {name: _r(np.nanmedian(sc[:, J[name]])) for name in BODY}
            out["score_p10"] = {name: _r(np.nanpercentile(sc[:, J[name]], 10)) for name in BODY}
        return out

    return {
        "fps": fps,
        "median_frame_interval_s": round(float(np.median(np.diff(t))), 6) if n > 1 else None,
        "hand": hand,
        "torso_length_px": round(unit, 2),
        "scores_present": scores is not None,
        "min_score": min_score,
        "peak": {
            "frame": int(frame_index[peak]),
            "seconds": round(float(t[peak]), 4),
            "wrist_speed_torso_per_s": round(float(speed[peak]), 3),
        },
        "window": {"start_frame": int(frame_index[lo]), "end_frame": int(frame_index[hi - 1]),
                   "before_s": before, "after_s": after},
        "clip": block(slice(0, n)),
        "swing_window": block(slice(lo, hi)),
    }


def _r(v) -> float | None:
    return None if v is None or not np.isfinite(v) else round(float(v), 4)


def summary_markdown(report: dict) -> str:
    w = report["swing_window"]
    hand = report["hand"]
    rows = []
    for joint in ("shoulder", "elbow", "wrist", "hip", "knee", "ankle"):
        name = f"{hand}_{joint}"
        score = w.get("score_median", {}).get(name, "—") if report["scores_present"] else "n/a"
        rows.append(f"| {name} | {w['visible'][name]:.0%} | "
                    f"{w['jitter_torso_per_frame2'][name]} | {score} |")
    lines = [
        f"Peak {hand}-wrist speed {report['peak']['wrist_speed_torso_per_s']} torso lengths/s "
        f"at frame {report['peak']['frame']} ({report['peak']['seconds']} s).",
        f"Swing window frames {report['window']['start_frame']}–{report['window']['end_frame']}; "
        f"longest hitting-wrist gap {w['hitting_wrist_longest_gap_ms']} ms; "
        f"{len(w['wrist_swap_frames'])} suspected wrist swaps.",
        "Per-keypoint scores in response: "
        + ("yes" if report["scores_present"] else "**no** — confidence must come from elsewhere"),
        "",
        "| Joint (hitting side) | Visible | Jitter (torso/frame²) | Median score |",
        "|---|---|---|---|",
        *rows,
    ]
    return "\n".join(lines)


def report_for(frames: list[dict], *, width: int, height: int, hand: str,
               fps: float | None = None, min_score: float | None = None) -> dict:
    """The full pipeline the CLI and ``run_pose_m0`` share."""
    persons, crowded = lock_player(frames)
    xy, scores, field = keypoint_arrays(persons, width=width, height=height)
    report = analyze(xy, scores, hand=hand, times=frame_times(frames, fps), fps=fps,
                     frame_index=np.array([f["index"] for f in frames]), min_score=min_score)
    report.update(score_field=field, frames_with_multiple_people=crowded,
                  frames_with_nobody=sum(p is None for p in persons))
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Keypoint quality during one swing (M0).")
    ap.add_argument("poses", type=Path,
                    help="poses.json (gateway, local_pose or swing run) or a cache entry")
    ap.add_argument("--fps", type=float,
                    help="real-time frame rate, only for poses without per-frame times")
    ap.add_argument("--width", type=int, required=True, help="inference video width, px")
    ap.add_argument("--height", type=int, required=True, help="inference video height, px")
    ap.add_argument("--hand", choices=("right", "left"), default="right")
    ap.add_argument("--min-score", type=float,
                    help="treat keypoints scored below this as unseen")
    ap.add_argument("--json", type=Path, help="write the full report here")
    args = ap.parse_args(argv)

    try:
        report = report_for(load_frames(args.poses), width=args.width, height=args.height,
                            hand=args.hand, fps=args.fps, min_score=args.min_score)
    except QualityError as exc:
        print(f"cannot assess: {exc}", file=sys.stderr)
        return 1
    print(summary_markdown(report))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
