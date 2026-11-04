"""M0: how good are ViTPose keypoints during a squash swing?

Reads one gateway pose response (a run's ``poses.json``, or a cached
``poses.<digest>.json`` from ``running/data/cache``) and reports, for the
hitting arm above all:

* visibility -- how often each joint was returned at all (``(0, 0)`` is the
  model's "not visible" sentinel)
* confidence -- per-keypoint scores, *if the response carries any*. Upstream
  never reads a score field, so whether one exists is itself an M0 finding:
  FR-7 and every metric's confidence depend on it.
* jitter -- median frame-to-frame acceleration, in torso lengths, which is
  what turns into noise on wrist speed and contact timing
* gaps -- the longest run of missing frames on the hitting wrist, in ms
* left/right swaps -- frames where the two wrists look exchanged
* the hitting wrist's speed peak, the contact proxy every later step leans on

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

KPT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
J = {name: i for i, name in enumerate(KPT_NAMES)}
BODY = [n for n in KPT_NAMES if not n.endswith(("eye", "ear")) and n != "nose"]

# Field names a per-keypoint score might come back under. The first one found
# on any person record is used and reported.
SCORE_FIELDS = ("kpts_scores", "kpts_score", "kpts_conf", "keypoint_scores", "scores")

# Two wrists exchanged between frames: the crossed pairing has to beat the
# straight one by this factor, and the wrists must be this far apart (torso
# lengths), so two hands together on the grip do not count as a swap.
SWAP_GAIN = 0.5
SWAP_MIN_SEPARATION = 0.25


# ── reading the response ─────────────────────────────────────────────────────
def load_payload(path: Path) -> dict:
    blob = json.loads(path.read_text())
    return blob["payload"] if "payload" in blob and "content" not in blob else blob


def frames_from_payload(payload: dict) -> list[dict]:
    """Same regrouping as ``running/src/pose.unwrap``, without its imports."""
    content = payload.get("content") or {}
    by_index: dict[int, list[dict]] = {int(f["frame_id"]): [] for f in content.get("frames") or []}
    for person in content.get("items") or []:
        if person.get("frame_id") is not None:
            by_index.setdefault(int(person["frame_id"]), []).append(person)
    return [{"index": i, "persons": by_index[i]} for i in sorted(by_index)]


def _iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def lock_player(frames: list[dict]) -> tuple[list[dict | None], int]:
    """One person per frame, by box continuity, falling back to the largest box.

    Returns the chosen record per frame and how many frames had more than one
    distinct person (IoU < 0.6 with each other), which is FR-2's "multiple
    people" warning.
    """
    chosen: list[dict | None] = []
    crowded = 0
    previous = None
    for frame in frames:
        persons = [p for p in frame["persons"] if p.get("bbox_xywh")]
        if not persons:
            chosen.append(None)
            continue
        distinct = []
        for p in persons:
            if all(_iou(p["bbox_xywh"], d["bbox_xywh"]) < 0.6 for d in distinct):
                distinct.append(p)
        crowded += len(distinct) > 1
        best = max(persons, key=lambda p: (
            _iou(p["bbox_xywh"], previous) if previous is not None else 0.0,
            p["bbox_xywh"][2] * p["bbox_xywh"][3]))
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
        raise ValueError("no frame has both shoulders and both hips; cannot set a unit")
    return float(np.median(d))


def _fill(x: np.ndarray) -> np.ndarray:
    ok = np.isfinite(x)
    if ok.all() or not ok.any():
        return x
    idx = np.arange(len(x))
    out = x.copy()
    out[~ok] = np.interp(idx[~ok], idx[ok], x[ok])
    return out


def _smooth3(x: np.ndarray) -> np.ndarray:
    """Median-3 then mean-3, edge-padded: the upstream gait.py defaults."""
    if len(x) < 3:
        return x
    p = np.pad(x, 1, mode="edge")
    med = np.array([np.median(p[i:i + 3]) for i in range(len(x))])
    p = np.pad(med, 1, mode="edge")
    return np.convolve(p, np.ones(3) / 3, mode="valid")


def wrist_speed(xy: np.ndarray, fps: float, hand: str, unit: float) -> np.ndarray:
    """Hitting-wrist speed in torso lengths per second, gaps filled, lightly smoothed."""
    w = xy[:, J[f"{hand}_wrist"]] / unit
    x, y = _smooth3(_fill(w[:, 0])), _smooth3(_fill(w[:, 1]))
    return np.hypot(np.gradient(x), np.gradient(y)) * fps


def longest_gap(missing: np.ndarray) -> int:
    best = run = 0
    for m in missing:
        run = run + 1 if m else 0
        best = max(best, run)
    return best


def jitter(xy: np.ndarray, unit: float) -> np.ndarray:
    """Per joint: median |second difference| in torso lengths per frame^2."""
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
    for t in range(1, len(xy)):
        pts = (lw[t - 1], rw[t - 1], lw[t], rw[t])
        if not all(np.isfinite(p).all() for p in pts):
            continue
        if np.linalg.norm(lw[t] - rw[t]) < SWAP_MIN_SEPARATION:
            continue
        straight = np.linalg.norm(lw[t] - lw[t - 1]) + np.linalg.norm(rw[t] - rw[t - 1])
        crossed = np.linalg.norm(lw[t] - rw[t - 1]) + np.linalg.norm(rw[t] - lw[t - 1])
        if crossed < SWAP_GAIN * straight:
            hits.append(t)
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


def analyze(xy: np.ndarray, scores: np.ndarray | None, *, fps: float, hand: str,
            frame_index: np.ndarray | None = None, before: float = 1.0,
            after: float = 0.75, min_score: float | None = None) -> dict:
    """Quality report for the whole clip and for the window around the speed peak.

    Visibility, jitter and gaps are reported on the raw keypoints. The speed
    peak (the contact proxy) is found on keypoints gated at *min_score*, and
    only where the hitting wrist was actually seen, never on a filled gap.
    """
    n = len(xy)
    frame_index = np.arange(n) if frame_index is None else frame_index
    unit = torso_length(xy)
    gated = gate(xy, scores, min_score)
    seen = np.isfinite(gated[:, J[f"{hand}_wrist"], 0])
    speed = wrist_speed(gated, fps, hand, unit)
    speed[~seen] = np.nan
    peak = int(np.nanargmax(speed))
    lo, hi = max(0, peak - round(before * fps)), min(n, peak + round(after * fps) + 1)

    def block(sl: slice) -> dict:
        part = xy[sl]
        # Seen = returned and, when gating, scored at or above min_score. A
        # model that always returns coordinates (MediaPipe does) would
        # otherwise report 100% visibility and no gaps while it is guessing.
        vis = np.isfinite(gated[sl][..., 0])
        jit = jitter(part, unit)
        hit = J[f"{hand}_wrist"]
        out = {
            "frames": int(part.shape[0]),
            "visible": {name: round(float(vis[:, J[name]].mean()), 3) for name in BODY},
            "jitter_torso_per_frame2": {name: (None if np.isnan(jit[J[name]])
                                               else round(float(jit[J[name]]), 5))
                                        for name in BODY},
            "hitting_wrist_longest_gap_ms": round(
                longest_gap(~vis[:, hit]) / fps * 1000, 1),
            "wrist_swap_frames": [int(frame_index[sl][t]) for t in wrist_swaps(part, unit)],
        }
        if scores is not None:
            sc = scores[sl]
            out["score_median"] = {name: _r(np.nanmedian(sc[:, J[name]])) for name in BODY}
            out["score_p10"] = {name: _r(np.nanpercentile(sc[:, J[name]], 10)) for name in BODY}
        return out

    return {
        "fps": fps,
        "hand": hand,
        "torso_length_px": round(unit, 2),
        "scores_present": scores is not None,
        "min_score": min_score,
        "peak": {
            "frame": int(frame_index[peak]),
            "seconds": round(float(frame_index[peak]) / fps, 4),
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Keypoint quality during one swing (M0).")
    ap.add_argument("poses", type=Path, help="poses.json or a cached poses.<digest>.json")
    ap.add_argument("--fps", type=float, required=True, help="real-time frame rate, e.g. 240")
    ap.add_argument("--width", type=int, required=True, help="inference video width, px")
    ap.add_argument("--height", type=int, required=True, help="inference video height, px")
    ap.add_argument("--hand", choices=("right", "left"), default="right")
    ap.add_argument("--min-score", type=float,
                    help="ignore keypoints scored below this when finding the speed peak")
    ap.add_argument("--json", type=Path, help="write the full report here")
    args = ap.parse_args(argv)

    frames = frames_from_payload(load_payload(args.poses))
    persons, crowded = lock_player(frames)
    xy, scores, field = keypoint_arrays(persons, width=args.width, height=args.height)
    report = analyze(xy, scores, fps=args.fps, hand=args.hand,
                     frame_index=np.array([f["index"] for f in frames]),
                     min_score=args.min_score)
    report["score_field"] = field
    report["frames_with_multiple_people"] = crowded
    report["frames_with_nobody"] = sum(p is None for p in persons)

    print(summary_markdown(report))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
