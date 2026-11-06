"""Synthetic poses: a side-on player who swings at fixed real-time instants.

For tests and offline runs only. Nothing it returns is evidence about squash.
The arm traces a circle *in pixels*: coordinates are normalized per axis, so
the arc is narrowed by the frame's aspect ratio. A circle in normalized units
is an ellipse on screen, and its speed peak moves off the middle of the swing.
"""

from __future__ import annotations

import numpy as np

from ..ingest import Rendition
from ..upstream.skeleton import KPT_NAMES

BODY = {
    "nose": (0.50, 0.20), "left_eye": (0.51, 0.19), "right_eye": (0.49, 0.19),
    "left_ear": (0.52, 0.20), "right_ear": (0.48, 0.20),
    "left_shoulder": (0.53, 0.30), "right_shoulder": (0.47, 0.30),
    "left_hip": (0.52, 0.52), "right_hip": (0.48, 0.52),
    "left_knee": (0.56, 0.68), "right_knee": (0.44, 0.68),
    "left_ankle": (0.58, 0.85), "right_ankle": (0.42, 0.85),
}


def pose_at(t: float, *, hand: str, swings: tuple[float, ...], aspect: float,
            rng: np.random.Generator | None = None, noise: float = 0.0) -> list[list[float]]:
    """COCO-17 normalized keypoints at time ``t`` (same clock as ``swings``)."""
    k = dict(BODY)
    nearest = min(swings, key=lambda s: abs(s - t)) if swings else 1e9
    phase = np.tanh((t - nearest) * 12)            # -1 backswing .. +1 follow-through
    if abs(t - nearest) > 0.6:
        phase = -1.0 if t < nearest else 1.0
        if abs(t - nearest) > 1.2:
            phase = -1.0                            # ready position between swings
    sx, sy = k[f"{hand}_shoulder"]
    ang = np.pi * (0.8 + 0.7 * (phase + 1))         # high behind, down at contact, high in front
    k[f"{hand}_elbow"] = (sx + 0.08 * np.cos(ang) / aspect, sy - 0.08 * np.sin(ang))
    k[f"{hand}_wrist"] = (sx + 0.16 * np.cos(ang) / aspect, sy - 0.16 * np.sin(ang))
    other = "left" if hand == "right" else "right"
    ox, oy = k[f"{other}_shoulder"]
    k[f"{other}_elbow"], k[f"{other}_wrist"] = (ox + 0.03, oy + 0.10), (ox + 0.06, oy + 0.18)
    jitter = (lambda: rng.normal(0, noise)) if (rng is not None and noise) else (lambda: 0.0)
    return [[round(float(x + jitter()), 4), round(float(y + jitter()), 4)]
            for x, y in (k[name] for name in KPT_NAMES)]


class FakePose:
    """Swings at ``swings``: real seconds into the *source* clip.

    Uses each rendition frame's real time, so a swing lands where it was
    asked for whatever the chunking, sampling rate or retiming.
    """

    name = "fake"

    def __init__(self, *, hand: str = "right", swings: tuple[float, ...] = (1.0,),
                 billable: bool = False, second_person: bool = False, noise: float = 0.0015,
                 seed: int = 0):
        self.hand, self.swings, self.billable = hand, tuple(swings), billable
        self.second_person, self.noise, self.seed = second_person, noise, seed
        self.calls = 0

    def describe(self) -> dict:
        return {"backend": self.name, "hand": self.hand, "swings": list(self.swings),
                "second_person": self.second_person, "noise": self.noise, "seed": self.seed}

    def check_ready(self) -> None:
        pass

    def estimate(self, rendition: Rendition) -> tuple[dict, dict | None]:
        self.calls += 1
        aspect = rendition.width / rendition.height
        rng = np.random.default_rng(self.seed)
        items = []
        for i, t in enumerate(rendition.real_times):
            items.append({"frame_id": i, "track_id": 0, "bbox_xywh": [0.36, 0.15, 0.30, 0.74],
                          "kpts_xy": pose_at(float(t), hand=self.hand, swings=self.swings,
                                             aspect=aspect, rng=rng, noise=self.noise),
                          "kpts_scores": [0.95] * 17})
            if self.second_person:
                still = pose_at(float(t), hand=self.hand, swings=(), aspect=aspect)
                shifted = [[min(0.99, x + 0.35), y] for x, y in still]
                items.append({"frame_id": i, "track_id": 1,
                              "bbox_xywh": [0.75, 0.20, 0.15, 0.60],
                              "kpts_xy": shifted, "kpts_scores": [0.9] * 17})
        frames = [{"frame_id": i} for i in range(rendition.n_frames)]
        payload = {"content": {"object": "vid.pose.kpts", "kpts_labels": KPT_NAMES,
                               "frames": frames, "items": items}}
        return payload, {"fake": True}
