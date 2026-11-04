"""ViTPose+ Large through the VLM Run Gateway: the RFP's first backend (FR-3).

The request is upstream's (``swing.upstream.pose``), with one rule added:
``video_max_frames`` is always the rendition's exact frame count, never the
gateway's 900-frame default. ``video_fps`` is set to the rendition's own rate,
because it is the detector cadence, not a sampling rate; sampling happens in
``swing.ingest`` before upload.
"""

from __future__ import annotations

from pathlib import Path

from .. import config as cfg
from ..timing import ffprobe_json
from ..upstream import pose as upstream_pose
from ..upstream.env import load_api_key


def _fps(video: Path) -> float:
    stream = next(s for s in ffprobe_json(video)["streams"] if s["codec_type"] == "video")
    num, _, den = (stream.get("avg_frame_rate") or "0/1").partition("/")
    return float(num) / float(den or 1) if float(den or 1) else 0.0


class VlmRunPose:
    name = "vlmrun"
    billable = True

    def __init__(self, *, model: str = cfg.POSE_MODEL, base_url: str = cfg.GATEWAY_BASE_URL,
                 timeout: float = cfg.REQUEST_TIMEOUT, precision: int = cfg.PRECISION,
                 key_search_from: Path = cfg.SQUASH_DIR):
        self.model, self.base_url, self.timeout, self.precision = model, base_url, timeout, precision
        self._key_from = key_search_from
        self._client = None

    def describe(self) -> dict:
        return {"backend": self.name, "model": self.model, "precision": self.precision,
                "every_frame": True, "contract": upstream_pose.CONTRACT}

    def _openai(self):
        if self._client is None:
            from openai import OpenAI
            api_key, _ = load_api_key(self._key_from)
            self._client = OpenAI(base_url=self.base_url, api_key=api_key,
                                  timeout=self.timeout, max_retries=1)
        return self._client

    def estimate(self, video: Path, *, n_frames: int) -> tuple[dict, dict | None]:
        video_b64, extra_body = upstream_pose.build_request(
            video, every_frame=True, fps=_fps(video), n_frames=n_frames,
            video_fps=None, video_max_frames=n_frames, precision=self.precision)
        return upstream_pose.request_poses(self._openai(), model=self.model,
                                           video_b64=video_b64, extra_body=extra_body)
