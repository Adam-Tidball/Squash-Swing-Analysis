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
from ..ingest import Rendition
from ..timing import rate, video_stream
from ..upstream import pose as upstream_pose
from ..upstream.env import load_api_key


class GatewayKeyMissing(RuntimeError):
    pass


class VlmRunPose:
    name = "vlmrun"
    billable = True

    def __init__(self, *, model: str = cfg.POSE_MODEL, base_url: str = cfg.GATEWAY_BASE_URL,
                 timeout: float = cfg.REQUEST_TIMEOUT, precision: int = cfg.PRECISION,
                 key_search_from: Path = cfg.SQUASH_DIR):
        self.model, self.base_url = model, base_url
        self.timeout, self.precision = timeout, precision
        self._key_from = key_search_from
        self._client = None

    def describe(self) -> dict:
        return {"backend": self.name, "model": self.model, "precision": self.precision,
                "every_frame": True, "contract": upstream_pose.CONTRACT}

    def _openai(self):
        if self._client is None:
            from openai import OpenAI
            try:
                api_key, _ = load_api_key(self._key_from)
            except RuntimeError as exc:
                # Upstream's text points at running/; say where it goes here.
                raise GatewayKeyMissing(
                    "no VLMRUN_API_KEY: export it, or copy .env.example to .env at the "
                    "repo root and add the key (get one at https://app.vlm.run)") from exc
            self._client = OpenAI(base_url=self.base_url, api_key=api_key,
                                  timeout=self.timeout, max_retries=1)
        return self._client

    def check_ready(self) -> None:
        """Fail before any work if the key is missing."""
        self._openai()

    def estimate(self, rendition: Rendition) -> tuple[dict, dict | None]:
        # Detector cadence = the file's own rate (stride 1); frames = exactly
        # the rendition's decoded count, the number the budget was checked on.
        # A subsampled rendition is variable-rate; the detector reaches stride 1
        # once video_fps is >= the decoded rate, so take the higher of the two
        # rates ffprobe reports.
        stream = video_stream(rendition.path) or {}
        fps = max(rate(stream.get("avg_frame_rate")) or 0.0,
                  rate(stream.get("r_frame_rate")) or 0.0)
        video_b64, extra_body = upstream_pose.build_request(
            rendition.path, every_frame=False, fps=fps, n_frames=rendition.n_frames,
            video_fps=fps, video_max_frames=rendition.n_frames, precision=self.precision)
        # upstream request_poses returns (payload, usage) despite its -> dict hint.
        return upstream_pose.request_poses(self._openai(), model=self.model,
                                           video_b64=video_b64, extra_body=extra_body)
