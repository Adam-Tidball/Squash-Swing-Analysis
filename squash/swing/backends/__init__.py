"""Pose backends behind one interface (FR-3)."""

from .base import BudgetExceeded, CachedPose, FrameBudget, PoseBackend, PoseTrack  # noqa: F401


def make(name: str, **kwargs) -> "PoseBackend":
    """``vlmrun`` (the RFP's gateway), ``mediapipe`` (local stand-in) or ``fake``."""
    if name == "vlmrun":
        from .vlmrun import VlmRunPose
        return VlmRunPose(**kwargs)
    if name == "mediapipe":
        from .mediapipe_local import MediaPipePose
        return MediaPipePose(**kwargs)
    if name == "fake":
        from .fake import FakePose
        return FakePose(**kwargs)
    raise ValueError(f"unknown pose backend {name!r}: vlmrun, mediapipe or fake")
