"""M0: does this clip's timing describe real time?

iPhone slo-mo reaches a laptop in one of two shapes, and they need opposite
handling:

* **native** -- the original .MOV, shared "as original" or via Image Capture.
  Frames carry ~240 fps timestamps, so presentation time *is* real time. The
  slow-motion ramp Photos plays is a separate timed-metadata track that only
  Apple's player reads; ffmpeg and OpenCV ignore it.
* **retimed** -- an export (AirDrop default, Messages, many cloud uploads). The
  ramp has been baked in: a ~30 fps file, slowed in the middle and possibly at
  normal speed at either end. Presentation time is *not* real time, and the
  speed-ups at the ends cannot be seen from timestamps at all.

This module only reads what the file says. It cannot prove a retimed file has
no normal-speed ramps; that is why ``--known-seconds`` exists (AC-4): film a
stopwatch or a known-length event and compare.

    python -m m0.timing_probe clip.mov [--known-seconds 3.20 --event-frames 768]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

# Below this the clip cannot be native slo-mo; above it, it cannot be a
# normal-speed export. 100 sits between the 60 fps ceiling of ordinary iPhone
# video and the 120/240 fps slo-mo modes.
HFR_MIN_FPS = 100.0

# A frame interval this far off the median counts as irregular. Native 240 fps
# files carry 1/600 s timebase rounding, so a few percent is normal jitter.
IRREGULAR_TOLERANCE = 0.25


@dataclass
class TimingReport:
    path: str
    codec: str | None
    width: int
    height: int
    rotation: int
    nominal_fps: float | None          # r_frame_rate
    average_fps: float | None          # avg_frame_rate
    measured_fps: float | None         # from packet timestamps
    n_frames: int
    media_seconds: float
    irregular_intervals: int           # intervals off the median by > tolerance
    apple_tags: dict = field(default_factory=dict)
    data_streams: list = field(default_factory=list)
    audio: dict | None = None          # real-time audio is the candidate contact signal
    verdict: str = "unknown"           # native | retimed | normal-speed | unknown
    real_time_scale: float | None = None   # real seconds per media second
    notes: list = field(default_factory=list)
    known_check: dict | None = None


def _rate(text: str | None) -> float | None:
    if not text or text in ("0/0", "0"):
        return None
    if "/" in text:
        num, den = text.split("/")
        return float(num) / float(den) if float(den) else None
    return float(text)


def ffprobe_json(path: Path) -> dict:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        capture_output=True, text=True, check=True)
    return json.loads(proc.stdout)


def frame_times(path: Path) -> np.ndarray:
    """Timestamps of every *decoded* frame, sorted, in media seconds.

    Decoded frames, not container packets: a screen recording or an edited
    file can hold packets that are never shown (one test file had 582 packets
    for 464 frames, 115 of them sharing a timestamp), and counting those
    makes frame index stop being a clock.
    """
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "frame=best_effort_timestamp_time", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True)
    values = []
    for line in proc.stdout.split():
        field = line.strip().strip(",").split(",")[0]
        if field and field != "N/A":
            values.append(float(field))
    return np.sort(np.array(values, dtype=float))


def classify(probe: dict, pts: np.ndarray, *, capture_fps: float | None = None) -> TimingReport:
    """Pure: turn ffprobe output and packet times into a verdict.

    ``capture_fps`` is the player's own statement of how the clip was filmed
    (e.g. 240). It is the only way to recover real time from a retimed file
    whose slow section is uniform, and it is recorded as an assumption, never
    inferred.
    """
    streams = probe.get("streams") or []
    video = next((s for s in streams if s.get("codec_type") == "video"), {})
    fmt = probe.get("format") or {}

    rotation = 0
    for side in video.get("side_data_list") or []:
        if "rotation" in side:
            rotation = int(side["rotation"])

    tags = {**(fmt.get("tags") or {}), **(video.get("tags") or {})}
    apple = {k: v for k, v in tags.items() if "apple" in k.lower()}
    data_streams = [
        {"index": s.get("index"), "codec_type": s.get("codec_type"),
         "codec_tag": s.get("codec_tag_string"),
         "handler": (s.get("tags") or {}).get("handler_name")}
        for s in streams if s.get("codec_type") in ("data", "unknown")
    ]

    audio = next(({"codec": s.get("codec_name"),
                   "sample_rate": int(s.get("sample_rate") or 0),
                   "seconds": float(s.get("duration") or 0.0)}
                  for s in streams if s.get("codec_type") == "audio"), None)

    dts = np.diff(pts) if pts.size > 1 else np.array([])
    duplicates = int(np.sum(dts <= 0))
    dts = dts[dts > 0]
    measured = float(1.0 / np.median(dts)) if dts.size else None
    irregular = int(np.sum(np.abs(dts / np.median(dts) - 1) > IRREGULAR_TOLERANCE)) if dts.size else 0
    irregular += duplicates
    media_seconds = float(pts[-1] - pts[0] + (np.median(dts) if dts.size else 0)) if pts.size else 0.0

    report = TimingReport(
        path=str(fmt.get("filename", "")),
        codec=video.get("codec_name"),
        width=int(video.get("width") or 0),
        height=int(video.get("height") or 0),
        rotation=rotation,
        nominal_fps=_rate(video.get("r_frame_rate")),
        average_fps=_rate(video.get("avg_frame_rate")),
        measured_fps=round(measured, 3) if measured else None,
        n_frames=int(pts.size),
        media_seconds=round(media_seconds, 4),
        irregular_intervals=irregular,
        apple_tags=apple,
        data_streams=data_streams,
        audio=audio,
    )

    fps = measured or report.average_fps or 0.0
    if fps >= HFR_MIN_FPS:
        report.verdict = "native"
        report.real_time_scale = 1.0
        report.notes.append(
            f"{fps:.1f} fps timestamps: presentation time is real time. "
            "Any slow-motion ramp is display metadata and is ignored.")
        if capture_fps and abs(capture_fps - fps) / capture_fps > 0.05:
            report.notes.append(
                f"--capture-fps {capture_fps} disagrees with the file's {fps:.1f} fps; "
                "the file wins.")
    elif capture_fps and capture_fps >= HFR_MIN_FPS:
        report.verdict = "retimed"
        report.real_time_scale = round(fps / capture_fps, 6)
        report.notes.append(
            f"{fps:.1f} fps file declared as {capture_fps:.0f} fps capture: "
            f"real seconds = media seconds x {report.real_time_scale}. "
            "Assumes the whole clip is slowed; Photos exports may keep "
            "normal-speed ramps at the start and end. Trim them or re-export "
            "the original.")
    elif fps:
        report.verdict = "normal-speed" if not apple and not data_streams else "unknown"
        report.notes.append(
            f"{fps:.1f} fps and no --capture-fps: treated as real time. If this "
            "was filmed in slo-mo, it is a retimed export -- pass --capture-fps "
            "or share the original file.")
    if irregular:
        report.notes.append(
            f"{irregular} frame intervals differ from the median by more than "
            f"{IRREGULAR_TOLERANCE:.0%} or repeat a timestamp: variable frame rate, "
            "dropped or duplicated frames. "
            "Frame index is not a clock here; use timestamps.")
    if data_streams:
        report.notes.append(
            f"{len(data_streams)} timed-metadata stream(s) present (iPhone slo-mo "
            "originals carry the slow-motion ramp here). ffmpeg drops them on convert.")
    if audio and report.media_seconds and audio["seconds"]:
        ratio = audio["seconds"] / report.media_seconds
        if abs(ratio - 1) > 0.02:
            report.notes.append(
                f"audio runs {audio['seconds']:.2f}s against {report.media_seconds:.2f}s "
                "of video: the two clocks disagree, so check which one is real time.")
    elif not audio:
        report.notes.append("no audio stream: ball-impact timing is unavailable for this clip.")
    return report


def known_duration_check(report: TimingReport, *, known_seconds: float,
                         event_frames: int) -> dict:
    """AC-4: an event of known real length, counted in frames, vs the clock we use."""
    fps = report.measured_fps or report.average_fps
    scale = report.real_time_scale or 1.0
    measured = event_frames / fps * scale if fps else float("nan")
    error = (measured - known_seconds) / known_seconds
    return {
        "known_seconds": known_seconds,
        "event_frames": event_frames,
        "measured_seconds": round(measured, 5),
        "relative_error": round(error, 5),
        "passes_ac4": bool(abs(error) <= 0.01),
    }


def probe(path: Path, *, capture_fps: float | None = None) -> TimingReport:
    return classify(ffprobe_json(path), frame_times(path), capture_fps=capture_fps)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("clip", type=Path)
    ap.add_argument("--capture-fps", type=float, help="how the clip was filmed, e.g. 240")
    ap.add_argument("--known-seconds", type=float,
                    help="real length of a timed event in the clip (AC-4)")
    ap.add_argument("--event-frames", type=int,
                    help="frames that event spans, counted by hand")
    ap.add_argument("--json", type=Path, help="also write the report here")
    args = ap.parse_args(argv)

    report = probe(args.clip, capture_fps=args.capture_fps)
    if args.known_seconds and args.event_frames:
        report.known_check = known_duration_check(
            report, known_seconds=args.known_seconds, event_frames=args.event_frames)

    out = json.dumps(asdict(report), indent=2)
    print(out)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(out)
    return 0 if report.verdict in ("native", "retimed", "normal-speed") else 2


if __name__ == "__main__":
    sys.exit(main())
