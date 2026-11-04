"""M0: does this clip's timing describe real time?

The logic lives in ``swing.timing`` (FR-1); this is its command line.

    python -m m0.timing_probe clip.mov [--known-seconds 3.20 --event-frames 768]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from swing.timing import (  # noqa: F401  (re-exported for the M0 tools and tests)
    HFR_MIN_FPS, IRREGULAR_TOLERANCE, TimingReport, classify, ffprobe_json,
    frame_times, known_duration_check, probe,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Does this clip's timing describe real time?")
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
