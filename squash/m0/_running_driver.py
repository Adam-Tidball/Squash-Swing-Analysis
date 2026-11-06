"""Runs ``running/main.py`` with M0's settings, in its own process.

Started by ``m0.run_pose_m0`` with ``cwd=running/`` so upstream's top-level
``config``, ``main`` and ``src`` imports resolve, and nothing they set or
patch leaks into the caller's interpreter. Takes one JSON argument:

    {"cut", "n_frames", "media_fps", "token", "fake": {...} | null}

Prints ``RUN_DIR=<path>`` as its last line.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def main() -> int:
    spec = json.loads(sys.argv[1])
    n, token = int(spec["n_frames"]), spec["token"]
    if spec.get("fake"):
        # Unconditionally fake: a real key in the environment must never be
        # reachable from a run that promised to be free.
        os.environ["VLMRUN_API_KEY"] = "fake-offline-key"

    sys.path.insert(0, os.getcwd())
    import config as cfg            # running/config.py
    import main as running_main     # running/main.py
    from src import pose

    cfg.INPUT_VIDEO = Path(spec["cut"])
    cfg.TRIM_SECONDS = None
    # Detector stride 1 by cadence, and exactly n frames billed: with
    # EVERY_FRAME on, upstream would replace VIDEO_MAX_FRAMES with OpenCV's
    # own count of its re-encoded copy, which is not the number checked
    # against the budget.
    cfg.EVERY_FRAME = False
    cfg.VIDEO_FPS = float(spec["media_fps"])
    cfg.VIDEO_MAX_FRAMES = n
    cfg.ANALYZE_GAIT = False
    cfg.SIDE_PANEL = False
    cfg.HIGHLIGHT_FEET = False
    cfg.FLASH_ON_CONTACT = False
    cfg.TRAIL_ON_ANKLES = False
    cfg.SAVE_CLEARANCE_PLOT = cfg.SAVE_STEPS_PLOT = cfg.SAVE_KNEE_PLOT = False
    cfg.MAX_PERSONS = 1
    # A unique folder name: upstream reuses a same-second folder otherwise.
    cfg.RUN_STAMP_FORMAT = f"%Y%m%d-%H%M%S-m0-{token}"

    real_build = pose.build_request

    def checked_build(video, **kw):
        if kw["n_frames"] != n:
            raise SystemExit(
                f"refusing: running decodes {kw['n_frames']} frames from its copy, "
                f"but {n} were budget-checked; the two counts must agree")
        b64, extra = real_build(video, **kw)
        if extra.get("video_max_frames") != n:
            raise SystemExit(f"refusing: request would bill {extra.get('video_max_frames')} "
                             f"frames, not the {n} checked")
        return b64, extra

    pose.build_request = checked_build
    if spec.get("fake"):
        from swing.backends.fake import pose_at
        from swing.upstream.skeleton import KPT_NAMES

        fake = spec["fake"]

        def fake_request(client, **kw):
            items = [{"frame_id": i, "track_id": 0, "bbox_xywh": [0.36, 0.15, 0.30, 0.74],
                      "kpts_xy": pose_at(i / cfg.VIDEO_FPS, hand=fake["hand"],
                                         swings=(fake["swing_at"],), aspect=fake["aspect"]),
                      "kpts_scores": [0.95] * 17}
                     for i in range(n)]
            payload = {"content": {"object": "vid.pose.kpts", "kpts_labels": KPT_NAMES,
                                   "frames": [{"frame_id": i} for i in range(n)],
                                   "items": items}}
            return payload, {"note": "fake gateway, nothing billed"}

        pose.request_poses = fake_request
        cfg.REUSE_POSES = False
        if running_main.pose is not pose:
            raise SystemExit("refusing: could not patch running's gateway call")

    code = running_main.main([])
    if code:
        return code
    found = sorted(Path(cfg.OUTPUT_DIR).glob(f"*-m0-{token}"))
    if len(found) != 1:
        raise SystemExit(f"expected one run folder for token {token}, found {len(found)}")
    print(f"RUN_DIR={found[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
