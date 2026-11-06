# Decisions

Choices made without asking, because they don't change cost or scope. Newest
first. Anything that changes cost or scope is an open question in
`PROPOSAL.md` instead.

| Date | Decision | Reason |
|---|---|---|
| 2026-11-05 | Subsampling uses ffmpeg `select` (first source frame at or after each slot) with `-fps_mode vfr`, not the `fps` filter | The `fps` filter put source frame 8k+3 in slot k (12.5 ms early at 30 fps, more than the 8 ms contact target) and duplicated frames above the source rate. `select` keeps real frames with their own timestamps and never duplicates. Verified with frame-ID clips in the tests. |
| 2026-11-05 | Seeks snap to the first source frame at or after the requested time (tolerance 1e-5 s) | An input seek labels that frame zero; labelling it with the requested time was off by up to one frame. The tolerance absorbs ffprobe's 6-decimal rounding, which otherwise skipped a frame at chunk seams. |
| 2026-11-05 | Coarse chunks are cut on the output frame grid | Cutting in seconds left sub-frame tail chunks that encoded empty files and crashed. |
| 2026-11-05 | The rendition cache key includes the media-time span, rate and real-time scale | Keyed on real-time values alone, a different `--capture-fps` reused renditions with the wrong frames and times (review H1). |
| 2026-11-05 | The budget is checked for the whole run before the first request, then per request; frames count as billed once sent | A run that can't finish now spends nothing. Counting a sent-but-failed request as billed is the conservative reading of an unknown bill. |
| 2026-11-05 | Exit codes 0 done, 1 refused, 2 usage / not built, 3 budget, 4 failed; key and model are checked before a run folder exists, and every later exit writes `run.json` | Crashes used to share code 1 with refusals and leave empty folders with no record of what was billed (review H2). |
| 2026-11-05 | `run.json` records the request (every option and argv), the effective config, and the commit plus a dirty flag | The config snapshot alone reported module defaults, not what a run used (review M8). |
| 2026-11-05 | M0's `running` driver runs in its own process, with `EVERY_FRAME` off, `VIDEO_FPS` at the window rate and `VIDEO_MAX_FRAMES` at the budget-checked count; it refuses if running's own frame count differs | Upstream's `EVERY_FRAME` overrode the frame count with a different file's count (M7). A subprocess keeps upstream's top-level `config`/`main`/`src` imports and patches out of the caller. |
| 2026-11-05 | Quality and key-frame tools work on real time per frame, resampled to an even grid before smoothing, and share one contact rule | Frame-index arithmetic turned a gap in the response into a peak 34× the true speed (M5), and the key-frame probe used an ungated rule of its own (M6). |
| 2026-11-05 | One OpenCV build (opencv-contrib-python, via mediapipe); the core package needs none; the wheel ships only `swing` | Three `cv2` providers in one environment break imports. A top-level `m0` package would clash; the M0 tools run from the checkout. |
| 2026-11-05 | `SWING_DATA_DIR` overrides where data, cache and runs go | A non-editable install has no project folder to write beside. |
| 2026-11-05 | The MediaPipe cache key includes the model file's SHA-256 | Model variants are often saved under the same file name. |
| 2026-11-04 | M1 started before the M0 review | Adam asked for it. The RFP's gate exists because real footage may change the design, so the points M0 could overturn sit behind interfaces (pose backend, timing, coarse rate). |
| 2026-11-04 | `--max-billed-frames` is required for the gateway backend, with no default | A default would be an invented cost limit. Requiring it means no run can bill without a stated ceiling. |
| 2026-11-04 | Coarse requests are chunked at 900 frames | It's the gateway's own default cap, and larger limits are unverified. M0 measures the real limit. |
| 2026-11-04 | Source cache identity = size + SHA-256 of the first and last 4 MB; renditions hashed in full | This survives renames and touches, and changes with any re-export or trim. Hashing a 2 GB original on every run is slow. |
| 2026-11-04 | Upstream `video.py` not copied; `swing.ingest` does renditions itself | M1 needs frame-exact seeking, resampling in real time, and a real-time map of every frame, which upstream's converter doesn't do. A copy would be dead code. |
| 2026-11-04 | The clipped-body check uses the largest person until the one-player lock (FR-6) lands in M2 | A cropped player is usually the largest person in frame. M2 replaces this with the locked player. |
| 2026-11-04 | `--capture-fps` equal to the file's rate declares normal-speed video | Ordinary iPhone video carries Apple metadata and was otherwise always "unknown". |
| 2026-11-03 | `--min-score` gate (0.5 in the YouTube check) applies to peak finding, visibility and gaps, not to jitter | It's a diagnostic setting, not a v1 threshold. Lost joints carry coordinates snapped onto other body parts, and on two clips they produced every false peak. |
| 2026-11-03 | Timing reads decoded frame timestamps, not packets | A screen recording had 582 packets for 464 frames. |
| 2026-11-03 | MediaPipe Pose Landmarker (heavy) as the local stand-in backend for M0 only | It's on PyPI, its model is on a host this environment can reach, and it returns per-keypoint visibility. The model card lists Apache-2.0; confirm before any adoption past M0. |
| 2026-11-03 | Each M0 key-frame question includes a "cannot tell" or "not visible" label | The RFP's rule: low confidence becomes Can't assess, never a guess. A question without an abstain option forces a guess. |
| 2026-11-03 | Key frames come from poses: contact = peak hitting-wrist speed; backswing and finish = highest wrist within 0.6 s either side | These are the RFP's contact proxy, plus the simplest definitions that keypoints support. 0.6 s is a starting value, to be checked by eye on the M0 crops. |
| 2026-11-03 | Vision crops pad the player box by 60% on every side (deadlift uses 15%) | The racket extends well past the body box. Deadlift only needed the back. |
| 2026-11-03 | Swing windows are cut by re-encoding, not by stream copy | A stream copy starts on a keyframe, which silently widens the window and the bill. |
| 2026-11-03 | M0 sets `video_max_frames` to the window's exact frame count and refuses runs over `--max-billed-frames` (default 500) | The RFP's frame-limit and cost-control rows. A 1.75 s window at 240 fps is 420 frames. **Corrected 2026-11-05:** as first written this had no effect, because upstream's `EVERY_FRAME` replaced it with its own count (review M7). |
| 2026-11-03 | M0 drives `running/main.py` with settings overrides (gait analysis, gait panel and foot markers off) rather than editing it | The RFP's M0 asks for the existing pipeline. This leaves upstream untouched. |
| 2026-11-03 | Upstream history imported as the base; `upstream` remote points at jeremyipark/vision-demos; no upstream file modified; `NOTICE` added | The repo was created standalone, not as a GitHub fork. This keeps attribution and clean merges. |
| 2026-11-03 | Wrist-swap rule: crossed pairing < 0.5 × straight pairing, and wrists ≥ 0.25 torso lengths apart | An M0 diagnostic only, not a v1 threshold. The separation floor stops two hands on the grip counting as a swap. |
| 2026-11-03 | Timing verdict boundary at 100 fps | This sits between ordinary iPhone video (≤ 60 fps) and slo-mo (120/240 fps). |
