# Decisions

Choices made without asking, because they don't change cost or scope. Newest
first. Anything that changes cost or scope is an open question in
`PROPOSAL.md` instead.

| Date | Decision | Reason |
|---|---|---|
| 2026-11-03 | `--min-score` gate (0.5 in the YouTube check) applies to peak finding, visibility and gaps, not to jitter | It's a diagnostic setting, not a v1 threshold. Lost joints carry coordinates snapped onto other body parts, and on two clips they produced every false peak. |
| 2026-11-03 | Timing reads decoded frame timestamps, not packets | A screen recording had 582 packets for 464 frames. |
| 2026-11-03 | MediaPipe Pose Landmarker (heavy) as the local stand-in backend for M0 only | It's on PyPI, its model is on a host this environment can reach, and it returns per-keypoint visibility. The model card lists Apache-2.0; confirm before any adoption past M0. |
| 2026-11-03 | Each M0 key-frame question includes a "cannot tell" or "not visible" label | The RFP's rule: low confidence becomes Can't assess, never a guess. A question without an abstain option forces a guess. |
| 2026-11-03 | Key frames come from poses: contact = peak hitting-wrist speed; backswing and finish = highest wrist within 0.6 s either side | These are the RFP's contact proxy, plus the simplest definitions that keypoints support. 0.6 s is a starting value, to be checked by eye on the M0 crops. |
| 2026-11-03 | Vision crops pad the player box by 60% on every side (deadlift uses 15%) | The racket extends well past the body box. Deadlift only needed the back. |
| 2026-11-03 | Swing windows are cut by re-encoding, not by stream copy | A stream copy starts on a keyframe, which silently widens the window and the bill. |
| 2026-11-03 | M0 sets `video_max_frames` to the window's exact frame count and refuses runs over `--max-billed-frames` (default 500) | The RFP's frame-limit and cost-control rows. A 1.75 s window at 240 fps is 420 frames. |
| 2026-11-03 | M0 drives `running/main.py` with settings overrides (gait analysis, gait panel and foot markers off) rather than editing it | The RFP's M0 asks for the existing pipeline. This leaves upstream untouched. |
| 2026-11-03 | Upstream history imported as the base; `upstream` remote points at jeremyipark/vision-demos; no upstream file modified; `NOTICE` added | The repo was created standalone, not as a GitHub fork. This keeps attribution and clean merges. |
| 2026-11-03 | Wrist-swap rule: crossed pairing < 0.5 × straight pairing, and wrists ≥ 0.25 torso lengths apart | An M0 diagnostic only, not a v1 threshold. The separation floor stops two hands on the grip counting as a swap. |
| 2026-11-03 | Timing verdict boundary at 100 fps | This sits between ordinary iPhone video (≤ 60 fps) and slo-mo (120/240 fps). |
