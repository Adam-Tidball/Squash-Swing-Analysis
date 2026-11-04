# Pre-M0 check: two YouTube coaching clips

**Not M0.** These are Mac screen recordings of a YouTube coaching video, run
through MediaPipe Pose Landmarker (heavy), **not** ViTPose. The footage and
frames are not in the repo. What this does show is how 2D pose behaves on a
squash arm, and what the M0 tools caught.

## Input

| Clip | Content used | Format | Timing |
|---|---|---|---|
| `…4.12.49PM.mov` | Forehand drive, side-on, 1.12–5.09 s (camera cuts at 1.12 and 5.12) | 2064×1166, 60 fps, no audio | Slowed in the edit: backswing to finish takes ~1.7 s, so not real time, and the factor is unknown |
| `…4.13.47PM.mov` | Backhand drive, mostly from behind, 0–7.1 s (cut at 7.12). YouTube end-cards cover part of the frame from ~3.5 s | 2064×1166, 60 fps, no audio | Unknown |

Hand: right. The player is right-handed.

## Findings

| Question | Forehand | Backhand |
|---|---|---|
| Wrist peak (gated at score 0.5) | 2.58 s, just before the ball-near-wrist frame at ~2.70 s | 2.70 s, in the contact zone (wrist on the handle at 2.90 s) |
| Hitting wrist seen in swing window | 62% | 62% |
| Longest wrist blind stretch | 600 ms (follow-through) | 317 ms (follow-through) |
| Hitting elbow seen | 79% | 21% (body hides it, back to camera) |
| Legs and hips | ~100% | 81–100% |

By eye, on drawn skeletons:

- The body and lunge are tracked well throughout.
- The wrist is tracked through backswing and contact, then lost in the
  follow-through, where it snaps onto the shoulder or hip.
- The visibility score falls to ≈0 on exactly those frames, so the score
  separates good frames from bad ones on these clips.
- The racket is sharp in some frames (forehand backswing, backhand contact)
  and a blur in others (forehand at 2.55 s).

## Bugs these clips exposed (fixed)

1. **Packet counts aren't frame counts.** The forehand recording has 582
   packets but 464 decoded frames, with 115 repeated packet timestamps and
   one 50 ms gap. `timing_probe` now reads decoded frame timestamps and
   counts repeats as irregular. `local_pose` picks frames by timestamp, not
   by seeking to a frame number.
2. **Lost joints made the fastest "motion".** Ungated, the forehand peak
   landed on the camera cut, and the backhand peak landed on noise under the
   end-cards. `keypoint_quality --min-score` now ignores low-scored
   keypoints when finding the peak, and counts visibility and gaps from the
   same gate.
3. **A segment boundary one frame past a cut** put the peak on the new
   scene. Detecting cuts must come before swing detection on any edited
   footage. Fixed-camera iPhone clips shouldn't have cuts, but FR-2 input
   checks should still look for them.

## What this means for the plan

- **G1 (keypoint scores):** MediaPipe returns usable per-keypoint visibility.
  Whether ViTPose via the gateway does is still the M0 question. If it
  doesn't, a local model is a credible source of confidence, or of poses
  outright (open question 2).
- **Follow-through checks** (RFP: Follow-through, Swing path after contact)
  will often be Can't assess from keypoints alone at 60 fps. They lean on the
  vision reads, or on 240 fps reducing blur, which M0 tests.
- **Timing from edited footage is unusable.** These clips can't inform any
  timing metric, AC-3 or AC-4.
