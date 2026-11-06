# Squash Swing Analysis: proposal and M0 plan

Response to the *Squash Swing Analysis Tool: Request for Proposal* (2026-11-03).
The RFP asks for this proposal and M0 first, with no building past M0 until
review. This document is the proposal. `m0/` holds the M0 tools, which are
ready to run once there is footage and a `VLMRUN_API_KEY`. `M0_NOTE.md` is the
note M0 will produce, with its slots still empty.

**Status (2026-11-05):** M1 foundations are built (`swing/`, see README) and
were revised after a code review. M0 tooling is built and tested offline. The
suite is 85 tests, including an end-to-end M0 run against a fake gateway. M0 itself has **not** run: there is no
footage, and this sandbox can't reach `vlm.run`. Nothing below is a measured
result on real squash video.

---

## 1. Reading of the requirements

V1 is a local Python CLI, built on the `running` demo in
[jeremyipark/vision-demos](https://github.com/jeremyipark/vision-demos). It
takes one fixed-camera iPhone slo-mo clip of solo drives and produces:

1. the swings, found automatically (two passes: cheap coarse, then full-rate
   windows only);
2. per-swing 2D body metrics in torso-length units, each with a confidence;
3. rubric verdicts (Meets / Needs work / Can't assess) for forehand and
   backhand drives, each linked to evidence;
4. an annotated video, per-swing clips and an HTML report that leads with at
   most two priorities;
5. history, baselines and comparisons, only across matching setups.

The binding rules I take from the RFP: never invent thresholds (calibrate them
from labelled reps and record each one), never present a proxy as a
measurement, keep metric logic pure and offline-testable, and stop for review
after M0 and after M2.

## 2. Conflicts and gaps found

| # | Issue | Why it matters | Proposed handling |
|---|---|---|---|
| G1 | **Per-keypoint confidence may not exist.** Upstream never reads a score field. It treats `(0,0)` as "not visible" and nothing more. | FR-7, every metric's confidence and "low confidence → Can't assess" all assume scores. | M0 checks the raw response (`keypoint_quality.py` reports `score_field`). If there are none, build confidence from visibility, jitter, limb-length consistency and left/right swap flags, and say so in the report. |
| G2 | **`video_fps` does not subsample.** Upstream's `pose.build_request` docstring says it is the detector cadence, that pose runs on every decoded frame, and that `video_max_frames` controls what is decoded and billed. | FR-4's "pose at 30 fps" cannot be requested from the gateway. A 240 fps clip with `video_fps=30` still bills every decoded frame up to the cap. | Subsample locally with ffmpeg before upload (`swing.ingest` selects real source frames and keeps their timestamps), and always set `video_max_frames` to the exact frame count. M0 confirms the billing. |
| G3 | **Retimed exports.** AirDrop and Messages exports bake the slow-motion ramp into a ~30 fps file, often with normal-speed sections at each end. Timestamps cannot reveal those sections. | FR-1/AC-4: every timing would be wrong by up to 8×, silently. | Require originals (Photos "Export Unmodified Original", or Image Capture). `swing.timing` labels each file `native`, `retimed`, `normal-speed` or `unknown`; analysis refuses `unknown` and accepts `normal-speed` with a warning (a slo-mo export with its metadata stripped lands here, and only `--capture-fps` can fix it). Run AC-4 with a filmed stopwatch. |
| G4 | **The coarse pass costs more than the fine pass** in the RFP's own example: 18,000 vs 16,800 frames. | Cost, and it's billed before any useful frame. | Section 6: move the coarse pass to 15 fps, or replace it with free audio onset detection, pending the M0 price. |
| G5 | **Peak wrist speed may lead contact.** In a wrist-led swing the wrist decelerates as the racket head releases through the ball. | AC-3 (median error ≤ 8 ms) may fail by construction, not by noise. | M0 and M2 measure the offset against hand labels. Ball-impact audio is the stronger contact reference and costs nothing (section 6). |
| G6 | **Not a GitHub fork.** `Adam-Tidball/Squash-Swing-Analysis` is a standalone repo. | Attribution and upstream merges. | Upstream's full history (26 commits, Apache-2.0) is imported as the base and `upstream` points at the original. LICENSE is kept, `NOTICE` added. No upstream file is modified. |
| G7 | **Gateway terms unverified.** This sandbox can't reach `vlm.run`/`docs.vlm.run`. | Pricing, limits and privacy feed the cost estimate and the privacy row. | M0 checks them with a key. One search result says the gateway is free during alpha with per-tier rate limits and gateway pricing "coming soon". That's unverified and may be out of date. |
| G8 | **AC-10 needs a coach.** 40 held-out coach-labelled drives, per check. | Without them no verdict can be shown (by the RFP's own rule). | Ship verdicts hidden behind calibration status. Before AC-10 the report shows measured values only, labelled "uncalibrated". Open question 3. |
| G9 | **Python version.** The RFP says 3.11 (matching upstream). | Only a note: the M0 tools also run on 3.13, which is what they were tested on here. | Keep 3.11 as the supported version via `environment.yml`. |

## 3. Architecture and module layout (M1 onward)

Upstream's `running/` stays untouched, so upstream changes still merge
cleanly. Reused modules are copied into `squash/swing/upstream/`, each with a
"Copied from vision-demos running/src/…" header listing what changed.
Everything that calls a model sits behind an interface, so everything else
tests offline. The tree below marks what exists after M1 (✓); the rest is the
plan for M2 onward.

```
squash/
├── swing/
│   ├── cli.py            ✓ swing probe | analyze; baseline | compare | history (M5)
│   ├── config.py         ✓ all tunables, each with its reason; snapshotted to run.json
│   ├── timing.py         ✓ the clock verdict: native / retimed / normal-speed / unknown  FR-1
│   ├── ingest.py         ✓ sources, real-time renditions, chunks, windows               FR-1
│   ├── checks.py         ✓ input checks before and after pose                           FR-2
│   ├── run.py            ✓ run folders and run.json
│   ├── geometry.py       ✓ IoU and box helpers
│   ├── backends/
│   │   ├── base.py       ✓ PoseBackend protocol, cache, frame budget                     FR-3
│   │   ├── vlmrun.py     ✓ ViTPose+ L via the gateway
│   │   ├── mediapipe_local.py ✓ local stand-in
│   │   ├── fake.py       ✓ synthetic poses for tests
│   │   ├── vision.py       VisionBackend protocol (fixed-choice questions)              FR-16
│   │   └── typesafe.py     System One reads, deadlift pattern
│   ├── track.py            one-player lock (box continuity + court position)            FR-6
│   ├── signal.py           gap fill, smoothing, confidence (from G1's outcome)          FR-7
│   ├── detect.py           coarse candidates, non-swing rejection                       FR-4, FR-10
│   ├── events.py           start / contact / end, manual override file                  FR-8
│   ├── metrics.py          the v1 metric table, pure functions on arrays                FR-9
│   ├── rubric.py           loads rubrics/*.yaml, returns verdict + evidence              FR-15
│   ├── calibrate.py        labels → thresholds, agreement per check                     FR-17, AC-10
│   ├── store.py            SQLite per-swing history, keyed by setup (view, hand, stroke)
│   ├── report/             video overlay, swing clips, contact sheet, report.html
│   └── upstream/         ✓ env.py, pose.py, skeleton.py (trimmed); drawing modules join in M3
├── rubrics/squash_canada_drive_v1.yaml
├── m0/                     the M0 tools in this commit
└── tests/                  synthetic keypoints with known angles/timings; no network   AC-8
```

| Upstream part | Plan |
|---|---|
| `running/src/video.py` | Not copied: `swing.ingest` needs frame-exact seeking, real-time subsampling and per-frame real times, which its converter doesn't do (DECISIONS). |
| `running/src/pose.py` | Reuse request, unwrap, cache and dedupe behind `PoseBackend`. Add one request per window and an explicit `video_max_frames`. |
| `skeleton.py`, `comet.py`, `text.py`, `env.py` | Reuse as is. Trails move from ankles to the hitting wrist. |
| `render.py`, `panel.py` | Adapt. The panel shows wrist speed and the swing's metrics, not cadence. |
| `gait.py` | Replace with swing logic. Keep its smoothing helpers and its rule that x and y go to the same pixel units first. |
| `dance_sync/src/similarity.py` | Review in M5 for curve alignment (FR-9 consistency, FR-13). |
| `deadlift/src/classify.py` | Pattern for `typesafe.py`: constrained labels, probabilities, cache keyed on the image and the question text (`m0/keyframe_probe.py` already caches this way). |

## 4. Gateway findings (what's known before M0)

From upstream code, verified by reading it:

- The endpoint is `https://gateway.vlm.run/v1/openai`, with model
  `usyd-community/vitpose-plus-large` and `extra_body`
  `{method: "pose", video_fps, video_max_frames, precision}`.
- The video goes as one base64 `data:` URL. Upstream converts to H.264 first
  and caps inference height at 1080 px, since ViTPose caps the long edge at
  2048 px internally.
- The response has a flat `items` list tagged with `frame_id`, plus a
  `frames` list that is authoritative for which frames exist.
- Per-frame records carry `track_id`, `bbox_xywh` and `kpts_xy`. No score
  field is read anywhere upstream (G1).
- `video_max_frames` defaults to 900 when unset.
- The vision route is `https://gateway.vlm.run/typesafe` (System One).
  deadlift settled on `google/gemma-4-26b-a4b-it`, and its usage block
  carries a `cost` the SDK drops, so it is read from the raw body.

Unverified, to check in M0: price per frame, request size and duration limits,
rate limits, retention of uploaded footage. A search result (not the page
itself) says the gateway is free in alpha with per-tier rate limits.

**Sending swing windows.** One request per window. A 1.75 s window at 240 fps
is 420 frames, a few MB as 1080p H.264 at CRF 18 (an estimate; M0 records the
real size), which is small next to the full-clip upload and keeps each
request retryable on its own. Batching windows
into one concatenated upload, with a frame map, is the fallback if per-request
overhead or rate limits dominate. M0 measures both the round-trip time and the
limits before choosing.

## 5. Cost estimate per practice session

Frame counts are exact. Price per frame is unknown until M0, so the cost is
given as a formula.

| Session (10 min at 240 fps, 40 swings) | Frames billed |
|---|---|
| Full clip, every frame | 144,000 |
| RFP plan: 30 fps coarse + 40 × 1.75 s fine | 18,000 + 16,800 = 34,800 (−76%) |
| 15 fps coarse + fine (recommended if pose stays the coarse detector) | 9,000 + 16,800 = 25,800 (−82%) |
| Audio-onset candidates + fine only | 0 + 16,800 = 16,800 (−88%) |
| Vision reads: 3 key frames × 40 swings | 120 image reads (separate price) |

Cost per session = frames × price per frame + 120 × price per read. M0 bills
about 2 × 420 = 840 pose frames and 6 vision reads, and records the gateway's
`usage` for each, which turns this formula into dollars.

## 6. Recommended changes to the requirements

1. **Add audio as a swing and contact signal (moves from "later" into M2).**
   iPhone slo-mo originals should keep real-time audio; M0 confirms this with
   `ffprobe`, and the recommendation falls away if they don't. A racket–ball impact is a
   sharp transient, located to well under a millisecond at 48 kHz.
   - It finds candidates for free (G4).
   - It gives a contact reference that peak wrist speed can't (G5).
   - It makes AC-3 measurable without hand labels on every swing.

   Keep the wrist-speed peak as the fallback and the cross-check: a transient
   with no wrist peak is a front-wall or floor bounce, not a swing (FR-10).
2. **FR-4: downsample locally, and 15 fps by default.** The gateway won't
   subsample (G2). A drive's fast phase lasts about 0.3 s, which is still 4–5
   samples at 15 fps, enough to place a 1.75 s window. Confirm on M0 footage.
3. **FR-1: require originals.** Refuse `unknown` timing. Accept `retimed`
   files only with `--capture-fps`, plus a warning that the ramps must be
   trimmed.
4. **FR-7 and the confidence model depend on G1.** If the response has no
   scores, every "confidence" in the RFP becomes a derived quality index, and
   the report should name it that.
5. **AC-3 target.** Keep 8 ms against audio-derived contact. Report the
   wrist-peak offset separately as a measured bias, not as a failure of the
   detector.
6. **Before AC-10 passes, show measurements without verdicts,** labelled
   "uncalibrated". That keeps the rule against unsupported verdicts without
   leaving the report empty.

## 7. M0 plan (ready to run)

Inputs Adam provides: one 240 fps original forehand drive clip and one
backhand, filmed per the capture protocol, plus a `VLMRUN_API_KEY` in the
repo-root `.env`.

```bash
conda env create -f squash/environment.yml && conda activate squash
cd squash
python -m m0.timing_probe ~/clips/fh.MOV                       # native? fps? metadata track?
python -m m0.run_pose_m0 ~/clips/fh.MOV --start 12.4 --duration 1.75 \
    --stroke forehand --hand right --max-billed-frames 500      # ~420 frames billed
python -m m0.keyframe_probe ../running/data/output/<run-id>    # 3 vision reads
# repeat for the backhand clip, then fill in M0_NOTE.md
```

What each M0 question gets answered by:

| M0 question (from the RFP) | Tool and field |
|---|---|
| Keypoint quality during the swing | `m0_quality.json`: visibility, jitter, longest wrist gap, wrist swaps, scores if any |
| Slo-mo timing | `timing_probe`: verdict, measured fps, metadata streams; AC-4 via `--known-seconds/--event-frames` |
| Gateway limits and measured cost | run's `metrics.json` and `usage`, request size, round trip |
| Can a vision model read racket position and face? | `m0_keyframes.json` plus the saved crops, checked by eye against the answers |
| Overlay alignment (AC-7 preview) | the run's `*_pose.mp4` |

After M0, I stop and send `M0_NOTE.md` for review.
