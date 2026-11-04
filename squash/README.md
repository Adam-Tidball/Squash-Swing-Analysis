# squash

Turns an iPhone slo-mo clip of solo squash practice into a per-swing
technique report. This project lives in a fork of
[jeremyipark/vision-demos](https://github.com/jeremyipark/vision-demos)
(Apache-2.0) and builds on its [`running`](../running) demo.

**Stage: M1 (foundations).** `swing analyze` ingests a clip with real-time
timing, runs the input checks and the coarse pose pass through a cached,
budgeted backend, and writes a run folder. Swing detection is M2. M1 was
started before the M0 review, at Adam's request; M0 itself still needs real
240 fps footage and a key. See [PROPOSAL.md](PROPOSAL.md) for the plan and
[DECISIONS.md](DECISIONS.md) for the choices made along the way.

## Set up

```bash
conda env create -f squash/environment.yml
conda activate squash
cp .env.example .env          # at the repo root; paste VLMRUN_API_KEY=
```

ffmpeg must be on the PATH. Footage, poses and run outputs stay under
`running/data/` and `squash/data/`, which are both gitignored.

Install the `swing` command into that environment (from the repo root):

```bash
pip install -e squash
```

## Analyze a clip (M1)

```bash
swing probe clip.MOV                         # what the clip's clock says

swing analyze clip.MOV --stroke forehand --view side --hand right \
    --backend vlmrun --max-billed-frames 20000
```

- `--backend vlmrun` is ViTPose+ L on the VLM Run Gateway, the RFP's model.
  It bills per frame, so `--max-billed-frames` is required. The run stops
  **before** a request that would exceed it.
- `--backend mediapipe` runs a local stand-in for free (model download below,
  saved to `squash/data/models/`). `--backend fake` is synthetic, for tests.
- `--trim 30` (with optional `--start 60`) analyzes part of a clip.
  `--capture-fps 240` declares a retimed slo-mo export, and `--capture-fps 60`
  normal-speed video with Apple metadata.
- Clips with ambiguous timing are refused with what to do, and clips too
  short for one swing are refused before anything is billed.

Each run writes `squash/data/output/<stamp>_<clip>/`:

| File | Holds |
|---|---|
| `run.json` | capture setup, timing verdict, renditions, backend, billed frames, checks, config snapshot, code version |
| `checks.json` | every input check, with its value and limit |
| `poses.json` | the raw model responses, plus the real time of every posed frame |

Renditions and pose responses are cached under `squash/data/cache/`, keyed
on content, so a re-run bills nothing (AC-5).

## Film

Use the original file. In Photos, choose File → Export → Export Unmodified
Original, or import with Image Capture. AirDrop and Messages exports bake in
the slow-motion ramp, which breaks timing. Film from a tripod at 240 fps, with
the whole body and the full swing in frame. Use side-on where it's safe,
otherwise behind the back glass.

## Run M0

From `squash/`:

```bash
# 1. Is the file's clock real time?
python -m m0.timing_probe ~/clips/fh.MOV

# 2. Pose on one swing window (every frame, billed frames capped)
python -m m0.run_pose_m0 ~/clips/fh.MOV --start 12.4 --duration 1.75 \
    --stroke forehand --hand right --max-billed-frames 500

# 3. Vision reads of the racket on backswing, contact and finish
python -m m0.keyframe_probe ../running/data/output/<run-id> --dry-run   # crops only
python -m m0.keyframe_probe ../running/data/output/<run-id>             # 3 reads
```

`--start` is where the swing window begins, in the clip's own seconds. Scrub
to just before the backswing. Each run writes `m0_summary.md`,
`m0_quality.json`, `m0_timing.json` and `m0_keyframes.json` next to the
overlay video in `running/data/output/<run-id>/`. Copy the findings into
[M0_NOTE.md](M0_NOTE.md).

To try the whole path without a key or cost, add `--fake-gateway` to step 2.
It runs a synthetic stick figure through the real pipeline. Nothing it
produces says anything about real footage.

## Without the gateway, or with outside clips

`m0.local_pose` runs MediaPipe Pose Landmarker locally and writes
gateway-shaped poses, so `keypoint_quality` reads them unchanged.
It is a stand-in, not ViTPose. Download its model (Apache-2.0) first, and keep
it out of git:

```bash
pip install mediapipe
curl -LO https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task
python -m m0.local_pose clip.mov --model pose_landmarker_heavy.task \
    --start 1.12 --duration 3.97 --out data/clip/poses.json
python -m m0.keypoint_quality data/clip/poses.json --fps 60 \
    --width 2064 --height 1166 --hand right --min-score 0.5
```

Clips from YouTube or other edited sources can show how pose handles a
squash arm. They can't support any timing result, because slow motion is
usually edited in. Keep a segment inside one camera shot: a cut reads as the
fastest motion in the clip. Only use footage you have permission to use.
Example: [m0/findings/2026-11-03-youtube-screen-recordings.md](m0/findings/2026-11-03-youtube-screen-recordings.md).

## Test

```bash
cd squash && python -m pytest
```

The tests use synthetic keypoints and generated clips, so they need no
network and no key.
