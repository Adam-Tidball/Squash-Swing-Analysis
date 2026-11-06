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
conda env create -f squash/environment.yml     # also installs the swing command
conda activate squash
cp .env.example .env          # at the repo root; paste VLMRUN_API_KEY=
```

Without conda: Python 3.11+, ffmpeg on the PATH, then
`pip install -e "squash[m0,test]"` from the repo root (`[m0]` brings
MediaPipe and the M0 tools' needs; the bare package needs neither).

Footage, poses and run outputs stay under `running/data/` and
`squash/data/`, which are both gitignored. Set `SWING_DATA_DIR` to keep
`swing`'s data somewhere else.

## Analyze a clip (M1)

```bash
swing probe clip.MOV                         # what the clip's clock says

swing analyze clip.MOV --stroke forehand --view side --hand right \
    --backend vlmrun --max-billed-frames 20000
```

- `--backend vlmrun` is ViTPose+ L on the VLM Run Gateway, the RFP's model.
  It bills per frame, so `--max-billed-frames` is required. A run whose
  uncached frames exceed it is refused before anything is sent. A missing
  API key is reported before any work starts.
- `--backend mediapipe` runs a local stand-in for free (model download below;
  default path `squash/data/models/pose_landmarker_heavy.task`, or pass
  `--mediapipe-model`). `--backend fake` is synthetic, for tests.
- `--trim 30` (with optional `--start 60`) analyzes part of a clip.
  `--coarse-fps 15` changes the coarse-pass rate; it never exceeds the clip's
  own rate, and frames are never duplicated.
- `--capture-fps 240` declares a retimed slo-mo export, and `--capture-fps 60`
  normal-speed video with Apple metadata. Clips with ambiguous timing are
  refused with what to do. One case can't be detected: a slo-mo export whose
  Apple metadata was stripped looks like ordinary video, so it's accepted
  with a warning and its timing is wrong until you pass `--capture-fps 240`
  or use the original file.
- A span too short for one swing (the whole clip, or what `--start`/`--trim`
  leave) is refused before anything is billed.
- `--output-dir` and `--cache-dir` move the run folders and the cache.

Exit codes: 0 done; 1 refused (bad input, or no poses); 2 usage error or not
built yet; 3 stopped by the frame budget; 4 failed (no key, no model, or a
gateway or model error). Once a run folder exists, every exit writes its
`run.json`, including failures, with what was already done and billed.

Each run writes `squash/data/output/<stamp>_<clip>/`:

| File | Holds |
|---|---|
| `run.json` | status, the request (every option), capture setup, timing verdict, renditions, backend, billed frames, checks, effective config, code version and whether the tree was dirty |
| `checks.json` | every input check, with its value and limit |
| `poses.json` | the raw model responses, plus the real time of every posed frame |

Renditions and pose responses are cached under `squash/data/cache/`, keyed
on content and on everything that shapes them (including `--capture-fps`),
so a re-run bills nothing (AC-5). A different ffmpeg build encodes slightly
different renditions, which miss the pose cache and are billed again.

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
    --stroke forehand --hand right --view side --max-billed-frames 500

# 3. Vision reads of the racket on backswing, contact and finish
python -m m0.keyframe_probe ../running/data/output/<run-id> --dry-run   # crops only
python -m m0.keyframe_probe ../running/data/output/<run-id>             # 3 reads
```

`--start` is where the swing window begins, in real seconds (the clip's own
seconds for an original). Scrub to just before the backswing. Add
`--min-score 0.5` if the model returns per-keypoint scores: lost joints then
count as unseen, and the key-frame probe uses the same gate. Answered vision
reads are cached in the run folder, so re-running step 3 doesn't pay again. Each run writes `m0_summary.md`,
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
pip install mediapipe        # already there with the [m0] extra
curl -LO https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task
python -m m0.local_pose clip.mov --model pose_landmarker_heavy.task \
    --start 1.12 --duration 3.97 --out data/clip/poses.json
python -m m0.keypoint_quality data/clip/poses.json \
    --width 2064 --height 1166 --hand right --min-score 0.5
```

Clips from YouTube or other edited sources can show how pose handles a
squash arm. They can't support any timing result, because slow motion is
usually edited in. `keypoint_quality` uses the per-frame times `local_pose`
records, so irregular timing in a screen recording is handled. Keep a
segment inside one camera shot: a cut reads as the fastest motion in the
clip. Only use footage you have permission to use.
Example: [m0/findings/2026-11-03-youtube-screen-recordings.md](m0/findings/2026-11-03-youtube-screen-recordings.md).

## Test

```bash
cd squash && python -m pytest
```

The tests use synthetic keypoints and generated clips, so they need no
network and no key. Set `SWING_MEDIAPIPE_MODEL` to a downloaded model file to
include the MediaPipe backend test. One test drives `running/main.py` end to
end with fake poses and cleans up after itself.
