# squash

Turns an iPhone slo-mo clip of solo squash practice into a per-swing
technique report. This project lives in a fork of
[jeremyipark/vision-demos](https://github.com/jeremyipark/vision-demos)
(Apache-2.0) and builds on its [`running`](../running) demo.

**Stage: M0 (feasibility).** Only the M0 tools exist so far. The full CLI
(`swing analyze …`) starts at M1, after the M0 review. See
[PROPOSAL.md](PROPOSAL.md) for the plan and [DECISIONS.md](DECISIONS.md) for
the choices made along the way.

## Set up

```bash
conda env create -f squash/environment.yml
conda activate squash
cp .env.example .env          # at the repo root; paste VLMRUN_API_KEY=
```

ffmpeg must be on the PATH. Footage, poses and run outputs stay under
`running/data/` and `squash/data/`, which are both gitignored.

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

## Test

```bash
cd squash && python -m pytest
```

The tests use synthetic keypoints and generated clips, so they need no
network and no key.
