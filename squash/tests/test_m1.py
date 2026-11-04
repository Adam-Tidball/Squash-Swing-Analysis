"""M1: ingest, input checks, pose interface, cache, budget, CLI. No network, no key."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from swing import backends, checks, cli, config, ingest
from swing.backends.base import CachedPose, FrameBudget, PoseChunk, join
from swing.backends.fake import FakePose
from swing.timing import TimingReport

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")


def make_clip(path: Path, *, fps: int, seconds: float, size: str = "320x240") -> Path:
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
                    "-i", f"testsrc=size={size}:rate={fps}", "-t", str(seconds),
                    "-pix_fmt", "yuv420p", str(path)], check=True)
    return path


@pytest.fixture(scope="session")
def clips(tmp_path_factory):
    d = tmp_path_factory.mktemp("clips")
    return {
        "native": make_clip(d / "native240.mp4", fps=240, seconds=3),
        "export30": make_clip(d / "export30.mp4", fps=30, seconds=4),
        "short": make_clip(d / "short240.mp4", fps=240, seconds=1),
        "plain60": make_clip(d / "plain60.mp4", fps=60, seconds=3),
    }


@pytest.fixture
def cache(tmp_path):
    return tmp_path / "cache"


# ── ingest (FR-1) ────────────────────────────────────────────────────────────
def test_native_source_is_real_time(clips):
    s = ingest.open_source(clips["native"])
    assert s.timing.verdict == "native" and s.scale == 1.0
    assert s.real_fps == pytest.approx(240, rel=1e-3)
    assert s.real_seconds == pytest.approx(3.0, rel=1e-3)


def test_retimed_export_maps_to_real_time(clips):
    s = ingest.open_source(clips["export30"], capture_fps=240)
    assert s.timing.verdict == "retimed"
    assert s.scale == pytest.approx(0.125, rel=1e-4) and s.real_seconds == pytest.approx(0.5, rel=1e-3)
    assert s.real_fps == pytest.approx(240, rel=1e-3)


def test_ambiguous_timing_is_refused(clips, monkeypatch):
    monkeypatch.setattr(ingest, "probe", lambda *a, **k: TimingReport(
        path="x", codec="h264", width=1, height=1, rotation=0, nominal_fps=30,
        average_fps=30, measured_fps=30, n_frames=10, media_seconds=1,
        irregular_intervals=0, verdict="unknown", notes=["ambiguous"]))
    with pytest.raises(ingest.IngestError, match="Export Unmodified Original"):
        ingest.open_source(clips["native"])


def test_declared_normal_speed_resolves_apple_metadata():
    from swing.timing import classify
    probe = {"streams": [{"codec_type": "video", "r_frame_rate": "60/1", "avg_frame_rate": "60/1",
                          "tags": {"com.apple.quicktime.make": "Apple"}}], "format": {}}
    pts = np.arange(120) / 60.0
    assert classify(probe, pts).verdict == "unknown"
    assert classify(probe, pts, capture_fps=60).verdict == "normal-speed"
    assert classify(probe, pts, capture_fps=90).verdict == "unknown"
    assert classify(probe, pts, capture_fps=240).verdict == "retimed"


def test_identity_follows_content_not_name(clips, tmp_path):
    copy = tmp_path / "renamed.mp4"
    shutil.copy(clips["native"], copy)
    assert ingest.source_identity(copy) == ingest.source_identity(clips["native"])
    with copy.open("ab") as f:
        f.write(b"\0")
    assert ingest.source_identity(copy) != ingest.source_identity(clips["native"])


def test_coarse_rendition_samples_in_real_time(clips, cache):
    s = ingest.open_source(clips["native"])
    r = ingest.make_rendition(s, fps=30, cache_dir=cache)
    assert r.n_frames == pytest.approx(90, abs=1)
    assert np.allclose(np.diff(r.real_times), 1 / 30, atol=1e-3)
    assert r.real_times[0] == pytest.approx(0.0, abs=1e-3)


def test_chunks_tile_without_overlap(clips, cache):
    s = ingest.open_source(clips["native"])
    chunks = ingest.coarse_renditions(s, fps=30, chunk_frames=40, cache_dir=cache)
    times = np.concatenate([c.real_times for c in chunks])
    assert len(chunks) == 3 and all(c.n_frames <= 40 for c in chunks)
    assert np.all(np.diff(times) > 0)                       # no duplicates, in order
    assert np.allclose(np.diff(times), 1 / 30, atol=2e-3)   # no gaps at the seams


def test_window_is_full_rate_and_frame_exact(clips, cache):
    s = ingest.open_source(clips["native"])
    w = ingest.window(s, 1.5, cache_dir=cache)
    assert w.n_frames == pytest.approx(420, abs=1)          # 1.75 s at 240 fps
    assert w.real_times[0] == pytest.approx(0.5, abs=1 / 240)
    assert w.real_times[-1] < 2.25


def test_retimed_window_uses_real_seconds(clips, cache):
    s = ingest.open_source(clips["export30"], capture_fps=240)
    r = ingest.make_rendition(s, start=0.1, duration=0.2, cache_dir=cache)
    # 0.2 real s of a 240 fps capture = 48 frames, stored at 30 fps media.
    assert r.n_frames == pytest.approx(48, abs=1)
    assert r.real_times[0] == pytest.approx(0.1, abs=1e-3)
    assert np.allclose(np.diff(r.real_times), 1 / 240, atol=1e-4)


def test_rendition_is_cached(clips, cache):
    s = ingest.open_source(clips["native"])
    a = ingest.make_rendition(s, fps=30, cache_dir=cache)
    mtime = a.path.stat().st_mtime_ns
    b = ingest.make_rendition(s, fps=30, cache_dir=cache)
    assert b.path == a.path and b.path.stat().st_mtime_ns == mtime


# ── pose interface, cache, budget (FR-3, AC-5) ───────────────────────────────
def test_cache_hit_bills_nothing_and_is_identical(clips, cache):
    s = ingest.open_source(clips["native"])
    r = ingest.make_rendition(s, fps=30, cache_dir=cache)
    fake = FakePose(billable=True)
    first = CachedPose(fake, budget=FrameBudget(1000), cache_dir=cache).run([r])
    second = CachedPose(fake, budget=FrameBudget(1000), cache_dir=cache).run([r])
    assert first.billed == r.n_frames and second.billed == 0 and fake.calls == 1
    assert second.all_cached
    assert np.array_equal(first.times, second.times)
    assert json.dumps(first.persons) == json.dumps(second.persons)


def test_budget_stops_before_the_call(clips, cache):
    s = ingest.open_source(clips["native"])
    r = ingest.make_rendition(s, fps=30, cache_dir=cache)
    fake = FakePose(billable=True)
    runner = CachedPose(fake, budget=FrameBudget(50), cache_dir=cache)
    with pytest.raises(backends.BudgetExceeded, match="Nothing was sent"):
        runner.run([r])
    assert fake.calls == 0 and not runner.cache_path(r).exists()


def test_budget_spans_chunks(clips, cache):
    s = ingest.open_source(clips["native"])
    chunks = ingest.coarse_renditions(s, fps=30, chunk_frames=40, cache_dir=cache)
    fake = FakePose(billable=True)
    with pytest.raises(backends.BudgetExceeded):
        CachedPose(fake, budget=FrameBudget(60), cache_dir=cache).run(chunks)
    assert fake.calls == 1          # first 40-frame chunk fit; the second would not


def test_backend_settings_change_the_cache_key(clips, cache):
    s = ingest.open_source(clips["native"])
    r = ingest.make_rendition(s, fps=30, cache_dir=cache)
    a = CachedPose(FakePose(seed=0), cache_dir=cache).cache_path(r)
    b = CachedPose(FakePose(seed=1), cache_dir=cache).cache_path(r)
    assert a != b


def test_join_puts_chunks_on_one_time_axis(clips, cache):
    s = ingest.open_source(clips["native"])
    chunks = ingest.coarse_renditions(s, fps=30, chunk_frames=40, cache_dir=cache)
    track = CachedPose(FakePose(), cache_dir=cache).run(chunks)
    assert len(track.times) == sum(c.n_frames for c in chunks)
    assert np.all(np.diff(track.times) > 0)


def test_join_rejects_frames_the_file_does_not_have(clips, cache):
    s = ingest.open_source(clips["native"])
    r = ingest.make_rendition(s, fps=30, cache_dir=cache)
    payload, _ = FakePose().estimate(r.path, n_frames=r.n_frames)
    payload["content"]["frames"].append({"frame_id": 10_000})
    with pytest.raises(RuntimeError, match="disagree"):
        join([PoseChunk(r, payload, None, 0, False, 0.0)])


def test_vlmrun_request_sets_exact_frame_count(clips, cache, monkeypatch):
    from swing.backends import vlmrun
    s = ingest.open_source(clips["native"])
    r = ingest.make_rendition(s, fps=30, cache_dir=cache)
    seen = {}

    def fake_request(client, *, model, video_b64, extra_body):
        seen.update(extra_body)
        return FakePose().estimate(r.path, n_frames=r.n_frames)

    monkeypatch.setattr(vlmrun.upstream_pose, "request_poses", fake_request)
    backend = vlmrun.VlmRunPose()
    backend._client = object()                  # never used: request is faked
    backend.estimate(r.path, n_frames=r.n_frames)
    assert seen["video_max_frames"] == r.n_frames
    assert seen["video_fps"] == pytest.approx(30, rel=1e-3)
    assert seen["method"] == "pose"


@pytest.mark.skipif(not Path(os.environ.get("SWING_MEDIAPIPE_MODEL", config.MEDIAPIPE_MODEL)).is_file(),
                    reason="MediaPipe model not downloaded")
def test_mediapipe_backend_returns_gateway_shape(clips, cache):
    pytest.importorskip("mediapipe")
    model = Path(os.environ.get("SWING_MEDIAPIPE_MODEL", config.MEDIAPIPE_MODEL))
    s = ingest.open_source(clips["native"])
    r = ingest.make_rendition(s, fps=30, duration=1.0, cache_dir=cache)
    track = CachedPose(backends.make("mediapipe", model=model), cache_dir=cache).run([r])
    assert len(track.times) == r.n_frames      # every frame accounted for, posed or empty


# ── input checks (FR-2) ──────────────────────────────────────────────────────
CAP = ingest.Capture("forehand", "side", "right")


def test_before_pose_flags_low_rate_and_short_clips(clips):
    low = {c.id: c for c in checks.before_pose(ingest.open_source(clips["plain60"]), CAP)}
    assert low["frame_rate"].level == "warn" and low["timing"].level == "warn"
    short = {c.id: c for c in checks.before_pose(ingest.open_source(clips["short"]), CAP)}
    assert short["duration"].level == "error"
    assert checks.worst(list(short.values())) == "error"


def test_after_pose_flags_second_person(clips, cache):
    s = ingest.open_source(clips["native"])
    r = ingest.make_rendition(s, fps=30, cache_dir=cache)
    track = CachedPose(FakePose(second_person=True), cache_dir=cache).run([r])
    found = {c.id: c for c in checks.after_pose(track)}
    assert found["multiple_people"].level == "warn"
    assert found["nobody"].level == "ok" and found["clipped_body"].level == "ok"


def test_after_pose_flags_cropped_player_and_empty_frames():
    def person(box):
        return {"bbox_xywh": box, "kpts_xy": [[0.5, 0.5]] * 17}
    track = backends.PoseTrack(
        times=np.arange(10) / 30.0,
        persons=[[person([0.0, 0.1, 0.3, 0.8])]] * 5 + [[]] * 5, width=320, height=240)
    found = {c.id: c for c in checks.after_pose(track)}
    assert found["clipped_body"].level == "warn" and found["clipped_body"].value == 1.0
    assert found["nobody"].level == "warn" and found["nobody"].value == 0.5


def test_mismatched_setups_are_never_compared():
    other = ingest.Capture("forehand", "back", "right")
    assert checks.compatible(CAP, other).level == "error"
    assert checks.compatible(CAP, ingest.Capture("forehand", "side", "right", 240)).level == "ok"


# ── CLI ──────────────────────────────────────────────────────────────────────
def analyze(clip, tmp_path, *extra):
    return cli.main(["analyze", str(clip), "--stroke", "forehand", "--view", "side",
                     "--hand", "right", "--output-dir", str(tmp_path / "out"),
                     "--cache-dir", str(tmp_path / "cache"), *extra])


def latest_run(tmp_path) -> dict:
    run = max((tmp_path / "out").iterdir(), key=lambda p: p.stat().st_mtime_ns)
    return json.loads((run / "run.json").read_text())


def test_cli_analyze_writes_a_complete_run(clips, tmp_path):
    assert analyze(clips["native"], tmp_path, "--backend", "fake") == 0
    run = latest_run(tmp_path)
    assert run["status"] == "pose_done"
    assert run["capture"] == {"stroke": "forehand", "view": "side", "hand": "right",
                              "capture_fps": None}
    assert run["source"]["timing"]["verdict"] == "native"
    assert run["pose"]["frames_posed"] == pytest.approx(90, abs=1)
    assert run["config"]["COARSE_FPS"] == config.COARSE_FPS
    assert {c["id"] for c in run["checks"]} >= {"timing", "frame_rate", "multiple_people"}
    run_dir = tmp_path / "out" / run["run_id"]
    assert (run_dir / "poses.json").is_file() and (run_dir / "checks.json").is_file()


def test_cli_rerun_comes_from_cache(clips, tmp_path):
    analyze(clips["native"], tmp_path, "--backend", "fake")
    analyze(clips["native"], tmp_path, "--backend", "fake")
    assert latest_run(tmp_path)["pose"]["all_cached"] is True


def test_cli_gateway_needs_a_budget(clips, tmp_path, capsys):
    assert analyze(clips["native"], tmp_path, "--backend", "vlmrun") == 1
    assert "--max-billed-frames" in capsys.readouterr().err


def test_cli_budget_stops_gateway_before_any_request(clips, tmp_path, monkeypatch):
    monkeypatch.delenv("VLMRUN_API_KEY", raising=False)   # proves nothing reached the client
    code = analyze(clips["native"], tmp_path, "--backend", "vlmrun", "--max-billed-frames", "10")
    assert code == 3
    run = latest_run(tmp_path)
    assert run["status"] == "budget_exceeded" and run["billed_before_stop"] == 0


def test_cli_refuses_short_clip_before_pose(clips, tmp_path):
    assert analyze(clips["short"], tmp_path, "--backend", "fake") == 1
    assert latest_run(tmp_path)["status"] == "refused"


def test_cli_trim_limits_the_pass(clips, tmp_path):
    analyze(clips["native"], tmp_path, "--backend", "fake", "--start", "1", "--trim", "1")
    assert latest_run(tmp_path)["pose"]["frames_posed"] == pytest.approx(30, abs=1)


def test_later_commands_say_when_they_arrive(capsys):
    assert cli.main(["history"]) == 2
    assert "M5" in capsys.readouterr().err
