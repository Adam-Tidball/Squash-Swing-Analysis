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


def make_clip(path: Path, *, fps: int, seconds: float | None = None, frames: int | None = None,
              size: str = "320x240") -> Path:
    length = ["-frames:v", str(frames)] if frames else ["-t", str(seconds)]
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
                    "-i", f"testsrc=size={size}:rate={fps}", *length,
                    "-pix_fmt", "yuv420p", str(path)], check=True)
    return path


def make_id_clip(path: Path, *, fps: int, frames: int) -> Path:
    """Each frame's brightness encodes its own index (16 + 4·(N mod 50)).

    Steps of 4 levels survive the renditions' lossy re-encode."""
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
                    "-i", f"color=c=black:s=64x64:r={fps}", "-frames:v", str(frames),
                    "-vf", "format=gray,geq=lum='16+4*mod(N\\,50)'",
                    "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv420p", str(path)], check=True)
    return path


def frame_ids(path: Path) -> list[int]:
    """The source index (mod 50) of every frame in an encoded id clip."""
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(path),
                          "-f", "rawvideo", "-pix_fmt", "gray", "-"],
                         capture_output=True, check=True).stdout
    a = np.frombuffer(raw, np.uint8).astype(float).reshape(-1, 64 * 64).mean(axis=1)
    return [int(round((m - 16) / 4)) % 50 for m in a]


@pytest.fixture(scope="session")
def clips(tmp_path_factory):
    d = tmp_path_factory.mktemp("clips")
    return {
        "native": make_clip(d / "native240.mp4", fps=240, seconds=3),
        "export30": make_clip(d / "export30.mp4", fps=30, seconds=4),
        "short": make_clip(d / "short240.mp4", fps=240, seconds=1),
        "plain60": make_clip(d / "plain60.mp4", fps=60, seconds=3),
        "plain25": make_clip(d / "plain25.mp4", fps=25, seconds=3),
        "odd721": make_clip(d / "odd721.mp4", fps=240, frames=721),
        "ids": make_id_clip(d / "ids240.mp4", fps=240, frames=480),
        "notavideo": (d / "notavideo.mov"),
    }


@pytest.fixture(scope="session", autouse=True)
def _not_a_video(clips):
    clips["notavideo"].write_bytes(b"this is not a video")


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
    assert s.scale == pytest.approx(0.125, rel=1e-4)
    assert s.real_seconds == pytest.approx(0.5, rel=1e-3)
    assert s.real_fps == pytest.approx(240, rel=1e-3)


def test_ambiguous_timing_is_refused(clips, monkeypatch):
    report = TimingReport(
        path="x", codec="h264", width=1, height=1, rotation=0, nominal_fps=30,
        average_fps=30, measured_fps=30, n_frames=10, media_seconds=1,
        irregular_intervals=0, verdict="unknown", notes=["ambiguous"])
    monkeypatch.setattr(ingest, "probe_with_times",
                        lambda *a, **k: (report, np.arange(10) / 30))
    with pytest.raises(ingest.IngestError, match="Export Unmodified Original"):
        ingest.open_source(clips["native"])


def test_non_video_is_refused_with_a_reason(clips):
    with pytest.raises(ingest.IngestError, match="ffprobe cannot read"):
        ingest.open_source(clips["notavideo"])


def test_declared_normal_speed_resolves_apple_metadata():
    from swing.timing import classify
    probe = {"streams": [{"codec_type": "video", "r_frame_rate": "60/1",
                          "avg_frame_rate": "60/1",
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


def test_media_seconds_keep_sub_frame_precision(clips):
    s = ingest.open_source(clips["odd721"])
    assert s.timing.media_seconds == pytest.approx(721 / 240, abs=1e-5)


# Which source frame each rendition frame really is (M2, M4, L1).
def test_coarse_frames_are_the_source_frames_they_are_labelled(clips, cache):
    s = ingest.open_source(clips["ids"])
    r = ingest.make_rendition(s, fps=30, duration=1.0, height=None, cache_dir=cache)
    ids = frame_ids(r.path)
    assert r.n_frames == 30
    assert ids == [(8 * k) % 50 for k in range(30)]
    assert np.allclose(r.real_times, np.arange(30) * 8 / 240, atol=1e-6)


def test_non_dividing_rate_keeps_true_frame_times(clips, cache):
    s = ingest.open_source(clips["ids"])
    r = ingest.make_rendition(s, fps=25, duration=1.0, height=None, cache_dir=cache)
    expected = [int(np.ceil(k * 240 / 25 - 1e-9)) for k in range(25)]   # first frame per slot
    assert frame_ids(r.path) == [e % 50 for e in expected]
    assert np.allclose(r.real_times, np.array(expected) / 240, atol=1e-6)


def test_rate_above_source_never_duplicates(clips, cache):
    s = ingest.open_source(clips["ids"])
    r = ingest.make_rendition(s, start=0.5, duration=0.1, fps=480, height=None,
                              cache_dir=cache)
    assert frame_ids(r.path) == [(120 + k) % 50 for k in range(24)]
    assert len(set(np.round(r.real_times, 6))) == r.n_frames


def test_seek_is_snapped_to_a_real_frame(clips, cache):
    s = ingest.open_source(clips["ids"])
    r = ingest.make_rendition(s, start=0.1234, duration=0.05, height=None, cache_dir=cache)
    assert frame_ids(r.path)[0] == 30                    # first frame at/after 0.1234 s
    assert r.real_times[0] == pytest.approx(30 / 240, abs=1e-6)
    assert r.start == pytest.approx(30 / 240, abs=1e-6)


def test_chunk_seams_hold_exactly_the_expected_frames(clips, cache):
    s = ingest.open_source(clips["ids"])
    chunks = ingest.coarse_renditions(s, fps=30, chunk_frames=7, cache_dir=cache)
    ids = [i for c in chunks for i in frame_ids(c.path)]
    assert ids == [(8 * k) % 50 for k in range(60)]
    times = np.concatenate([c.real_times for c in chunks])
    assert np.allclose(times, np.arange(60) * 8 / 240, atol=1e-6)


def test_chunks_tile_without_overlap(clips, cache):
    s = ingest.open_source(clips["native"])
    chunks = ingest.coarse_renditions(s, fps=30, chunk_frames=40, cache_dir=cache)
    times = np.concatenate([c.real_times for c in chunks])
    assert len(chunks) == 3 and [c.n_frames for c in chunks] == [40, 40, 10]
    assert np.all(np.diff(times) > 0)
    assert np.allclose(np.diff(times), 1 / 30, atol=2e-3)


def test_sub_frame_tail_does_not_crash(clips, cache):
    # M1: 721 frames at 240 fps = 3.0042 s; 1 s chunks used to leave a 4 ms tail.
    s = ingest.open_source(clips["odd721"])
    chunks = ingest.coarse_renditions(s, fps=30, chunk_frames=30, cache_dir=cache)
    assert [c.n_frames for c in chunks] == [30, 30, 30]
    trimmed = ingest.coarse_renditions(s, fps=30, chunk_frames=30, duration=1.02,
                                       cache_dir=cache)
    assert [c.n_frames for c in trimmed] == [30]


def test_empty_span_is_refused(clips, cache):
    s = ingest.open_source(clips["native"])
    with pytest.raises(ingest.IngestError, match="nothing to analyze"):
        ingest.coarse_renditions(s, start=5.0, cache_dir=cache)
    with pytest.raises(ingest.IngestError, match="invalid"):
        ingest.coarse_renditions(s, fps=0, cache_dir=cache)


def test_window_is_full_rate_and_frame_exact(clips, cache):
    s = ingest.open_source(clips["native"])
    w = ingest.window(s, 1.5, cache_dir=cache)
    assert w.n_frames == pytest.approx(420, abs=1)          # 1.75 s at 240 fps
    assert w.real_times[0] == pytest.approx(0.5, abs=1 / 240)
    assert w.real_times[-1] < 2.25


def test_window_outside_the_clip_is_refused(clips, cache):
    s = ingest.open_source(clips["native"])
    with pytest.raises(ingest.IngestError, match="outside the clip"):
        ingest.window(s, 5.0, cache_dir=cache)


def test_retimed_window_uses_real_seconds(clips, cache):
    s = ingest.open_source(clips["export30"], capture_fps=240)
    r = ingest.make_rendition(s, start=0.1, duration=0.2, cache_dir=cache)
    assert r.n_frames == pytest.approx(48, abs=1)
    assert r.real_times[0] == pytest.approx(0.1, abs=1e-3)
    assert np.allclose(np.diff(r.real_times), 1 / 240, atol=1e-4)


def test_changing_capture_fps_never_reuses_a_rendition(clips, cache):
    # H1: same file, same real-time request, different declared capture rate.
    at240 = ingest.make_rendition(ingest.open_source(clips["export30"], capture_fps=240),
                                  fps=None, cache_dir=cache)
    at120 = ingest.make_rendition(ingest.open_source(clips["export30"], capture_fps=120),
                                  fps=None, cache_dir=cache)
    assert at240.path != at120.path
    assert np.allclose(np.diff(at240.real_times), 1 / 240, atol=1e-4)
    assert np.allclose(np.diff(at120.real_times), 1 / 120, atol=1e-4)


def test_rendition_is_cached(clips, cache):
    s = ingest.open_source(clips["native"])
    a = ingest.make_rendition(s, fps=30, cache_dir=cache)
    mtime = a.path.stat().st_mtime_ns
    b = ingest.make_rendition(s, fps=30, cache_dir=cache)
    assert b.path == a.path and b.path.stat().st_mtime_ns == mtime


def test_portrait_rotation_is_applied(tmp_path, cache):
    clip = tmp_path / "rot.mov"
    src = make_clip(tmp_path / "land.mp4", fps=240, seconds=2)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-display_rotation", "90",
                    "-i", str(src), "-c", "copy", str(clip)], check=True)
    r = ingest.make_rendition(ingest.open_source(clip), fps=30, cache_dir=cache)
    assert (r.width, r.height) == (240, 320)


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


def test_budget_refuses_the_whole_run_before_any_request(clips, cache):
    s = ingest.open_source(clips["native"])
    chunks = ingest.coarse_renditions(s, fps=30, chunk_frames=40, cache_dir=cache)
    fake = FakePose(billable=True)
    runner = CachedPose(fake, budget=FrameBudget(60), cache_dir=cache)
    with pytest.raises(backends.BudgetExceeded, match="Nothing was sent"):
        runner.run(chunks)                     # 90 frames needed, 60 allowed
    assert fake.calls == 0 and not any(runner.cache_path(c).exists() for c in chunks)


def test_cached_chunks_do_not_count_against_the_budget(clips, cache):
    s = ingest.open_source(clips["native"])
    chunks = ingest.coarse_renditions(s, fps=30, chunk_frames=40, cache_dir=cache)
    fake = FakePose(billable=True)
    CachedPose(fake, budget=FrameBudget(40), cache_dir=cache).run(chunks[:1])
    track = CachedPose(fake, budget=FrameBudget(50), cache_dir=cache).run(chunks)
    assert track.billed == 50 and fake.calls == 3      # 40 cached, 40 + 10 new


def test_per_request_charge_says_what_was_spent():
    budget = FrameBudget(50)
    budget.charge(40)
    with pytest.raises(backends.BudgetExceeded, match="40 frames were already billed"):
        budget.charge(20)


def test_backend_settings_change_the_cache_key(clips, cache):
    s = ingest.open_source(clips["native"])
    r = ingest.make_rendition(s, fps=30, cache_dir=cache)
    a = CachedPose(FakePose(seed=0), cache_dir=cache).cache_path(r)
    b = CachedPose(FakePose(seed=1), cache_dir=cache).cache_path(r)
    assert a != b


def test_mediapipe_cache_key_follows_model_content(tmp_path):
    from swing.backends.mediapipe_local import MediaPipePose
    a, b = tmp_path / "a" / "m.task", tmp_path / "b" / "m.task"
    for path, data in ((a, b"lite"), (b, b"heavy")):
        path.parent.mkdir()
        path.write_bytes(data)
    assert MediaPipePose(a).describe() != MediaPipePose(b).describe()


def test_join_puts_chunks_on_one_time_axis(clips, cache):
    s = ingest.open_source(clips["native"])
    chunks = ingest.coarse_renditions(s, fps=30, chunk_frames=40, cache_dir=cache)
    track = CachedPose(FakePose(), cache_dir=cache).run(chunks)
    assert len(track.times) == sum(c.n_frames for c in chunks)
    assert np.all(np.diff(track.times) > 0)


def test_fake_swings_land_at_real_source_times(clips, cache):
    from swing.upstream.skeleton import INDEX
    s = ingest.open_source(clips["native"])
    chunks = ingest.coarse_renditions(s, fps=30, chunk_frames=40, cache_dir=cache)
    track = CachedPose(FakePose(swings=(2.0,), noise=0), cache_dir=cache).run(chunks)
    wrist_y = np.array([ps[0]["kpts_xy"][INDEX["right_wrist"]][1] for ps in track.persons])
    lowest = track.times[wrist_y >= wrist_y.max() - 1e-6]   # contact: arm straight down
    assert np.all(np.abs(lowest - 2.0) < 0.05)               # once, at 2.0 s, in chunk 2


def test_join_rejects_frames_the_file_does_not_have(clips, cache):
    s = ingest.open_source(clips["native"])
    r = ingest.make_rendition(s, fps=30, cache_dir=cache)
    payload, _ = FakePose().estimate(r)
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
        return FakePose().estimate(r)

    monkeypatch.setattr(vlmrun.upstream_pose, "request_poses", fake_request)
    backend = vlmrun.VlmRunPose()
    backend._client = object()                  # never used: request is faked
    backend.estimate(r)
    assert seen["video_max_frames"] == r.n_frames
    assert seen["video_fps"] >= 30 - 1e-3        # detector stride 1
    assert seen["method"] == "pose"


@pytest.mark.skipif(not Path(os.environ.get("SWING_MEDIAPIPE_MODEL",
                                            config.MEDIAPIPE_MODEL)).is_file(),
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


def test_duration_check_uses_the_analyzed_span(clips):
    found = {c.id: c for c in checks.before_pose(ingest.open_source(clips["native"]), CAP,
                                                 start=1.0, duration=0.2)}
    assert found["duration"].level == "error" and found["duration"].value == 0.2


def test_coarse_rate_above_source_is_reported(clips):
    found = {c.id: c for c in checks.before_pose(ingest.open_source(clips["plain25"]), CAP,
                                                 coarse_fps=30)}
    assert found["coarse_rate"].level == "info"


def test_after_pose_flags_second_person(clips, cache):
    s = ingest.open_source(clips["native"])
    r = ingest.make_rendition(s, fps=30, cache_dir=cache)
    track = CachedPose(FakePose(second_person=True), cache_dir=cache).run([r])
    found = {c.id: c for c in checks.after_pose(track)}
    assert found["multiple_people"].level == "warn"
    assert found["nobody"].level == "ok" and found["clipped_body"].level == "ok"


def _track(persons):
    return backends.PoseTrack(times=np.arange(len(persons)) / 30.0, persons=persons,
                              width=320, height=240)


def test_after_pose_flags_cropped_player_and_empty_frames():
    def person(box):
        return {"bbox_xywh": box, "kpts_xy": [[0.5, 0.5]] * 17}
    found = {c.id: c for c in checks.after_pose(
        _track([[person([0.0, 0.1, 0.3, 0.8])]] * 5 + [[]] * 5))}
    assert found["clipped_body"].level == "warn" and found["clipped_body"].value == 1.0
    assert found["nobody"].level == "warn" and found["nobody"].value == 0.5


def test_after_pose_tolerates_people_without_boxes():
    found = {c.id: c for c in checks.after_pose(_track([[{"kpts_xy": []}]] * 4))}
    assert found["nobody"].value == 1.0


def test_mismatched_setups_are_never_compared():
    other = ingest.Capture("forehand", "back", "right")
    assert checks.compatible(CAP, other).level == "error"
    assert checks.compatible(CAP, ingest.Capture("forehand", "side", "right", 240)).level == "ok"


# ── CLI ──────────────────────────────────────────────────────────────────────
def analyze(clip, tmp_path, *extra):
    return cli.main(["analyze", str(clip), "--stroke", "forehand", "--view", "side",
                     "--hand", "right", "--output-dir", str(tmp_path / "out"),
                     "--cache-dir", str(tmp_path / "cache"), *extra])


def runs(tmp_path) -> list[Path]:
    out = tmp_path / "out"
    return sorted(out.iterdir(), key=lambda p: p.stat().st_mtime_ns) if out.exists() else []


def latest_run(tmp_path) -> dict:
    return json.loads((runs(tmp_path)[-1] / "run.json").read_text())


def test_cli_analyze_writes_a_complete_run(clips, tmp_path):
    assert analyze(clips["native"], tmp_path, "--backend", "fake") == 0
    run = latest_run(tmp_path)
    assert run["status"] == "pose_done"
    assert run["capture"] == {"stroke": "forehand", "view": "side", "hand": "right",
                              "capture_fps": None}
    assert run["source"]["timing"]["verdict"] == "native"
    assert run["pose"]["frames_posed"] == pytest.approx(90, abs=1)
    assert {c["id"] for c in run["checks"]} >= {"timing", "frame_rate", "multiple_people"}
    assert set(run["code_version"]) == {"commit", "dirty"}
    run_dir = runs(tmp_path)[-1]
    assert (run_dir / "poses.json").is_file() and (run_dir / "checks.json").is_file()


def test_cli_records_what_the_run_used(clips, tmp_path):
    analyze(clips["native"], tmp_path, "--backend", "fake", "--coarse-fps", "15",
            "--start", "0.5", "--trim", "2")
    run = latest_run(tmp_path)
    assert run["request"]["coarse_fps"] == 15 and run["request"]["start"] == 0.5
    assert run["request"]["trim"] == 2 and run["request"]["backend"] == "fake"
    assert run["config"]["COARSE_FPS"] == 15


def test_cli_rerun_comes_from_cache(clips, tmp_path):
    analyze(clips["native"], tmp_path, "--backend", "fake")
    analyze(clips["native"], tmp_path, "--backend", "fake")
    assert latest_run(tmp_path)["pose"]["all_cached"] is True


def test_cli_gateway_needs_a_budget(clips, tmp_path, capsys):
    assert analyze(clips["native"], tmp_path, "--backend", "vlmrun") == cli.REFUSED
    assert "--max-billed-frames" in capsys.readouterr().err
    assert runs(tmp_path) == []


def test_cli_budget_stops_gateway_before_any_request(clips, tmp_path, monkeypatch):
    from swing.backends import vlmrun
    monkeypatch.setattr(vlmrun.VlmRunPose, "check_ready", lambda self: None)
    monkeypatch.setattr(vlmrun.VlmRunPose, "estimate",
                        lambda self, r: pytest.fail("a request was sent"))
    code = analyze(clips["native"], tmp_path, "--backend", "vlmrun", "--max-billed-frames", "10")
    assert code == cli.BUDGET
    run = latest_run(tmp_path)
    assert run["status"] == "budget_exceeded" and run["billed_before_stop"] == 0


def test_cli_missing_key_fails_before_any_run_folder(clips, tmp_path, monkeypatch, capsys):
    from swing.backends import vlmrun

    def no_key(_):
        raise RuntimeError("no key")
    monkeypatch.setattr(vlmrun, "load_api_key", no_key)
    code = analyze(clips["native"], tmp_path, "--backend", "vlmrun", "--max-billed-frames", "999")
    assert code == cli.FAILED and "VLMRUN_API_KEY" in capsys.readouterr().err
    assert runs(tmp_path) == []


def test_cli_missing_model_fails_before_any_run_folder(clips, tmp_path):
    code = analyze(clips["native"], tmp_path, "--backend", "mediapipe",
                   "--mediapipe-model", str(tmp_path / "missing.task"))
    assert code == cli.FAILED and runs(tmp_path) == []


def test_cli_backend_error_mid_run_is_recorded(clips, tmp_path, monkeypatch):
    calls = {"n": 0}
    real = FakePose.estimate

    def flaky(self, rendition):
        calls["n"] += 1
        if calls["n"] == 2:
            raise ConnectionError("gateway went away")
        return real(self, rendition)

    monkeypatch.setattr(FakePose, "estimate", flaky)
    monkeypatch.setattr(config, "CHUNK_FRAMES", 40)
    monkeypatch.setattr(cli.cfg, "CHUNK_FRAMES", 40)
    monkeypatch.setattr(cli, "coarse_renditions",
                        lambda *a, **k: ingest.coarse_renditions(*a, chunk_frames=40, **k))
    assert analyze(clips["native"], tmp_path, "--backend", "fake") == cli.FAILED
    run = latest_run(tmp_path)
    assert run["status"] == "failed" and "gateway went away" in run["error"]
    assert len(run["chunks_done"]) == 1


def test_cli_no_poses_is_not_success(clips, tmp_path, monkeypatch):
    def empty(self, rendition):
        return ({"content": {"object": "vid.pose.kpts", "frames": [], "items": []}}, None)
    monkeypatch.setattr(FakePose, "estimate", empty)
    assert analyze(clips["native"], tmp_path, "--backend", "fake") == cli.REFUSED
    assert latest_run(tmp_path)["status"] == "no_poses"


@pytest.mark.parametrize("extra", [["--start", "10"], ["--start", "-1"], ["--trim", "0"],
                                   ["--trim", "-1"], ["--coarse-fps", "0"]])
def test_cli_invalid_spans_are_refused_up_front(clips, tmp_path, extra):
    assert analyze(clips["native"], tmp_path, "--backend", "fake", *extra) == cli.REFUSED
    assert runs(tmp_path) == []


def test_cli_refuses_short_clip_before_pose(clips, tmp_path):
    assert analyze(clips["short"], tmp_path, "--backend", "fake") == cli.REFUSED
    assert latest_run(tmp_path)["status"] == "refused"


def test_cli_refuses_a_short_trimmed_span(clips, tmp_path):
    assert analyze(clips["native"], tmp_path, "--backend", "fake",
                   "--start", "1", "--trim", "0.2") == cli.REFUSED
    assert latest_run(tmp_path)["status"] == "refused"


def test_cli_trim_limits_the_pass(clips, tmp_path):
    analyze(clips["native"], tmp_path, "--backend", "fake", "--start", "1", "--trim", "1.8")
    assert latest_run(tmp_path)["pose"]["frames_posed"] == pytest.approx(54, abs=1)


def test_later_commands_say_when_they_arrive(capsys):
    assert cli.main(["history"]) == cli.USAGE
    assert "M5" in capsys.readouterr().err
