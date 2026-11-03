"""M0 tool tests on synthetic keypoints and generated clips: no network, no key."""

from __future__ import annotations

import shutil
import subprocess

import numpy as np
import pytest

from m0 import keypoint_quality as kq
from m0 import keyframe_probe, timing_probe
from m0.run_pose_m0 import fake_payload


# ── keypoints ────────────────────────────────────────────────────────────────
def standing(n: int) -> np.ndarray:
    """A still figure in pixels: torso exactly 100 px, hips at y=500."""
    xy = np.full((n, 17, 2), np.nan)
    pose = {
        "left_shoulder": (510, 400), "right_shoulder": (490, 400),
        "left_hip": (510, 500), "right_hip": (490, 500),
        "left_elbow": (515, 450), "right_elbow": (485, 450),
        "left_wrist": (520, 490), "right_wrist": (480, 490),
        "left_knee": (515, 600), "right_knee": (485, 600),
        "left_ankle": (515, 700), "right_ankle": (485, 700),
    }
    for name, p in pose.items():
        xy[:, kq.J[name]] = p
    return xy


def test_torso_length_is_hip_mid_to_shoulder_mid():
    assert kq.torso_length(standing(10)) == pytest.approx(100.0)


def test_wrist_speed_in_torso_lengths_per_second():
    fps, n = 240.0, 241
    xy = standing(n)
    # Right wrist moves 2 torso lengths (200 px) in 1 s at constant speed.
    xy[:, kq.J["right_wrist"], 0] = 480 + np.linspace(0, 200, n)
    speed = kq.wrist_speed(xy, fps, "right", kq.torso_length(xy))
    assert np.median(speed) == pytest.approx(2.0, rel=1e-3)


def test_peak_and_window_find_the_fast_frame():
    fps, n = 240.0, 600
    xy = standing(n)
    t = np.arange(n)
    xy[:, kq.J["right_wrist"], 0] = 480 + 150 * np.tanh((t - 400) / 10)
    report = kq.analyze(xy, None, fps=fps, hand="right")
    assert abs(report["peak"]["frame"] - 400) <= 1
    assert report["window"]["start_frame"] == 400 - 240
    assert report["window"]["end_frame"] == 400 + 180


def test_gap_reported_in_ms():
    fps = 240.0
    xy = standing(100)
    xy[:, kq.J["right_wrist"], 0] += np.arange(100)       # something to peak on
    xy[40:52, kq.J["right_wrist"]] = np.nan               # 12 frames = 50 ms
    report = kq.analyze(xy, None, fps=fps, hand="right", before=1, after=1)
    assert report["clip"]["hitting_wrist_longest_gap_ms"] == pytest.approx(50.0)
    assert report["clip"]["visible"]["right_wrist"] == pytest.approx(0.88)


def test_wrist_swap_detected_once():
    xy = standing(20)
    l, r = kq.J["left_wrist"], kq.J["right_wrist"]
    xy[10:, [l, r]] = xy[10:, [r, l]]                     # labels exchange at frame 10
    assert kq.wrist_swaps(xy, 100.0) == [10]


def test_scores_read_when_present_and_absent_otherwise():
    person = {"bbox_xywh": [0, 0, 1, 1], "kpts_xy": [[0.5, 0.5]] * 17,
              "kpts_scores": [0.9] * 17}
    _, scores, field = kq.keypoint_arrays([person], width=100, height=100)
    assert field == "kpts_scores" and scores[0, 0] == pytest.approx(0.9)
    del person["kpts_scores"]
    _, scores, field = kq.keypoint_arrays([person], width=100, height=100)
    assert scores is None and field is None


def test_zero_zero_is_not_visible():
    person = {"bbox_xywh": [0, 0, 1, 1], "kpts_xy": [[0, 0]] + [[0.5, 0.5]] * 16}
    xy, _, _ = kq.keypoint_arrays([person], width=100, height=100)
    assert np.isnan(xy[0, 0]).all() and np.isfinite(xy[0, 1]).all()


def test_aspect_ratio_applied_before_lengths():
    # Normalized coords on a 2:1 frame: a 0.1 horizontal step is 2x a 0.1 vertical one.
    person = {"bbox_xywh": [0, 0, 1, 1], "kpts_xy": [[0.1, 0.0]] + [[0.0, 0.1]] * 16}
    xy, _, _ = kq.keypoint_arrays([person], width=200, height=100)
    assert xy[0, 0, 0] == pytest.approx(20.0) and xy[0, 1, 1] == pytest.approx(10.0)


def test_lock_player_follows_continuity_and_counts_crowding():
    a = {"bbox_xywh": [0.1, 0.1, 0.2, 0.6], "kpts_xy": []}
    big = {"bbox_xywh": [0.6, 0.1, 0.3, 0.8], "kpts_xy": []}
    frames = [{"index": 0, "persons": [a]}, {"index": 1, "persons": [a, big]}]
    chosen, crowded = kq.lock_player(frames)
    assert chosen[1] is a and crowded == 1


def test_fake_payload_round_trips_through_the_reader():
    payload = fake_payload(240, 240.0, "right")
    frames = kq.frames_from_payload(payload)
    persons, crowded = kq.lock_player(frames)
    xy, _, _ = kq.keypoint_arrays(persons, width=1280, height=720)
    report = kq.analyze(xy, None, fps=240.0, hand="right")
    assert crowded == 0 and abs(report["peak"]["frame"] - 120) <= 3


def test_key_frames_order():
    payload = fake_payload(240, 240.0, "right")
    frames = kq.frames_from_payload(payload)
    persons, _ = kq.lock_player(frames)
    xy, _, _ = kq.keypoint_arrays(persons, width=1280, height=720)
    keys = keyframe_probe.key_frames(xy, np.arange(240), fps=240.0, hand="right")
    assert keys["backswing"] < keys["contact"] < keys["finish"]


def test_every_question_can_abstain():
    for spec in keyframe_probe.QUESTIONS.values():
        assert {"cannot_tell", "not_visible"} & set(spec["labels"])


# ── timing ───────────────────────────────────────────────────────────────────
def probe_dict(fps_text: str, tags: dict | None = None, data: bool = False) -> dict:
    streams = [{"codec_type": "video", "codec_name": "hevc", "width": 1920, "height": 1080,
                "r_frame_rate": fps_text, "avg_frame_rate": fps_text, "tags": tags or {}}]
    if data:
        streams.append({"index": 2, "codec_type": "data", "codec_tag_string": "mebx",
                        "tags": {"handler_name": "Core Media Metadata"}})
    return {"streams": streams, "format": {"filename": "x.mov"}}


def test_native_240_is_real_time():
    pts = np.arange(480) / 240.0
    probe = probe_dict("240/1", data=True)
    probe["streams"].append({"codec_type": "audio", "codec_name": "aac",
                             "sample_rate": "48000", "duration": "2.0"})
    r = timing_probe.classify(probe, pts)
    assert r.audio == {"codec": "aac", "sample_rate": 48000, "seconds": 2.0}
    assert not any("clocks disagree" in n for n in r.notes)
    assert r.verdict == "native" and r.real_time_scale == 1.0
    assert r.measured_fps == pytest.approx(240.0) and r.data_streams


def test_retimed_export_needs_declared_capture_rate():
    pts = np.arange(300) / 30.0
    r = timing_probe.classify(probe_dict("30/1", tags={"com.apple.quicktime.make": "Apple"}), pts)
    assert r.verdict == "unknown"
    r = timing_probe.classify(probe_dict("30/1"), pts, capture_fps=240)
    assert r.verdict == "retimed" and r.real_time_scale == pytest.approx(0.125)


def test_known_duration_check_ac4():
    r = timing_probe.classify(probe_dict("240/1"), np.arange(1000) / 240.0)
    ok = timing_probe.known_duration_check(r, known_seconds=2.0, event_frames=480)
    assert ok["passes_ac4"] and ok["relative_error"] == pytest.approx(0.0, abs=1e-6)
    bad = timing_probe.known_duration_check(r, known_seconds=2.0, event_frames=60)
    assert not bad["passes_ac4"]


def test_irregular_intervals_flagged():
    pts = np.concatenate([np.arange(100) / 240.0, 100 / 240.0 + 0.1 + np.arange(100) / 240.0])
    r = timing_probe.classify(probe_dict("240/1"), pts)
    assert r.irregular_intervals == 1


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")
def test_probe_real_generated_clip(tmp_path):
    clip = tmp_path / "c.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
                    "-i", "testsrc=size=320x240:rate=240", "-t", "1",
                    "-pix_fmt", "yuv420p", str(clip)], check=True)
    r = timing_probe.probe(clip)
    assert r.verdict == "native" and r.n_frames == 240
    assert r.media_seconds == pytest.approx(1.0, rel=0.01)
