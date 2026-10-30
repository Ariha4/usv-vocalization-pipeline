"""Tests for streaming USV detection on synthetic recordings.

Each synthetic recording is low-level noise with 25 kHz tone bursts ("calls") placed at known
times, including calls that straddle frame boundaries. The audio is fed to the streaming
detector in small blocks, the way a live input would deliver it.
"""

import numpy as np
import pytest

from usv_pipeline.live import StreamingUSVDetector, run_live

FS = 250_000
PARAMS = {
    "freq_range": [20000, 30000],
    "window_size": 30,
    "on_std_multiplier": 1,
    "off_std_multiplier": 0.5,
    "amp_std_multiplier": 1.5,
    "twentyfive": False,
    "call_duration": [0.008, 0.15],
}

# (start, duration) in seconds. 2.48 and 3.49 straddle the half-frame hops at 2.5 s and 3.5 s;
# 4.75 ends inside the guard zone of the frame that ends at 5.0 s.
CALL_TIMES = [(1.20, 0.05), (2.48, 0.06), (3.49, 0.04), (4.75, 0.05), (6.30, 0.08)]


def make_recording(duration=7.0, calls=CALL_TIMES, seed=0):
    rng = np.random.default_rng(seed)
    audio = rng.normal(0, 0.01, int(duration * FS)).astype(np.float32)
    for start, length in calls:
        first = int(start * FS)
        sample_times = np.arange(int(length * FS)) / FS
        audio[first:first + len(sample_times)] += 0.3 * np.sin(2 * np.pi * 25_000 * sample_times)
    return audio


def make_detector(audio):
    # Calibrate on the first 2 s of session audio (one call included), as in a real session.
    return StreamingUSVDetector.from_calibration(audio[: int(2.0 * FS)], FS, PARAMS)


def stream(detector, audio, block_seconds):
    block_size = int(block_seconds * FS)
    calls = []
    for first in range(0, len(audio), block_size):
        calls.extend(detector.feed(audio[first:first + block_size]))
    calls.extend(detector.flush())
    return calls


def test_detects_every_call_once():
    audio = make_recording()
    calls = stream(make_detector(audio), audio, block_seconds=0.01)

    assert len(calls) == len(CALL_TIMES)
    for (true_start, true_length), (start, stop) in zip(CALL_TIMES, calls):
        assert abs(start - true_start) < 0.01
        assert abs(stop - (true_start + true_length)) < 0.015


def test_block_size_does_not_change_results():
    audio = make_recording()
    small_blocks = stream(make_detector(audio), audio, block_seconds=0.007)
    large_blocks = stream(make_detector(audio), audio, block_seconds=0.25)

    assert small_blocks == large_blocks


def test_calls_are_reported_soon_after_they_end():
    audio = make_recording()
    detector = make_detector(audio)
    block_size = int(0.01 * FS)
    delays = []
    for first in range(0, len(audio), block_size):
        for start, stop in detector.feed(audio[first:first + block_size]):
            delays.append(detector.samples_received / FS - stop)

    assert delays, "expected calls before the end of the stream"
    assert max(delays) <= 1.0     # never more than one frame behind


def test_noise_only_stream_reports_nothing():
    calibration_audio = make_recording()
    quiet_audio = make_recording(calls=[], seed=1)
    assert stream(make_detector(calibration_audio), quiet_audio, block_seconds=0.01) == []


def test_exported_thresholds_skip_calibration():
    audio = make_recording()
    calibrated = make_detector(audio)
    exported = dict(PARAMS, onset_params={"on_thresh": calibrated.on_thresh},
                    offset_params={"off_thresh": calibrated.off_thresh}, amp_thresh=calibrated.amp_thresh)
    block_size = int(0.01 * FS)
    blocks = ((audio[first:first + block_size], FS) for first in range(0, len(audio), block_size))

    calls = run_live(blocks, exported, on_call=lambda message: None)

    assert calls == stream(make_detector(audio), audio, block_seconds=0.01)


def test_run_live_calibrates_then_detects():
    audio = make_recording()
    block_size = int(0.01 * FS)
    blocks = ((audio[first:first + block_size], FS) for first in range(0, len(audio), block_size))

    calls = run_live(blocks, PARAMS, calibration_seconds=2.0, on_call=lambda message: None)

    assert len(calls) == len(CALL_TIMES)


def test_rejects_sample_rate_that_is_not_a_multiple_of_1000():
    with pytest.raises(ValueError):
        StreamingUSVDetector(44_100, (20000, 30000), 1.0, 1.0, 1.0)
