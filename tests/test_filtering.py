"""Tests for butterworth_hilbert_filter and get_env using synthetic signals."""

import numpy as np

from usv_pipeline.usv_library import butterworth_hilbert_filter, get_env


def make_tone(freq, fs, duration, amplitude=1.0):
    t = np.arange(0, duration, 1 / fs)
    return amplitude * np.sin(2 * np.pi * freq * t)


def test_bandpass_passes_in_band_tone():
    fs = 200_000
    tone = make_tone(freq=25_000, fs=fs, duration=0.05)
    filtered, envelope = butterworth_hilbert_filter(tone, fs, (20_000, 30_000))

    assert filtered.shape == tone.shape
    assert envelope.shape == tone.shape
    # An in-band tone should retain most of its amplitude.
    assert envelope.max() > 0.5 * np.abs(tone).max()


def test_bandpass_attenuates_out_of_band_tone():
    fs = 200_000
    tone = make_tone(freq=5_000, fs=fs, duration=0.05)
    _, envelope = butterworth_hilbert_filter(tone, fs, (20_000, 30_000))

    # A tone well outside the passband should be strongly attenuated.
    assert envelope.max() < 0.3 * np.abs(tone).max()


def test_get_env_returns_expected_shapes():
    fs = 200_000
    tone = make_tone(freq=25_000, fs=fs, duration=0.1)

    filtered, envelope, time, downsampled_env, env_derivative = get_env(tone, fs, (20_000, 30_000))

    assert filtered.shape == tone.shape
    assert envelope.shape == tone.shape
    assert time.shape == tone.shape
    # Downsampled to ~1kHz from fs.
    expected_len = len(tone) // (fs // 1000)
    assert abs(len(downsampled_env) - expected_len) <= 1
    assert len(env_derivative) == len(downsampled_env)
