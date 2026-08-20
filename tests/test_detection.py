"""Tests for onset/offset detection, pairing, and nearest-neighbor logic
using small, hand-constructed synthetic envelopes (1 sample = 1ms)."""

import numpy as np

from usv_pipeline.usv_library import (
    get_onsets,
    get_offsets,
    verify_usv,
    get_nearest_neighbors,
)


def test_get_onsets_detects_rising_edge():
    env_derivative = np.zeros(50)
    env_derivative[20] = 2.0  # single spike above threshold

    onsets, onset_times = get_onsets(env_derivative, start_time=0, window_size=5, on_thresh=0.5)

    assert sum(onsets) == 1
    assert onsets[20] == 1
    assert onset_times == [0.02]


def test_get_onsets_offset_by_start_time():
    env_derivative = np.zeros(30)
    env_derivative[10] = 2.0

    _, onset_times = get_onsets(env_derivative, start_time=1.5, window_size=3, on_thresh=0.5)

    assert onset_times == [1.5 + 0.010]


def test_get_offsets_detects_falling_edge():
    downsampled_env = np.concatenate([np.full(20, 2.0), np.full(30, 0.0)])

    offsets, offset_times = get_offsets(downsampled_env, window_size=5, off_thresh=0.5, start_time=0)

    assert sum(offsets) == 1
    assert offset_times[0] == 0.019


def test_verify_usv_pairs_onset_with_offset_above_amplitude_threshold():
    downsampled_env = np.concatenate([np.full(20, 0.0), np.full(30, 2.0), np.full(20, 0.0)])
    env_derivative = np.gradient(downsampled_env)

    onsets, _ = get_onsets(env_derivative, start_time=0, window_size=5, on_thresh=0.5)
    offsets, _ = get_offsets(downsampled_env, window_size=5, off_thresh=0.5, start_time=0)

    pairs, orphans = verify_usv(downsampled_env, onsets, offsets, amp_thresh=1.0, start_time=0)

    assert pairs == [(0.019, 0.049)]
    assert len(orphans) == 0


def test_verify_usv_drops_pairs_below_amplitude_threshold():
    downsampled_env = np.concatenate([np.full(20, 0.0), np.full(30, 0.2), np.full(20, 0.0)])
    env_derivative = np.gradient(downsampled_env)

    onsets, _ = get_onsets(env_derivative, start_time=0, window_size=5, on_thresh=0.05)
    offsets, _ = get_offsets(downsampled_env, window_size=5, off_thresh=0.05, start_time=0)

    # Amplitude threshold set above the signal's peak -> should be dropped.
    pairs, _ = verify_usv(downsampled_env, onsets, offsets, amp_thresh=1.0, start_time=0)

    assert pairs == []


def test_get_nearest_neighbors_matches_expected_distances():
    pairs = np.array([(0.0, 0.01), (0.05, 0.06), (0.2, 0.21)])

    distances = get_nearest_neighbors(pairs)

    assert distances[0] == 0.04  # to next pair only
    assert distances[1] == 0.04  # min(to prev, to next)
    assert distances[2] == 0.14  # to prev pair only


def test_get_nearest_neighbors_handles_single_pair():
    pairs = np.array([(0.0, 0.01)])
    assert get_nearest_neighbors(pairs) == [float("inf")]
