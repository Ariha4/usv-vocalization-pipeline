"""Tests for trial-alignment helpers (assigning USVs and lever presses to trials)."""

from usv_pipeline.usv_library import get_trial_nums, get_open


def test_get_trial_nums_assigns_events_to_correct_trial():
    start_times = [0.5, 5.5, 10.5, 15.5]
    trial_starts = [0, 10]
    trial_duration = 10

    trial_nums = get_trial_nums(start_times, trial_starts, trial_duration)

    assert trial_nums == [1, 1, 2, 2]


def test_get_open_computes_latency_from_trial_start():
    lever_press = [1.0, 12.0]
    trial_starts = [0, 10]
    trial_duration = 10

    opens, latency = get_open(lever_press, trial_starts, trial_duration)

    assert opens[1] == 1.0
    assert opens[2] == 12.0
    assert latency[1] == 1.0
    assert latency[2] == 2.0


def test_get_open_leaves_trial_none_when_no_press_occurs():
    lever_press = [1.0]
    trial_starts = [0, 10]
    trial_duration = 10

    opens, latency = get_open(lever_press, trial_starts, trial_duration)

    assert opens[1] == 1.0
    assert opens[2] is None
    assert latency[2] is None
