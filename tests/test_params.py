"""Tests for parameter-file validation (usv_pipeline/params.py)."""

import json
from pathlib import Path

import pytest

from usv_pipeline.params import ParamsError, load_params, validate_params

DEFAULT_PARAMS_DIR = Path(__file__).resolve().parent.parent / "default_params"

STREAMING_PARAMS = {
    "freq_range": [20000, 30000],
    "window_size": 30,
    "on_std_multiplier": 1,
    "off_std_multiplier": 0.5,
    "amp_std_multiplier": 1.5,
}

EXPORTED_PARAMS = {
    "freq_range": [20000, 30000],
    "window_size": 30,
    "onset_params": {"on_thresh": 0.4},
    "offset_params": {"off_thresh": 0.2},
    "amp_thresh": 0.3,
}


def problems_for(params, **kwargs):
    with pytest.raises(ParamsError) as excinfo:
        validate_params(params, **kwargs)
    return excinfo.value.problems


@pytest.mark.parametrize("path", sorted(DEFAULT_PARAMS_DIR.glob("*.json")), ids=lambda path: path.name)
def test_shipped_default_params_are_valid_for_offline_runs(path):
    load_params(path, fs=250_000, offline=True)


def test_valid_streaming_and_exported_params_pass_unchanged():
    assert validate_params(STREAMING_PARAMS, fs=250_000) is STREAMING_PARAMS
    assert validate_params(EXPORTED_PARAMS, fs=250_000) is EXPORTED_PARAMS


def test_every_problem_is_reported_at_once():
    problems = problems_for({"freq_range": [30000, 20000], "window_size": 30.0})

    assert any("freq_range" in problem for problem in problems)
    assert any("window_size" in problem for problem in problems)
    for key in ("on_std_multiplier", "off_std_multiplier", "amp_std_multiplier"):
        assert f"missing required key '{key}'" in problems


@pytest.mark.parametrize("freq_range", [[20000], [0, 30000], [30000, 20000], ["20k", "30k"], 25000])
def test_bad_freq_ranges_are_rejected(freq_range):
    problems = problems_for(dict(STREAMING_PARAMS, freq_range=freq_range))
    assert any("freq_range" in problem for problem in problems)


def test_band_above_nyquist_is_rejected_only_when_sample_rate_is_known():
    params = dict(STREAMING_PARAMS, freq_range=[20000, 90000])

    validate_params(params)                    # sample rate unknown: nothing to compare against
    validate_params(params, fs=250_000)        # Nyquist 125 kHz
    problems = problems_for(params, fs=96_000)  # Nyquist 48 kHz
    assert len(problems) == 1
    assert "Nyquist" in problems[0] and "48000" in problems[0]


@pytest.mark.parametrize("window_size", [30.0, 0, -5, True, "30"])
def test_window_size_must_be_a_positive_integer(window_size):
    # A float here used to fail deep inside numpy.pad; True would silently mean 1 ms.
    problems = problems_for(dict(STREAMING_PARAMS, window_size=window_size))
    assert any("window_size" in problem for problem in problems)


@pytest.mark.parametrize("value", [-1, "1.5", None, float("nan")])
def test_multipliers_must_be_non_negative_numbers(value):
    problems = problems_for(dict(STREAMING_PARAMS, amp_std_multiplier=value))
    assert any("amp_std_multiplier" in problem for problem in problems)


def test_exported_thresholds_replace_the_multipliers_for_streaming_only():
    validate_params(EXPORTED_PARAMS)
    problems = problems_for(EXPORTED_PARAMS, offline=True)
    assert "missing required key 'on_std_multiplier'" in problems


@pytest.mark.parametrize("missing", ["onset_params", "offset_params", "amp_thresh"])
def test_partially_exported_thresholds_are_rejected(missing):
    params = {key: value for key, value in EXPORTED_PARAMS.items() if key != missing}
    problems = problems_for(params)
    assert any("incomplete" in problem for problem in problems)


def test_twentyfive_requires_valid_call_duration():
    assert "missing required key 'call_duration' (needed when twentyfive is true)" in problems_for(
        dict(STREAMING_PARAMS, twentyfive=True)
    )
    problems = problems_for(dict(STREAMING_PARAMS, twentyfive=True, call_duration=[0.15, 0.008]))
    assert any("call_duration" in problem for problem in problems)
    problems = problems_for(dict(STREAMING_PARAMS, twentyfive="yes"))
    assert any("twentyfive must be true or false" in problem for problem in problems)


def test_offline_runs_need_neighbor_distance_for_25khz_calls():
    params = json.loads((DEFAULT_PARAMS_DIR / "25kHz_with_trial_starts.json").read_text())
    del params["neighbor_distance"]

    validate_params(params)   # streaming detection doesn't use it
    assert any("neighbor_distance" in problem for problem in problems_for(params, offline=True))


def test_load_params_reports_bad_json_with_the_file_name(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text('{"freq_range": [20000, 30000],}')

    with pytest.raises(ParamsError, match="broken.json") as excinfo:
        load_params(path)
    assert "not valid JSON" in excinfo.value.problems[0]
