"""
Loading and validating detection parameter files.

Parameter files (see default_params/) are hand-edited JSON, and files exported by an offline run
(usv_class.export_params) get reused on new recordings. A typo or a value that doesn't fit the
recording used to surface far from its cause: a missing key as a bare KeyError, a window_size of
30.0 as a numpy padding error, and a frequency band above the Nyquist limit as a scipy filter error
one full frame into a live stream. validate_params checks everything up front and reports every
problem at once, naming the key and what it should be.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

MULTIPLIER_KEYS = ("on_std_multiplier", "off_std_multiplier", "amp_std_multiplier")
OFFLINE_KEYS = ("chunk_duration", "frame_duration", "params_source")


class ParamsError(ValueError):
    """A parameter file is missing keys or has values the detector can't use."""

    def __init__(self, problems, source=None):
        self.problems = list(problems)
        where = f" in {source}" if source else ""
        details = "\n".join(f"  - {problem}" for problem in self.problems)
        super().__init__(f"Invalid detection parameters{where}:\n{details}")


def _is_number(value):
    # bool is a subclass of int, but `"window_size": true` is a mistake, not the number 1.
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _check_pair(params, key, problems, minimum_low=0.0):
    """Check a [low, high] pair of numbers with minimum_low <= low < high. Returns it, or None."""
    value = params[key]
    if not isinstance(value, (list, tuple)) or len(value) != 2 or not all(map(_is_number, value)):
        problems.append(f"{key} must be a list of two numbers [low, high], got {value!r}")
        return None
    low, high = value
    if low < minimum_low or low >= high:
        problems.append(f"{key} must satisfy {minimum_low:g} <= low < high, got {list(value)}")
        return None
    return low, high


def _check_freq_range(params, fs, problems):
    band = _check_pair(params, "freq_range", problems)
    if band is None:
        return
    low, high = band
    if low == 0:
        problems.append("freq_range low edge must be above 0 Hz for a bandpass filter")
    if fs is not None and high >= fs / 2:
        problems.append(
            f"freq_range upper edge {high:g} Hz must be below the Nyquist frequency ({fs / 2:g} Hz) "
            f"for a {fs:g} Hz recording; use a higher sample rate or lower the band"
        )


def _check_exported_thresholds(params, problems):
    onset_params = params.get("onset_params")
    offset_params = params.get("offset_params")
    if not isinstance(onset_params, dict) or not _is_number(onset_params.get("on_thresh")):
        problems.append("onset_params.on_thresh must be a number (exported thresholds are incomplete)")
    if not isinstance(offset_params, dict) or not _is_number(offset_params.get("off_thresh")):
        problems.append("offset_params.off_thresh must be a number (exported thresholds are incomplete)")
    if not _is_number(params.get("amp_thresh")):
        problems.append("amp_thresh must be a number (exported thresholds are incomplete)")


def has_exported_thresholds(params):
    """True if the file carries thresholds from an offline run, so no calibration is needed."""
    return "onset_params" in params or "offset_params" in params or "amp_thresh" in params


def validate_params(params, fs=None, offline=False, source=None):
    """Check a parameter dict and return it unchanged, or raise ParamsError listing every problem.

    fs: sample rate of the audio the parameters will be applied to, if known. Enables the
        Nyquist check on freq_range.
    offline: also require the keys only the offline pipeline (usv_class) uses.
    source: file name to mention in the error message.
    """
    if not isinstance(params, dict):
        raise ParamsError([f"expected a JSON object, got {type(params).__name__}"], source)

    required = ["freq_range", "window_size"]
    exported = has_exported_thresholds(params)
    if offline or not exported:
        required += MULTIPLIER_KEYS   # thresholds have to be calibrated from these
    if offline:
        required += [*OFFLINE_KEYS, "twentyfive"]
    problems = [f"missing required key '{key}'" for key in required if key not in params]

    if "freq_range" in params:
        _check_freq_range(params, fs, problems)

    window_size = params.get("window_size")
    if "window_size" in params and (
        isinstance(window_size, bool) or not isinstance(window_size, int) or window_size < 1
    ):
        problems.append(f"window_size must be a positive whole number of milliseconds, got {window_size!r}")

    for key in MULTIPLIER_KEYS:
        if key in params and not (_is_number(params[key]) and params[key] >= 0):
            problems.append(f"{key} must be a number >= 0, got {params[key]!r}")

    if offline:
        for key in ("chunk_duration", "frame_duration"):
            if key in params and not (_is_number(params[key]) and params[key] > 0):
                problems.append(f"{key} must be a positive number of seconds, got {params[key]!r}")

    if exported:
        _check_exported_thresholds(params, problems)

    twentyfive = params.get("twentyfive", False)
    if not isinstance(twentyfive, bool):
        problems.append(f"twentyfive must be true or false, got {twentyfive!r}")
    elif twentyfive:
        if "call_duration" not in params:
            problems.append("missing required key 'call_duration' (needed when twentyfive is true)")
        else:
            _check_pair(params, "call_duration", problems)
        if offline and "neighbor_distance" not in params:
            problems.append("missing required key 'neighbor_distance' (needed when twentyfive is true)")
        elif "neighbor_distance" in params and not (
            _is_number(params["neighbor_distance"]) and params["neighbor_distance"] > 0
        ):
            problems.append(f"neighbor_distance must be a positive number, got {params['neighbor_distance']!r}")

    if problems:
        raise ParamsError(problems, source)
    return params


def load_params(path, fs=None, offline=False):
    """Read a parameter JSON file and validate it. Raises ParamsError on bad JSON or bad values."""
    path = Path(path)
    try:
        with open(path) as params_file:
            params = json.load(params_file)
    except json.JSONDecodeError as error:
        raise ParamsError([f"not valid JSON: {error}"], path) from error
    return validate_params(params, fs=fs, offline=offline, source=path)
