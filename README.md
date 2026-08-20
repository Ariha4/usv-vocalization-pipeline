# usv-vocalization-pipeline

A signal-processing pipeline for automatically detecting ultrasonic vocalizations
(USVs) in rodent audio recordings, plus utilities for aligning behavioral event
markers (e.g. lever presses, trial starts) recorded on a separate acquisition
system to the audio timeline.

[![CI](https://github.com/Ariha4/usv-vocalization-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/Ariha4/usv-vocalization-pipeline/actions/workflows/ci.yml)

Originally built for behavioral neuroscience research at the Neural Crossroads
Research Laboratory, where I self-identified the need for an automated USV
detection workflow (replacing fully manual review of hours-long recordings) and
built the pipeline independently.

**Note on data:** this repo contains the detection/analysis code only. Real
experiment recordings and session data are intentionally excluded (see
`USV_DATA/`) since they're unpublished lab research data, not code.

## What it does

Rodents (and other species) produce vocalizations well above the range of human
hearing. Manually reviewing hours of raw audio to find and time-stamp these
calls doesn't scale. This pipeline automates that:

1. **Bandpass filter + envelope extraction** — a Butterworth bandpass filter
   isolates the frequency band of interest (e.g. 20-30 kHz or 40-90 kHz calls),
   then a Hilbert transform extracts the signal's amplitude envelope.
2. **Adaptive thresholding** — onset/offset thresholds are computed per-recording
   from the statistics of quiet ("noise floor") segments, rather than using a
   single fixed threshold across all recordings.
3. **Onset/offset detection** — the envelope's derivative and amplitude are
   scanned to find call start and end times, with a configurable minimum
   quiet-window requirement to avoid false triggers.
4. **Pairing and filtering** — detected onsets/offsets are paired into candidate
   calls, filtered by minimum amplitude, and (for 25 kHz calls) filtered further
   by call duration and inter-call spacing.
5. **Feature extraction** — each detected call gets a feature vector (RMS,
   zero-crossing rate, spectral centroid/bandwidth/flatness/rolloff/contrast,
   pitch via YIN, mel spectrogram) for downstream classification/analysis.
6. **Cross-modal alignment** — a separate module detects a known 5-beep
   alignment signal (via bandpass+envelope onset detection on audio, or by
   parsing EVT32 event markers from a behavioral acquisition system's CSV
   export) so that behavioral event timestamps and audio timestamps can be
   synced to a shared clock.
7. **Trial alignment + rater workflow** — USVs are mapped to trial numbers and
   labeled relative to lever-press ("open") times; a CLI-driven rater setup
   flow supports multi-rater manual validation with per-rater tracking.

## Repo layout

```
usv_pipeline/
  usv_library.py    # core signal-processing functions (filtering, onset/
                     # offset detection, pairing, alignment, feature extraction)
  usv_class.py       # USV class: wraps a recording + params, runs the full
                       # detect -> extract -> export pipeline
  detect_beeps.py     # standalone alignment-beep detector (audio-based)
  evt32_detector.py   # standalone alignment-beep detector (CSV/EVT32-based)
  rater_setup.py       # CLI helpers for the manual-validation workflow
default_params/
  25kHz_with_trial_starts.json   # example tuned detection parameters
tests/
  test_filtering.py    # bandpass/envelope tests on synthetic tones
  test_detection.py     # onset/offset/pairing tests on synthetic envelopes
  test_trials.py         # trial-assignment / lever-press-latency tests
```

## Getting started

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt

pytest -v      # run the test suite
ruff check .   # lint
```

## Testing approach

The core signal-processing and detection logic (filtering, onset/offset
detection, pairing, trial alignment) is tested with small, hand-constructed
synthetic signals — sine tones at known frequencies for the filter tests, and
short synthetic envelope arrays with known onset/offset points for the
detection tests. This keeps the test suite fast and dependency-free (no real
audio files needed) while still exercising the actual algorithms rather than
mocks.

## Known limitations / in-progress pieces

This is research code, and a few pieces are left as-is for transparency rather
than polished or removed:
- `usv_class.py` has a couple of methods marked `# IN PROGRESS` /
  `# NOT NEEDED FOR NOW` by the original author (threshold auto-optimization,
  a manual 25 kHz sorting workflow) that aren't part of the main pipeline path.
- Some helper functions assume a specific lab acquisition format (Plexon EVT32
  markers, a specific trial-starts CSV layout) and would need adaptation for a
  different setup.

## License

MIT — see [LICENSE](LICENSE).
