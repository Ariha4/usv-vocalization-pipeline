"""
Streaming (live) USV detection.

The offline pipeline in usv_class.py reads a whole recording from disk. This module runs the
same detection steps (bandpass + Hilbert envelope, derivative onsets, quiet-window offsets,
amplitude-verified pairing) on audio that arrives a few milliseconds at a time, and reports
each call shortly after it ends.

How it works
------------
- Incoming samples are appended to a buffer.
- Once the buffer holds one full frame (default 1 s), the frame is processed with the existing
  usv_library functions, then the buffer advances by half a frame (the hop). Consecutive frames
  overlap by 50%, so a call that is cut off at the end of one frame is seen whole in the next.
- From each frame we only report calls that end before the last `guard_duration` seconds of
  the frame. Calls closer to the edge are left for the next frame, which sees them in full.
- Calls that start in the first `window_size` ms of a later frame are skipped: the previous frame
  already covered that stretch, and an onset there can be an artifact of the frame boundary.
- A call is never reported twice: new calls that overlap one already reported are skipped.

Thresholds come from one of two places:
- A parameter file exported by an earlier offline run (usv_class.export_params writes
  on_thresh, off_thresh and amp_thresh). Detection starts immediately.
- A calibration window at the start of the stream (default 5 s), using the same mean + k * std
  rule and multipliers as the offline pipeline. Like the offline pipeline, which computes its
  statistics over the whole session, the window should contain typical session audio, calls
  included. A window of pure silence gives thresholds that are too low.

Audio sources
-------------
- iter_wav_blocks(): replays a WAV file in small blocks, optionally at real-time speed. This is
  how the streaming path is tested against real recordings.
- iter_microphone_blocks(): reads a live input device through the optional `sounddevice`
  package. USVs sit at 20-90 kHz, so this needs an ultrasonic microphone and an audio interface
  that samples at 192-250 kHz. A built-in laptop microphone (44.1/48 kHz) cannot capture them.

Command line
------------
    python -m usv_pipeline.live recording.wav --params default_params/25kHz_with_trial_starts.json --realtime
    python -m usv_pipeline.live --mic --fs 250000 --params default_params/25kHz_with_trial_starts.json
"""

from __future__ import annotations

import argparse
import json
import queue
import time

import numpy as np

from usv_pipeline import usv_library as ul


class StreamingUSVDetector:
    """Detects USVs in audio that arrives in small blocks."""

    def __init__(self, fs, freq_range, on_thresh, off_thresh, amp_thresh,
                 window_size=30, frame_duration=1.0, guard_duration=0.2, call_duration=None):
        if fs % 1000 != 0:
            raise ValueError("Sample rate must be a multiple of 1000 Hz (the envelope is downsampled to 1 kHz).")
        if guard_duration >= frame_duration / 2:
            raise ValueError("guard_duration must be shorter than half a frame.")

        self.fs = fs
        self.freq_range = freq_range
        self.on_thresh = on_thresh
        self.off_thresh = off_thresh
        self.amp_thresh = amp_thresh
        self.window_size = window_size
        self.call_duration = call_duration  # optional (min_seconds, max_seconds) filter

        self.frame_length = int(fs * frame_duration)
        self.hop_length = self.frame_length // 2
        self.guard_duration = guard_duration

        self.buffer = np.zeros(0, dtype=np.float32)
        self.buffer_start_time = 0.0       # stream time (s) of buffer[0]
        self.samples_received = 0
        self.reported_calls = []            # every call reported so far, as (start, stop)

    @classmethod
    def from_exported_params(cls, fs, params, frame_duration=1.0, guard_duration=0.2):
        """Build a detector from thresholds saved by an earlier offline run (export_params)."""
        call_duration = tuple(params["call_duration"]) if params.get("twentyfive") else None
        return cls(fs, tuple(params["freq_range"]),
                   params["onset_params"]["on_thresh"], params["offset_params"]["off_thresh"],
                   params["amp_thresh"], window_size=params["window_size"],
                   frame_duration=frame_duration, guard_duration=guard_duration,
                   call_duration=call_duration)

    @classmethod
    def from_calibration(cls, noise_samples, fs, params, frame_duration=1.0, guard_duration=0.2):
        """Build a detector whose thresholds come from a noise-only stretch of audio.

        Uses the same statistics (usv_library.get_thresholds_helper) and the same
        on/off/amp std multipliers as the offline pipeline.
        """
        freq_range = tuple(params["freq_range"])
        _, _, _, downsampled_env, env_derivative = ul.get_env(np.asarray(noise_samples), fs, freq_range)
        on_avg, on_std, off_avg, off_std = ul.get_thresholds_helper(downsampled_env, env_derivative)

        on_thresh = on_avg + on_std * params["on_std_multiplier"]
        off_thresh = off_avg + off_std * params["off_std_multiplier"]
        amp_thresh = off_avg + off_std * params["amp_std_multiplier"]

        call_duration = tuple(params["call_duration"]) if params.get("twentyfive") else None
        return cls(fs, freq_range, on_thresh, off_thresh, amp_thresh,
                   window_size=params["window_size"], frame_duration=frame_duration,
                   guard_duration=guard_duration, call_duration=call_duration)

    def feed(self, samples):
        """Add a block of samples. Returns the calls that were finalized by this block."""
        samples = np.asarray(samples, dtype=np.float32).ravel()
        self.buffer = np.concatenate([self.buffer, samples])
        self.samples_received += len(samples)

        new_calls = []
        while len(self.buffer) >= self.frame_length:
            frame = self.buffer[:self.frame_length]
            frame_end_time = self.buffer_start_time + self.frame_length / self.fs
            report_before = frame_end_time - self.guard_duration
            new_calls.extend(self._process_frame(frame, self.buffer_start_time, report_before))

            self.buffer = self.buffer[self.hop_length:]
            self.buffer_start_time += self.hop_length / self.fs
        return new_calls

    def flush(self):
        """Process whatever is left in the buffer at the end of the stream."""
        if len(self.buffer) < self.fs // 10:   # under 100 ms: too short to analyze
            return []
        frame = self.buffer
        self.buffer = np.zeros(0, dtype=np.float32)
        return self._process_frame(frame, self.buffer_start_time, report_before=float("inf"))

    def _process_frame(self, frame, frame_start_time, report_before):
        _, _, _, downsampled_env, env_derivative = ul.get_env(frame, self.fs, self.freq_range)
        onsets, _ = ul.get_onsets(env_derivative, frame_start_time, self.window_size, self.on_thresh)
        offsets, _ = ul.get_offsets(downsampled_env, self.window_size, self.off_thresh, frame_start_time)
        frame_calls, _ = ul.verify_usv(downsampled_env, onsets, offsets, self.amp_thresh, frame_start_time)

        skip_before = frame_start_time + self.window_size / 1000 if frame_start_time > 0 else -1.0

        new_calls = []
        for start, stop in frame_calls:
            if start < skip_before:
                continue   # covered by the previous frame
            if stop >= report_before:
                continue   # too close to the frame edge; the next frame will see it whole
            if self.call_duration is not None:
                length = stop - start
                if length <= self.call_duration[0] or length > self.call_duration[1]:
                    continue
            if self._overlaps_reported(start, stop):
                continue
            call = (round(start, 3), round(stop, 3))
            self.reported_calls.append(call)
            new_calls.append(call)
        return new_calls

    def _overlaps_reported(self, start, stop):
        # Only recent calls can overlap, so check from the newest backwards.
        for reported_start, reported_stop in reversed(self.reported_calls):
            if reported_stop < start - 1.0:
                break
            if start <= reported_stop and stop >= reported_start:
                return True
        return False


def iter_wav_blocks(wav_path, block_duration=0.01, realtime=False):
    """Yield (samples, fs) blocks from a WAV file, optionally paced at real-time speed."""
    import soundfile as sf

    fs = sf.info(wav_path).samplerate
    block_size = max(1, int(fs * block_duration))
    stream_start = time.monotonic()
    samples_sent = 0
    for block in sf.blocks(wav_path, blocksize=block_size, dtype="float32", always_2d=True):
        mono_block = block[:, 0]
        if realtime:
            due_time = stream_start + samples_sent / fs
            wait = due_time - time.monotonic()
            if wait > 0:
                time.sleep(wait)
        samples_sent += len(mono_block)
        yield mono_block, fs


def iter_microphone_blocks(fs, block_duration=0.01, device=None):
    """Yield (samples, fs) blocks from a live input device (needs the `sounddevice` package)."""
    try:
        import sounddevice as sd
    except ImportError as error:
        raise ImportError("Live microphone input needs `pip install sounddevice`.") from error

    block_queue = queue.Queue()

    def on_audio(indata, frames, time_info, status):
        block_queue.put(indata[:, 0].copy())

    with sd.InputStream(samplerate=fs, channels=1, blocksize=int(fs * block_duration),
                        dtype="float32", device=device, callback=on_audio):
        while True:
            yield block_queue.get(), fs


def run_live(blocks, params, calibration_seconds=5.0, frame_duration=1.0, on_call=print, all_calls=None):
    """Detect calls until the stream ends.

    Uses exported thresholds from `params` when present; otherwise calibrates on the first
    `calibration_seconds` of the stream. Detected calls are appended to `all_calls` as they
    arrive, so a caller that stops the stream early still keeps them.
    """
    calibration_parts = []
    calibration_samples = 0
    detector = None
    if all_calls is None:
        all_calls = []
    has_exported_thresholds = "onset_params" in params

    for samples, fs in blocks:
        if detector is None and has_exported_thresholds:
            detector = StreamingUSVDetector.from_exported_params(fs, params, frame_duration)
            on_call("Using thresholds from the parameter file. Listening...")

        if detector is None:
            calibration_parts.append(samples)
            calibration_samples += len(samples)
            if calibration_samples >= calibration_seconds * fs:
                noise = np.concatenate(calibration_parts)
                detector = StreamingUSVDetector.from_calibration(noise, fs, params, frame_duration)
                on_call(f"Calibrated on {calibration_samples / fs:.1f} s of audio. Listening...")
                for call in detector.feed(noise):
                    all_calls.append(call)
                    _report(call, detector, on_call)
            continue

        for call in detector.feed(samples):
            all_calls.append(call)
            _report(call, detector, on_call)

    if detector is not None:
        for call in detector.flush():
            all_calls.append(call)
            _report(call, detector, on_call)
    return all_calls


def _report(call, detector, on_call):
    start, stop = call
    stream_time = detector.samples_received / detector.fs
    delay = stream_time - stop
    on_call(f"USV  {start:9.3f}s - {stop:9.3f}s  ({(stop - start) * 1000:5.1f} ms)  "
            f"reported {delay:4.2f}s after it ended")


def write_audacity_labels(calls, path):
    with open(path, "w") as labels_file:
        for index, (start, stop) in enumerate(calls, 1):
            labels_file.write(f"{start}\t{stop}\tUSV {index}\n")


def main():
    parser = argparse.ArgumentParser(description="Streaming USV detection from a WAV file or a live input.")
    parser.add_argument("wav", nargs="?", help="WAV file to stream (omit when using --mic)")
    parser.add_argument("--params", required=True, help="detection parameter JSON (see default_params/)")
    parser.add_argument("--realtime", action="store_true", help="replay the WAV file at real-time speed")
    parser.add_argument("--mic", action="store_true", help="read from a live input device instead of a file")
    parser.add_argument("--fs", type=int, default=250000, help="sample rate for --mic")
    parser.add_argument("--device", default=None, help="input device name or index for --mic")
    parser.add_argument("--calibration-seconds", type=float, default=5.0)
    parser.add_argument("--frame-duration", type=float, default=1.0)
    parser.add_argument("--labels", help="also write detected calls as an Audacity label file")
    args = parser.parse_args()

    with open(args.params) as params_file:
        params = json.load(params_file)

    if args.mic:
        blocks = iter_microphone_blocks(args.fs, device=args.device)
    elif args.wav:
        blocks = iter_wav_blocks(args.wav, realtime=args.realtime)
    else:
        parser.error("give a WAV file or --mic")

    calls = []
    try:
        run_live(blocks, params, args.calibration_seconds, args.frame_duration, all_calls=calls)
    except KeyboardInterrupt:
        print("Stopped.")

    print(f"{len(calls)} calls detected.")
    if args.labels:
        write_audacity_labels(calls, args.labels)
        print(f"Audacity labels written to {args.labels}")


if __name__ == "__main__":
    main()
