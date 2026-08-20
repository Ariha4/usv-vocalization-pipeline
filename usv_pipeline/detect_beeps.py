#!/usr/bin/env python3
"""
Beep Detection with Onset Detection - Simple Single Plot Version
Shows amplitude envelope with onsets from 50-53 seconds
"""

import numpy as np
import soundfile as sf
import sys
import os
from scipy import signal
import matplotlib.pyplot as plt

def butterworth_hilbert_filter(data, fs, band):
    """
    Apply a Butterworth bandpass filter and Hilbert transform to the input data.
    
    Parameters:
    - data: 1D numpy array of the signal to be filtered.
    - fs: Sampling frequency of the signal.
    - band: Tuple (low_freq, high_freq) specifying the frequency band for the filter.
    
    Returns:
    - filtered: The bandpass filtered signal.
    - amplitude_envelope: The amplitude envelope of the filtered signal.
    """
    data = np.asarray(data)
    
    nyq = 0.5 * fs
    low, high = band[0] / nyq, band[1] / nyq
    b, a = signal.butter(3, [low, high], btype='band')
    filtered = signal.filtfilt(b, a, data)
    
    analytic_signal = signal.hilbert(filtered)
    amplitude_envelope = np.abs(analytic_signal)
    
    return filtered, amplitude_envelope


def detect_onset(envelope, threshold, min_distance_samples):
    """Detect onset points where signal crosses threshold going upward."""
    above_threshold = envelope > threshold
    crossings = np.diff(above_threshold.astype(int))
    onset_indices = np.where(crossings == 1)[0] + 1
    
    if len(onset_indices) > 0:
        filtered_onsets = [onset_indices[0]]
        for onset in onset_indices[1:]:
            if onset - filtered_onsets[-1] >= min_distance_samples:
                filtered_onsets.append(onset)
        onset_indices = np.array(filtered_onsets)
    
    return onset_indices


def detect_alignment_beeps_onset(wav_file_path, output_label_file=None):
    """Detect 5 alignment beeps using onset detection."""
    
    print(f"Loading audio file: {wav_file_path}")
    
    # Load first 2 minutes
    info = sf.info(wav_file_path)
    sample_rate = info.samplerate
    max_samples = int(120 * sample_rate)
    
    print(f"Sample rate: {sample_rate} Hz")
    print("Analyzing first 2 minutes of audio...")
    
    audio_data = sf.read(wav_file_path, frames=max_samples)[0]
    
    print(f"Loaded {len(audio_data) / sample_rate:.2f} seconds")
    
    # Convert to mono
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)
    
    # Apply bandpass filter (1.5-4 kHz) and get envelope
    print("\nApplying bandpass filter: 1.5 kHz - 4.0 kHz")
    band = (2500, 2800)
    filtered, envelope = butterworth_hilbert_filter(audio_data, sample_rate, band)
    
    print(f"Filtered signal shape: {filtered.shape}")
    print(f"Envelope max: {np.max(envelope):.4f}")
    
    # Set threshold
    threshold = 0.15 * np.max(envelope)
    print(f"Onset detection threshold: {threshold:.4f}")
    
    # Detect onsets
    min_distance_samples = int(0.1 * sample_rate)
    onset_indices = detect_onset(envelope, threshold, min_distance_samples)
    
    print(f"\nFound {len(onset_indices)} onset points")
    
    if len(onset_indices) == 0:
        print("ERROR: No onsets detected")
        return []
    
    # Convert to timestamps
    onset_times = onset_indices / sample_rate
    
    # Create time axis
    time_axis = np.arange(len(envelope)) / sample_rate
    
    # SINGLE PLOT: Envelope with onsets (50-53 seconds)
    print("\nGenerating amplitude envelope plot...")
    
    plt.figure(figsize=(14, 6))
    plt.plot(time_axis, envelope, color='blue', linewidth=0.5, label='Amplitude Envelope')
    
    # Add onsets
    plt.scatter(onset_times, envelope[onset_indices], color='green', s=200, zorder=5, 
                marker='x', linewidths=3, label=f'Detected Onsets ({len(onset_times)})')
    
    plt.xlim([50, 53])
    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel('Amplitude', fontsize=12)
    plt.title('Amplitude Envelope (1.5-4 kHz Bandpass Filtered)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save plot
    plot_filename = os.path.splitext(os.path.basename(wav_file_path))[0] + "_envelope.png"
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    print(f"✓ Plot saved: {plot_filename}")
    plt.close()
    
    # Show detected onsets
    print("\nDetected onsets:")
    for i in range(min(20, len(onset_times))):
        minutes = int(onset_times[i] // 60)
        seconds = onset_times[i] % 60
        print(f"  Onset {i+1}: {onset_times[i]:.6f}s ({minutes}:{seconds:06.3f})")
    
    # Look for 5 beeps with 254ms spacing
    EXPECTED_SPACING_MS = 254
    EXPECTED_SPACING_SEC = EXPECTED_SPACING_MS / 1000
    SPACING_TOLERANCE_SEC = 0.010
    
    print(f"\nSearching for 5 beeps spaced ~{EXPECTED_SPACING_MS}ms apart...")
    
    for i in range(len(onset_times)):
        cluster = [onset_times[i]]
        j = i + 1
        
        while j < len(onset_times) and len(cluster) < 5:
            spacing = onset_times[j] - cluster[-1]
            
            if abs(spacing - EXPECTED_SPACING_SEC) < SPACING_TOLERANCE_SEC:
                cluster.append(onset_times[j])
                j += 1
            else:
                break
        
        if len(cluster) == 5:
            spacings = np.diff(cluster)
            avg_spacing = np.mean(spacings)
            spacing_error = abs(avg_spacing - EXPECTED_SPACING_SEC)
            
            if spacing_error < 0.01:
                print(f"\n{'='*60}")
                print("ALIGNMENT BEEPS DETECTED (ONSETS):")
                print('='*60)
                
                labels = []
                for idx, beep_time in enumerate(cluster, 1):
                    start_time = beep_time
                    end_time = beep_time + 0.001
                    label_name = f"Beep_{idx}_onset"
                    
                    labels.append({
                        'start': start_time,
                        'end': end_time,
                        'name': label_name
                    })
                    
                    minutes = int(beep_time // 60)
                    seconds = beep_time % 60
                    
                    if idx == 1:
                        spacing_info = ""
                    else:
                        spacing_ms = (beep_time - cluster[idx-2]) * 1000
                        spacing_info = f" | Spacing: {spacing_ms:.3f}ms"
                    
                    print(f"{label_name}: {start_time:.6f}s ({minutes}:{seconds:06.3f}){spacing_info}")
                
                # Spacing summary
                print(f"\n{'='*60}")
                print("SPACING SUMMARY:")
                print('='*60)
                spacings_ms = np.array(spacings) * 1000
                for k in range(len(spacings_ms)):
                    print(f"Beep_{k+1} to Beep_{k+2}: {spacings_ms[k]:.3f}ms")
                
                print(f"\nAverage: {np.mean(spacings_ms):.3f}ms")
                print(f"Std dev: {np.std(spacings_ms):.3f}ms")
                print(f"Min: {np.min(spacings_ms):.3f}ms")
                print(f"Max: {np.max(spacings_ms):.3f}ms")
                print(f"Variation: {np.max(spacings_ms) - np.min(spacings_ms):.3f}ms")
                
                print("\nFIRST BEEP ONSET TIMESTAMP:")
                print(f"  {cluster[0]:.6f} seconds")
                
                # Generate output filename
                if output_label_file is None:
                    base_name = os.path.splitext(os.path.basename(wav_file_path))[0]
                    output_label_file = f"{base_name}_onset_labels.txt"
                
                # Write Audacity labels
                write_audacity_labels(labels, output_label_file)
                print(f"\n✓ Audacity labels saved to: {output_label_file}")
                
                print('='*60)
                
                return cluster
    
    print("\nERROR: Could not find 5 beeps with expected spacing")
    return []


def write_audacity_labels(labels, output_file):
    """Write labels to Audacity format."""
    with open(output_file, 'w') as f:
        for label in labels:
            f.write(f"{label['start']:.6f}\t{label['end']:.6f}\t{label['name']}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python detect_beeps.py <wav_file> [output_label_file]")
        sys.exit(1)
    
    wav_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    beeps = detect_alignment_beeps_onset(wav_file, output_file)
    
    if len(beeps) > 0:
        print("\n✓ Success! Use first onset timestamp for alignment.")
    else:
        sys.exit(1)
