#!/usr/bin/env python3
"""
CSV Beep Filter Script - Flexible Version
Processes CSV files to identify alignment beeps from EVT32 events
Saves output to current working directory
"""

import pandas as pd
import numpy as np
import sys
import os

def find_alignment_beeps_in_csv(csv_file_path, output_dir=None):
    """
    Find 5 alignment beeps from EVT32 events in CSV file.
    
    Args:
        csv_file_path: Path to input CSV file
        output_dir: Directory to save output (default: current directory)
    
    Returns:
        DataFrame with the 5 alignment beeps
    """
    
    print(f"Loading CSV file: {csv_file_path}")
    
    # Read CSV file without header
    df = pd.read_csv(csv_file_path, header=None)
    
    print(f"Total rows in CSV: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    
    # Use fixed column indices based on known CSV structure
    event_col = 0  # First column: Event type (EVT32, Frame Marker, etc.)
    time_col = 4   # Fifth column: Timestamp
    
    print(f"\nUsing columns: Event (col {event_col}), Time (col {time_col})")
    print("First few rows:")
    print(df.iloc[:3, [event_col, time_col]])
    
    # Filter for EVT32
    evt32_df = df[df[event_col] == 'EVT32'].copy()
    evt32_df = evt32_df.sort_values(by=time_col).reset_index(drop=True)
    
    print(f"\nFound {len(evt32_df)} EVT32 events")
    
    if len(evt32_df) == 0:
        print("ERROR: No EVT32 events found in CSV")
        return None
    
    # Show all EVT32 events
    print("\nAll EVT32 events (chronological):")
    for idx, row in evt32_df.iterrows():
        print(f"  {idx+1}. Time: {row[time_col]:.4f}s")
    
    # Known characteristics of alignment beeps
    EXPECTED_SPACING_MS = 254  # milliseconds
    EXPECTED_SPACING_SEC = EXPECTED_SPACING_MS / 1000
    SPACING_TOLERANCE_SEC = 0.010  # 10ms tolerance
    
    print(f"\nSearching for 5 beeps spaced ~{EXPECTED_SPACING_MS}ms apart...")
    print(f"Tolerance: ±{SPACING_TOLERANCE_SEC*1000:.1f}ms")
    
    # Convert timestamps to numpy array
    timestamps = evt32_df[time_col].values
    
    # Look for 5 consecutive events with consistent spacing
    best_cluster = None
    best_score = float('inf')
    
    for i in range(len(timestamps) - 4):
        cluster_times = timestamps[i:i+5]
        spacings = np.diff(cluster_times)
        spacing_errors = np.abs(spacings - EXPECTED_SPACING_SEC)
        max_error = np.max(spacing_errors)
        avg_error = np.mean(spacing_errors)
        
        if max_error < SPACING_TOLERANCE_SEC:
            avg_spacing_ms = np.mean(spacings) * 1000
            spacing_variation = (np.max(spacings) - np.min(spacings)) * 1000
            
            print(f"\n  Candidate cluster starting at index {i+1}:")
            print(f"    Times: {[f'{t:.4f}' for t in cluster_times]}")
            print(f"    Spacings: {[f'{s*1000:.2f}' for s in spacings]} ms")
            print(f"    Average spacing: {avg_spacing_ms:.2f}ms")
            print(f"    Spacing error: {avg_error*1000:.2f}ms")
            print(f"    Variation: {spacing_variation:.2f}ms")
            
            if avg_error < best_score:
                best_score = avg_error
                best_cluster = (i, cluster_times)
    
    if best_cluster is None:
        print("\nERROR: Could not find 5 EVT32 events with ~254ms spacing")
        return None
    
    # Extract the best cluster
    start_idx, cluster_times = best_cluster
    beep_df = evt32_df.iloc[start_idx:start_idx+5].copy()
    
    # Calculate spacings for each beep
    spacings_list = [None]  # First beep has no previous spacing
    timestamps = beep_df[time_col].values
    for i in range(1, 5):
        spacing_ms = (timestamps[i] - timestamps[i-1]) * 1000
        spacings_list.append(f"{spacing_ms:.3f} ms")
    
    # Add informative columns
    beep_df['Beep_Label'] = [f'Beep_{i}' for i in range(1, 6)]
    beep_df['Spacing_from_Previous'] = spacings_list
    beep_df['Beep_Number'] = range(1, 6)
    
    # Rename columns for clarity
    beep_df = beep_df.rename(columns={
        event_col: 'Event_Type',
        time_col: 'Timestamp_Seconds'
    })
    
    # Reorder columns
    output_columns = ['Beep_Number', 'Beep_Label', 'Event_Type', 'Timestamp_Seconds', 'Spacing_from_Previous']
    output_columns = [col for col in output_columns if col in beep_df.columns]
    beep_df = beep_df[output_columns]
    
    print("\n" + "="*60)
    print("ALIGNMENT BEEPS IDENTIFIED:")
    print("="*60)
    
    for idx, row in beep_df.iterrows():
        beep_num = row['Beep_Label']
        timestamp = row['Timestamp_Seconds']
        print(f"{beep_num}: {timestamp:.6f}s")
    
    # Calculate spacing statistics
    spacings = np.diff(beep_df['Timestamp_Seconds'].values)
    avg_spacing_ms = np.mean(spacings) * 1000
    spacing_variation_ms = (np.max(spacings) - np.min(spacings)) * 1000
    
    print("\nSpacing verification:")
    print(f"  Average spacing: {avg_spacing_ms:.3f}ms")
    print(f"  Expected spacing: {EXPECTED_SPACING_MS}ms")
    print(f"  Error: {abs(avg_spacing_ms - EXPECTED_SPACING_MS):.3f}ms")
    print(f"  Variation: {spacing_variation_ms:.3f}ms")
    
    print("\nFIRST BEEP TIMESTAMP:")
    print(f"  {beep_df.iloc[0]['Timestamp_Seconds']:.6f} seconds")
    
    # Generate output filename in current directory or specified output directory
    base_name = os.path.splitext(os.path.basename(csv_file_path))[0]
    
    if output_dir is None:
        output_dir = os.getcwd()  # Current working directory
    
    output_file = os.path.join(output_dir, f"{base_name}_alignment_beeps.xlsx")
    
    # Save to Excel
    beep_df.to_excel(output_file, index=False, engine='openpyxl')
    
    print(f"\n{'='*60}")
    print(f"✓ Alignment beeps saved to: {output_file}")
    print('='*60)
    
    return beep_df


def main():
    if len(sys.argv) < 2:
        print("Usage: python evt32_detector.py <data_folder> [csv_filename]")
        print("\nExamples:")
        print('  python evt32_detector.py "/path/to/data/folder"')
        print('  python evt32_detector.py "/path/to/data/folder" "events.csv"')
        print('  python evt32_detector.py "/path/to/events.csv"')
        print("\nThe script will:")
        print("  1. Find EVT32 events in the CSV")
        print("  2. Identify 5 beeps spaced ~254ms apart")
        print("  3. Save Excel file to CURRENT directory")
        sys.exit(1)
    
    arg1 = sys.argv[1]
    
    # Check if arg1 is a file or folder
    if os.path.isfile(arg1):
        # Direct path to CSV file
        csv_file = arg1
    elif os.path.isdir(arg1):
        # Folder path provided
        data_folder = arg1
        
        # Check if CSV filename provided as second argument
        if len(sys.argv) > 2:
            csv_filename = sys.argv[2]
            csv_file = os.path.join(data_folder, csv_filename)
        else:
            # Find CSV files in folder
            csv_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
            
            if len(csv_files) == 0:
                print(f"ERROR: No CSV files found in {data_folder}")
                sys.exit(1)
            
            if len(csv_files) == 1:
                csv_file = os.path.join(data_folder, csv_files[0])
                print(f"Found CSV file: {csv_files[0]}\n")
            else:
                print(f"Found {len(csv_files)} CSV files:")
                for i, f in enumerate(csv_files, 1):
                    print(f"  {i}. {f}")
                csv_file = os.path.join(data_folder, csv_files[0])
                print(f"\nUsing first file: {csv_files[0]}\n")
    else:
        print(f"ERROR: Not a valid file or folder: {arg1}")
        sys.exit(1)
    
    if not os.path.exists(csv_file):
        print(f"ERROR: File not found: {csv_file}")
        sys.exit(1)
    
    try:
        result = find_alignment_beeps_in_csv(csv_file, output_dir=None)
        
        if result is None:
            sys.exit(1)
        
        print("\n✓ Success! The 5 alignment beeps have been identified and exported.")
        print(f"✓ Output saved to current directory: {os.getcwd()}")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()