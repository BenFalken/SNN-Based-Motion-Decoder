import numpy as np
import h5py
import scipy.io as sio
from typing import Tuple
from pynwb import NWBHDF5IO
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Jenkins Loader (NWB format - continuous time)
# ============================================================

def load_jenkins_data(
    filename: str,
    reference_label: str = 'go_cue_time'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, np.ndarray]:
    """
    Load Jenkins NWB file and align neural/behavioral data to a reference marker.
    
    This loader handles continuous time data (in seconds) and converts to 1ms bins.
    All trials are aligned to a common reference marker (default: go cue) and 
    padded to have identical duration based on the earliest start and latest stop 
    times across all trials.
    
    Parameters
    ----------
    filename : str
        Path to NWB format file containing neural recordings and behavioral data.
    reference_label : str, optional
        Name of the trial marker to use as alignment reference (t=0 for each trial).
        Default: 'go_cue_time'.
    
    Returns
    -------
    spikes : np.ndarray
        Binary spike raster of shape (time_bins, n_trials, n_neurons).
        Values are 1 where spikes occur, 0 otherwise.
        Time resolution: 1ms bins.
    hand_velocity : np.ndarray
        Hand velocity data of shape (time_bins, n_trials, 2).
        Contains [vx, vy] velocity components at 1ms resolution.
        Computed as numerical derivative of position data.
    lfp_data : np.ndarray
        LFP data of shape (time_bins, n_trials, n_electrodes, n_channels).
        Aligned to the same time window as spikes and velocity.
    behavioral_markers : pd.DataFrame
        Timing of behavioral events (ms) relative to reference marker.
        Shape: (n_trials, n_markers). Each value represents time offset from 
        the reference marker (negative = before reference, positive = after).
    marker_labels : np.ndarray
        Names of behavioral markers (columns of behavioral_markers).
    
    Notes
    -----
    - All trials are padded to the same duration to create rectangular arrays
    - Duration is determined by the trial with earliest relative start and 
      latest relative stop across all trials
    - Spike times are binned at 1ms resolution
    - Hand position and LFP timestamps may have irregular sampling due to tracking 
      imperfections; padding accounts for gaps at trial boundaries
    - LFP data is concatenated across recording segments (segment 1 + segment 2)
    """
    # Load NWB file
    io = NWBHDF5IO(filename, 'r')
    nwb_file = io.read()
    
    # Extract hand position data (continuous tracking)
    hand_position = nwb_file.processing['behavior'].data_interfaces['Position']['Hand'].data[:]
    hand_timestamps = nwb_file.processing['behavior'].data_interfaces['Position']['Hand'].timestamps[:]
    
    # Compute hand velocity from position using numerical derivative
    hand_velocity = np.diff(hand_position, axis=0)
    velocity_timestamps = (hand_timestamps[:-1] + hand_timestamps[1:]) / 2
    
    # Extract trial structure and timing markers
    trial_data = nwb_file.intervals['trials'].to_dataframe()
    
    # Extract spike times for all units (neurons)
    unit_data = [
        nwb_file.units[unit_id]['spike_times'].iloc[0][:] 
        for unit_id in range(len(nwb_file.units))
    ]

    # ========================================
    # Extract and concatenate LFP data across segments
    # ========================================
    electrode_series = nwb_file.processing['ecephys'].data_interfaces['Processed'].electrical_series
    electrode_names = list(electrode_series.keys())
    
    # Group electrodes by array (A or B)
    arrays = {}
    for name in electrode_names:
        array_id = name[0]  # 'A' or 'B'
        if array_id not in arrays:
            arrays[array_id] = []
        arrays[array_id].append(name)
    
    # Sort segments within each array (A001, A002 or B001, B002)
    for array_id in arrays:
        arrays[array_id].sort()
    
    # Concatenate segments for each array
    LFP = []
    array_names = sorted(arrays.keys())
    
    for array_id in array_names:
        segments = arrays[array_id]
        print(f"Processing array {array_id} with segments: {segments}")
        
        # Load all segments for this array
        segment_data = []
        for segment_name in segments:
            series = electrode_series[segment_name]
            data = series.data[:]  # Shape: (time, channels)
            segment_data.append(data)
            print(f"  {segment_name}: {data.shape}")
        
        # Concatenate segments along time axis
        concatenated_data = np.concatenate(segment_data, axis=0)
        print(f"  Concatenated shape: {concatenated_data.shape}")
        
        # Transpose to (channels, time) for consistency with original code
        LFP.append(concatenated_data.T)
    
    # Create unified timestamp array based on concatenated length
    # Use the minimum length across arrays to ensure all timestamps are valid
    sampling_rate = 1000.0  # Hz
    min_samples = min(lfp.shape[1] for lfp in LFP)
    lfp_timestamps = np.arange(min_samples) / sampling_rate
    
    print(f"\nLFP concatenated shapes: {[lfp.shape for lfp in LFP]}")
    print(f"LFP timestamps: {len(lfp_timestamps)} samples ({lfp_timestamps[-1]:.2f} seconds)")
    print(f"Trial time range: {trial_data['start_time'].min():.2f} to {trial_data['stop_time'].max():.2f} seconds")
    
    # Truncate LFP data to minimum length across arrays
    LFP = [lfp[:, :min_samples] for lfp in LFP]
    
    io.close()
    
    num_trials = len(trial_data)
    num_neurons = len(unit_data)
    num_electrodes = len(LFP)
    num_lfp_channels = LFP[0].shape[0] if len(LFP) > 0 else 0
    
    # ========================================
    # STEP 1: Determine trial alignment window
    # ========================================
    min_start = 0.0
    max_stop = 0.0
    
    for i in range(num_trials):
        ref_time = trial_data.iloc[i][reference_label]
        start_rel = trial_data.iloc[i]['start_time'] - ref_time
        stop_rel = trial_data.iloc[i]['stop_time'] - ref_time
        
        min_start = min(min_start, start_rel)
        max_stop = max(max_stop, stop_rel)
    
    max_duration = int(1000 * (max_stop - min_start))
    print(f"\nTrial window: {min_start:.3f} to {max_stop:.3f} seconds relative to {reference_label}")
    print(f"Max duration: {max_duration} ms")
    
    # ========================================
    # STEP 2: Align behavioral markers
    # ========================================
    marker_labels = trial_data.columns[[0, 1, 2, 3, 5, 6]].to_numpy()
    reference_col = trial_data[reference_label].to_numpy()
    min_start_col = np.full((num_trials, 1), min_start)
    
    behavioral_markers = (
        trial_data[marker_labels].to_numpy() 
        - reference_col[:, np.newaxis] 
        - min_start_col
    )
    
    behavioral_markers = pd.DataFrame(
        1000 * behavioral_markers, 
        columns=marker_labels
    )
    
    # ========================================
    # STEP 3: Initialize output arrays
    # ========================================
    X = np.zeros((num_trials, max_duration, num_neurons), dtype=np.float32)
    y = np.zeros((num_trials, max_duration, 2), dtype=np.float32)
    lfp_array = np.zeros((num_trials, max_duration, num_electrodes, num_lfp_channels), dtype=np.float32)
    
    # ========================================
    # STEP 4: Populate spike rasters
    # ========================================
    for trial_idx in range(num_trials):
        ref_time = trial_data.iloc[trial_idx][reference_label]
        window_start = ref_time + min_start
        window_stop = ref_time + max_stop
        
        for neuron_idx in range(num_neurons):
            spike_times = unit_data[neuron_idx]
            valid_spikes = spike_times[
                (spike_times >= window_start) & 
                (spike_times < window_stop)
            ]
            spike_times_rel = valid_spikes - window_start
            spike_bins = np.clip(
                np.array(1000 * spike_times_rel, dtype=int),
                0,
                max_duration - 1
            )
            X[trial_idx, spike_bins, neuron_idx] = 1
    
    # ========================================
    # STEP 5: Populate hand velocity data
    # ========================================
    for trial_idx in range(num_trials):
        ref_time = trial_data.iloc[trial_idx][reference_label]
        window_start = ref_time + min_start
        window_stop = ref_time + max_stop
        
        valid_velocity_mask = (
            (velocity_timestamps >= window_start) & 
            (velocity_timestamps < window_stop)
        )
        
        if not valid_velocity_mask.any():
            continue
        
        first_velocity_timestamp = np.min(velocity_timestamps[valid_velocity_mask])
        start_padding_len = int(1000 * (first_velocity_timestamp - window_start))
        start_padding_len = max(0, start_padding_len)
        
        y_trial = hand_velocity[valid_velocity_mask]
        end_idx = min(start_padding_len + y_trial.shape[0], max_duration)
        actual_len = end_idx - start_padding_len
        
        if actual_len > 0:
            y[trial_idx, start_padding_len:end_idx] = y_trial[:actual_len]
    
    # ========================================
    # STEP 6: Populate LFP data
    # ========================================
    trials_outside_lfp_range = 0
    
    for trial_idx in range(num_trials):
        ref_time = trial_data.iloc[trial_idx][reference_label]
        window_start = ref_time + min_start
        window_stop = ref_time + max_stop
        
        # Check if trial is within LFP recording range
        if window_stop > lfp_timestamps[-1]:
            trials_outside_lfp_range += 1
            continue
        
        # Find LFP samples within trial window
        valid_lfp_mask = (
            (lfp_timestamps >= window_start) & 
            (lfp_timestamps < window_stop)
        )
        
        if not valid_lfp_mask.any():
            continue
        
        # Calculate padding needed if LFP starts after window start
        first_lfp_timestamp = np.min(lfp_timestamps[valid_lfp_mask])
        start_padding_len = int(1000 * (first_lfp_timestamp - window_start))
        start_padding_len = max(0, start_padding_len)
        
        # Process each electrode array
        for electrode_idx in range(num_electrodes):
            # Extract LFP for this trial (channels x time)
            lfp_trial = LFP[electrode_idx][:, valid_lfp_mask]
            
            # Calculate how much data we can fit
            end_idx = min(start_padding_len + lfp_trial.shape[1], max_duration)
            actual_len = end_idx - start_padding_len
            
            # Place LFP data in the aligned array
            if actual_len > 0:
                # Transpose to get (time x channels)
                lfp_array[trial_idx, start_padding_len:end_idx, electrode_idx, :] = lfp_trial[:, :actual_len].T
    
    lfp_array = np.reshape(lfp_array, lfp_array.shape[:-2]+(192,))

    if trials_outside_lfp_range > 0:
        print(f"\nWarning: {trials_outside_lfp_range} trials extend beyond LFP recording range and were skipped for LFP data")

    print(f"\nFinal data shapes:")
    print(f"  Spikes (X): {X.shape}")
    print(f"  Hand velocity (y): {y.shape}")
    print(f"  LFP: {lfp_array.shape}")
    print(f"  Behavioral markers: {behavioral_markers.shape}")
    print(f"Data is of shape: {X.shape[0]} trials, {X.shape[1]} time points, {X.shape[2]} neurons")

    return X, y, lfp_array, behavioral_markers, marker_labels