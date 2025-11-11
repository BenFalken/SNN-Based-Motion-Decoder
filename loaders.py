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

def load_jenkins_file(
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

'''
def load_jenkins_file(
    filename: str,
    reference_label: str = 'go_cue_time'
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, np.ndarray]:
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
    - Hand position timestamps may have irregular sampling due to tracking 
      imperfections; padding accounts for gaps at trial boundaries
    """
    # Load NWB file
    io = NWBHDF5IO(filename, 'r')
    nwb_file = io.read()
    
    # Extract hand position data (continuous tracking)
    hand_position = nwb_file.processing['behavior'].data_interfaces['Position']['Hand'].data[:]
    hand_timestamps = nwb_file.processing['behavior'].data_interfaces['Position']['Hand'].timestamps[:]
    
    # Compute hand velocity from position using numerical derivative
    # Note: This reduces the length by 1, velocity[i] = position[i+1] - position[i]
    hand_velocity = np.diff(hand_position, axis=0)
    # Adjust timestamps to align with velocity samples (use midpoints)
    velocity_timestamps = (hand_timestamps[:-1] + hand_timestamps[1:]) / 2
    
    # Extract trial structure and timing markers
    trial_data = nwb_file.intervals['trials'].to_dataframe()
    
    # Extract spike times for all units (neurons)
    unit_data = [
        nwb_file.units[unit_id]['spike_times'].iloc[0][:] 
        for unit_id in range(len(nwb_file.units))
    ]

    # Extract LFP
    electrodes = list(nwb_file.processing['ecephys'].data_interfaces['Processed'].electrical_series.keys())
    LFP = [[] for _ in electrodes]
    
    for i, electrode in enumerate(electrodes):
        data = nwb_file.processing['ecephys'].data_interfaces['Processed'].electrical_series[electrode].data[:]
        LFP[i] = data.T

    print(LFP[0].shape, LFP[1].shape, LFP[2].shape, LFP[3].shape)


    io.close()
    
    num_trials = len(trial_data)
    num_neurons = len(unit_data)
    
    # ========================================
    # STEP 1: Determine trial alignment window
    # ========================================
    # Find the earliest start and latest stop times relative to the reference marker
    # across all trials. This ensures all trials fit within a common time window.
    
    min_start = 0.0  # Most negative time relative to reference (earliest start)
    max_stop = 0.0   # Most positive time relative to reference (latest stop)
    
    for i in range(num_trials):
        ref_time = trial_data.iloc[i][reference_label]
        start_rel = trial_data.iloc[i]['start_time'] - ref_time
        stop_rel = trial_data.iloc[i]['stop_time'] - ref_time
        
        min_start = min(min_start, start_rel)
        max_stop = max(max_stop, stop_rel)
    
    # Convert to milliseconds and get total duration
    max_duration = int(1000 * (max_stop - min_start))
    
    # ========================================
    # STEP 2: Align behavioral markers
    # ========================================
    # Transform all marker times to be relative to reference marker and 
    # shifted to start at 0 (by subtracting min_start)
    
    marker_labels = trial_data.columns[[0, 1, 2, 3, 5, 6]].to_numpy()
    reference_col = trial_data[reference_label].to_numpy()
    min_start_col = np.full((num_trials, 1), min_start)
    
    # Calculate: (marker_time - reference_time) - min_start
    # This gives time in seconds relative to the start of the aligned window
    behavioral_markers = (
        trial_data[marker_labels].to_numpy() 
        - reference_col[:, np.newaxis] 
        - min_start_col
    )
    
    # Convert to milliseconds
    behavioral_markers = pd.DataFrame(
        1000 * behavioral_markers, 
        columns=marker_labels
    )
    
    # ========================================
    # STEP 3: Initialize output arrays
    # ========================================
    # Create arrays with dimensions: (time_bins, trials, features)
    
    X = np.zeros((max_duration, num_trials, num_neurons), dtype=np.float32)
    y = np.zeros((max_duration, num_trials, 2), dtype=np.float32)
    
    # ========================================
    # STEP 4: Populate spike rasters
    # ========================================
    # For each trial and neuron, convert continuous spike times to binary raster
    
    for trial_idx in range(num_trials):
        ref_time = trial_data.iloc[trial_idx][reference_label]
        
        # Define the time window for this trial in absolute time
        window_start = ref_time + min_start
        window_stop = ref_time + max_stop
        
        for neuron_idx in range(num_neurons):
            spike_times = unit_data[neuron_idx]
            
            # Select spikes within the trial window
            valid_spikes = spike_times[
                (spike_times >= window_start) & 
                (spike_times < window_stop)
            ]
            
            # Convert to relative time (seconds from window start)
            spike_times_rel = valid_spikes - window_start
            
            # Convert to millisecond bins and clip to valid range
            spike_bins = np.clip(
                np.array(1000 * spike_times_rel, dtype=int),
                0,
                max_duration - 1
            )
            
            # Mark bins with spikes as 1
            X[trial_idx, spike_bins, neuron_idx] = 1
    
    # ========================================
    # STEP 5: Populate hand velocity data
    # ========================================
    # Align hand velocity data to the same time window as spikes.
    # Note: Hand tracking timestamps may be irregular due to system imperfections
    # (drift, gaps, etc.), so we use padding to account for misalignment at boundaries.
    
    for trial_idx in range(num_trials):
        ref_time = trial_data.iloc[trial_idx][reference_label]
        window_start = ref_time + min_start
        window_stop = ref_time + max_stop
        
        # Find velocity samples within trial window
        valid_velocity_mask = (
            (velocity_timestamps >= window_start) & 
            (velocity_timestamps < window_stop)
        )
        
        if not valid_velocity_mask.any():
            # No velocity data available for this trial window
            continue
        
        # Calculate padding needed if velocity tracking starts after window start
        # This accounts for irregular sampling and gaps in the tracking data
        first_velocity_timestamp = np.min(velocity_timestamps[valid_velocity_mask])
        start_padding_len = int(1000 * (first_velocity_timestamp - window_start))
        
        # Ensure padding is within valid range
        start_padding_len = max(0, start_padding_len)
        
        # Extract velocity for this trial
        y_trial = hand_velocity[valid_velocity_mask]
        
        # Calculate how much data we can actually fit
        end_idx = min(start_padding_len + y_trial.shape[0], max_duration)
        actual_len = end_idx - start_padding_len
        
        # Place velocity data in the aligned array
        if actual_len > 0:
            y[start_padding_len:end_idx, trial_idx] = y_trial[:actual_len]
    
    return X, y, behavioral_markers, marker_labels
'''

# ============================================================
# Chowdhury Loader (MAT format - discrete time bins)
# ============================================================

def load_chowdhury_file(
    filename: str,
    reference_label: str = 'idx_goCueTime'
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, np.ndarray]:
    """
    Load Chowdhury .mat file and align neural/behavioral data to a reference marker.
    
    This loader handles discrete time-binned data sampled at 1ms resolution.
    All trials are aligned to a common reference marker and padded to have 
    identical duration based on the earliest start and latest stop times 
    across all trials.
    
    Parameters
    ----------
    filename : str
        Path to .mat file containing neural recordings and behavioral data.
    reference_label : str, optional
        Name of the trial marker to use as alignment reference (t=0 for each trial).
        Default: 'idx_goCueTime'.
    
    Returns
    -------
    spikes : np.ndarray
        Spike raster of shape (time_bins, n_trials, n_neurons).
        Time bins are sampled at 1ms resolution.
    hand_velocity : np.ndarray
        Hand velocity data of shape (time_bins, n_trials, 2).
        Contains [vx, vy] velocity components at 1ms resolution.
    behavioral_markers : pd.DataFrame
        Timing of behavioral events (time bins) relative to reference marker.
        Shape: (n_trials, n_markers). Each value represents time offset from 
        the reference marker (negative = before reference, positive = after).
    marker_labels : np.ndarray
        Names of behavioral markers (columns of behavioral_markers).
    
    Notes
    -----
    - Trials with missing markers (NaN values) are automatically dropped as 
      some behavioral events may not occur in all trials
    - All trials are padded to the same duration to create rectangular arrays
    - Duration is determined by the trial with earliest relative start and 
      latest relative stop across all trials
    - Data is already discretized at 1ms resolution in the source file
    """
    # Load MATLAB file
    mat = sio.loadmat(filename, squeeze_me=True, struct_as_record=False)
    trial_data_struct = mat['trial_data']
    
    # Convert MATLAB struct to dictionary for easier access
    data = {
        field: getattr(trial_data_struct, field) 
        for field in trial_data_struct._fieldnames
    }
    
    # Extract hand velocity data
    hand_velocity = data['vel']
    
    # ========================================
    # STEP 1: Extract and align behavioral markers
    # ========================================
    # Find all marker fields (those containing 'idx' in their name)
    
    marker_labels = np.array([k for k in data.keys() if 'idx' in k])
    behavioral_markers = {}
    
    # For each marker type, extract the first valid occurrence within each trial
    for label in marker_labels[2:]:  # Skip 'idx_startTime' and 'idx_endTime'
        behavioral_markers[label] = []
        
        for start, end in zip(data['idx_startTime'], data['idx_endTime']):
            time_points = data[label]
            
            # Find marker occurrences within this trial's time window
            valid_times = time_points[(time_points > start) & (time_points < end)]
            
            if valid_times.size == 0:
                # No valid marker found in this trial - mark as NaN
                # These trials will be dropped later
                behavioral_markers[label].append(np.nan)
            else:
                # Use the first occurrence
                behavioral_markers[label].append(valid_times[0])
        
        behavioral_markers[label] = np.array(behavioral_markers[label])
    
    # Add trial boundaries
    behavioral_markers['idx_startTime'] = np.array(data['idx_startTime'])
    behavioral_markers['idx_endTime'] = np.array(data['idx_endTime'])
    
    # Create DataFrame and remove trials with missing markers
    trial_data = pd.DataFrame(
        np.column_stack([behavioral_markers[marker] for marker in marker_labels]),
        columns=marker_labels
    )
    trial_data = trial_data.dropna()
    num_trials = len(trial_data)
    
    # ========================================
    # STEP 2: Determine trial alignment window
    # ========================================
    # Find the earliest start and latest stop times relative to the reference marker
    
    min_start = 0  # Most negative time relative to reference (earliest start)
    max_stop = 0   # Most positive time relative to reference (latest stop)
    
    for i in range(num_trials):
        ref_time = trial_data.iloc[i][reference_label]
        start_rel = trial_data.iloc[i]['idx_startTime'] - ref_time
        stop_rel = trial_data.iloc[i]['idx_endTime'] - ref_time
        
        min_start = min(min_start, start_rel)
        max_stop = max(max_stop, stop_rel)
    
    # Total duration in time bins (already at 1ms resolution)
    max_duration = int(max_stop - min_start)
    
    # ========================================
    # STEP 3: Align behavioral markers
    # ========================================
    # Transform all marker times to be relative to reference and shifted to start at 0
    
    reference_col = trial_data[reference_label].to_numpy()
    min_start_col = np.full((num_trials, 1), min_start)
    
    # Calculate: (marker_time - reference_time) - min_start
    behavioral_markers = (
        trial_data.to_numpy() 
        - reference_col[:, np.newaxis] 
        - min_start_col
    )
    
    behavioral_markers = pd.DataFrame(
        behavioral_markers,
        columns=marker_labels
    )
    
    # ========================================
    # STEP 4: Initialize output arrays
    # ========================================
    
    unit_data = np.atleast_2d(data['S1_spikes'])
    num_neurons = unit_data.shape[1]
    
    X = np.zeros((max_duration, num_trials, num_neurons), dtype=np.float32)
    y = np.zeros((max_duration, num_trials, 2), dtype=np.float32)
    
    # ========================================
    # STEP 5: Extract aligned neural and kinematic data
    # ========================================
    # For discrete data, we slice the arrays using the aligned time windows
    
    for trial_idx in range(num_trials):
        ref_time = trial_data.iloc[trial_idx][reference_label]
        
        # Define the slice indices in the original data
        slice_start = int(ref_time + min_start)
        slice_stop = int(ref_time + max_stop)
        
        # Extract spike data for all neurons in this trial
        for neuron_idx in range(num_neurons):
            raster = unit_data[:, neuron_idx]
            trial_raster = raster[slice_start:slice_stop]
            
            # Handle potential length mismatch due to integer rounding
            actual_len = min(len(trial_raster), max_duration)
            X[trial_idx, :actual_len, neuron_idx] = trial_raster[:actual_len]
        
        # Extract hand velocity data for this trial
        # Note: velocity is one sample shorter than position due to np.diff
        # Adjust slice accordingly
        velocity_slice_start = slice_start
        velocity_slice_stop = min(slice_stop, len(hand_velocity))
        
        y_trial = hand_velocity[velocity_slice_start:velocity_slice_stop]
        actual_len = min(len(y_trial), max_duration)
        y[:actual_len, trial_idx] = y_trial[:actual_len]
    
    return X, y, behavioral_markers, marker_labels


# ============================================================
# Dispatcher
# ============================================================

def get_all_data_from_file(
    experiment: str,
    filename: str,
    reference_label: str = None
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, np.ndarray]:
    """
    Load neural and behavioral data using the appropriate loader for the experiment.
    
    Parameters
    ----------
    experiment : str
        Name of the experiment dataset. Options: 'Jenkins', 'Chowdhury'.
    filename : str
        Path to the data file.
    reference_label : str, optional
        Name of the behavioral marker to use as alignment reference.
        If None, uses the default for each experiment type.
    
    Returns
    -------
    spikes : np.ndarray
        Neural spike data of shape (time, trials, neurons).
    hand_velocity : np.ndarray
        Hand velocity of shape (time, trials, 2).
    behavioral_markers : pd.DataFrame
        Timing of behavioral events relative to reference marker.
    marker_labels : np.ndarray
        Names of the behavioral markers.
    
    Raises
    ------
    ValueError
        If experiment type is not recognized.
    
    Examples
    --------
    >>> # Load Jenkins data with default reference (go_cue_time)
    >>> spikes, velocity, markers, labels = get_all_data_from_file(
    ...     'Jenkins', 'data.nwb'
    ... )
    
    >>> # Load Chowdhury data with custom reference
    >>> spikes, velocity, markers, labels = get_all_data_from_file(
    ...     'Chowdhury', 'data.mat', reference_label='idx_targetTime'
    ... )
    """
    if experiment == 'Jenkins':
        if reference_label is None:
            reference_label = 'go_cue_time'
        return load_jenkins_file(filename, reference_label)
    
    elif experiment == 'Chowdhury':
        if reference_label is None:
            reference_label = 'idx_goCueTime'
        return load_chowdhury_file(filename, reference_label)
    
    else:
        raise ValueError(
            f"Unknown experiment type: '{experiment}'. "
            f"Supported types: 'Jenkins', 'Chowdhury'"
        )