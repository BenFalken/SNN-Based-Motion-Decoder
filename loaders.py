import numpy as np
import h5py
import scipy.io as sio
from typing import List, Tuple
from pynwb import NWBHDF5IO

# ============================================================
# Shared Alignment Helper
# ============================================================

def align_to_reference(
    spike_trials: List[np.ndarray],
    marker_trials: List[np.ndarray],
    marker_labels: np.ndarray,
    reference_label: str
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Align all spike times and marker times to a reference marker (e.g. 'go_cue_time').

    Parameters
    ----------
    spike_trials : list of np.ndarray
        List of per-trial spike times (absolute).
    marker_trials : list of np.ndarray
        List of per-trial marker times (absolute).
    marker_labels : np.ndarray
        Marker labels array (must contain `reference_label`).
    reference_label : str
        Name of the marker to align all times to.

    Returns
    -------
    aligned_spikes : list of np.ndarray
        Spike times shifted so that the reference marker is at t = 0.
    aligned_markers : list of np.ndarray
        Marker times shifted by the same amount.
    """
    ref_idx = np.where(marker_labels == reference_label)[0]
    if ref_idx.size == 0:
        raise ValueError(f"Reference marker '{reference_label}' not found in marker_labels.")
    ref_idx = ref_idx[0]

    aligned_spikes, aligned_markers = [], []
    for spikes, markers in zip(spike_trials, marker_trials):
        ref_time = markers[ref_idx]
        aligned_spikes.append(spikes - ref_time)
        aligned_markers.append(markers - ref_time)
    return aligned_spikes, aligned_markers

# ============================================================
# Diomedi Loader
# ============================================================

def load_diomedi_file(
    filename: str,
    reference_label: str = 'Move out on'
) -> Tuple[List[List[np.ndarray]], List[List[np.ndarray]], np.ndarray, int]:
    """
    Load Diomedi-format HDF5 neural data and align to a reference marker.

    Parameters
    ----------
    filename : str
        Path to the .h5 data file.
    reference_label : str
        Marker name to align spike and marker times to (default: 'go_cue_time').

    Returns
    -------
    spikes_per_unit : list
        Outer list over units; inner list over trials, each is np.ndarray of spike times (ms, relative).
    markers_per_unit : list
        Outer list over units; inner list over trials, each is np.ndarray of marker times (ms, relative).
    marker_labels : np.ndarray
        Array of marker names.
    num_units : int
        Number of neural units.
    """
    all_spikes, all_markers = [], []

    with h5py.File(filename, 'r') as f:
        unit_names = list(f['DATA'])
        num_units = len(unit_names)

        # Marker labels are stored in the attributes of the first unit/condition/trial
        marker_labels = f['/DATA/unit_001/condition_01/trial_01/event_markers'].attrs['Marker labels']

        for unit in unit_names:
            unit_spikes, unit_markers = [], []
            group_name = f'/DATA/{unit}'

            def collect(name, node):
                if isinstance(node, h5py.Dataset):
                    if name.endswith('event_markers'):
                        unit_markers.append(node[()])
                    elif name.endswith('spike_trains'):
                        unit_spikes.append(node[()])

            f[group_name].visititems(collect)

            # Align to reference marker
            aligned_spikes, aligned_markers = align_to_reference(
                unit_spikes, unit_markers, marker_labels, reference_label
            )

            all_spikes.append(aligned_spikes)
            all_markers.append(aligned_markers)

    return all_spikes, all_markers, marker_labels, num_units

# ============================================================
# Jenkins Loader
# ============================================================

def load_jenkins_file(
    filename: str,
    reference_label: str = 'go_cue_time'
) -> Tuple[List[List[np.ndarray]], List[List[np.ndarray]], np.ndarray, int]:
    """
    Load Jenkins NWB file, extract spike times and trial markers, and align to reference marker.

    Parameters
    ----------
    filename : str
        Path to NWB file.
    reference_label : str
        Marker name to align to (default: 'go_cue_time').

    Returns
    -------
    spikes_per_unit, markers_per_unit, marker_labels, num_units
    """
    io = NWBHDF5IO(filename, 'r')
    trial_data = io.read()
    trials_df = trial_data.trials.to_dataframe()
    units_df = trial_data.units.to_dataframe()
    io.close()

    # Select relevant columns for markers
    marker_labels = trials_df.columns[[0, 1, 2, 3, 5, 6]].to_numpy()
    marker_matrix = trials_df[marker_labels].to_numpy()
    num_trials = marker_matrix.shape[0]
    num_units = len(units_df)

    # Build spike lists
    spikes_per_unit = [[] for _ in range(num_units)]
    for trial_idx in range(num_trials):
        start_time, stop_time = marker_matrix[trial_idx, 0], marker_matrix[trial_idx, 1]
        for unit_idx in range(num_units):
            spike_times = units_df.iloc[unit_idx]['spike_times']
            trial_spikes = spike_times[(spike_times >= start_time) & (spike_times < stop_time)]
            spikes_per_unit[unit_idx].append(trial_spikes)

    # Duplicate markers per unit
    markers_per_unit = [marker_matrix.copy() for _ in range(num_units)]

    # Align to reference marker for each unit
    for unit_idx in range(num_units):
        spikes_per_unit[unit_idx], markers_per_unit[unit_idx] = align_to_reference(
            spikes_per_unit[unit_idx],
            list(markers_per_unit[unit_idx]),  # convert array of rows to list of rows
            marker_labels,
            reference_label
        )

    # Convert to 1000 Hz
    for unit_idx in range(num_units):
        for trial, spike_data in enumerate(spikes_per_unit[unit_idx]):
            spikes_per_unit[unit_idx][trial] = spike_data*1000
            markers_per_unit[unit_idx][trial] = markers_per_unit[unit_idx][trial]*1000

    return spikes_per_unit, markers_per_unit, marker_labels, num_units

# ============================================================
# Chowdhury Loader
# ============================================================

def load_chowdhury_file(
    filename: str,
    reference_label: str = 'idx_goCueTime'
) -> Tuple[List[List[np.ndarray]], List[List[np.ndarray]], np.ndarray, int]:
    """
    Load Chowdhury .mat file, extract spike times and marker times per trial, and align to reference marker.

    Parameters
    ----------
    filename : str
        Path to the .mat file.
    reference_label : str
        Marker name to align to (default: 'idx_goCueTime').

    Returns
    -------
    spikes_per_unit, markers_per_unit, marker_labels, num_units
    """
    mat = sio.loadmat(filename, squeeze_me=True, struct_as_record=False)
    trial_data = mat['trial_data']

    # Convert MATLAB struct to dict
    data = {field: getattr(trial_data, field) for field in trial_data._fieldnames}

    # Marker labels are all keys that contain 'idx'
    marker_labels = np.array([k for k in data.keys() if 'idx' in k])
    start_times = np.atleast_1d(data['idx_startTime'])
    end_times = np.atleast_1d(data['idx_endTime'])
    spike_matrix = np.atleast_2d(data['S1_spikes'])
    num_units = spike_matrix.shape[1]

    # Build marker trials (absolute times)
    marker_trials = []
    for start, end in zip(start_times, end_times):
        trial_markers = []
        for label in marker_labels:
            idx_times = np.atleast_1d(data[label])
            valid_idx = idx_times[(idx_times >= start) & (idx_times <= end)]
            if valid_idx.size > 0:
                trial_markers.append(valid_idx[0] - start)
            else:
                trial_markers.append(np.nan)
        marker_trials.append(np.array(trial_markers))

    # Build spike trials
    spikes_per_unit = []
    for unit_idx in range(num_units):
        unit_trials = []
        for start, end in zip(start_times, end_times):
            trial_spikes = np.where(spike_matrix[start:end, unit_idx])[0]
            unit_trials.append(trial_spikes)
        spikes_per_unit.append(unit_trials)

    # Duplicate markers per unit
    markers_per_unit = [marker_trials.copy() for _ in range(num_units)]

    # Align each unit to reference marker
    for unit_idx in range(num_units):
        spikes_per_unit[unit_idx], markers_per_unit[unit_idx] = align_to_reference(
            spikes_per_unit[unit_idx],
            markers_per_unit[unit_idx],
            marker_labels,
            reference_label
        )

    return spikes_per_unit, markers_per_unit, marker_labels, num_units

# ============================================================
# Dispatcher
# ============================================================

def get_all_data_from_file(experiment: str, filename: str):
    """
    Dispatch to the appropriate loader based on experiment name.
    """
    if experiment == 'Diomedi':
        return load_diomedi_file(filename)
    elif experiment == 'Jenkins':
        return load_jenkins_file(filename)
    elif experiment == 'Chowdhury':
        return load_chowdhury_file(filename)
    else:
        raise ValueError(f"Unknown experiment type: {experiment}")