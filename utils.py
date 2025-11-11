from imports import *
from const import *
from typing import Tuple, Optional

def select_file(experiment):
    '''Prompts user to select the exact file they want'''
    current_dir = os.getcwd()
    # Path to open the file dialog window in the data branch
    path_folder_data = os.path.join(current_dir, f"data/{experiment}")
    # Select H5 dataset file
    root = tk.Tk()
    root.withdraw()
    filename = filedialog.askopenfilename(initialdir=path_folder_data)
    return filename

def bin_and_window_data(
    X: np.ndarray,
    y: np.ndarray,
    LFP: np.ndarray,
    behavioral_markers: np.ndarray,
    bin_size_ms: int = 50,
    window_size_ms: int = 500,
    stride_ms: Optional[int] = None,
    original_bin_ms: int = 1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Bin and window data while maintaining (trials, time, features) ordering.
    """

    # Default stride
    if stride_ms is None:
        stride_ms = bin_size_ms

    # Derived parameters
    bins_per_bin = bin_size_ms // original_bin_ms
    bins_per_window = window_size_ms // bin_size_ms
    bins_per_stride = stride_ms // bin_size_ms

    n_trials, n_time_bins, n_neurons = X.shape
    n_lfp_channels = LFP.shape[2]

    print("Binning configuration:")
    print(f"  Original resolution: {original_bin_ms}ms")
    print(f"  Target bin size: {bin_size_ms}ms ({bins_per_bin} original bins)")
    print(f"  Window size: {window_size_ms}ms ({bins_per_window} target bins)")
    print(f"  Stride: {stride_ms}ms ({bins_per_stride} target bins)")
    print(f"  Window overlap: {window_size_ms - stride_ms}ms")

    # ========================================
    # STEP 1: Bin along time axis
    # ========================================
    n_binned_time = n_time_bins // bins_per_bin

    def bin_data(data, mode="sum"):
        """Bin data along time axis."""
        data = data[:, :n_binned_time * bins_per_bin, ...]  # trim excess
        reshaped = data.reshape(n_trials, n_binned_time, bins_per_bin, -1)
        if mode == "sum":
            return reshaped.sum(axis=2)
        elif mode == "mean":
            return reshaped.mean(axis=2)
        else:
            raise ValueError("mode must be 'sum' or 'mean'")

    X_binned = bin_data(X, mode="sum")
    y_binned = bin_data(y, mode="mean")
    LFP_binned = bin_data(LFP, mode="mean")

    behavioral_markers_windowed = behavioral_markers / bin_size_ms

    print("\nBinned data shapes:")
    print(f"  X_binned: {X_binned.shape}")
    print(f"  y_binned: {y_binned.shape}")
    print(f"  LFP_binned: {LFP_binned.shape}")

    # ========================================
    # STEP 2: Sliding windows
    # ========================================
    windows_per_trial = (n_binned_time - bins_per_window) // bins_per_stride + 1
    total_windows = n_trials * windows_per_trial

    X_windowed = np.zeros((total_windows, bins_per_window, n_neurons), dtype=np.float32)
    y_windowed = np.zeros((total_windows, bins_per_window, 2), dtype=np.float32)
    LFP_windowed = np.zeros((total_windows, bins_per_window, n_lfp_channels), dtype=np.float32)
    trial_indices = np.zeros(total_windows, dtype=np.int32)
    window_start_times = np.zeros(total_windows, dtype=np.float32)

    window_idx = 0
    for trial_idx in range(n_trials):
        for start_bin in range(0, n_binned_time - bins_per_window + 1, bins_per_stride):
            end_bin = start_bin + bins_per_window

            X_windowed[window_idx] = X_binned[trial_idx, start_bin:end_bin, :]
            y_windowed[window_idx] = y_binned[trial_idx, start_bin:end_bin, :]
            LFP_windowed[window_idx] = LFP_binned[trial_idx, start_bin:end_bin, :]

            trial_indices[window_idx] = trial_idx
            window_start_times[window_idx] = start_bin * bin_size_ms
            window_idx += 1

    # Trim to actual number
    X_windowed = X_windowed[:window_idx]
    y_windowed = y_windowed[:window_idx]
    LFP_windowed = LFP_windowed[:window_idx]
    trial_indices = trial_indices[:window_idx]
    window_start_times = window_start_times[:window_idx]

    print("\nWindowed data shapes:")
    print(f"  X_windowed: {X_windowed.shape}")
    print(f"  y_windowed: {y_windowed.shape}")
    print(f"  LFP_windowed: {LFP_windowed.shape}")
    print(f"  Windows per trial: {windows_per_trial}")
    print(f"  Total windows: {window_idx}")

    metadata = {
        "trial_indices": trial_indices,
        "window_start_times": window_start_times,
        "bin_size_ms": bin_size_ms,
        "window_size_ms": window_size_ms,
        "stride_ms": stride_ms,
        "bins_per_window": bins_per_window
    }

    return X_windowed, y_windowed, LFP_windowed, behavioral_markers_windowed, metadata

def create_decoder_dataset(
    X_windowed: np.ndarray,
    y_windowed: np.ndarray,
    LFP_windowed: np.ndarray,
    metadata: dict,
    prediction_offset: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    
    n_windows = X_windowed.shape[0]
    
    # Determine target time point
    target_bin = min(X_windowed.shape[1] - 1 + prediction_offset, X_windowed.shape[1] - 1)
    
    # Extract targets (velocity at target time)
    targets = y_windowed[:, target_bin, :]

    print(f"\nDecoder dataset created:")
    print(f"  Spike inputs: {X_windowed.shape}")
    print(f"  LFP inputs: {LFP_windowed.shape}")
    print(f"  Targets: {targets.shape}")
    return (X_windowed, LFP_windowed), targets

def process_velocities(y):
    '''Return trial-averaged velocity traces'''
    T = np.arange(y.shape[1])
    # Compute mean velocity across trial (centered at cue time)
    mean_x_vel = np.mean(y[:, :, 0], axis=0)
    mean_y_vel = np.mean(y[:, :, 1], axis=0)
    # Compute velocity errors across trial (centered at cue time)
    x_vel_err = 2*np.std(y[:, :, 0], axis=0)/np.sqrt(y.shape[0])
    y_vel_err = 2*np.std(y[:, :, 1], axis=0)/np.sqrt(y.shape[0])
    return T, mean_x_vel, mean_y_vel, x_vel_err, y_vel_err

def process_LFP(LFP):
    mu = np.mean(LFP, axis=0)
    err = np.std(LFP, axis=0) / np.sqrt(LFP.shape[0])
    return mu, err

def plot_neural_behavioral_data(X, y, LFP, behavioral_markers, marker_labels, neuron_idx=40, binned=False):
    """
    Create a comprehensive visualization combining spike rasters, hand velocity, 
    and LFP data with shared time axes.
    
    Parameters
    ----------
    X : np.ndarray
        Spike data (n_trials, time_bins, n_neurons)
    y : np.ndarray
        Hand velocity (n_trials, time_bins, 2)
    LFP : np.ndarray
        LFP data (n_trials, time_bins, n_electrodes, n_channels)
    behavioral_markers : pd.DataFrame
        Behavioral event timestamps (n_trials, n_markers)
    marker_labels : np.ndarray
        Names of behavioral markers
    neuron_idx : int
        Index of neuron to display
    """
    # Process data
    T, mean_x_vel, mean_y_vel, x_vel_err, y_vel_err = process_velocities(y)
    mean_LFP, LFP_err = process_LFP(LFP)
    
    # Create color map for behavioral markers
    clmap = plt.cm.tab10(np.linspace(0, 1, len(marker_labels)))
    
    # Create figure with 3 subplots sharing x-axis
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    title = f'Neural and Behavioral Data - Unit {neuron_idx}'
    if binned:
        title = f'{title} (binned)'

    fig.suptitle(title, 
                 fontsize=14, fontweight='bold', y=0.995)
    
    # ==========================================
    # Panel 1: Spike Raster
    # ==========================================
    ax1 = axes[0]
    
    # Plot spike raster for selected neuron
    if not binned:
        for trial in range(X.shape[0]):
            spike_times = np.where(X[trial, :, neuron_idx])[0]
            ax1.scatter(spike_times, trial * np.ones(spike_times.size), 
                    color='black', s=1, alpha=0.6, rasterized=True)
    else:
        ax1.pcolormesh(zscore(X[:, :, neuron_idx], axis=1), vmin=-3, vmax=3, cmap='RdBu_r')
    
    # Overlay behavioral markers
    for trial in range(X.shape[0]):
        markers = behavioral_markers.iloc[trial].to_numpy()
        for i, marker in enumerate(markers):
            if binned:
                marker_alpha = 0.3
            else:
                marker_alpha = 0.8
            if not np.isnan(marker):  # Skip NaN markers
                ax1.scatter(marker, trial, color=clmap[i], s=8, 
                           alpha=marker_alpha, edgecolors='white', linewidths=0.3,
                           zorder=10)
    
    ax1.set_ylabel('Trial Number', fontsize=11, fontweight='bold')
    ax1.set_title('Spike Raster with Behavioral Events', fontsize=11, loc='left')
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax1.set_ylim(-0.5, X.shape[0] - 0.5)
    
    # Create custom legend for behavioral markers
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                  markerfacecolor=clmap[i], markersize=6,
                                  label=marker_labels[i], markeredgecolor='white',
                                  markeredgewidth=0.3)
                      for i in range(len(marker_labels))]
    ax1.legend(handles=legend_elements, loc='upper right', 
              fontsize=8, framealpha=0.9, ncol=1)
    
    # ==========================================
    # Panel 2: Hand Velocity
    # ==========================================
    ax2 = axes[1]
    
    # Plot X velocity
    ax2.plot(T, mean_x_vel, linewidth=1.5, label='X Velocity', 
            color='#2E86AB', alpha=0.9)
    ax2.fill_between(T, mean_x_vel - x_vel_err, mean_x_vel + x_vel_err, 
                     alpha=0.3, color='#2E86AB')
    
    # Plot Y velocity
    ax2.plot(T, mean_y_vel, linewidth=1.5, label='Y Velocity', 
            color='#A23B72', alpha=0.9)
    ax2.fill_between(T, mean_y_vel - y_vel_err, mean_y_vel + y_vel_err, 
                     alpha=0.3, color='#A23B72')
    
    # Add zero reference line
    ax2.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    
    ax2.set_ylabel('Velocity (a.u.)', fontsize=11, fontweight='bold')
    ax2.set_title('Mean Hand Velocity ± SEM', fontsize=11, loc='left')
    ax2.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # ==========================================
    # Panel 3: LFP Signal
    # ==========================================
    ax3 = axes[2]
    
    # Plot mean LFP
    ax3.plot(T, mean_LFP[:, neuron_idx], linewidth=1.5, label='Mean LFP', 
            color='#F18F01', alpha=0.9)
    ax3.fill_between(T, mean_LFP[:, neuron_idx] - LFP_err[:, neuron_idx], mean_LFP[:, neuron_idx] + LFP_err[:, neuron_idx], 
                     alpha=0.3, color='#F18F01')
    
    # Add zero reference line
    ax3.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    
    if not binned:
        ax3.set_xlabel('Time (ms)', fontsize=11, fontweight='bold')
    if not binned:
        ax3.set_xlabel('Bins', fontsize=11, fontweight='bold')

    ax3.set_ylabel('LFP Amplitude (μV)', fontsize=11, fontweight='bold')
    ax3.set_title('Mean Local Field Potential ± SEM', fontsize=11, loc='left')
    ax3.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax3.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # ==========================================
    # Final adjustments
    # ==========================================
    plt.tight_layout()
    
    # Add vertical line at t=0 (reference marker) across all panels
    for ax in axes:
        ax.axvline(x=0, color='red', linestyle='--', linewidth=1.5, 
                  alpha=0.7, label='Reference (Go Cue)', zorder=5)
    
    return fig, axes