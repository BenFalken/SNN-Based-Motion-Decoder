from __future__ import annotations
import os
import tkinter as tk
from tkinter import filedialog
from typing import Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import zscore


# ======================================================================
# File Selection
# ======================================================================

def select_file(experiment: str) -> str:
    """
    Prompt user to select a specific data file within a given experiment.

    Parameters
    ----------
    experiment : str
        Name of the experiment folder inside `data/`.

    Returns
    -------
    str
        Full path to the selected file.
    """
    base_path = os.path.join(os.getcwd(), "data", experiment)
    root = tk.Tk()
    root.withdraw()
    filename = filedialog.askopenfilename(initialdir=base_path)
    return filename


# ======================================================================
# Data Binning and Windowing
# ======================================================================

def bin_and_window_data(
    X: np.ndarray,
    y: np.ndarray,
    LFP: np.ndarray,
    bin_size_ms: int,
    window_size_ms: int,
    stride_ms: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bin and window neural, behavioral, and LFP data while preserving
    (trials, time, features) structure.

    Parameters
    ----------
    X : np.ndarray
        Spike data, shape (n_trials, n_timepoints, n_neurons).
    y : np.ndarray
        Hand velocity data, shape (n_trials, n_timepoints, 2).
    LFP : np.ndarray
        LFP data, shape (n_trials, n_timepoints, n_channels).
    bin_size_ms : int
        Duration of each bin in milliseconds.
    window_size_ms : int
        Size of the temporal window to analyze in milliseconds.
    stride_ms : int
        Stride (overlap step) in milliseconds.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Binned spike, velocity, and LFP arrays.
    """
    # Restrict data to window

    if not window_size_ms:
        window_size_ms = X.shape[1]

    X_win, y_win, LFP_win = X[:, :window_size_ms, :], y[:, :window_size_ms, :], LFP[:, :window_size_ms, :]
    total_bins = np.arange(0, X_win.shape[1] - bin_size_ms, stride_ms)

    def bin_data(data: np.ndarray, func: callable) -> np.ndarray:
        binned = np.empty((data.shape[0], len(total_bins), data.shape[2]), dtype=float)
        for i, start in enumerate(total_bins):
            segment = data[:, start:start + bin_size_ms, :]
            binned[:, i, :] = func(segment, axis=1)
        return binned

    X_binned = bin_data(X_win, np.sum)
    y_binned = bin_data(y_win, np.mean)
    LFP_binned = bin_data(LFP_win, np.mean)

    return X_binned, y_binned, LFP_binned


# ======================================================================
# Decoder Dataset Construction
# ======================================================================

def create_decoder_dataset(
    X_windowed: np.ndarray,
    y_windowed: np.ndarray,
    LFP_windowed: np.ndarray,
    prediction_offset: int = 0
) -> Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]:
    """
    Construct input-target pairs for a decoding model.

    Parameters
    ----------
    X_windowed : np.ndarray
        Binned or windowed spike data, shape (n_trials, n_bins, n_neurons).
    y_windowed : np.ndarray
        Binned or windowed hand velocity data, shape (n_trials, n_bins, 2).
    LFP_windowed : np.ndarray
        Binned or windowed LFP data, shape (n_trials, n_bins, n_channels).
    prediction_offset : int, optional
        Temporal offset (in bins) between neural data and target velocity.

    Returns
    -------
    Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]
        ((X_inputs, LFP_inputs), targets)
    """
    target_bin = min(y_windowed.shape[1] - 1 + prediction_offset, y_windowed.shape[1] - 1)
    targets = y_windowed[:, target_bin, :]

    print("\nDecoder dataset created:")
    print(f"  Spike inputs: {X_windowed.shape}")
    print(f"  LFP inputs: {LFP_windowed.shape}")
    print(f"  Targets: {targets.shape}")

    return (X_windowed, LFP_windowed), targets


# ======================================================================
# Data Processing Helpers
# ======================================================================

def process_velocities(y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute trial-averaged hand velocity and its SEM.

    Parameters
    ----------
    y : np.ndarray
        Hand velocity data (n_trials, n_timepoints, 2).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Time points, mean X velocity, mean Y velocity, X SEM, Y SEM.
    """
    T = np.arange(y.shape[1])
    mean_x, mean_y = np.mean(y[:, :, 0], axis=0), np.mean(y[:, :, 1], axis=0)
    sem_x = 2 * np.std(y[:, :, 0], axis=0) / np.sqrt(y.shape[0])
    sem_y = 2 * np.std(y[:, :, 1], axis=0) / np.sqrt(y.shape[0])
    return T, mean_x, mean_y, sem_x, sem_y


def process_LFP(LFP: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute mean and SEM for LFP data across trials.

    Parameters
    ----------
    LFP : np.ndarray
        Local field potential data (n_trials, n_timepoints, n_channels).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Mean and SEM arrays, both of shape (n_timepoints, n_channels).
    """
    mean = np.mean(LFP, axis=0)
    sem = np.std(LFP, axis=0) / np.sqrt(LFP.shape[0])
    return mean, sem


# ======================================================================
# Plotting
# ======================================================================

def plot_neural_behavioral_data(
    X: np.ndarray,
    y: np.ndarray,
    LFP: np.ndarray,
    behavioral_markers: pd.DataFrame,
    marker_labels: list[str],
    neuron_idx: int = 40,
    binned: bool = False
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Visualize spike rasters, mean velocity, and LFP traces aligned to behavioral markers.

    Parameters
    ----------
    X : np.ndarray
        Spike data (n_trials, time_bins, n_neurons).
    y : np.ndarray
        Hand velocity (n_trials, time_bins, 2).
    LFP : np.ndarray
        LFP data (n_trials, time_bins, n_channels).
    behavioral_markers : pd.DataFrame
        DataFrame with behavioral event timestamps per trial.
    marker_labels : list of str
        Labels for each behavioral marker.
    neuron_idx : int, default=40
        Index of neuron to visualize.
    binned : bool, default=False
        If True, display binned spike intensities instead of rasters.

    Returns
    -------
    Tuple[plt.Figure, np.ndarray]
        Matplotlib figure and axes array.
    """
    T, mean_x, mean_y, err_x, err_y = process_velocities(y)
    mean_LFP, err_LFP = process_LFP(LFP)

    clmap = plt.cm.tab10(np.linspace(0, 1, len(marker_labels)))
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(
        f"Neural and Behavioral Data - Unit {neuron_idx}" + (" (binned)" if binned else ""),
        fontsize=14, fontweight='bold', y=0.995
    )

    # ----------------- Spike Raster -----------------
    ax1 = axes[0]
    if not binned:
        for trial in range(X.shape[0]):
            spikes = np.where(X[trial, :, neuron_idx])[0]
            ax1.scatter(spikes, np.full_like(spikes, trial), color='black', s=1, alpha=0.6, rasterized=True)
    else:
        ax1.pcolormesh(zscore(X[:, :, neuron_idx], axis=1), vmin=-3, vmax=3, cmap='RdBu_r')

    if behavioral_markers:
        # Overlay markers
        for trial in range(X.shape[0]):
            for i, marker in enumerate(behavioral_markers.iloc[trial].to_numpy()):
                if not np.isnan(marker):
                    ax1.scatter(marker, trial, color=clmap[i], s=8, alpha=0.6, edgecolors='white', linewidths=0.3)

    ax1.set_ylabel("Trial", fontsize=11, fontweight='bold')
    ax1.set_title("Spike Raster with Behavioral Events", loc='left', fontsize=11)
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax1.legend(
        handles=[plt.Line2D([0], [0], marker='o', color='w',
                            markerfacecolor=clmap[i], markersize=6,
                            label=marker_labels[i], markeredgecolor='white', markeredgewidth=0.3)
                 for i in range(len(marker_labels))],
        loc='upper right', fontsize=8, framealpha=0.9
    )

    # ----------------- Hand Velocity -----------------
    ax2 = axes[1]
    ax2.plot(T, mean_x, color='#2E86AB', label='X Velocity', linewidth=1.5)
    ax2.fill_between(T, mean_x - err_x, mean_x + err_x, color='#2E86AB', alpha=0.3)
    ax2.plot(T, mean_y, color='#A23B72', label='Y Velocity', linewidth=1.5)
    ax2.fill_between(T, mean_y - err_y, mean_y + err_y, color='#A23B72', alpha=0.3)
    ax2.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax2.set_ylabel("Velocity (a.u.)", fontsize=11, fontweight='bold')
    ax2.set_title("Mean Hand Velocity ± SEM", loc='left', fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    # ----------------- LFP -----------------
    ax3 = axes[2]
    ax3.plot(T, mean_LFP[:, neuron_idx], linewidth=1.5, label='Mean LFP')
    ax3.fill_between(T, mean_LFP[:, neuron_idx] - err_LFP[:, neuron_idx],
                     mean_LFP[:, neuron_idx] + err_LFP[:, neuron_idx], alpha=0.3)
    ax3.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax3.set_xlabel("Time (ms)" if not binned else "Bins", fontsize=11, fontweight='bold')
    ax3.set_ylabel("LFP Amplitude (μV)", fontsize=11, fontweight='bold')
    ax3.set_title("Mean Local Field Potential ± SEM", loc='left', fontsize=11)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    # ----------------- Final touch -----------------
    for ax in axes:
        ax.axvline(0, color='red', linestyle='--', linewidth=1.2, alpha=0.7)

    plt.tight_layout()
    return fig, axes
