import torch
import numpy as np
from scipy.signal import welch
import cupy as cp
from cupyx.scipy.signal import welch
from torch.utils.dlpack import to_dlpack, from_dlpack
import os

from metrics.MMD import mmd_
from metrics.RMSE import RMSE
from metrics.PRD import PRD

from metrics.Metric import run_metric

import matplotlib.pyplot as plt


def PSD(signal: torch.Tensor, fs=100.0, nperseg=512, noverlap=256, nfft=512):
    """
    Computes the normalized Power Spectral Density (PSD) for each channel of the input signal using Welch's method on the GPU.
    Handles zero-power signals by setting normalized PSD to zero at all frequencies.

    Args:
        signal (torch.Tensor): Input signal tensor of shape (N, C, T) or (C, T).
        fs (float, optional): Sampling frequency of the signal in Hz. Default is 100.0.
        nperseg (int, optional): Length of each segment for Welch's method. Default is 512.
        noverlap (int, optional): Number of points to overlap between segments. Default is 256 (50%).
        nfft (int, optional): Number of FFT points. Default is 512.

    Returns:
        tuple:
            Pxx_norm (torch.Tensor): Normalized PSD estimates of shape (N, C, nfft//2+1).
            f (torch.Tensor): Frequency bins of shape (nfft//2+1,).
    """
    if signal.ndim == 2:
        signal = signal.unsqueeze(0)

    # Check if signal is torch else move to torch
    if not isinstance(signal, torch.Tensor):
        signal = torch.tensor(signal, dtype=torch.float32)

    # Check if signal is already on GPU, move if needed
    device = signal.device
    if device.type != 'cuda':
        x = signal.contiguous().cuda()
    else:
        x = signal.contiguous()

    # Convert to CuPy array
    x_cupy = cp.fromDlpack(to_dlpack(x))

    # Call CuPy's welch (runs on GPU)
    f_cupy, Pxx_cupy = welch(x_cupy,
                            fs=fs,
                            window=cp.hamming(nperseg),
                            nperseg=nperseg,
                            noverlap=noverlap,
                            nfft=nfft)
 
    # Convert results back to torch.Tensor on the same device
    f = from_dlpack(f_cupy.toDlpack()).to(device)
    Pxx = from_dlpack(Pxx_cupy.toDlpack()).to(device)

    # FIXED: Handle zero-power signals
    total_power = torch.sum(Pxx, dim=-1, keepdim=True)
    min_power = 1e-8
    
    # Create mask for zero-power signals
    zero_power_mask = total_power < min_power
    
    # Safe division: avoid division by zero
    safe_total_power = torch.where(zero_power_mask, 
                                   torch.ones_like(total_power), 
                                   total_power)
    
    # Normalize
    Pxx_norm = Pxx / safe_total_power
    
    # Set zero-power signals to zero at all frequencies
    Pxx_norm = torch.where(zero_power_mask, torch.zeros_like(Pxx_norm), Pxx_norm)

    return Pxx_norm, f


def PSD_MMD(X, Y):
    """
    Computes the Maximum Mean Discrepancy (MMD) between two PSD results.

    Args:
        X (torch.Tensor): First dataset tensor of shape (N, C, T).
        Y (torch.Tensor): Second dataset tensor of shape (N, C, T).

    Returns:
        float: MMD value.
    """
    # visualize_psd(X[0], Y[0])
    psd_result1, _ = PSD(X)
    psd_result2, _ = PSD(Y)
    mmd_value = mmd_(psd_result1, psd_result2)
    return mmd_value


def PSD_RMSE(X, Y):
    """
    Computes the Root Mean Square Error (RMSE) between two PSD results.

    Args:
        X (torch.Tensor): First dataset tensor of shape (N, C, T).
        Y (torch.Tensor): Second dataset tensor of shape (N, C, T).

    Returns:
        float: RMSE value.
    """
    # visualize_psd(X[0], Y[0])
    psd_result1, _ = PSD(X)
    psd_result2, _ = PSD(Y)
    rmse_value = RMSE(psd_result1, psd_result2)
    return rmse_value


def PSD_PRD(X, Y):
    """
    Computes the Power Ratio Density (PRD) between two PSD results.

    Args:
        X (torch.Tensor): First dataset tensor of shape (N, C, T).
        Y (torch.Tensor): Second dataset tensor of shape (N, C, T).

    Returns:
        float: PRD value.
    """
    # print("PSD-PRD")
    # visualize_psd(X[0], Y[1])
    # visualize_batch_avg_psd(X, Y)
    psd_result1, _ = PSD(X)
    psd_result2, _ = PSD(Y)
    prd_value = PRD(psd_result1, psd_result2)
    return prd_value


def visualize_psd(x, y, save_path='psd'):
    """
    Visualizes the Power Spectral Density (PSD) of two signals' first channel (0)
    with thesis-quality styling.

    Args:
        x (torch.Tensor): First signal tensor of shape (C, T).
        y (torch.Tensor): Second signal tensor of shape (C, T).
        save_path (str): Base path for saving figures (without extension).
    """
    # Set up publication-quality figure styling
    plt.rcParams.update({
        'font.family': 'DejaVu Serif',  # Professional serif font
        'mathtext.fontset': 'cm',       # Computer Modern for math text
        'font.size': 11,
        'axes.labelsize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 16,
        'figure.figsize': (11, 4),      # Adjusted width for 2 subplots
        'figure.dpi': 300,              # High DPI for print
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',        # Avoid wasted space
        'savefig.pad_inches': 0.05,     # Minimal padding
    })
    
    # Colorblind-friendly colors
    colors = ['#000000', '#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#9467BD', '#8C564B']
    
    x = x
    y = y

    # Calculate PSD
    psd_x, freq = PSD(x)
    # Print frequency bin width
    if freq.numel() > 1:
        bin_width = (freq[1] - freq[0]).item()
        print(f"Frequency bin width: {bin_width:.4f} Hz")
    else:
        print("Frequency bin width: N/A (only one frequency bin)")
    psd_y, _ = PSD(y)

    # Print max absolute error between PSDs (first channel)
    max_abs_error = torch.max(torch.abs(psd_x[0, 0, :] - psd_y[0, 0, :])).item()
    print(f"Max absolute error between PSDs (channel 0): {max_abs_error:.6f}")

    # Calculate percentage overlap between the two PSDs (first channel)
    min_psd = torch.min(psd_x[0, 0, :], psd_y[0, 0, :])
    overlap = torch.sum(min_psd).item()
    # Both PSDs are normalized, so their sum is 1
    percentage_overlap = overlap * 100
    print(f"Percentage overlap between PSDs (channel 0): {percentage_overlap:.2f}%")

    # Create figure with two subplots (time domain, PSD)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    
    # Plot the time domain signal (first channel)
    axes[0].plot(x[0].cpu().numpy(), color=colors[1], linewidth=2, 
                label='Unshifted')
    axes[0].plot(y[0].cpu().numpy(), color=colors[2], linewidth=2, 
                label='Shifted')
    axes[0].set_title('Channel 0', fontsize=16)
    axes[0].set_xlabel('Time', fontsize=16)
    axes[0].set_ylabel('Amplitude (mV)', fontsize=16)
    axes[0].grid(True, alpha=0.2, linestyle='-', linewidth=0.5)

    # Plot the PSD (first channel)
    axes[1].plot(freq.cpu().numpy(), psd_x[0, 0, :].cpu().numpy(),
                color=colors[1], linewidth=0.3, label='PSD unshifted')
    axes[1].fill_between(freq.cpu().numpy(), 0, psd_x[0, 0, :].cpu().numpy(),
                        alpha=0.3, color=colors[1])
    axes[1].plot(freq.cpu().numpy(), psd_y[0, 0, :].cpu().numpy(), 
                color=colors[2], linewidth=0.3, label='PSD shifted')
    axes[1].fill_between(freq.cpu().numpy(), 0, psd_y[0, 0, :].cpu().numpy(),
                        alpha=0.3, color=colors[2])
    axes[1].set_title('Channel 0', fontsize=16)
    axes[1].set_xlabel('Frequency (Hz)', fontsize=16)
    axes[1].set_ylabel('Normalized PSD (-)', fontsize=16)
    axes[1].grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
    axes[1].set_ylim(top=1)
    axes[1].set_xlim(0, 2)  # Use same x-axis range as theoretical plot

    # Add a shared legend below all plots
    handles = [
        plt.Line2D([0], [0], color=colors[1], linewidth=2, label='Unshifted'),
        plt.Line2D([0], [0], color=colors[2], linewidth=2, label='Shifted'),
    ]
    
    fig.legend(
        handles=handles,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.12),  # Position below the plots
        ncol=2,                       # Two columns in the legend
        frameon=False,
        handlelength=2.5,             # Longer legend lines
        fontsize=16,                  # Larger legend text
    )
    
    # Adjust spacing
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)  # Add space at bottom for legend

    # Save with higher quality and transparent background
    plt.savefig(f"{save_path}.pdf", transparent=True)
    plt.savefig(f"{save_path}.png", dpi=300)
    plt.close()



def visualize_batch_avg_psd(x, y, save_path='psd_batch_avg', unshifted_mean=1, std=0.1):
    """
    Visualizes the average Power Spectral Density (PSD) of batches of signals compared
    with theoretical Gaussian frequency distributions.

    Args:
        x (torch.Tensor): First signal tensor of shape (B, C, T), where B is batch size.
        y (torch.Tensor): Second signal tensor of shape (B, C, T), where B is batch size.
        save_path (str): Base path for saving figures (without extension).
        unshifted_mean (float): Mean of the unshifted Gaussian distribution.
        std (float): Standard deviation of both Gaussian distributions, default 0.2.
    """
    # Calculate shifted_mean from unshifted_mean
    shift = 0
    shifted_mean = unshifted_mean + shift

    # Set up publication-quality figure styling
    plt.rcParams.update({
        'font.family': 'DejaVu Serif',  # Professional serif font
        'mathtext.fontset': 'cm',       # Computer Modern for math text
        'font.size': 11,
        'axes.labelsize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 16,
        'figure.figsize': (11, 4),      # Adjusted width for 2 subplots
        'figure.dpi': 300,              # High DPI for print
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',        # Avoid wasted space
        'savefig.pad_inches': 0.05,     # Minimal padding
    })
    
    # Colorblind-friendly colors
    colors = ['#000000', '#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#9467BD', '#8C564B']
    
    # Ensure inputs are batched
    assert len(x.shape) == 3, "Input 'x' must be batched with shape (B, C, T)"
    assert len(y.shape) == 3, "Input 'y' must be batched with shape (B, C, T)"
    
    batch_size = x.shape[0]
    print(f"Processing batch of size: {batch_size}")
    
    # Calculate PSD for both batches at once
    psd_x, freq = PSD(x)  # Should return (B, C, F) and freq
    psd_y, _ = PSD(y)     # Should return (B, C, F) and freq
    
    # Average PSDs across the batch dimension
    psd_x_avg = torch.mean(psd_x, dim=0)  # Now (C, F)
    psd_y_avg = torch.mean(psd_y, dim=0)  # Now (C, F)
    
    # Print frequency bin width
    if freq.numel() > 1:
        bin_width = (freq[1] - freq[0]).item()
        print(f"Frequency bin width: {bin_width:.4f} Hz")
    else:
        print("Frequency bin width: N/A (only one frequency bin)")
    
    # Print max absolute error between average PSDs (first channel)
    max_abs_error = torch.max(torch.abs(psd_x_avg[0, :] - psd_y_avg[0, :])).item()
    print(f"Max absolute error between average PSDs (channel 0): {max_abs_error:.6f}")

    # Calculate percentage overlap between the two average PSDs (first channel)
    min_psd = torch.min(psd_x_avg[0, :], psd_y_avg[0, :])
    overlap = torch.sum(min_psd).item()
    # Both PSDs are normalized, so their sum is 1
    percentage_overlap = overlap * 100
    print(f"Percentage overlap between average PSDs (channel 0): {percentage_overlap:.2f}%")

    # Create figure with two subplots (theoretical distributions, PSD)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    
    # Get the frequency range from PSD calculation for consistent axes
    f_max = float(freq[-1].cpu().numpy())
    
    # Generate theoretical Gaussian distributions using the PSD frequency range
    theoretical_freq = torch.linspace(0, f_max, 1000)
    
    # Gaussian function (without normalization)
    def gaussian(x, mu, sigma):
        return torch.exp(-0.5 * ((x - mu) / sigma) ** 2)
    
    # Calculate Gaussian distributions (without normalization)
    unshifted_gaussian = gaussian(theoretical_freq, unshifted_mean, std)
    shifted_gaussian = gaussian(theoretical_freq, shifted_mean, std)
    # Normalize so their sum is 1
    unshifted_gaussian = unshifted_gaussian / unshifted_gaussian.sum()
    shifted_gaussian = shifted_gaussian / shifted_gaussian.sum()
    
    # Plot theoretical Gaussian distributions
    axes[0].plot(theoretical_freq.cpu().numpy(), unshifted_gaussian.cpu().numpy(), 
                color=colors[1], linewidth=1)
    axes[0].fill_between(theoretical_freq.cpu().numpy(), 0, unshifted_gaussian.cpu().numpy(),
                        alpha=0.3, color=colors[1])
    axes[0].plot(theoretical_freq.cpu().numpy(), shifted_gaussian.cpu().numpy(), 
                color=colors[2], linewidth=1)
    axes[0].fill_between(theoretical_freq.cpu().numpy(), 0, shifted_gaussian.cpu().numpy(),
                        alpha=0.3, color=colors[2])
    axes[0].set_title('Theoretical frequency distributions', fontsize=16)
    axes[0].set_xlabel('Frequency (Hz)', fontsize=16)
    axes[0].set_ylabel('Probability density', fontsize=16)
    axes[0].grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
    axes[0].set_ylim(top=1)
    axes[0].set_xlim(0, 2)  # Use same x-axis range as PSD plot
    
    # Plot the average PSD (first channel)
    axes[1].plot(freq.cpu().numpy(), psd_x_avg[0, :].cpu().numpy(),
                color=colors[1], linewidth=1)
    axes[1].fill_between(freq.cpu().numpy(), 0, psd_x_avg[0, :].cpu().numpy(),
                        alpha=0.3, color=colors[1])
    axes[1].plot(freq.cpu().numpy(), psd_y_avg[0, :].cpu().numpy(), 
                color=colors[2], linewidth=1)
    axes[1].fill_between(freq.cpu().numpy(), 0, psd_y_avg[0, :].cpu().numpy(),
                        alpha=0.3, color=colors[2])
    axes[1].set_title('Average PSD (Channel 0)', fontsize=16)
    axes[1].set_xlabel('Frequency (Hz)', fontsize=16)
    axes[1].set_ylabel('Normalized PSD', fontsize=16)
    axes[1].grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
    axes[1].set_ylim(top=1)
    axes[1].set_xlim(0, 2)  # Use same x-axis range as theoretical plot
    
    # Add a shared legend below all plots
    handles = [
        plt.Line2D([0], [0], color=colors[1], linewidth=2, label='Unshifted'),
        plt.Line2D([0], [0], color=colors[2], linewidth=2, label='Shifted'),
    ]
    
    fig.legend(
        handles=handles,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.12),  # Position below the plots
        ncol=2,                       # Two columns in the legend
        frameon=False,
        handlelength=2.5,             # Longer legend lines
        fontsize=16,                  # Larger legend text
    )
    
    # Adjust spacing
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)  # Add space at bottom for legend

    # Save with higher quality and transparent background
    plt.savefig(f"{save_path}.pdf", transparent=True)
    plt.savefig(f"{save_path}.png", dpi=300)
    plt.close()


def run_psd_mmd(X, Y, labels, **kwargs):
    """
    Run the PSD MMD metric on two datasets.

    Parameters
    ----------
    X : torch.Tensor or np.ndarray
        First dataset (e.g., real data).
    Y : torch.Tensor or np.ndarray
        Second dataset (e.g., generated/reference data).
    labels : list or np.ndarray
        Labels or identifiers for the samples.
    **kwargs : dict
        Additional keyword arguments passed to `run_metric`.

    Returns
    -------
    tuple
        overall : float
            Mean metric value computed over all pairwise samples.
        per_class : list of float
            List of mean metric values computed per class (NaN if no samples for a class).
    """
    PSD_MMD.__name__ = 'PSD MMD'
    return run_metric(PSD_MMD, X, Y, labels, **kwargs)


def run_psd_rmse(X, Y, labels, **kwargs):
    """
    Run the PSD RMSE metric on two datasets.

    Parameters
    ----------
    X : torch.Tensor or np.ndarray
        First dataset (e.g., real data).
    Y : torch.Tensor or np.ndarray
        Second dataset (e.g., generated/reference data).
    labels : list or np.ndarray
        Labels or identifiers for the samples.
    **kwargs : dict
        Additional keyword arguments passed to `run_metric`.

    Returns
    -------
    tuple
        overall : float
            Mean metric value computed over all pairwise samples.
        per_class : list of float
            List of mean metric values computed per class (NaN if no samples for a class).
    """
    PSD_RMSE.__name__ = 'PSD RMSE'
    return run_metric(PSD_RMSE, X, Y, labels, **kwargs)


def run_psd_prd(X, Y, labels, **kwargs):
    """
    Run the PSD PRD metric on two datasets.

    Parameters
    ----------
    X : torch.Tensor or np.ndarray
        First dataset (e.g., real data).
    Y : torch.Tensor or np.ndarray
        Second dataset (e.g., generated/reference data).
    labels : list or np.ndarray
        Labels or identifiers for the samples.
    **kwargs : dict
        Additional keyword arguments passed to `run_metric`.

    Returns
    -------
    tuple
        overall : float
            Mean metric value computed over all pairwise samples.
        per_class : list of float
            List of mean metric values computed per class (NaN if no samples for a class).
    """
    PSD_PRD.__name__ = 'PSD PRD'
    return run_metric(PSD_PRD, X, Y, labels, **kwargs)


def match_train_val(train_data, train_labels, val_data, val_labels):
    """
    Match training and validation data based on labels and order.

    This function aligns the training data and features to the validation set by matching each validation label
    to a corresponding training label. For each label in val_labels, it finds a matching label in train_labels,
    extracts the corresponding entry from train_data and train_features, and removes it from the pool to prevent
    duplicate matches. This ensures the matched training data has the same label distribution and order as the
    validation data.

    Args:
        train_data (torch.Tensor): Training data samples.
        train_features (torch.Tensor): Training feature representations.
        train_labels (torch.Tensor): Training labels (same shape as val_labels).
        val_data (torch.Tensor): Validation data samples.
        val_labels (torch.Tensor): Validation labels.

    Returns:
        tuple: (matched_train_data, matched_train_features), both torch.Tensor, aligned to val_labels order.
               If matching fails, returns empty tensors with appropriate shape.
    """
    # Check if signal is torch else move to torch
    if not isinstance(train_data, torch.Tensor):
        train_data = torch.tensor(train_data, dtype=torch.float32)
    if not isinstance(train_labels, torch.Tensor):
        train_labels = torch.tensor(train_labels, dtype=torch.float32)
    if not isinstance(val_data, torch.Tensor):
        val_data = torch.tensor(val_data, dtype=torch.float32)
    if not isinstance(val_labels, torch.Tensor):
        val_labels = torch.tensor(val_labels, dtype=torch.float32)
    # Move everything to CPU for indexing to save GPU memory
    train_data_cpu = train_data.cpu()
    train_labels_cpu = train_labels.cpu()
    val_labels_cpu = val_labels.cpu()

    available_indices = list(range(train_labels_cpu.shape[0]))
    matched_indices = []

    for i in range(val_labels_cpu.shape[0]):
        label = val_labels_cpu[i]
        found = False
        for idx in available_indices:
            if torch.equal(train_labels_cpu[idx], label):
                matched_indices.append(idx)
                available_indices.remove(idx)
                found = True
                break
        if not found:
            print(f"\t\t[WARNING] No matching label found for {label} in training data.")

    if len(matched_indices) == 0:
        print("\t\t[WARNING] No matching training data found.")
        return (
            torch.empty((0, train_data.shape[1]), device=train_data.device),
        )

    matched_train_data = train_data_cpu[matched_indices].to(train_data.device)

    if matched_train_data.shape != val_data.shape:
        print(f"\t\t[WARNING] Matched training data shape {matched_train_data.shape} does not match validation data shape {val_data.shape}.")
        return (
            torch.empty((0, train_data.shape[1]), device=train_data.device),
        )
    return matched_train_data


def visualize_avg_psd():
    """
    Visualizes the average Power Spectral Density (PSD) of a full dataset.
    This function is a placeholder and should be implemented as needed.
    """

    DATA_PATH = 'data/ptbxl/'

    # Set up publication-quality figure styling
    plt.rcParams.update({
        'font.family': 'DejaVu Serif',  # Professional serif font
        'mathtext.fontset': 'cm',       # Computer Modern for math text
        'font.size': 11,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 12,
        'figure.figsize': (16, 12),     # Larger figure for subplots
        'figure.dpi': 300,              # High DPI for print
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',        # Avoid wasted space
        'savefig.pad_inches': 0.05,     # Minimal padding
    })
    
    # Colorblind-friendly colors
    colors = ['#000000', '#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#9467BD', '#8C564B']

    CURRENT_LEAD_NAMES = ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'III', 'aVR', 'aVL', 'aVF']
    STANDARD_LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

    # Load validation dataset
    val_data_file = os.path.join(DATA_PATH, 'val_data.npy')
    val_labels_file = os.path.join(DATA_PATH, 'val_labels.npy')
    
    # Load train dataset
    train_data_file = os.path.join(DATA_PATH, 'train_data.npy')
    train_labels_file = os.path.join(DATA_PATH, 'train_labels.npy')
    
    if not os.path.exists(val_data_file) or not os.path.exists(val_labels_file):
        raise FileNotFoundError(f"Validation ECG data files not found in {DATA_PATH}")
    
    if not os.path.exists(train_data_file) or not os.path.exists(train_labels_file):
        raise FileNotFoundError(f"Train ECG data files not found in {DATA_PATH}")
    
    val_data = np.load(val_data_file)  # (n_samples, n_channels, n_timepoints)
    val_labels = np.load(val_labels_file)  # (n_samples, n_labels)
    
    train_data = np.load(train_data_file)  # (n_samples, n_channels, n_timepoints)
    train_labels = np.load(train_labels_file)  # (n_samples, n_labels)

    train_data = match_train_val(train_data, train_labels, val_data, val_labels)
    
    # Ensure inputs are batched
    assert len(val_data.shape) == 3, "Input 'val_data' must be batched with shape (B, C, T)"
    assert len(train_data.shape) == 3, "Input 'train_data' must be batched with shape (B, C, T)"
    
    val_dataset_size = val_data.shape[0]
    train_dataset_size = train_data.shape[0]
    print(f"Processing validation batch of size: {val_dataset_size}")
    print(f"Processing train batch of size: {train_dataset_size}")
    
    # Calculate PSD for both datasets
    psd_val_data, freq = PSD(val_data)  # Should return (B, C, F) and freq
    psd_train_data, _ = PSD(train_data)  # Should return (B, C, F) and freq

    # Average PSDs across the batch dimension
    psd_val_data_avg = torch.mean(psd_val_data, dim=0)  # Now (C, F)
    psd_train_data_avg = torch.mean(psd_train_data, dim=0)  # Now (C, F)

    reorder_indices = [CURRENT_LEAD_NAMES.index(lead) for lead in STANDARD_LEAD_NAMES 
                    if lead in CURRENT_LEAD_NAMES]
    
    # Reorder the signals
    psd_val_data_avg = psd_val_data_avg[reorder_indices]
    psd_train_data_avg = psd_train_data_avg[reorder_indices]
    
    # Print frequency bin width
    if freq.numel() > 1:
        bin_width = (freq[1] - freq[0]).item()
        print(f"Frequency bin width: {bin_width:.4f} Hz")
    else:
        print("Frequency bin width: N/A (only one frequency bin)")
    
    # Create subplots (3 rows, 4 columns for 12 leads)
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()  # Flatten for easier indexing
    
    # Calculate spacing for x-axis
    max_freq = min(50, freq.max().cpu().numpy())
    freq_spacing = max_freq * 0.05  # 5% spacing
    
    # Plot the average PSD for each channel in separate subplots
    for channel in range(min(psd_val_data_avg.shape[0], 12)):  # Plot up to 12 channels
        lead_name = STANDARD_LEAD_NAMES[channel] if channel < len(STANDARD_LEAD_NAMES) else f'Channel {channel}'
        
        # Plot validation data in blue
        axes[channel].plot(freq.cpu().numpy(), psd_val_data_avg[channel, :].cpu().numpy(),
                          color=colors[1], linewidth=1, label='Validation')
        axes[channel].fill_between(freq.cpu().numpy(), 0, psd_val_data_avg[channel, :].cpu().numpy(),
                                 alpha=0.3, color=colors[1])
        
        # Plot train data in orange
        axes[channel].plot(freq.cpu().numpy(), psd_train_data_avg[channel, :].cpu().numpy(),
                          color=colors[2], linewidth=1, label='Train')
        axes[channel].fill_between(freq.cpu().numpy(), 0, psd_train_data_avg[channel, :].cpu().numpy(),
                                 alpha=0.3, color=colors[2])
        
        axes[channel].set_title(f'Lead {lead_name}', fontsize=12)
        axes[channel].set_xlabel('Frequency (Hz)', fontsize=10)
        axes[channel].set_ylabel('Normalized PSD', fontsize=10)
        axes[channel].grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
        axes[channel].set_xlim(-freq_spacing, max_freq + freq_spacing)  # Add spacing before and after
        
        # Add legend only to the first subplot to avoid clutter
        if channel == 0:
            axes[channel].legend(fontsize=10)
    
    # Hide any unused subplots
    for i in range(psd_val_data_avg.shape[0], 12):
        axes[i].set_visible(False)
    
    plt.tight_layout()

    # Save with higher quality and transparent background
    save_path = f"PTBXL_train_val_avg_psd"
    plt.savefig(f"{save_path}.pdf", transparent=True)
    plt.savefig(f"{save_path}.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    visualize_avg_psd()
