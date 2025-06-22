import dtaidistance
from dtaidistance import preprocessing, dtw_ndim
import numpy as np
import torch
from metrics.Metric import run_metric
import matplotlib.pyplot as plt

def pairwise_dtw_distance(x, x_og, y, y_og, visualize=False):
    dtw_score, paths = dtaidistance.dtw.warping_paths(
            x, 
            y, 
            # window=100,
            # psi=(50, 50, 50, 50),
            # psi_neg=True,
            use_ndim=True,
            )
    best_path = dtaidistance.dtw.best_path(paths)

    # Only keep the best path until just before a boundary (maxx or maxy) is hi
    # maxx = x.shape[0] - 1
    # maxy = y.shape[0] - 1
    # filtered_path = []
    # for i, j in best_path:
    #     if i == maxx or j == maxy:
    #         filtered_path.append((i, j))
    #         break
    #     filtered_path.append((i, j))
    # best_path = filtered_path

    dtw_score_val = np.sum(np.array([np.sum((x[i] - y[j])**2) for (i, j) in best_path]))

    if visualize:
        visualize_dtw(dtw_score, paths, best_path, x, x_og, y, y_og)
        print(f"DTW score: {dtw_score_val}")

    return dtw_score_val


def DTW_matrix(X, Y):
    signals = np.concatenate((X, Y), axis=0)
    block = ((0, X.shape[0]), (X.shape[0], signals.shape[0]))

    dtw_matrix = dtw_ndim.distance_matrix_fast(
        signals, 
        block=block, 
        # psi=(50, 50, 50, 50),
        # psi_neg=True,
        compact=True
        )

    dtw_matrix = np.array(dtw_matrix) ** 2

    return dtw_matrix


def DTW(X, Y, visualize=False):
    # Put on CPU if needed
    if isinstance(X, torch.Tensor):
        X = X.cpu().numpy()
    if isinstance(Y, torch.Tensor):
        Y = Y.cpu().numpy()
    # Convert the signals to numpy arrays if they are not already
    if not isinstance(X, np.ndarray):
        X = np.array(X, dtype=np.double)
    if not isinstance(Y, np.ndarray):
        Y = np.array(Y, dtype=np.double)
    # And also convert to double if needed
    if X.dtype != np.double:
        X = X.astype(np.double)
    if Y.dtype != np.double:
        Y = Y.astype(np.double)

    X, X_og, Y, Y_og = preprocess(X, Y)

    if visualize:
        dtw_score = pairwise_dtw_distance(X[0], X_og[0], Y[1], Y_og[1], visualize)

    dtw_distance_matrix = DTW_matrix(X, Y)
    
    # mean_dtw_distance_ = torch.tensor(np.mean(dtw_distance_matrix_), dtype=torch.float32).cuda()
    # Compute the mean distance and convert to torch tensor
    mean_dtw_distance = torch.tensor(np.mean(dtw_distance_matrix), dtype=torch.float32).cuda()

    # if mean_dtw_distance_ != mean_dtw_distance:
    #     print(f"Warning: mean dtw distance mismatch: {mean_dtw_distance_} != {mean_dtw_distance}")

    return mean_dtw_distance

def preprocess(X, Y):
    # Reshape signals if needed
    if X.ndim == 1:
        X = X.reshape(1, -1, 1)
        print(f"Reshaped X to: {X.shape}")
    if Y.ndim == 1:
        Y = Y.reshape(1, -1, 1)
        print(f"Reshaped Y to: {Y.shape}")
    
    if X.ndim != 3 or Y.ndim != 3:
        raise ValueError(f"Both signals must be 3D arrays. Current shapes - X: {X.shape}, Y: {Y.shape}")
    
    # Check if channels and timesteps match
    if X.shape[1] != Y.shape[1]:
        raise ValueError(f"Number of channels must match. X: {X.shape[1]}, Y: {Y.shape[1]}")
    if X.shape[2] != Y.shape[2]:
        raise ValueError(f"Number of timesteps must match. X: {X.shape[2]}, Y: {Y.shape[2]}")
    
    # Transform to (n_samples, n_timesteps, n_channels) as required by dtw_ndim.distance_matrix
    X = X.transpose(0, 2, 1)
    Y = Y.transpose(0, 2, 1)

    # X_ = preprocessing.differencing(X, smooth=0.2)
    # Y_ = preprocessing.differencing(Y, smooth=0.2)
    # X = 2 * (X - np.min(X, axis=1, keepdims=True)) / (np.max(X, axis=1, keepdims=True) - np.min(X, axis=1, keepdims=True)) - 1
    # Y = 2 * (Y - np.min(Y, axis=1, keepdims=True)) / (np.max(Y, axis=1, keepdims=True) - np.min(Y, axis=1, keepdims=True)) - 1

    return X, X, Y, Y

def visualize_dtw(dtw_score, paths, best_path, x, x_og, y, y_og):
    """
    Visualize the accumulated cost matrix, local cost matrix, and warping path.
    Handles multidimensional signals (n_timesteps, n_channels).
    """
    # If x/y are multidimensional, plot the first channel for visualization
    if x.ndim == 2:
        x_plot = x[:, 0]
        x_og_plot = x_og[:, 0]
    else:
        x_plot = x
        x_og_plot = x_og
    if y.ndim == 2:
        y_plot = y[:, 0]
        y_og_plot = y_og[:, 0]
    else:
        y_plot = y
        y_og_plot = y_og

    # square the accumulated cost matrix
    paths = np.array(paths) ** 2
    local_costs = np.array([np.sum((x[i] - y[j])**2) for (i, j) in best_path])
    # Calculate accumulated costs along the path
    accumulated_costs = np.cumsum(local_costs)

    # Set up better styling for publication-quality figures
    plt.rcParams.update({
        'font.family': 'DejaVu Serif',
        'mathtext.fontset': 'cm',
        'font.size': 12,
        'axes.labelsize': 16,
        'axes.titlesize': 16,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 16,
        'figure.figsize': (18, 6.5),  # Slightly taller to accommodate legend below
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
    })

    # Improved color scheme - colorblind-friendly
    colors = ['#000000', '#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#9467BD', '#8C564B']

    # Create figure with 3 axes arranged horizontally
    fig, axs = plt.subplots(1, 3, figsize=(18, 6.5))

    # Accumulated cost matrix
    ax = axs[0]
    # Set vmax=60 to limit the colorbar maximum to 60
    im = ax.imshow(paths, origin='lower', aspect='auto', cmap='viridis', vmin=0, vmax=3200)
    ax.set_title('Accumulated cost matrix (mV²)')
    ax.set_xlabel('Unshifted (timesteps)')
    ax.set_ylabel('Shifted (timesteps)')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    # Plot warping path
    path_x, path_y = zip(*best_path)
    warping_line = ax.plot(path_y, path_x, color='red', linewidth=2, label='Warping path')[0]
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
    
    # Place legend below plot
    ax.legend(handles=[warping_line], loc='upper center', 
              bbox_to_anchor=(0.5, -0.15), frameon=False)

    # Local cost matrix (difference along path)
    ax = axs[1]
    local_cost_line = ax.plot(local_costs, color=colors[1], linewidth=2, label='Local costs')[0]
    ax.set_title('Cost along path (mV²)')
    ax.set_xlabel('Path index')
    ax.set_ylabel('Local cost')
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
    
    # Set consistent y-axis scale for local cost plot
    ax.set_ylim(0, 8)  # Fixed limit as specified
    
    # Create a secondary y-axis for accumulated cost
    ax2 = ax.twinx()
    accumulated_line = ax2.plot(accumulated_costs, color=colors[2], linewidth=2, 
                               linestyle='--', label='Accumulated cost')[0]
    ax2.set_ylabel('Accumulated cost', color=colors[2])
    ax2.tick_params(axis='y', labelcolor=colors[2])
    
    # Set the max value for accumulated cost y-axis to 60
    ax2.set_ylim(0, 200)
    
    # Combined legend for both lines
    lines = [local_cost_line, accumulated_line]
    labels = [line.get_label() for line in lines]
    ax.legend(lines, labels, loc='upper center', 
              bbox_to_anchor=(0.5, -0.15), frameon=False, ncol=2)

    # Plot the signals with vertical separation and warping path connections
    ax = axs[2]
    
    # Calculate vertical offset for better visualization
    x_offset = 0
    y_offset = 1.5 * (max(np.max(x_plot), np.max(y_plot)) - min(np.min(x_plot), np.min(y_plot)))
    
    # Plot signals with offset
    line_x = ax.plot(np.arange(len(x_plot)), x_plot + x_offset, color=colors[1], 
                     linewidth=2, label='Unshifted')[0]
    line_y = ax.plot(np.arange(len(y_plot)), y_plot - y_offset, color=colors[2], 
                     linewidth=2, label='Shifted')[0]

    # Draw warping path connections - only every 5th connection
    for idx, (i, j) in enumerate(best_path):
        # Only draw every 10th connection
        if idx % 10 == 0:
            ax.plot([i, j], [x_plot[i] + x_offset, y_plot[j] - y_offset], 
                    color='gray', alpha=0.3, linestyle='-', linewidth=0.5)
    
    ax.set_title('DTW warping path connections')
    ax.set_xlabel('Timesteps')
    # Remove y-ticks and label for clarity
    ax.set_yticks([])
    ax.set_ylabel('')
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)

    # Place legend below plot
    ax.legend(handles=[line_x, line_y], loc='upper center',
              bbox_to_anchor=(0.5, -0.15), frameon=False, ncol=2)

    plt.tight_layout(rect=[0, 0.05, 1, 1])  # Allow space for legends at bottom
    # Save as both PNG and PDF
    plt.savefig("dtw.png", dpi=300, transparent=True)
    plt.savefig("dtw.pdf", transparent=True)
    plt.close()

def run_dtw(X, Y, labels, **kwargs):
    """
    Run DTW metric on a model against training and test data.
    """
    DTW.__name__ = 'DTW'
    return run_metric(DTW, X, Y, labels, **kwargs)
