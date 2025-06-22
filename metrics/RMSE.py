import torch
import numpy as np
from metrics.Metric import run_metric
import matplotlib.pyplot as plt


def RMSE_distance(x, y):
    """
    Compute the RMSE between two sets of points.
    Mean is taken over time and channels.
    
    Parameters:
    -----------
    x : torch.Tensor
        Set of points, shape (n_channels, n_timepoints)
    y : torch.Tensor
        Set of points, shape (n_channels, n_timepoints)
    
    Returns:
    --------
    RMSE : float
        RMSE between the two sets of points
    """
    return torch.sqrt(torch.mean((x - y)**2))


def RMSE_matrix(dataset1, dataset2, visualize=False):
    """
    Compute the RMSE matrix between two datasets.
    
    Parameters:
    -----------
    dataset1 : torch.Tensor
        First dataset, shape (n_samples, n_channels, n_timepoints)
    dataset2 : torch.Tensor
        Second dataset, shape (n_samples, n_channels, n_timepoints)
    
    Returns:
    --------
    RMSE_matrix : torch.Tensor
        RMSE matrix between the two datasets
    """
    if visualize:
        visualize_rmse(dataset1[0], dataset2[0])
    diff_squared = torch.mean((dataset1[:, None, :, :] - dataset2[None, :, :, :])**2, dim=(2,3))
    return torch.sqrt(diff_squared)


def RMSE(dataset1, dataset2):
    if isinstance(dataset1, torch.Tensor):
        dataset1 = dataset1.float().cuda()
    elif isinstance(dataset1, np.ndarray):
        dataset1 = torch.from_numpy(dataset1).float().cuda()
    if isinstance(dataset2, torch.Tensor):
        dataset2 = dataset2.float().cuda()
    elif isinstance(dataset2, np.ndarray):
        dataset2 = torch.from_numpy(dataset2).float().cuda()

    # Reshape signals if needed
    if dataset1.ndim == 1:
        dataset1 = dataset1.reshape(1, -1, 1)
        print(f"Reshaped dataset1 to: {dataset1.shape}")
    if dataset2.ndim == 1:
        dataset2 = dataset2.reshape(1, -1, 1)
        print(f"Reshaped dataset2 to: {dataset2.shape}")
    
    if dataset1.ndim != 3 or dataset2.ndim != 3:
        raise ValueError(f"Both signals must be 3D arrays. Current shapes - dataset1: {dataset1.shape}, dataset2: {dataset2.shape}")
    
    # Check if channels and timesteps match
    if dataset1.shape[1] != dataset2.shape[1]:
        raise ValueError(f"Number of channels must match. dataset1: {dataset1.shape[1]}, dataset2: {dataset2.shape[1]}")
    if dataset1.shape[2] != dataset2.shape[2]:
        raise ValueError(f"Number of timesteps must match. dataset1: {dataset1.shape[2]}, dataset2: {dataset2.shape[2]}")

    rmse = RMSE_matrix(dataset1, dataset2, visualize=False)

    # TODO: matrix should be stored somewhere for statistical analysis

    rmse = torch.mean(rmse)
    return rmse


def visualize_rmse(x, y):
    """
    Visualize the RMSE score for each channel pointwise (per timepoint).
    Each channel is plotted in a separate subplot. The absolute error is shown as a line plot next to each channel.
    
    Parameters:
    -----------
    rmse_score : float
        RMSE score
    x : torch.Tensor
        First signal, shape (n_channels, n_timepoints)
    y : torch.Tensor
        Second signal, shape (n_channels, n_timepoints)
    
    Returns:
    --------
    None
    """
    rmse_score = RMSE_distance(x, y)
    n_channels = x.shape[0]
    n_timepoints = x.shape[1]

    fig, axs = plt.subplots(n_channels, 2, figsize=(10, 2*n_channels), sharex=True)
    for i in range(n_channels):
        axs[i, 0].plot(x[i].cpu().numpy(), label='Signal 1', color='blue')
        axs[i, 0].plot(y[i].cpu().numpy(), label='Signal 2', color='orange')
        axs[i, 0].set_title(f'Channel {i+1} RMSE: {rmse_score.item():.4f}')
        axs[i, 0].legend()
        axs[i, 1].plot(np.arange(n_timepoints), np.abs((x[i] - y[i]).cpu().numpy()), label='Error', color='red')
        axs[i, 1].set_title(f'Channel {i+1} Error')
        axs[i, 1].legend()
        axs[i, 1].set_xlabel('Timepoints')
        axs[i, 1].set_ylabel('Error (mV)')
    
    plt.xlabel('Timepoints')
    plt.tight_layout()
    plt.savefig(f"rmse.png", dpi=300)
    plt.close()

    
def run_rmse(X, Y, labels):
    """
    Run RMSE metric on a model against training and test data.
    
    Parameters:
    -----------
    model : torch.nn.Module
        Model to evaluate
    train_data : torch.Tensor
        Training data
    test_data : torch.Tensor
        Test data
    n_samples : int, optional
        Number of samples to generate from the model (default: 100)
    model_data : torch.Tensor, optional
        Pre-generated model data (default: None)
    model_labels : torch.Tensor, optional
        Labels for pre-generated model data (default: None)
    option : str, optional
        RMSE computation option (default: 'default', min: 'min')
        
    Returns:
    --------
    tuple:
        (metric_test_model, metric_test_train) dictionaries containing RMSE values
    """
    RMSE.__name__ = 'RMSE'
    return run_metric(RMSE, X, Y, labels)
