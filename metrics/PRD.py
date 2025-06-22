import torch
import numpy as np
from metrics.Metric import run_metric
import matplotlib.pyplot as plt


def PRD_distance(x, y):
    """
    Compute the PRD between two sets of points.
    
    Parameters:
    -----------
    x : torch.Tensor
        Set of points, shape (n_channels, n_timepoints)
    y : torch.Tensor
        Set of points, shape (n_channels, n_timepoints)
    
    Returns:
    --------
    PRD : float
        PRD between the two sets of points
    """
    
    # Compute the PRD more efficiently
    diff_squared = torch.sum((x - y)**2)
    x_squared = torch.sum(x**2) + 1e-6  # Avoid division by zero
    PRD = torch.sqrt(diff_squared / x_squared) * 100
    
    return PRD


def PRD_matrix(dataset1, dataset2, visualize=False):
    """
    Compute the PRD matrix between two datasets.
    
    Parameters:
    -----------
    dataset1 : torch.Tensor
        First dataset, shape (n_samples, n_channels, n_timepoints)
    dataset2 : torch.Tensor
        Second dataset, shape (n_samples, n_channels, n_timepoints)
    
    Returns:
    --------
    PRD_matrix : torch.Tensor
        PRD matrix between the two datasets
    """
    if visualize:
        visualize_prd(dataset1[0], dataset2[0])
   
    # Compute (x-y)^2 term efficiently using broadcasting
    diff_squared = torch.sum((dataset1[:, None, :, :] - dataset2[None, :, :, :])**2, dim=(2,3))
    
    # Original non-symmetric version
    # x_squared = torch.sum(dataset1**2, dim=(1,2))[:, None]
    # Compute denominator term (x^2 + y^2)/2
    # This is a symmetric version of the PRD
    x_squared = (torch.sum(dataset1**2, dim=(1,2))[:, None] + torch.sum(dataset2**2, dim=(1,2))[None, :])/2
    
    # Compute final PRD matrix
    PRD_matrix = torch.sqrt(diff_squared / (x_squared + 1e-6)) * 100
    
    return PRD_matrix


def PRD(dataset1, dataset2):

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

    prd = PRD_matrix(dataset1, dataset2, visualize=False)

    # TODO: matrix should be stored somewhere for statistical analysis

    prd = torch.mean(prd)
    return prd


def visualize_prd(x, y):
    """
    Visualize the PRD score for each channel pointwise (per timepoint).
    Each channel is plotted in a separate subplot. The relative, absolute error is shown as a line plot next to each channel.

    
    Parameters:
    -----------
    x : torch.Tensor
        Set of points, shape (n_channels, n_timepoints)
    y : torch.Tensor
        Set of points, shape (n_channels, n_timepoints)
    
    Returns:
    --------
    None
    """
    prd_score = PRD_distance(x, y)
    n_channels = x.shape[0]
    n_timepoints = x.shape[1]

    prd_time = torch.sqrt((x - y)**2 / torch.sum(x**2)) * 100

    fig, axs = plt.subplots(n_channels, 2, figsize=(10, 2*n_channels), sharex=True)
    fig.suptitle(f'PRD: {prd_score.item():.2f} %', fontsize=16)
    for i in range(n_channels):
        axs[i, 0].plot(x[i].cpu().numpy(), label='Signal 1', color='blue')
        axs[i, 0].plot(y[i].cpu().numpy(), label='Signal 2', color='orange')
        axs[i, 0].set_title(f'Channel {i+1}')
        axs[i, 0].legend()
        axs[i, 1].plot(np.arange(n_timepoints), prd_time[i].cpu().numpy(), label='Relative error', color='red')
        axs[i, 1].set_title(f'Channel {i+1} Relative error')
        axs[i, 1].legend()
        axs[i, 1].set_xlabel('Timepoints')
        axs[i, 1].set_ylabel('Relative error')
    
    plt.xlabel('Timepoints')
    plt.tight_layout()
    plt.savefig(f"prd.png", dpi=300)
    plt.close()


def run_prd(X, Y, labels, **kwargs):
    """
    Run PRD metric on a model against training and test data.
    
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
        PRD computation option (default: 'default', min: 'min')
        
    Returns:
    --------
    tuple:
        (metric_test_model, metric_test_train) dictionaries containing PRD values
    """
    # Call the dtw function with option filled in
    PRD.__name__ = 'PRD'
    return run_metric(PRD, X, Y, labels, **kwargs)
