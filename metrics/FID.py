import torch
import numpy as np
from scipy import linalg
from metrics.Metric import run_metric


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=0.1):
    """Calculate the Frechet distance between two multivariate Gaussians.
    
    Stable version by Dougal J. Sutherland.
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert (
        mu1.shape == mu2.shape
    ), "Training and test mean vectors have different lengths"
    assert (
        sigma1.shape == sigma2.shape
    ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; "
            "adding %s to diagonal of cov estimates"
        ) % eps
        # print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=0.1):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


# WIP
# import torch

# def frechet_distance_torch(mu1, mu2, sigma1, sigma2, eps=1e-6):
#     """Calculate the Frechet distance between two multivariate Gaussians using PyTorch.
    
#     Args:
#         mu1 (torch.Tensor): Mean of first distribution (1D)
#         mu2 (torch.Tensor): Mean of second distribution (1D)
#         sigma1 (torch.Tensor): Covariance of first distribution (2D)
#         sigma2 (torch.Tensor): Covariance of second distribution (2D)
#         eps (float): Small constant for numerical stability
        
#     Returns:
#         float: Frechet distance
#     """
#     # Ensure vectors are at least 1D
#     if mu1.dim() == 0:
#         mu1 = mu1.unsqueeze(0)
#     if mu2.dim() == 0:
#         mu2 = mu2.unsqueeze(0)
        
#     # Ensure matrices are at least 2D
#     if sigma1.dim() == 0 or sigma1.dim() == 1:
#         sigma1 = sigma1.view(1, -1)
#     if sigma2.dim() == 0 or sigma2.dim() == 1:
#         sigma2 = sigma2.view(1, -1)
    
#     assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
#     assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"
    
#     diff = mu1 - mu2
    
#     # Add eps to diagonals for numerical stability
#     offset = torch.eye(sigma1.size(0), device=sigma1.device) * eps
#     sigma1_eps = sigma1 + offset
#     sigma2_eps = sigma2 + offset
    
#     # Compute matrix square root of product
#     prod = sigma1_eps @ sigma2_eps

#     # Try to use torch.linalg.sqrtm if available (PyTorch >= 1.9)
#     try:
#         sqrt_prod = torch.linalg.sqrtm(prod)
#         if torch.is_complex(sqrt_prod):
#             if not torch.allclose(sqrt_prod.imag, torch.zeros_like(sqrt_prod.imag), atol=1e-3):
#                 raise ValueError("Complex values in sqrtm result")
#             sqrt_prod = sqrt_prod.real
#     except Exception as e:
#         print("Warning: torch.linalg.sqrtm not available or failed, falling back to eigen-decomposition. Results may be inaccurate.")
#         vals, vecs = torch.linalg.eigh(prod)
#         # Take real part of sqrt of possibly negative eigenvalues (like SciPy)
#         sqrt_vals = torch.sqrt(vals + 0j).real
#         sqrt_prod = vecs @ torch.diag(sqrt_vals) @ vecs.T

#     tr_covmean = torch.trace(sqrt_prod)
#     return diff.dot(diff) + torch.trace(sigma1) + torch.trace(sigma2) - 2 * tr_covmean


def FID(X_features, Y_features):
    """Calculate Frechet Inception Distance between two feature sets."""
    # Ensure input tensor types are correct
    if not isinstance(X_features, np.ndarray):
        X_features = X_features.float().cpu().numpy()
    if not isinstance(Y_features, np.ndarray):
        Y_features = Y_features.float().cpu().numpy()

    # Ensure input tensors are 2D
    if len(X_features.shape) != 2 or len(Y_features.shape) != 2:
        raise ValueError("Input tensors must be 2D")
    # Ensure input tensors have the same shape
    if X_features.shape[1] != Y_features.shape[1]:
        raise ValueError("Input tensors must have the same number of features")
    
    # Ensure we have enough samples for covariance calculation
    if X_features.shape[0] < 2 or Y_features.shape[0] < 2:
        raise ValueError("Need at least 2 samples per distribution to compute FID")

    # Calculate mean and covariance statistics
    mu1 = np.mean(X_features, axis=0)
    sigma1 = np.cov(X_features, rowvar=False)
    
    mu2 = np.mean(Y_features, axis=0)
    sigma2 = np.cov(Y_features, rowvar=False)

    # Debug: print means and covariances
    # print("NumPy FID: mu1", mu1[:5], "mu2", mu2[:5])
    # print("NumPy FID: sigma1", sigma1[:2, :2], "sigma2", sigma2[:2, :2])

    # Calculate FID
    fid_value = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    
    # Convert to torch tensor and ensure it's non-negative
    return torch.tensor(max(fid_value, 0.0)).float().cuda()

# WIP
# def FID_torch(X_features, Y_features):
#     """Calculate Frechet Inception Distance between two feature sets using PyTorch."""
#     # Ensure input tensor types are correct
#     if not isinstance(X_features, torch.Tensor):
#         X_features = torch.tensor(X_features, dtype=torch.float32)
#     if not isinstance(Y_features, torch.Tensor):
#         Y_features = torch.tensor(Y_features, dtype=torch.float32)
    
#     # Move to the same device if needed
#     X_features = X_features.cuda()
#     Y_features = Y_features.cuda()
    
#     # Ensure input tensors are 2D
#     if len(X_features.shape) != 2 or len(Y_features.shape) != 2:
#         raise ValueError("Input tensors must be 2D")
#     # Ensure input tensors have the same shape
#     if X_features.shape[1] != Y_features.shape[1]:
#         raise ValueError("Input tensors must have the same number of features")
    
#     # Ensure we have enough samples for covariance calculation
#     if X_features.shape[0] < 2 or Y_features.shape[0] < 2:
#         raise ValueError("Need at least 2 samples per distribution to compute FID")

#     # Calculate mean and covariance statistics in PyTorch
#     mu1 = torch.mean(X_features, dim=0)
#     # For covariance, we need to center the data first
#     X_centered = X_features - mu1
#     # Compute covariance with Bessel's correction (dividing by n-1)
#     sigma1 = torch.matmul(X_centered.t(), X_centered) / (X_features.shape[0] - 1)
    
#     mu2 = torch.mean(Y_features, dim=0)
#     Y_centered = Y_features - mu2
#     sigma2 = torch.matmul(Y_centered.t(), Y_centered) / (Y_features.shape[0] - 1)

#     # Debug: print means and covariances
#     print("Torch FID: mu1", mu1[:5].cpu().numpy(), "mu2", mu2[:5].cpu().numpy())
#     print("Torch FID: sigma1", sigma1[:2, :2].cpu().numpy(), "sigma2", sigma2[:2, :2].cpu().numpy())

#     # Calculate FID using the PyTorch function
#     fid_value = frechet_distance_torch(mu1, mu2, sigma1, sigma2)
    
#     # Ensure it's non-negative and return as a tensor
#     return torch.tensor(max(fid_value, 0.0)).float().cuda()


def run_fid(X, Y, labels, **kwargs):
    """
    Run FID metric on a model against training and test data.
    
    Parameters:
    -----------
    model : torch.nn.Module
        Model to evaluate
    train_data : torch.Tensor
        Training data
    train_labels : torch.Tensor
        Training labels
    test_data : torch.Tensor
        Test data
    test_labels : torch.Tensor
        Test labels
    n_samples : int, optional
        Number of samples to generate from model (default: 100)
    model_data : torch.Tensor, optional
        Pre-generated model data (if None, will generate from model)
    model_labels : torch.Tensor, optional
        Labels for pre-generated model data
    classifier : torch.nn.Module, optional
        Classifier model to extract features
    project : str, optional
        Project name for logging
        
    Returns:
    --------
    tuple:
        (fid_test_model, fid_test_train) dictionaries containing FID values
    """
    FID.__name__ = 'FID'

    return run_metric(FID, X, Y, labels, **kwargs)
