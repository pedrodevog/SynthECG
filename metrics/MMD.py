import torch
from metrics.Metric import run_metric


def median_pairwise_distance_gpu(X, Y=None, verbose=False):
    """
    Heuristic for bandwidth of the RBF. Median pairwise distance of joint data.
    If Y is missing, just calculate it from X:
        this is so that, during training, as Y changes, we can use a fixed
        bandwidth (and save recalculating this each time we evaluated the mmd)
    At the end of training, we do the heuristic "correctly" by including
    both X and Y.

    Original implementation from:
    GitHub:     https://github.com/ratschlab/RGAN/blob/master/mmd.py#L172
    Paper:      https://arxiv.org/pdf/1706.02633v2

    Parameters:
        X: Input array/tensor
        Y: Optional second array/tensor (if None, uses X)
        verbose: Whether to print debug statements (default: False)
    """
    if verbose:
        print("\nStarting median_pairwise_distance...")
        print(f"Input shapes - X: {X.shape}, Y: {'same as X' if Y is None else Y.shape}")

    if Y is None:
        Y = X       # WARNING: Memory inefficient - doubles memory usage
        if verbose:
            print("Y is None - copying X (this doubles memory usage)")
   
    if verbose:
        print("\nComputing pairwise distances...")
    
    if len(X.shape) == 2:
        if verbose:
            print("Computing for 2D matrices...")
        X_sqnorms = torch.einsum('...i,...i', X, X)
        Y_sqnorms = torch.einsum('...i,...i', Y, Y)
        XY = torch.einsum('ia,ja', X, Y)
        if verbose:
            print(f"Intermediate results shapes: X_sqnorms: {X_sqnorms.shape}, XY: {XY.shape}")
    
    elif len(X.shape) == 3:
        if verbose:
            print("Computing for 3D tensors...")
        X_sqnorms = torch.einsum('...ij,...ij', X, X)
        Y_sqnorms = torch.einsum('...ij,...ij', Y, Y)
        XY = torch.einsum('iab,jab', X, Y)
        if verbose:
            print(f"Intermediate results shapes: X_sqnorms: {X_sqnorms.shape}, XY: {XY.shape}")
    else:
        raise ValueError(f"Unexpected input shape: {X.shape}")

    if verbose:
        print("\nComputing final distances matrix...")
    distances = torch.sqrt(torch.clamp(X_sqnorms.reshape(-1, 1) - 2*XY + Y_sqnorms.reshape(1, -1), min=0.0))
    if verbose:
        print(f"Distance matrix shape: {distances.shape}")

    result = torch.median(distances)
    if verbose:
        print(f"Final median distance: {result}")

    # Addition to avoid zero distance
    # This is a hack to avoid zero distance, which can happen if the two distributions are identical
    if result.item() < 1e-6:
        eps = 0
        result = torch.where(result == 0, torch.ones_like(result) * eps, result)

    return result


def my_kernel_gpu(X, Y, sigma):
    """
    PyTorch GPU version of the kernel calculation
    
    Original implementation from:
    GitHub:     https://github.com/ratschlab/RGAN/blob/master/eugenium_mmd.py#L21
    Paper:      https://arxiv.org/pdf/1706.02633v2
    """
    gamma = 1 / (2 * sigma**2)
    if len(X.shape) == 2:
        X_sqnorms = torch.einsum('...i,...i', X, X)
        Y_sqnorms = torch.einsum('...i,...i', Y, Y)
        XY = torch.einsum('ia,ja', X, Y)
    elif len(X.shape) == 3:
        X_sqnorms = torch.einsum('...ij,...ij', X, X)
        Y_sqnorms = torch.einsum('...ij,...ij', Y, Y)
        XY = torch.einsum('iab,jab', X, Y)
    else:
        raise ValueError(X)
    Kxy = torch.exp(-gamma*(X_sqnorms.reshape(-1, 1) - 2*XY + Y_sqnorms.reshape(1, -1)))
    return Kxy


def mmd(X, Y, sigma=1.0, biased=False):
    """
    Calculates the MMD between two distributions with the RBF kernel.
    Given a bandwidth sigma.
    If unbiased gives a negative MMD, it sets the MMD to 0.
    PyTorch GPU version

    Original implementation from:
    GitHub:     https://github.com/ratschlab/RGAN/blob/master/mmd.py#L79
    Paper:      https://arxiv.org/pdf/1706.02633v2
    """
    XX = my_kernel_gpu(X, X, sigma)
    YY = my_kernel_gpu(Y, Y, sigma)
    XY = my_kernel_gpu(X, Y, sigma)

    m = XX.shape[0]
    n = YY.shape[0]

    if biased:
        mmd2 = (XX.sum() / (m * m)
              + YY.sum() / (n * n)
              - 2 * XY.sum() / (m * n))                 # Always non-negative
    else:
        # Use diag instead of trace to avoid issues with non-square matrices
        diag_X = torch.diag(XX)
        diag_Y = torch.diag(YY)

        mmd2 = ((XX.sum() - diag_X.sum()) / (m * (m - 1))
            + (YY.sum() - diag_Y.sum()) / (n * (n - 1))
            - 2 * XY.sum() / (m * n))                   # Can give negative value              
        
        # if mmd2 < 0:
        #     mmd2 = (XX.sum() / (m * m) 
        #             + YY.sum() / (n * n) 
        #             - 2 * XY.sum() / (m * n))           # Replace with biased version

    return torch.max(torch.tensor(0.0, device=mmd2.device), mmd2)


def mmd_(X, Y):
    """
    Organizes the MMD calculation.
    """
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X).float().cuda()
    if not isinstance(Y, torch.Tensor):
        Y = torch.tensor(Y).float().cuda()

    # Fix the dimension check
    if not (X.ndim in [2, 3] and Y.ndim in [2, 3]):
        raise ValueError(f"Both signals must be 2- or 3D arrays. Current shapes - dataset1: {X.shape}, dataset2: {Y.shape}")
    
    # Check if channels and timesteps match
    if X.shape[1] != Y.shape[1]:
        raise ValueError(f"Number of channels/features must match. dataset1: {X.shape[1]}, dataset2: {Y.shape[1]}")

    sigma = median_pairwise_distance_gpu(X, Y)
    if sigma.item() == 0:
        return torch.tensor(0.0, device=X.device)
    mmd_result = mmd(X, Y, sigma)
    return mmd_result


def run_mmd(X, Y, labels, **kwargs):
    """
    Run MMD metric on two datasets.

    Parameters:
    -----------
    X : torch.Tensor or np.ndarray
        First dataset (e.g., real data).
    Y : torch.Tensor or np.ndarray
        Second dataset (e.g., generated/reference data). 
    labels : list or np.ndarray
        Labels or identifiers for the samples.
    **kwargs : dict
        Additional keyword arguments to pass to the metric.

    Returns:
    --------
    tuple:
        overall : scalar
            Mean metric value computed over all pairwise samples.
        per_class : list
            List of mean metric values computed per class (NaN if no samples for a class).
    """
    if X.dim() == 2:
        mmd_.__name__ = 'KID'
    else:
        mmd_.__name__ = 'MMD'

    return run_metric(mmd_, X, Y, labels, **kwargs)