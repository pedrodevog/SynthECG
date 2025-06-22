"""
Fréchet Distance (dFD) Metric implementation.
Copied from: https://github.com/spiros/discrete_frechet/blob/master/frechetdist.py
"""


# input signal dataset shape: (num_samples, num_channels, num_timesteps)
# input signal shape: (num_channels, num_timesteps)
# expected signal shape: (num_timesteps, num_channels)
# curve1 = [(0, 0), (1, 1), (2, 2)]
# curve2 = [(0, 0), (1, 2), (2, 3)]

# frechet_distance = Frechet()
# distance = frechet_distance.calculate(curve1, curve2)
# print(f"Fréchet Distance: {distance}")

import torch
import numpy as np
from joblib import Parallel, delayed
from functools import partial
from metrics.Metric import run_metric
import sys
sys.setrecursionlimit(10000)  # Increase as needed, but be careful
import matplotlib.pyplot as plt
from numba import jit
import math
import time

__all__ = ['frdist']

def _c(ca, i, j, p, q):

    if ca[i, j] > -1:
        return ca[i, j]
    elif i == 0 and j == 0:
        ca[i, j] = np.linalg.norm(p[i]-q[j])
    elif i > 0 and j == 0:
        ca[i, j] = max(_c(ca, i-1, 0, p, q), np.linalg.norm(p[i]-q[j]))
    elif i == 0 and j > 0:
        ca[i, j] = max(_c(ca, 0, j-1, p, q), np.linalg.norm(p[i]-q[j]))
    elif i > 0 and j > 0:
        ca[i, j] = max(
            min(
                _c(ca, i-1, j, p, q),
                _c(ca, i-1, j-1, p, q),
                _c(ca, i, j-1, p, q)
            ),
            np.linalg.norm(p[i]-q[j])
            )
    else:
        ca[i, j] = float('inf')

    return ca[i, j]


def _c_m(ca, i, j, p, q, path):
    """
    Implementation of _c which also returns cost matrix and path.
    """
    if path == []:
        path = [(i, j)]
    else:
        path.append((i, j))

    if ca[i, j] > -1:
        return ca[i, j], ca, path
    elif i == 0 and j == 0:
        ca[i, j] = np.linalg.norm(p[i]-q[j])
    elif i > 0 and j == 0:
        val, ca, path = _c_m(ca, i-1, 0, p, q, path)
        ca[i, j] = max(val, np.linalg.norm(p[i]-q[j]))
    elif i == 0 and j > 0:
        val, ca, path = _c_m(ca, 0, j-1, p, q, path)
        ca[i, j] = max(val, np.linalg.norm(p[i]-q[j]))
    elif i > 0 and j > 0:
        val1, ca1, path1 = _c_m(ca, i-1, j, p, q, path)
        val2, ca2, path2 = _c_m(ca1, i-1, j-1, p, q, path)
        val3, ca3, path3 = _c_m(ca2, i, j-1, p, q, path)

        min_val = min(val1, val2, val3)
        if min_val == val1:
            ca1[i, j] = max(
                min_val,
                np.linalg.norm(p[i]-q[j])
            )
            ca = ca1
            path = path1
        elif min_val == val2:
            ca2[i, j] = max(
                min_val,
                np.linalg.norm(p[i]-q[j])
            )
            ca = ca2
            path = path2
        else:
            ca3[i, j] = max(
                min_val,
                np.linalg.norm(p[i]-q[j])
            )
            ca = ca3
            path = path3
    else:
        ca[i, j] = float('inf')

    return ca[i, j], ca, path


def _c_m_iterative(p, q):
    """
    Iterative implementation of _c_m which returns distance, cost matrix, and path.
    """
    len_p, len_q = len(p), len(q)
    
    # Initialize cost matrix with -1 (uncomputed)
    ca = np.ones((len_p, len_q), dtype=np.float64) * -1
    
    # Initialize traceback matrix to store decisions
    # 0: came from (i-1, j)
    # 1: came from (i-1, j-1)
    # 2: came from (i, j-1)
    traceback = np.zeros((len_p, len_q), dtype=np.int8)
    
    # Fill the matrix bottom-up
    # Base case (0,0)
    ca[0, 0] = np.linalg.norm(p[0] - q[0])
    
    # Fill first column (i > 0, j = 0)
    for i in range(1, len_p):
        ca[i, 0] = max(ca[i-1, 0], np.linalg.norm(p[i] - q[0]))
        traceback[i, 0] = 0  # Always from above for first column
    
    # Fill first row (i = 0, j > 0)
    for j in range(1, len_q):
        ca[0, j] = max(ca[0, j-1], np.linalg.norm(p[0] - q[j]))
        traceback[0, j] = 2  # Always from left for first row
    
    # Fill the rest of the matrix
    for i in range(1, len_p):
        for j in range(1, len_q):
            # Compare three possible previous cells
            val1 = ca[i-1, j]      # from above
            val2 = ca[i-1, j-1]    # from diagonal
            val3 = ca[i, j-1]      # from left
            
            # Find minimum value and its direction
            min_val = min(val1, val2, val3)
            
            # Store the direction
            if min_val == val1:
                traceback[i, j] = 0
            elif min_val == val2:
                traceback[i, j] = 1
            else:
                traceback[i, j] = 2
            
            # Calculate the final value for this cell
            ca[i, j] = max(min_val, np.linalg.norm(p[i] - q[j]))
    
    # Reconstruct the path by backtracking
    path = []
    i, j = len_p - 1, len_q - 1
    
    # Start from the end and follow the traceback
    while True:
        path.append((i, j))
        
        if i == 0 and j == 0:
            break
            
        # Move according to the traceback matrix
        direction = traceback[i, j]
        if direction == 0:      # from above
            i -= 1
        elif direction == 1:    # from diagonal
            i -= 1
            j -= 1
        else:                   # from left
            j -= 1
    
    # Reverse the path to get it in forward order
    path.reverse()
    
    return ca[len_p-1, len_q-1], ca, path


def _c_m_iterative_optimized(p, q):
    """
    Optimized iterative implementation of _c_m which returns distance, cost matrix, and path.
    """
    len_p, len_q = len(p), len(q)
    
    # Pre-compute distance matrix using vectorization
    # This avoids repeatedly calculating the same distances
    dist_matrix = np.zeros((len_p, len_q), dtype=np.float64)
    for i in range(len_p):
        dist_matrix[i, :] = np.linalg.norm(p[i].reshape(1, -1) - q, axis=1)
    
    # Initialize cost matrix and traceback matrix
    ca = np.ones((len_p, len_q), dtype=np.float64) * -1
    traceback = np.zeros((len_p, len_q), dtype=np.int8)
    
    # Base case (0,0)
    ca[0, 0] = dist_matrix[0, 0]
    
    # Fill first column (i > 0, j = 0)
    for i in range(1, len_p):
        ca[i, 0] = max(ca[i-1, 0], dist_matrix[i, 0])
        traceback[i, 0] = 0
    
    # Fill first row (i = 0, j > 0)
    for j in range(1, len_q):
        ca[0, j] = max(ca[0, j-1], dist_matrix[0, j])
        traceback[0, j] = 2
    
    # Fill the rest of the matrix with optimized access pattern
    # Process in column-major order for better cache locality in NumPy
    for j in range(1, len_q):
        for i in range(1, len_p):
            val1 = ca[i-1, j]      # from above
            val2 = ca[i-1, j-1]    # from diagonal
            val3 = ca[i, j-1]      # from left
            
            # Find minimum value using NumPy's min function
            candidates = np.array([val1, val2, val3])
            min_idx = np.argmin(candidates)
            min_val = candidates[min_idx]
            
            # Store the direction (0, 1, or 2)
            traceback[i, j] = min_idx
            
            # Calculate the final value for this cell
            ca[i, j] = max(min_val, dist_matrix[i, j])
    
    # Reconstruct the path (same as before)
    path = []
    i, j = len_p - 1, len_q - 1
    
    while True:
        path.append((i, j))
        
        if i == 0 and j == 0:
            break
            
        direction = traceback[i, j]
        if direction == 0:      # from above
            i -= 1
        elif direction == 1:    # from diagonal
            i -= 1
            j -= 1
        else:                   # from left
            j -= 1
    
    path.reverse()
    
    return ca[len_p-1, len_q-1], ca, path


def frdist(p, q, visualize=False):
    """
    Computes the discrete Fréchet distance between
    two curves. The Fréchet distance between two curves in a
    metric space is a measure of the similarity between the curves.
    The discrete Fréchet distance may be used for approximately computing
    the Fréchet distance between two arbitrary curves,
    as an alternative to using the exact Fréchet distance between a polygonal
    approximation of the curves or an approximation of this value.

    This is a Python 3.* implementation of the algorithm produced
    in Eiter, T. and Mannila, H., 1994. Computing discrete Fréchet distance.
    Tech. Report CD-TR 94/64, Information Systems Department, Technical
    University of Vienna.
    http://www.kr.tuwien.ac.at/staff/eiter/et-archive/cdtr9464.pdf

    Function dF(P, Q): real;
        input: polygonal curves P = (u1, . . . , up) and Q = (v1, . . . , vq).
        return: δdF (P, Q)
        ca : array [1..p, 1..q] of real;
        function c(i, j): real;
            begin
                if ca(i, j) > −1 then return ca(i, j)
                elsif i = 1 and j = 1 then ca(i, j) := d(u1, v1)
                elsif i > 1 and j = 1 then ca(i, j) := max{ c(i − 1, 1), d(ui, v1) }
                elsif i = 1 and j > 1 then ca(i, j) := max{ c(1, j − 1), d(u1, vj) }
                elsif i > 1 and j > 1 then ca(i, j) :=
                max{ min(c(i − 1, j), c(i − 1, j − 1), c(i, j − 1)), d(ui, vj ) }
                else ca(i, j) = ∞
                return ca(i, j);
            end; /* function c */

        begin
            for i = 1 to p do for j = 1 to q do ca(i, j) := −1.0;
            return c(p, q);
        end.

    Parameters
    ----------
    P : Input curve - two dimensional array of points
    Q : Input curve - two dimensional array of points

    Returns
    -------
    dist: float64
        The discrete Fréchet distance between curves `P` and `Q`.

    Examples
    --------
    >>> from frechetdist import frdist
    >>> P=[[1,1], [2,1], [2,2]]
    >>> Q=[[2,2], [0,1], [2,4]]
    >>> frdist(P,Q)
    >>> 2.0
    >>> P=[[1,1], [2,1], [2,2]]
    >>> Q=[[1,1], [2,1], [2,2]]
    >>> frdist(P,Q)
    >>> 0
    """
    p = np.array(p, np.float64)
    q = np.array(q, np.float64)

    len_p = len(p)
    len_q = len(q)

    if len_p == 0 or len_q == 0:
        raise ValueError('Input curves are empty.')
        
    # ca = (np.ones((len_p, len_q), dtype=np.float64) * -1)
    # dist = _c(ca, len_p-1, len_q-1, p, q)
    # dist_, ca, path = _c_m_iterative(p, q)

    dist, ca, path = _c_m_iterative_optimized(p, q)

    # if dist != dist_:
    #     print(f"Warning: Distances do not match: {dist} != {dist_}")

    if visualize:
        print(f"Visualized, distance: {dist}")
        visualize_fd(dist, ca, path, p, p, q, q)

    return dist


@jit(nopython=True)
def _bresenham_pairs(x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    """Generates the diagonal coordinates for the Fréchet distance calculation"""
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    max_dim = max(dx, dy) + 1  # Add 1 to include both endpoints
    pairs = np.zeros((max_dim, 2), dtype=np.int64)
    
    # Set the starting point
    pairs[0, 0] = x0
    pairs[0, 1] = y0
    
    # If both points are the same, return just one point
    if dx == 0 and dy == 0:
        return pairs[:1]
    
    x, y = x0, y0
    sx = 1 if x1 > x0 else -1
    sy = 1 if y1 > y0 else -1
    
    count = 1  # We already added the first point
    
    if dx > dy:
        err = dx // 2
        while x != x1:
            x += sx
            err -= dy
            if err < 0:
                y += sy
                err += dx
            pairs[count, 0] = x
            pairs[count, 1] = y
            count += 1
    else:
        err = dy // 2
        while y != y1:
            y += sy
            err -= dx
            if err < 0:
                x += sx
                err += dy
            pairs[count, 0] = x
            pairs[count, 1] = y
            count += 1
    
    # Return only the filled portion of the array
    return pairs[:count]

@jit(nopython=True)
def _get_corner_min_array(f_mat: np.ndarray, i: int, j: int) -> float:
    """Gets the minimum value from the neighboring cells in the cost matrix
    
    Parameters
    ----------
    f_mat : Cost matrix
    i, j : Current cell coordinates
    
    Returns
    -------
    float : Minimum value from neighboring cells
    """
    if i > 0 and j > 0:
        a = min(f_mat[i - 1, j - 1],
                f_mat[i, j - 1],
                f_mat[i - 1, j])
    elif i == 0 and j == 0:
        a = f_mat[i, j]
    elif i == 0:
        a = f_mat[i, j - 1]
    else:  # j == 0:
        a = f_mat[i - 1, j]
    return a

@jit(nopython=True)
def _fast_distance_matrix(p: np.ndarray, q: np.ndarray, diag: np.ndarray, 
                          dist_func) -> np.ndarray:
    """Compute the distance matrix efficiently using diagonal seeding
    
    Parameters
    ----------
    p, q : Input curves
    diag : Diagonal coordinates
    dist_func : Distance function
    
    Returns
    -------
    np.ndarray : Distance matrix
    """
    n_diag = diag.shape[0]
    diag_max = 0.0
    i_min = 0
    j_min = 0
    p_count = p.shape[0]
    q_count = q.shape[0]

    # Create the distance array
    dist = np.full((p_count, q_count), np.inf, dtype=np.float64)

    # Fill in the diagonal with the seed distance values
    for k in range(n_diag):
        i0 = diag[k, 0]
        j0 = diag[k, 1]
        d = dist_func(p[i0], q[j0])
        diag_max = max(diag_max, d)
        dist[i0, j0] = d

    for k in range(n_diag - 1):
        i0 = diag[k, 0]
        j0 = diag[k, 1]
        p_i0 = p[i0]
        q_j0 = q[j0]

        for i in range(i0 + 1, p_count):
            if np.isinf(dist[i, j0]):
                d = dist_func(p[i], q_j0)
                if d < diag_max or i < i_min:
                    dist[i, j0] = d
                else:
                    break
            else:
                break
        i_min = i

        for j in range(j0 + 1, q_count):
            if np.isinf(dist[i0, j]):
                d = dist_func(p_i0, q[j])
                if d < diag_max or j < j_min:
                    dist[i0, j] = d
                else:
                    break
            else:
                break
        j_min = j
    return dist

@jit(nopython=True)
def _fast_frechet_matrix(dist: np.ndarray, diag: np.ndarray, 
                         p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Compute the Fréchet distance matrix from the distance matrix
    
    Parameters
    ----------
    dist : Distance matrix
    diag : Diagonal coordinates
    p, q : Input curves
    
    Returns
    -------
    np.ndarray : Fréchet distance matrix
    """
    for k in range(diag.shape[0]):
        i0 = diag[k, 0]
        j0 = diag[k, 1]

        for i in range(i0, p.shape[0]):
            if np.isfinite(dist[i, j0]):
                c = _get_corner_min_array(dist, i, j0)
                if c > dist[i, j0]:
                    dist[i, j0] = c
            else:
                break

        # Add 1 to j0 to avoid recalculating the diagonal
        for j in range(j0 + 1, q.shape[0]):
            if np.isfinite(dist[i0, j]):
                c = _get_corner_min_array(dist, i0, j)
                if c > dist[i0, j]:
                    dist[i0, j] = c
            else:
                break
    return dist

@jit(nopython=True)
def euclidean(p: np.ndarray, q: np.ndarray) -> float:
    """Euclidean distance between two points
    
    Parameters
    ----------
    p, q : Points
    
    Returns
    -------
    float : Euclidean distance
    """
    d = p - q
    return math.sqrt(np.dot(d, d))

@jit(nopython=True)
def fdfd_matrix(p: np.ndarray, q: np.ndarray, dist_func=euclidean) -> float:
    """Fast discrete Fréchet distance using matrix approach
    
    Parameters
    ----------
    p, q : Input curves
    dist_func : Distance function (default: euclidean)
    
    Returns
    -------
    float : Fréchet distance between curves p and q
    """
    diagonal = _bresenham_pairs(0, 0, p.shape[0]-1, q.shape[0]-1)
    ca = _fast_distance_matrix(p, q, diagonal, dist_func)
    ca = _fast_frechet_matrix(ca, diagonal, p, q)
    return ca[p.shape[0]-1, q.shape[0]-1]

class FastDiscreteFrechet:
    """Fast implementation of discrete Fréchet distance algorithm"""
    
    def __init__(self, dist_func=euclidean):
        """Initialize with a distance function
        
        Parameters
        ----------
        dist_func : Distance function (default: euclidean)
        """
        self.dist_func = dist_func
        self.ca = np.zeros((1, 1))
        # Warm up JIT
        self.distance(np.array([[0.0, 0.0], [1.0, 1.0]]),
                     np.array([[0.0, 0.0], [1.0, 1.0]]))
    
    def distance(self, p: np.ndarray, q: np.ndarray) -> float:
        """Calculate the discrete Fréchet distance between two curves
        
        Parameters
        ----------
        p, q : Input curves as arrays of points
        
        Returns
        -------
        float : Fréchet distance between curves p and q
        """
        # Ensure inputs are numpy arrays
        p = np.asarray(p, dtype=np.float64)
        q = np.asarray(q, dtype=np.float64)
        
        # Handle empty curves
        if p.shape[0] == 0 or q.shape[0] == 0:
            raise ValueError('Input curves cannot be empty')
        
        # Special case: single point curves
        if p.shape[0] == 1 and q.shape[0] == 1:
            return self.dist_func(p[0], q[0])
            
        # Get diagonal path
        diagonal = _bresenham_pairs(0, 0, p.shape[0]-1, q.shape[0]-1)
        
        # Calculate distance matrix
        ca = _fast_distance_matrix(p, q, diagonal, self.dist_func)
        
        # Calculate Fréchet distance matrix
        ca = _fast_frechet_matrix(ca, diagonal, p, q)
        
        self.ca = ca  # Store for inspection if needed
        return ca[p.shape[0]-1, q.shape[0]-1]


def dFD_distance(x, y):
    """
    Compute the Fréchet Distance between two sets of points.
    """
    x = x.transpose(1, 0)
    y = y.transpose(1, 0)

    # # Measure time for frdist
    # start = time.perf_counter()
    # d = frdist(x, y)
    # t1 = time.perf_counter() - start

    # # Create a FastDiscreteFrechet instance
    fdf = FastDiscreteFrechet()

    # # Measure time for FastDiscreteFrechet
    # start = time.perf_counter()
    d_ = fdf.distance(x, y)
    # t2 = time.perf_counter() - start

    # print(f"frdist time: {t1:.6f} seconds")
    # print(f"FastDiscreteFrechet time: {t2:.6f} seconds")

    # if d != d_:
    #     print(f"Warning: Distances do not match: {d} != {d_}")
    # else:
    #     print(f"Distances match: {d} == {d_}")

    return d_

def _compute_row(i, dataset1, dataset2):
    """Helper function to compute distances for one row"""
    return [dFD_distance(dataset1[i], dataset2[j]) for j in range(dataset2.shape[0])]

def dFD_matrix(dataset1, dataset2):
    """
    Compute the Fréchet Distance matrix between two datasets using parallel processing.
    """
    # Convert to numpy arrays if they're torch tensors
    if torch.is_tensor(dataset1):
        dataset1 = dataset1.cpu().numpy()
    if torch.is_tensor(dataset2):
        dataset2 = dataset2.cpu().numpy()
    
    # Parallel computation of distances
    n_jobs = -1  # Use all available cores
    results = Parallel(n_jobs=n_jobs)(
        delayed(_compute_row)(i, dataset1, dataset2) 
        for i in range(dataset1.shape[0])
    )
    
    # Convert results to torch tensor
    fd_matrix = torch.tensor(np.array(results), dtype=torch.float32)
    
    return fd_matrix


def dFD(dataset1, dataset2, visualize=True):
    """
    Compute the Fréchet Distance between two datasets.
    
    Parameters:
    -----------
    dataset1 : torch.Tensor
        First dataset, shape (n_samples, n_channels, n_timepoints)
    dataset2 : torch.Tensor
        Second dataset, shape (n_samples, n_channels, n_timepoints)
    option : str
        Option for distance calculation ('default' or 'matrix')
    
    Returns:
    --------
    dFD : float or torch.Tensor
        Fréchet Distance between the two datasets
    """
    
    if not isinstance(dataset1, np.ndarray):
        dataset1 = dataset1.float().cpu().numpy()
    if not isinstance(dataset2, np.ndarray):
        dataset2 = dataset2.float().cpu().numpy()

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

    if visualize:
        frdist(dataset1[0].transpose(1, 0), dataset2[0].transpose(1, 0), visualize)
    fd = dFD_matrix(dataset1, dataset2)
    fd = torch.mean(fd)

    return fd


def visualize_fd(fd_score, matrix, best_path, x, x_og, y, y_og):
    """
    Visualize the cost matrix, local costs, and Frechet distance warping path.
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

    # Use matrix as is - already contains Euclidean distances
    local_costs = np.array([matrix[i, j] for (i, j) in best_path])

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

    # Cost matrix
    ax = axs[0]
    # Set vmax to a fixed value for better comparison
    im = ax.imshow(matrix, origin='lower', aspect='auto', cmap='viridis')
    ax.set_title('Cost matrix (mV)')
    ax.set_xlabel('Unshifted (timesteps)')
    ax.set_ylabel('Shifted (timesteps)')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Plot warping path
    path_x, path_y = zip(*best_path)
    warping_line = ax.plot(path_y, path_x, color='red', linewidth=2, label='Warping path')[0]
    
    # Find the maximum value coordinates along the path
    max_idx = np.argmax(local_costs)
    max_i, max_j = best_path[max_idx]
    
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
    
    # Place legend below plot with all items in one line
    ax.legend(handles=[warping_line], loc='upper center', 
              bbox_to_anchor=(0.5, -0.15), frameon=False, ncol=2)

    # Local distances plot
    ax = axs[1]
    local_cost_line = ax.plot(local_costs, color=colors[1], linewidth=2, label='Local distances')[0]
    
    # Mark the maximum distance point
    max_point = ax.plot(max_idx, local_costs[max_idx], 'o', color=colors[4], markersize=5, label='Max dist.: {:.2f} mV'.format(local_costs[max_idx]))[0]

    ax.set_title('Distance along path')
    ax.set_xlabel('Path index')
    ax.set_ylabel('Distance (mV)')
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
    # Set consistent y-axis scale for local cost plot
    ax.set_ylim(1.8, 2.6)
    
    # Legend for the distances plot
    ax.legend(handles=[local_cost_line, max_point], loc='upper center', 
              bbox_to_anchor=(0.5, -0.15), frameon=False, ncol=2, 
              handletextpad=0.5, columnspacing=1.0)

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

    # Draw warping path connections - only every 10th connection
    for idx, (i, j) in enumerate(best_path):
        if idx % 10 == 0:
            ax.plot([i, j], [x_plot[i] + x_offset, y_plot[j] - y_offset], 
                    color='gray', alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Highlight the maximum distance connection
    max_i, max_j = best_path[max_idx]
    max_warp = ax.plot([max_i, max_j], [x_plot[max_i] + x_offset, y_plot[max_j] - y_offset], 
            color=colors[4], linewidth=2, alpha=0.8, label='Max conn.')[0]
    
    ax.set_title('FD warping path connections')
    ax.set_xlabel('Timesteps')
    # Remove y-ticks and label for clarity
    ax.set_yticks([])
    ax.set_ylabel('')
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)

    # Place legend below plot with all items on one line
    ax.legend(handles=[line_x, line_y, max_warp], loc='upper center',
              bbox_to_anchor=(0.5, -0.15), frameon=False, ncol=3)

    plt.tight_layout(rect=[0, 0.05, 1, 1])  # Allow space for legends at bottom
    # Save as both PNG and PDF
    plt.savefig("dfd.png", dpi=300, transparent=True)
    plt.savefig("dfd.pdf", transparent=True)
    plt.close()


def run_dfd(X, Y, labels, **kwargs):
    """
    Run dFD metric on a model against training and test data.
    
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
        dFD computation option (default: 'default', min: 'min')
        
    Returns:
    --------
    tuple:
        (metric_test_model, metric_test_train) dictionaries containing dFD values
    """
    dFD.__name__ = 'dFD'
    return run_metric(dFD, X, Y, labels, **kwargs)


if __name__ == "__main__":
    # Example usage
    curve1 = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]])
    curve2 = np.array([[0, 0], [1, 2], [2, 3], [3, 4], [4, 5]])
    
    distance = frdist(curve1, curve2)

    fdf = FastDiscreteFrechet()
    distance_optimized = fdf.distance(curve1, curve2)

    print(f"Fréchet Distance: {distance}")
    print(f"Optimized Fréchet Distance: {distance_optimized}")