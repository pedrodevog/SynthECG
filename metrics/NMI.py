import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import mutual_info_score
from tqdm import tqdm
import math
from datetime import datetime


def compute_joint_histogram(x, y, num_bins=64, normalize=True, visualize=False):
    """
    Compute the joint histogram between two univariate time series.
    
    Parameters:
    -----------
    x : array-like
        First time series
    y : array-like
        Second time series
    num_bins : int, optional
        Number of bins to use for each dimension (default: 64)
    normalize : bool, optional
        Whether to normalize the histogram to get a joint probability distribution (default: True)
    
    Returns:
    --------
    joint_hist : 2D numpy array
        The joint histogram/distribution
    x_edges : 1D numpy array
        The edges of the bins for x
    y_edges : 1D numpy array
        The edges of the bins for y
    """
    # Compute the joint histogram
    joint_hist, x_edges, y_edges = np.histogram2d(x, y, bins=num_bins)
    
    # Normalize if requested (to get a probability distribution)
    if normalize:
        joint_hist = joint_hist / np.sum(joint_hist)

    # Plot the joint histogram
    if visualize:
        plt.imshow(joint_hist.T, origin='lower', cmap='viridis', aspect='auto')
        plt.colorbar(label='Joint Probability')
        plt.title('Joint Histogram')
        plt.xlabel('x bins')
        plt.ylabel('y bins')
        plt.savefig('joint_histogram.png', dpi=300)
        plt.close()
    
    return joint_hist, x_edges, y_edges


def compute_mutual_information(joint_dist, normalized=True):
    """
    Compute the (optionally normalized) mutual information based on a joint probability distribution.

    Parameters:
    -----------
    joint_dist : 2D numpy array
        Joint probability distribution (normalized joint histogram)
    normalized : bool, optional
        If True, return normalized MI (NMI) as (H(x) + H(y)) / H(x, y). If False, return MI in bits.

    Returns:
    --------
    mi : float
        Mutual information value (normalized if normalized=True)
    """
    # Ensure input is a probability distribution
    if not np.isclose(np.sum(joint_dist), 1.0):
        raise ValueError("Input must be a normalized joint distribution (sum=1)")

    # Compute marginal distributions
    p_x = np.sum(joint_dist, axis=1)  # sum over y
    p_y = np.sum(joint_dist, axis=0)  # sum over x

    # Compute entropies
    hx = -np.sum(p_x[p_x > 0] * np.log2(p_x[p_x > 0]))
    hy = -np.sum(p_y[p_y > 0] * np.log2(p_y[p_y > 0]))
    hxy = -np.sum(joint_dist[joint_dist > 0] * np.log2(joint_dist[joint_dist > 0]))

    # Compute MI
    mi = hx + hy - hxy

    if normalized:
        if hxy > 0:
            return (hx + hy) / hxy - 1
        else:
            return 0.0
    else:
        return mi


def NMI(signals, num_bins=32):
    """
    Compute mutual information statistics across multiple signals.
    
    Parameters:
    -----------
    signals : 3D numpy array
        Array of signals with shape (num_signals, num_leads, signal_length)
    num_bins : int, optional
        Number of bins for histograms (default: 64)
    
    Returns:
    --------
    mi_mean : 2D numpy array
        Mean mutual information between leads
    mi_std : 2D numpy array
        Standard deviation of mutual information between leads
    """
    num_signals = signals.shape[0]
    num_leads = signals.shape[1]
    
    # Initialize array to store MI values for all signals
    all_mi_values = np.zeros((num_signals, num_leads, num_leads))
    
    # Use tqdm for progress bar
    for signal_idx in tqdm(range(num_signals), desc="Computing NMI for signals"):
        current_signal = signals[signal_idx]
        
        # Calculate MI for each lead pair
        for i in range(num_leads):
            for j in range(i):  # j < i for lower triangular
                joint_hist, _, _ = compute_joint_histogram(current_signal[i], current_signal[j], num_bins=num_bins)
                mi = compute_mutual_information(joint_hist)
                # mi = mutual_information_alternative(current_signal[i], current_signal[j], num_bins=num_bins)
                all_mi_values[signal_idx, i, j] = mi
                all_mi_values[signal_idx, j, i] = mi  # Mirror for easier averaging
    
    # Calculate mean and standard deviation across signals
    mi_mean = np.mean(all_mi_values, axis=0)
    mi_std = np.std(all_mi_values, axis=0)
    
    return mi_mean, mi_std


def PCC(signals):
    """
    Compute Pearson correlation coefficients (PCC) between leads for multiple signals.
    
    Parameters:
    -----------
    signals : 3D numpy array
        Array of signals with shape (num_signals, num_leads, signal_length)
    
    Returns:
    --------
    pcc_matrix : 2D numpy array
        Matrix of Pearson correlation coefficients between leads
    """
    num_signals = signals.shape[0]
    num_leads = signals.shape[1]
    
    # Initialize array to store PCC values for all signals
    all_pcc_values = np.zeros((num_signals, num_leads, num_leads))
    
    # Use tqdm for progress bar
    for signal_idx in tqdm(range(num_signals), desc="Computing PCC for signals"):
        current_signal = signals[signal_idx]
        
        # Calculate PCC for each lead pair
        for i in range(num_leads):
            for j in range(i):  # j < i for lower triangular
                pcc, _ = stats.pearsonr(current_signal[i], current_signal[j])
                all_pcc_values[signal_idx, i, j] = pcc
                all_pcc_values[signal_idx, j, i] = pcc  # Mirror for easier averaging
    
    # Calculate mean PCC across signals
    pcc_mean = np.mean(all_pcc_values, axis=0)
    pcc_std = np.std(all_pcc_values, axis=0)
    
    return pcc_mean, pcc_std


def mutual_information_alternative(x, y, num_bins=64, normalized=True):
    """
    Alternative implementation using sklearn's mutual_info_score, with optional normalization.

    Parameters:
    -----------
    x : array-like
        First time series
    y : array-like
        Second time series
    num_bins : int, optional
        Number of bins for discretization (default: 64)
    normalized : bool, optional
        If True, return normalized MI (NMI) in [0, 1]. If False, return MI in nats.

    Returns:
    --------
    mi : float
        Mutual information value (normalized if normalized=True)
    """
    # Discretize the data
    x_binned = np.digitize(x, np.linspace(min(x), max(x), num_bins))
    y_binned = np.digitize(y, np.linspace(min(y), max(y), num_bins))

    # Compute mutual information (in nats)
    mi = mutual_info_score(x_binned, y_binned)

    if normalized:
        # Compute marginal entropies
        hx = mutual_info_score(x_binned, x_binned)
        hy = mutual_info_score(y_binned, y_binned)
        denom = hx + hy
        if denom > 0:
            return 2 * mi / denom
        else:
            return 0.0
    else:
        return mi


def plot_all_leads_separately(signal, lead_names, save_path='all_leads_plot.png'):
    """
    Plot all ECG leads in separate subplots.
    
    Parameters:
    -----------
    signal : numpy.ndarray
        ECG signal with shape (num_leads, signal_length)
    lead_names : list
        List of lead names
    save_path : str
        Path to save the figure
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import math
    
    num_leads = signal.shape[0]
    
    # Calculate grid dimensions (trying to make it as square as possible)
    grid_size = math.ceil(math.sqrt(num_leads))
    nrows = grid_size
    ncols = math.ceil(num_leads / nrows)
    
    # Create the figure
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 12))
    
    # Flatten axes array for easier indexing
    axes = axes.flatten()
    
    # Plot each lead in a separate subplot
    for i in range(num_leads):
        ax = axes[i]
        ax.plot(signal[i], 'b-')
        ax.set_title(f'Lead {lead_names[i]}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Amplitude')
        ax.grid(True, alpha=0.3)
    
    # Remove any unused subplots
    for i in range(num_leads, len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    return fig


def visualize_NMI(mi_mean, mi_std, lead_names, model="", colorbar=True):
    num_leads = len(lead_names)

    plt.rcParams.update({
        'font.family': 'DejaVu Serif',  # Professional serif font
        'mathtext.fontset': 'cm',       # Computer Modern for math text
        'font.size': 11,
        'axes.labelsize': 14,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 16,
        # 'figure.figsize': (11, 4),      # Adjusted width for 2 subplots
        'figure.dpi': 300,              # High DPI for print
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',        # Avoid wasted space
        'savefig.pad_inches': 0.05,     # Minimal padding
    })

    # Create a mask for the upper triangular part (including diagonal)
    mask = np.triu(np.ones_like(mi_mean, dtype=bool))

    # Set up the figure with conditional sizing based on colorbar
    if colorbar:
        plt.figure(figsize=(12, 10))
    else:
        plt.figure(figsize=(10, 10))  # Square figure to maintain matrix ratios
    
    ax = plt.gca()

    # Remove the box around the plot
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Create a masked array
    masked_mi_mean = np.ma.array(mi_mean, mask=mask)

    # Display the mean MI matrix with scale from 0 to 1
    im = plt.imshow(masked_mi_mean, cmap=plt.cm.Blues, interpolation='nearest',
                    vmin=0, 
                    vmax=0.40
                    )

    # Add colorbar only if requested
    if colorbar:
        # Create a shorter colorbar positioned closer to the matrix
        # Place the colorbar more to the left (closer to the matrix) and add more padding for the label
        cbar = plt.colorbar(im, shrink=0.5, pad=0)  # smaller pad moves it closer to the matrix
        cbar.set_label('Mean NMI', fontsize=20, labelpad=20)  # increase labelpad for more space
        cbar.ax.tick_params(labelsize=18)

    # Add lead labels
    plt.xticks(np.arange(num_leads), lead_names, rotation=0, fontsize=16)
    plt.yticks(np.arange(num_leads), lead_names, fontsize=16)

    max_val = min(1.0, np.nanmax(mi_mean))
    # max_val = 1

    # Add text annotations with BOTH mean and std values (lower triangle)
    for i in range(num_leads):
        for j in range(i):  # j < i for lower triangular
            # Calculate text color based on background darkness
            val = mi_mean[i, j]
            color_intensity = val / max_val if max_val > 0 else 0
            text_color = 'white' if color_intensity > 0.6 else 'black'
            
            # Add mean value in normal size
            plt.text(j, i-0.15, f"{mi_mean[i, j]:.2f}",
                ha="center", va="center", color=text_color,
                fontsize=13, fontweight='bold')
            
            # Add std value in slightly bigger size below the mean
            plt.text(j, i+0.15, f"±{mi_std[i, j]:.2f}",
                ha="center", va="center", color=text_color,
                fontsize=11)

    plt.tight_layout()

    # Remove the box
    plt.box(False)

    if model:
        plt.savefig(f'NMI_{model}.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'NMI_{model}.pdf', bbox_inches='tight')
    else:
        plt.savefig('NMI.png', dpi=300, bbox_inches='tight')
        plt.savefig('NMI.pdf', bbox_inches='tight')

    plt.close()

def visualize_NMI_class(mi_mean, mi_std, lead_names, model=""):
    class_names = ["CD", "HYP", "MI", "NORM", "STTC", "Unlabeled"]

    num_classes = mi_mean.shape[0]
    num_leads = len(lead_names)

    plt.rcParams.update({
        'font.family': 'DejaVu Serif',
        'mathtext.fontset': 'cm',
        'font.size': 11,
        'axes.labelsize': 14,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 16,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
    })

    # Determine subplot grid size
    ncols = int(math.ceil(np.sqrt(num_classes)))
    nrows = int(math.ceil(num_classes / ncols))

    # Make subplots bigger
    fig, axes = plt.subplots(nrows, ncols, figsize=(7*ncols, 7*nrows), dpi=300)
    axes = axes.flatten()

    # For consistent colorbar, fix vmin/vmax
    vmin = 0
    vmax = 0.42

    ims = []
    for c in range(num_classes):
        ax = axes[c]
        # Create a mask for the upper triangular part (including diagonal)
        mask = np.triu(np.ones_like(mi_mean[c], dtype=bool))
        masked_mi_mean = np.ma.array(mi_mean[c], mask=mask)

        im = ax.imshow(masked_mi_mean, cmap=plt.cm.Blues, interpolation='nearest', vmin=vmin, vmax=vmax)
        ims.append(im)
        max_val = min(1.0, np.nanmax(mi_mean[c]))

        # Add lead labels
        ax.set_xticks(np.arange(num_leads))
        ax.set_yticks(np.arange(num_leads))
        ax.set_xticklabels(lead_names, rotation=0, fontsize=12)
        ax.set_yticklabels(lead_names, fontsize=12)

        # Add text annotations (lower triangle)
        for i in range(num_leads):
            for j in range(i):
                val = mi_mean[c, i, j]
                color_intensity = val / max_val if max_val > 0 else 0
                text_color = 'white' if color_intensity > 0.6 else 'black'
                ax.text(j, i-0.15, f"{mi_mean[c, i, j]:.2f}",
                        ha="center", va="center", color=text_color,
                        fontsize=9, fontweight='bold')
                ax.text(j, i+0.15, f"±{mi_std[c, i, j]:.2f}",
                        ha="center", va="center", color=text_color,
                        fontsize=8)

        # Set the title closer to the matrix by reducing pad and using set_title
        ax.set_title(class_names[c], fontsize=22, fontweight='bold', pad=0)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Hide unused subplots
    for i in range(num_classes, len(axes)):
        axes[i].set_visible(False)

    # Add a single common colorbar to the right of all subplots
    # Position colorbar directly next to the plots with no gap
    fig.subplots_adjust(right=0.94)  # Give more space to subplots
    cbar_ax = fig.add_axes([0.94, 0.25, 0.025, 0.5])  # [left, bottom, width, height] - shorter and chunkier
    cbar = fig.colorbar(ims[0], cax=cbar_ax)
    cbar.set_label('Mean NMI', fontsize=16, labelpad=18)
    cbar.ax.tick_params(labelsize=15)

    # Use subplots_adjust instead of tight_layout to avoid conflicts with manual colorbar positioning
    fig.subplots_adjust(left=0.05, bottom=0.05, top=0.95, right=0.94, wspace=0.1, hspace=0.15)
    if model:
        plt.savefig(f'NMI_class_subplots_{model}.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'NMI_class_subplots_{model}.pdf', bbox_inches='tight')
    else:
        plt.savefig('NMI_class_subplots.png', dpi=300, bbox_inches='tight')
        plt.savefig('NMI_class_subplots.pdf', bbox_inches='tight')
    plt.close()


def save_NMI_results(mi_mean, mi_std, lead_names, model="", mi_mean_c=None, mi_std_c=None, class_names=None):
    """
    Save NMI results to a text file with detailed statistics.
    
    Parameters:
    -----------
    mi_mean : 2D numpy array
        Mean mutual information between leads
    mi_std : 2D numpy array
        Standard deviation of mutual information between leads
    lead_names : list
        List of lead names
    model : str
        Model name for filename
    mi_mean_c : 3D numpy array, optional
        Mean MI per class with shape (num_classes, num_leads, num_leads)
    mi_std_c : 3D numpy array, optional
        Std MI per class with shape (num_classes, num_leads, num_leads)
    class_names : list, optional
        List of class names
    """
    filename = f'results/NMI_results_{model}.txt' if model else 'results/NMI_results.txt'
    
    with open(filename, 'w') as f:
        f.write(f"Normalized Mutual Information (NMI) Results\n")
        f.write(f"Model: {model}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*60 + "\n\n")
        
        # Overall NMI statistics
        f.write("OVERALL NMI STATISTICS\n")
        f.write("-"*30 + "\n")
        
        # Summary statistics
        lower_tri_mask = np.tril(np.ones_like(mi_mean, dtype=bool), k=-1)
        mean_values = mi_mean[lower_tri_mask]
        std_values = mi_std[lower_tri_mask]
        
        f.write(f"Overall Mean NMI: {np.mean(mean_values):.4f} ± {np.mean(std_values):.4f}\n")
        f.write(f"Min NMI: {np.min(mean_values):.4f}\n")
        f.write(f"Max NMI: {np.max(mean_values):.4f}\n")
        f.write(f"Median NMI: {np.median(mean_values):.4f}\n\n")
        
        # Detailed lead pair results
        f.write("DETAILED LEAD PAIR RESULTS\n")
        f.write("-"*30 + "\n")
        f.write(f"{'Lead Pair':<15} {'Mean NMI':<10} {'Std NMI':<10}\n")
        f.write("-"*35 + "\n")
        
        for i in range(len(lead_names)):
            for j in range(i):
                f.write(f"{lead_names[j]}-{lead_names[i]:<10} {mi_mean[i,j]:.4f}     {mi_std[i,j]:.4f}\n")
        
        # Class-specific results if provided
        if mi_mean_c is not None and mi_std_c is not None and class_names is not None:
            f.write("\n\nCLASS-SPECIFIC NMI STATISTICS\n")
            f.write("-"*30 + "\n")
            
            for c, class_name in enumerate(class_names):
                f.write(f"\nClass: {class_name}\n")
                f.write("~"*20 + "\n")
                
                # Summary for this class
                class_mean_values = mi_mean_c[c][lower_tri_mask]
                class_std_values = mi_std_c[c][lower_tri_mask]
                
                if np.any(class_mean_values > 0):  # Check if class has data
                    f.write(f"Class Mean NMI: {np.mean(class_mean_values):.4f} ± {np.mean(class_std_values):.4f}\n")
                    f.write(f"Class Min NMI: {np.min(class_mean_values):.4f}\n")
                    f.write(f"Class Max NMI: {np.max(class_mean_values):.4f}\n\n")
                    
                    # Top 5 highest NMI pairs for this class
                    pairs_with_values = []
                    for i in range(len(lead_names)):
                        for j in range(i):
                            pairs_with_values.append((f"{lead_names[j]}-{lead_names[i]}", 
                                                    mi_mean_c[c, i, j], mi_std_c[c, i, j]))
                    
                    pairs_with_values.sort(key=lambda x: x[1], reverse=True)
                    f.write("Top 5 Lead Pairs:\n")
                    for k, (pair, mean_val, std_val) in enumerate(pairs_with_values[:5]):
                        f.write(f"  {k+1}. {pair:<12} {mean_val:.4f} ± {std_val:.4f}\n")
                else:
                    f.write("No data available for this class\n")
        
        f.write(f"\n\nResults saved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


# # Example usage for ECG data analysis
if __name__ == "__main__":
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    data_dir = 'data/ptbxl'
    signals = np.load(os.path.join(data_dir, 'test_data.npy'))
    labels = np.load(os.path.join(data_dir, 'test_labels.npy'))
    one_signal = signals[0]

    # Calculate MI between all leads for a single signal
    num_leads = one_signal.shape[0]
    # Define the standard 12-lead ECG order
    standard_lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
    # If your signals are not in this order, you need to reorder them accordingly.
    # For example, if your signals are in the order ['I', 'II', 'V1', ...], create a mapping:
    # Map from your current order to the standard order
    current_lead_names = ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'III', 'aVR', 'aVL', 'aVF']
    # Find indices to reorder signals to standard order
    reorder_indices = [current_lead_names.index(lead) for lead in standard_lead_names if lead in current_lead_names]

    class_names = ["CD", "HYP", "MI", "NORM", "STTC", "Unlabeled"]
    
    # Reorder the signal for plotting and analysis
    one_signal = one_signal[reorder_indices]
    signals = signals[:, reorder_indices, :]
    num_leads = len(reorder_indices)
    lead_names = [standard_lead_names[i] for i in range(num_leads)]

    # Compute MI statistics across signals
    mi_mean, mi_std = NMI(signals, num_bins=32)

    visualize_NMI(mi_mean, mi_std, lead_names)

    num_classes = labels.shape[1]

    mi_mean_c = np.zeros((num_classes+1, num_leads, num_leads))
    mi_std_c = np.zeros((num_classes+1, num_leads, num_leads))
    for c in range(num_classes):
        idx_c = np.where(labels[:, c] == 1)[0]
        if len(idx_c) > 0:
            mi_mean_c[c], mi_std_c[c] = NMI(signals[idx_c], num_bins=32)
    # Calculate for signals with no classes (all zeros in label row)
    idx_none = np.where(labels.sum(axis=1) == 0)[0]
    if len(idx_none) > 0:
        mi_mean_c[-1], mi_std_c[-1] = NMI(signals[idx_none], num_bins=32)
    
    visualize_NMI_class(mi_mean_c, mi_std_c, lead_names)

    save_NMI_results(mi_mean, mi_std, lead_names, "Baseline", mi_mean_c, mi_std_c, class_names)

    MODELS = ['SSSD', 'DSAT', 'WaveGan', 'P2P']

    # Load ECG data
    import glob
    for model in MODELS:
        data_dir = f'experiments/test/ECG_{model}/run_1/'
        data_files = glob.glob(os.path.join(data_dir, 'data_*.npy'))
        label_files = glob.glob(os.path.join(data_dir, 'labels_*.npy'))
        
        if not data_files:
            raise FileNotFoundError(f"No data files found matching pattern 'data_*.npy' in {data_dir}")
        if not label_files:
            raise FileNotFoundError(f"No label files found matching pattern 'labels_*.npy' in {data_dir}")
        
        signals = np.load(data_files[0])  # Load the first matching file
        labels = np.load(label_files[0])  # Load the first matching file

        signals = signals[:, reorder_indices, :]
        num_leads = len(reorder_indices)
        lead_names = [standard_lead_names[i] for i in range(num_leads)]

        # Compute MI statistics across signals
        mi_mean, mi_std = NMI(signals, num_bins=32)
        
        # ----------------------
        # FIXED VISUALIZATION CODE - Combined Mean and Std MI Matrix (Lower Triangle)
        # ----------------------
        if model == 'DSAT':
            visualize_NMI(mi_mean, mi_std, lead_names, model, colorbar=True)
        else:
            visualize_NMI(mi_mean, mi_std, lead_names, model, colorbar=True)

        print("\nComputing NMI statistics per class...")

        num_classes = labels.shape[1]

        mi_mean_c = np.zeros((num_classes+1, num_leads, num_leads))
        mi_std_c = np.zeros((num_classes+1, num_leads, num_leads))
        for c in range(num_classes):
            idx_c = np.where(labels[:, c] == 1)[0]
            if len(idx_c) > 0:
                mi_mean_c[c], mi_std_c[c] = NMI(signals[idx_c], num_bins=32)
        # Calculate for signals with no classes (all zeros in label row)
        idx_none = np.where(labels.sum(axis=1) == 0)[0]
        if len(idx_none) > 0:
            mi_mean_c[-1], mi_std_c[-1] = NMI(signals[idx_none], num_bins=32)
        
        visualize_NMI_class(mi_mean_c, mi_std_c, lead_names, model)

        save_NMI_results(mi_mean, mi_std, lead_names, model, mi_mean_c, mi_std_c, class_names)

    # MI calculation between leads 1 and 2
    lead1 = one_signal[0]
    lead2 = one_signal[1]
    joint_hist, x_edges, y_edges = compute_joint_histogram(lead1, lead2, num_bins=64, visualize=True)
    mi = compute_mutual_information(joint_hist)
    print(f"NMI between lead I and lead II: {mi:.4f}")
    print(f"NMI (alternative method) between lead I and lead II: {mutual_information_alternative(lead1, lead2, num_bins=64):.4f}")
   
    # show the joint histogram
    plt.imshow(joint_hist, origin='lower', cmap='viridis')
    plt.colorbar(label='Joint Probability')
    plt.title(f'Joint Histogram (NMI = {mi:.4f})')
    plt.xlabel('Lead I bins')
    plt.ylabel('Lead II bins')
    plt.savefig('joint_histogram_leadI_leadII.png', dpi=300)
    plt.close()
    
    # show the signals
    plt.figure(figsize=(10, 4))
    plt.plot(lead1, 'b-', label='Lead 1')
    plt.plot(lead2, 'r-', alpha=0.7, label='Lead 2')
    plt.title('Lead I and Lead II Signals')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('leadI_leadII_signals.png', dpi=300)
    plt.close()
