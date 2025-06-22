import torch
import numpy as np
import matplotlib.pyplot as plt
import wandb
# Importing custom metrics
from metrics.MMD import run_mmd
from metrics.PSD import run_psd_mmd, run_psd_prd
from metrics.FID import run_fid
from metrics.TXTY import run_TSTR, run_TRTS


def wandb_figure(data, labels, log_dict, prefix=""):
    sample_rate = 100
    data = data.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    
    n_samples, n_channels, n_timepoints = data.shape
    timestamp = np.linspace(0, n_timepoints / sample_rate, n_timepoints)

    # Log sample plots
    for idx in range(4):
        # Create subplot for all channels in this sample
        fig, axes = plt.subplots(n_channels, 1, figsize=(12, 4*n_channels))
        fig.suptitle(f'Sample {idx} (Class {np.argmax(labels[idx, :])})')
        
        for ch in range(n_channels):
            ax = axes[ch] if n_channels > 1 else axes
            ax.plot(timestamp, data[idx, ch, :])
            ax.set_title(f'Channel {ch+1}')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Amplitude (mV)')
            ax.grid(True)
        
        plt.tight_layout()
        log_dict[f"{prefix}sample_{idx}_plot"] = wandb.Image(plt)
        plt.close()

    return log_dict


def log_real_vs_fake(ref_bank, fake_samples, log_dict, prefix="real_vs_fake_", sample_rate=100):
    """
    Log a side-by-side comparison of reference statistics (mean ± std) and fake samples.
    
    For each class in the reference bank:
      - Left column: the reference mean with shaded ±1 std deviation range
      - Right column: the corresponding generated fake sample
    
    Args:
        ref_bank (dict): Mapping from class index to tuple of (mean, std) arrays, each with shape (channels, timepoints)
        fake_samples (dict): Fake samples with shape (num_classes, channels, timepoints)
        log_dict (dict): Dictionary to update with wandb images
        prefix (str): Prefix for the log keys
        sample_rate (int): Sample rate for constructing the time vector
    """
    class_names = ["CD", "HYP", "MI", "NORM", "STTC"]

    # Get dimensions from the first class in ref_bank
    first_class = next(iter(ref_bank.values()))
    n_timepoints = first_class[0].shape[-1]
    
    time_vec = np.linspace(0, n_timepoints / sample_rate, n_timepoints)
    
    # Loop over every class in the reference bank
    for cls, (ref_signal, mean, std, _) in ref_bank.items():
        # Check that fake_samples has an entry for this class
        if cls not in fake_samples:
            print(f"Warning: Fake sample for class {cls} not available; skipping.")
            continue
        
        # Get corresponding data from fake samples
        ref_fake, mean_fake, std_fake, n_samples = fake_samples[cls]
        
        # Convert tensors to numpy arrays
        ref_signal = ref_signal.cpu().detach().numpy() if torch.is_tensor(ref_signal) else ref_signal
        mean = mean.cpu().detach().numpy() if torch.is_tensor(mean) else mean
        std = std.cpu().detach().numpy() if torch.is_tensor(std) else std
        ref_fake = ref_fake.cpu().detach().numpy() if torch.is_tensor(ref_fake) else ref_fake
        mean_fake = mean_fake.cpu().detach().numpy() if torch.is_tensor(mean_fake) else mean_fake
        std_fake = std_fake.cpu().detach().numpy() if torch.is_tensor(std_fake) else std_fake
        
        n_channels = mean.shape[0]
        
        # Create a subplot for each channel with two columns
        fig, axes = plt.subplots(n_channels, 2, figsize=(12, 3 * n_channels))
        if n_channels == 1:
            axes = np.expand_dims(axes, axis=0)
        
        for ch in range(n_channels):
            # Plot mean and std range
            axes[ch, 0].plot(time_vec, ref_signal[ch], 'r-', label='Reference')
            axes[ch, 0].plot(time_vec, mean[ch], 'b-', label='Mean')
            axes[ch, 0].fill_between(time_vec, 
                                   mean[ch] - std[ch],
                                   mean[ch] + std[ch],
                                   color='b', alpha=0.2, label='±1 std')
            axes[ch, 0].set_title(f'Class {cls} Reference - Channel {ch+1}')
            axes[ch, 0].set_xlabel("Time (s)")
            axes[ch, 0].set_ylabel("Amplitude")
            axes[ch, 0].grid(True)
            axes[ch, 0].legend()
            
            # Plot fake sample
            axes[ch, 1].plot(time_vec, ref_fake[ch], 'r-', label='Reference')
            axes[ch, 1].plot(time_vec, mean_fake[ch], 'b-', label='Mean')
            axes[ch, 1].fill_between(time_vec, 
                                   mean_fake[ch] - std_fake[ch],
                                   mean_fake[ch] + std_fake[ch],
                                   color='b', alpha=0.2, label='±1 std')
            axes[ch, 1].set_title(f'Class {cls} Reference - Channel {ch+1}')
            axes[ch, 1].set_xlabel("Time (s)")
            axes[ch, 1].set_ylabel("Amplitude")
            axes[ch, 1].grid(True)
            axes[ch, 1].legend()
        
        plt.tight_layout()
        log_dict[f"{class_names[cls]} (n = {n_samples})"] = wandb.Image(fig)
        plt.close(fig)
    
    return log_dict


def create_reference_bank(samples, labels, num_classes):
    """
    Create a reference bank with mean and std statistics per timepoint for each class and channel.
    
    Args:
        real_samples (np.ndarray): Real signals with shape (N, channels, timepoints).
        labels (np.ndarray): One-hot encoded labels with shape (N, num_classes).
        num_classes (int): Total number of classes.
    
    Returns:
        dict: A mapping from class index to tuple of (mean, std) arrays.
              mean and std have shape (channels, timepoints)
    """
    # Convert to numpy if they're tensors
    if torch.is_tensor(samples):
        samples = samples.cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()
    
    class_indices = np.argmax(labels, axis=1)
    ref_bank = {}
    
    for cls in range(num_classes):
        # Get all samples for this class
        indices = np.where(class_indices == cls)[0]
        if len(indices) > 0:
            n_samples = len(indices)
            class_samples = samples[indices]
            ref_signal = class_samples[0]  # Use the first sample as reference
            # Calculate mean and std per timepoint and channel
            mean = np.mean(class_samples, axis=0)  # shape: (channels, timepoints)
            std = np.std(class_samples, axis=0)    # shape: (channels, timepoints)
            ref_bank[cls] = (ref_signal, mean, std, n_samples)
        else:
            print(f"Warning: No samples found for class {cls}.")
    
    return ref_bank


def log_metrics(log_dict, name_metric, metric, baseline):
    class_names = ["CD", "HYP", "MI", "NORM", "STTC"]

    # Special handling for TXTY metrics (TRTR, TSTR, TRTS): only overall AUROC, no std or per-class
    if name_metric in ["TRTR", "TSTR", "TRTS"]:
        # metric and baseline are scalars (overall AUROC)
        log_dict[name_metric] = metric
        log_dict[f'{name_metric}_baseline'] = baseline
        # No std or per-class metrics for TXTY
        return log_dict

    # Check if metric and baseline are tensors with multiple elements
    if len(metric) == 2:
        overall_tensor, per_tensors = metric
        # Save raw tensor values to txt file 
        # ! This will overwrite the file each time, so ensure you save the model results fast enough !
        # TODO: consider putting the name of the model in the filename so that it doesn't overwrite previous results
        with open(f'results/{name_metric}_raw_values.txt', 'w') as f:
            f.write(f"Raw {name_metric} values:\n")
            f.write(str(overall_tensor.cpu().numpy().tolist()))

        # Compute mean
        overall_mean = overall_tensor.mean().item()
        overall_std = overall_tensor.std().item() if len(overall_tensor) > 1 else 0.0

        per_class_means, per_class_stds = [], []
        for vals in per_tensors:
            if vals is not None and len(vals) > 0:
                per_class_means.append(vals.mean().item())
                per_class_stds.append(vals.std().item() if len(vals) > 1 else 0.0)
            else:
                per_class_means.append(float('nan'))
                per_class_stds.append(float('nan'))
    else:
        overall_mean, overall_std, per_class_means, per_class_stds = metric

    if len(baseline) == 2:
        overall_tensor, per_tensors = baseline
        with open(f'results/baseline_{name_metric}_raw_values.txt', 'w') as f:
            f.write(f"Baseline raw {name_metric} values:\n")
            f.write(str(overall_tensor.cpu().numpy().tolist()))
        # baseline_std and baseline_per_class_stds are not used in this case
        # Compute mean
        baseline_mean = overall_tensor.mean().item()
        baseline_std = overall_tensor.std().item() if len(overall_tensor) > 1 else 0.0

        baseline_per_class_means, baseline_per_class_stds = [], []
        for vals in per_tensors:
            if vals is not None and len(vals) > 0:
                baseline_per_class_means.append(vals.mean().item())
                baseline_per_class_stds.append(vals.std().item() if len(vals) > 1 else 0.0)
            else:
                baseline_per_class_means.append(float('nan'))
                baseline_per_class_stds.append(float('nan'))
    else:
        baseline_mean, baseline_std, baseline_per_class_means, baseline_per_class_stds = baseline

    # Log overall metrics
    log_dict[name_metric] = overall_mean
    # log_dict[f'{name_metric}_std'] = overall_std
    log_dict[f'{name_metric}_baseline'] = baseline_mean
    # log_dict[f'{name_metric}_baseline_std'] = baseline_std

    # Only log per-class metrics if they exist
    if per_class_means and baseline_per_class_means:
        for c in range(len(per_class_means)):
            class_metric = per_class_means[c]
            class_metric_std = per_class_stds[c]
            class_baseline = baseline_per_class_means[c]
            class_baseline_std = baseline_per_class_stds[c]

            # Log per-class metrics
            log_dict[f'{name_metric}_{class_names[c]}'] = class_metric
            # log_dict[f'{name_metric}_{class_names[c]}_std'] = class_metric_std
            log_dict[f'{name_metric}_{class_names[c]}_baseline'] = class_baseline
            # log_dict[f'{name_metric}_{class_names[c]}_baseline_std'] = class_baseline_std

    return log_dict


def sample_by_class(data, classes, classifier, project=None):
    """
    Sample all available data for each class and process features.
    This improved version doesn't artificially limit the number of samples.
    
    Args:
        data (torch.Tensor): Data samples
        classes (torch.Tensor): Class indices for each sample
        classifier: Model for feature extraction
        project (str): Project name (for conditional processing)
        
    Returns:
        tuple: (samples by class, features by class)
    """
    samples_list = {}
    samples_list_features = {}
    for c in torch.unique(classes):
        # Get all samples for this class
        mask = (classes == c).nonzero(as_tuple=True)[0]
        n_available = len(mask)
        if n_available == 0:
            continue
            
        # Use all available samples
        samples_list[f'{c}'] = data[mask].cuda()
        
        # Process features in batches to avoid CUDA memory issues
        batch_size = 32  # Adjust based on your GPU memory
        features = []
        for i in range(0, n_available, batch_size):
            batch = samples_list[f'{c}'][i:i+batch_size]
            if project == "ECG":
                batch_features = classifier(batch).cuda().detach()
            else:
                batch_features = classifier(batch).cuda().detach()
            features.append(batch_features)
        
        samples_list_features[f'{c}'] = torch.cat(features, dim=0) if features else torch.tensor([]).cuda()
        
    return samples_list, samples_list_features


def balance_samples(train_subset, test_subset, model_subset, n_per_class=None):
    """
    Balance the number of samples across all datasets to ensure fair comparison.
    Either uses all available samples or limits to n_per_class samples per class.
    
    Args:
        train_subset (dict): Training samples by class
        test_subset (dict): Test samples by class
        model_subset (dict): Model samples by class
        n_per_class (int, optional): Number of samples per class to use (if limited)
        
    Returns:
        tuple: Balanced (train_subset, test_subset, model_subset)
    """
    balanced_train = {}
    balanced_test = {}
    balanced_model = {}
    
    # Process each class
    for c in train_subset.keys():
        if c not in test_subset or c not in model_subset:
            continue
            
        # Find minimum number of samples available across all datasets
        min_samples = min(
            train_subset[c].shape[0],
            test_subset[c].shape[0],
            model_subset[c].shape[0]
        )
        
        # If n_per_class is specified, cap at that value
        if n_per_class is not None:
            min_samples = min(min_samples, n_per_class)
        
        # Randomly select samples if needed
        if train_subset[c].shape[0] > min_samples:
            perm = torch.randperm(train_subset[c].shape[0], device='cuda')
            balanced_train[c] = train_subset[c][perm[:min_samples]]
        else:
            balanced_train[c] = train_subset[c]
            
        if test_subset[c].shape[0] > min_samples:
            perm = torch.randperm(test_subset[c].shape[0], device='cuda')
            balanced_test[c] = test_subset[c][perm[:min_samples]]
        else:
            balanced_test[c] = test_subset[c]
            
        if model_subset[c].shape[0] > min_samples:
            perm = torch.randperm(model_subset[c].shape[0], device='cuda')
            balanced_model[c] = model_subset[c][perm[:min_samples]]
        else:
            balanced_model[c] = model_subset[c]
    
    return balanced_train, balanced_test, balanced_model


def select_samples(train_data, train_labels, test_data, test_labels, model_data, model_labels, n_per_class, classifier, project):
# Convert to torch tensors and move to CUDA
    train_data = torch.as_tensor(train_data).cuda().float()
    train_labels = torch.as_tensor(train_labels).cuda().float()
    
    test_data = torch.as_tensor(test_data).cuda().float()
    test_labels = torch.as_tensor(test_labels).cuda().float()
    
    model_data = torch.as_tensor(model_data).cuda().float()
    model_labels = torch.as_tensor(model_labels).cuda().float()
    
    # Convert one-hot to class indices
    train_classes = torch.argmax(train_labels, dim=1)  
    test_classes = torch.argmax(test_labels, dim=1)
    model_classes = torch.argmax(model_labels, dim=1)

    # Print class distribution info for debugging
    print("\nClass distributions:")
    for name, classes in [("Train", train_classes), ("Test", test_classes), ("Model", model_classes)]:
        class_counts = {}
        for c in torch.unique(classes):
            class_counts[c.item()] = (classes == c).sum().item()
        print(f"{name} data: {class_counts}")

    # Get samples and features for each class
    train_subset, train_subset_features = sample_by_class(train_data, train_classes, classifier, project)
    test_subset, test_subset_features = sample_by_class(test_data, test_classes, classifier, project) 
    model_subset, model_subset_features = sample_by_class(model_data, model_classes, classifier, project)
    
    # Balance samples to ensure fair comparison
    if n_per_class is not None:
        train_subset, test_subset, model_subset = balance_samples(
            train_subset, test_subset, model_subset, n_per_class
        )
        
        train_subset_features, test_subset_features, model_subset_features = balance_samples(
            train_subset_features, test_subset_features, model_subset_features, n_per_class
        )
    
    # Print final sample counts per class
    print("\nFinal balanced samples counts:")
    for c in train_subset:
        print(f"Class {c}: Train={train_subset[c].shape[0]}, Test={test_subset[c].shape[0] if c in test_subset else 0}, Model={model_subset[c].shape[0] if c in model_subset else 0}")

    return (train_subset, test_subset, model_subset), (train_subset_features, test_subset_features, model_subset_features)


def match_train_val(train_data, train_features, train_labels, val_data, val_labels):
    """
    Match training and validation data based on labels and order.

    This function aligns the training data and features to the validation set by matching each validation label
    to a corresponding training label. For each label in val_labels, it finds a matching label in train_labels,
    extracts the corresponding entry from train_data and train_features, and removes it from the pool to prevent
    duplicate matches. This ensures the matched training data has the same label distribution and order as the
    validation data.

    If an exact match is not found for a validation label, the function falls back to using cosine similarity
    to find the closest matching training label. The cosine similarity is computed between normalized label
    vectors, and the training sample with the highest similarity score is selected.

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
    if not isinstance(train_features, torch.Tensor):
        train_features = torch.tensor(train_features, dtype=torch.float32)
    if not isinstance(train_labels, torch.Tensor):
        train_labels = torch.tensor(train_labels, dtype=torch.float32)
    if not isinstance(val_data, torch.Tensor):
        val_data = torch.tensor(val_data, dtype=torch.float32)
    if not isinstance(val_labels, torch.Tensor):
        val_labels = torch.tensor(val_labels, dtype=torch.float32)

    # Move everything to CPU for indexing to save GPU memory
    train_data_cpu = train_data.cpu()
    train_features_cpu = train_features.cpu()
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
            # Find the closest matching label using cosine similarity
            train_labels_subset = train_labels_cpu[available_indices]
            # Normalize vectors for cosine similarity
            label_norm = label / (torch.norm(label) + 1e-8)
            train_labels_norm = train_labels_subset / (torch.norm(train_labels_subset, dim=1, keepdim=True) + 1e-8)
            # Compute cosine similarity (higher is better)
            cosine_similarities = torch.mm(train_labels_norm, label_norm.unsqueeze(1)).squeeze()
            closest_idx_relative = torch.argmax(cosine_similarities)
            closest_idx = available_indices[closest_idx_relative]
            matched_indices.append(closest_idx)
            available_indices.remove(closest_idx)
            print(f"\t\t[WARNING] No exact match for {label}, using closest label {train_labels_cpu[closest_idx]} (cosine similarity: {cosine_similarities[closest_idx_relative]:.4f})")

    if len(matched_indices) == 0:
        print("\t\t[WARNING] No matching training data found.")
        return (
            torch.empty((0, train_data.shape[1]), device=train_data.device),
            torch.empty((0, train_features.shape[1]), device=train_features.device)
        )

    matched_train_data = train_data_cpu[matched_indices].to(train_data.device)
    matched_train_features = train_features_cpu[matched_indices].to(train_features.device)

    if matched_train_data.shape != val_data.shape:
        print(f"\t\t[WARNING] Matched training data shape {matched_train_data.shape} does not match validation data shape {val_data.shape}.")
        return (
            torch.empty((0, train_data.shape[1]), device=train_data.device),
            torch.empty((0, train_features.shape[1]), device=train_features.device)
        )
    return matched_train_data, matched_train_features


def evaluate(
        log_dict,
        real_train,
        real_val,
        fake_train,
        fake_val,
        baseline_metrics
    ):
    """
    Evaluate only the model metrics using pre-calculated baselines
    """
    train_data, train_labels, train_features = real_train
    val_data, val_labels, val_features = real_val
    model_train_data, model_train_labels, _ = fake_train
    model_val_data, model_val_labels, model_val_features = fake_val

    # Group for signal metrics
    signal_metric_data = (val_data, model_val_data, val_labels)
    
    # Group for (Classifier) feature metrics
    feature_metric_data = (val_features, model_val_features, val_labels)
    
    # Group for TSTR and TRTS
    TSTR_metric_data = (model_train_data, model_train_labels, val_data, val_labels)
    TRTS_metric_data = (train_data, train_labels, model_val_data, model_val_labels)

    # Signal metrics
    # RMSE_metric = run_rmse(*signal_metric_data)
    # log_dict = log_metrics(log_dict, "RMSE", RMSE_metric, baseline_metrics["RMSE"])

    # PRD_metric = run_prd(*signal_metric_data)
    # log_dict = log_metrics(log_dict, "PRD", PRD_metric, baseline_metrics["PRD"])

    # dFD_metric = run_dfd(*signal_metric_data)
    # log_dict = log_metrics(log_dict, "dFD", dFD_metric, baseline_metrics["dFD"])

    # DTW_metric = run_dtw(*signal_metric_data)
    # log_dict = log_metrics(log_dict, "DTW", DTW_metric, baseline_metrics["DTW"])

    # Feature metrics
    MMD_metric = run_mmd(*signal_metric_data, analytics=False, return_raw=True)
    log_dict = log_metrics(log_dict, "MMD", MMD_metric, baseline_metrics["MMD"])

    # PSDRMSE_metric = run_psd_rmse(*signal_metric_data)
    # log_dict = log_metrics(log_dict, "PSD RMSE", PSDRMSE_metric, baseline_metrics["PSD RMSE"])

    PSDMMD_metric = run_psd_mmd(*signal_metric_data, analytics=False, return_raw=True)
    log_dict = log_metrics(log_dict, "PSD-MMD", PSDMMD_metric, baseline_metrics["PSD-MMD"])

    PSDPRD_metric = run_psd_prd(*signal_metric_data, analytics=False, return_raw=True)
    log_dict = log_metrics(log_dict, "PSD-PRD", PSDPRD_metric, baseline_metrics["PSD-PRD"])

    # (Classifier) feature metrics
    FID_metric = run_fid(*feature_metric_data, analytics=False, return_raw=True)
    log_dict = log_metrics(log_dict, "FID", FID_metric, baseline_metrics["FID"])

    KID_metric = run_mmd(*feature_metric_data, analytics=False, return_raw=True)
    log_dict = log_metrics(log_dict, "KID", KID_metric, baseline_metrics["KID"])

    # Classifier metrics
    _, TSTR_metric, _ = run_TSTR(*TSTR_metric_data)
    _, TRTS_metric, _ = run_TRTS(*TRTS_metric_data, model_checkpoint_path='./txty_results/TRTR/best_model.pth')
    log_dict = log_metrics(log_dict, "TSTR", TSTR_metric, baseline_metrics["TRTR"])
    log_dict = log_metrics(log_dict, "TRTS", TRTS_metric, baseline_metrics["TRTR"])

    return log_dict