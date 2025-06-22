import torch
from tqdm import tqdm
import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import math


def run_metric(metric_fn, X, Y, labels, batch_size=256, seed=42, use_tqdm=True, analytics=True, return_raw=False):
    """
    Run a metric function on two datasets (X, Y), reporting overall and per-class results.
    Uses MultilabelStratifiedKFold for overall stratified batching and exact-size per-class batching.
    Optionally prints post-batch analytics and warnings for heavy oversampling.

    Parameters:
    -----------
    metric_fn : callable
        Metric function that takes two tensors (X, Y) as input and returns a scalar value.
    X : torch.Tensor
        Groundtruth.
    Y : torch.Tensor
        Synthetic/reference.
    labels : torch.Tensor
        Multi-label class conditions for each sample (shape: [num_samples, num_classes]).
    batch_size : int or None
        If set, all batches will have exactly this size. Batch size refers to the number of samples in each dataset batch, 
        e.g. batch_size_X = batch_size_Y = batch_size. Effectively, 2*batch_size samples are processed.
    seed : int
        Random seed for reproducibility.
    use_tqdm : bool
        Whether to use tqdm progress bar for batch processing.
    analytics : bool
        Whether to print batch distribution analytics and warnings.
    return_raw : bool
        If True, returns the raw batch results as torch tensors instead of calculating means and standard deviations.

    Returns:
    --------
    If return_raw is False (default):
        tuple:
            overall_mean : scalar
            overall_std : scalar
            per_class_means : list
            per_class_stds : list
    If return_raw is True:
        tuple:
            overall_vals : torch.Tensor
                Tensor of metric values for each overall batch.
            per_vals : list of torch.Tensor
                List of tensors of metric values for each class's batches.
    """
    num_samples, num_classes = labels.shape
    device = labels.device

    # Set seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    generator = torch.Generator().manual_seed(seed)

    # Fast path: no batching
    if batch_size is None or batch_size >= num_samples:
        overall = metric_fn(X, Y)
        per_class = []
        for c in range(num_classes):
            idx = torch.where(labels[:, c] == 1)[0]
            per_class.append(metric_fn(X[idx], Y[idx]) if len(idx) else float('nan'))
        return float(overall), 0.0, [float(x) for x in per_class], [0.0] * num_classes

    # ===================== OVERALL METRIC BATCHES WITH MSKFOLD =====================
    n_splits = math.ceil(num_samples / batch_size)
    if n_splits < 1:
        raise ValueError(f"Batch size {batch_size} too large for dataset of {num_samples} samples.")

    labels_np = labels.cpu().numpy().astype(int)
    X_dummy = np.zeros((num_samples, 1))
    mskf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    overall_batches = []
    for _, test_idx in mskf.split(X_dummy, labels_np):
        if len(test_idx) != batch_size:
            if len(test_idx) > batch_size:
                test_idx = np.random.choice(test_idx, size=batch_size, replace=False)
            else:
                extra = np.random.choice(
                    np.setdiff1d(np.arange(num_samples), test_idx),
                    size=(batch_size - len(test_idx)), replace=True
                )
                test_idx = np.concatenate([test_idx, extra])
        overall_batches.append(torch.tensor(test_idx, device=device, dtype=torch.long))
    
    # Check for duplicate samples across batches
    if analytics:
        all_indices = torch.cat(overall_batches)
        unique_indices = torch.unique(all_indices)
        total_batch_samples = len(all_indices)
        unique_samples = len(unique_indices)
        overlap_samples = total_batch_samples - unique_samples
        print(f"Batch overlap analysis: {overlap_samples}/{total_batch_samples} samples appear in multiple batches ({overlap_samples/total_batch_samples*100:.1f}%)")
        
        # Find which samples appear in multiple batches
        if overlap_samples > 0:
            sample_to_batches = {}
            for batch_idx, batch_indices in enumerate(overall_batches):
                for sample_idx in batch_indices:
                    sample_idx_item = sample_idx.item()
                    if sample_idx_item not in sample_to_batches:
                        sample_to_batches[sample_idx_item] = []
                    sample_to_batches[sample_idx_item].append(batch_idx)
            
            # Report duplicated samples
            duplicated_samples = {k: v for k, v in sample_to_batches.items() if len(v) > 1}
            print(f"Duplicated samples ({len(duplicated_samples)} samples):")
            for sample_idx, batch_list in list(duplicated_samples.items())[:10]:  # Show first 10
                print(f"  Sample {sample_idx}: appears in batches {batch_list}")
            if len(duplicated_samples) > 10:
                print(f"  ... and {len(duplicated_samples) - 10} more duplicated samples")

    # ===================== PER-CLASS EXACT-SIZE BATCHES WITH OVERSAMPLING WARNINGS =====================
    class_batches = {}
    class_distribution = []
    for c in range(num_classes):
        idx_c = torch.where(labels[:, c] == 1)[0]
        count_c = len(idx_c)
        class_distribution.append(count_c)
        if analytics and count_c < 2 * batch_size:
            print(f"WARNING: Class {c} has only {count_c} samples (< 2*batch_size={2*batch_size}); heavy oversampling may occur.")

        if count_c == 0:
            class_batches[c] = []
            continue

        perm = torch.randperm(count_c, generator=generator)
        shuffled = idx_c[perm]

        if count_c < batch_size:
            repeats = (batch_size + count_c - 1) // count_c
            reps = shuffled.repeat(repeats)
            perm2 = torch.randperm(len(reps), generator=generator)
            class_batches[c] = [reps[perm2[:batch_size]]]
        else:
            full = count_c // batch_size
            batches = [shuffled[i * batch_size:(i + 1) * batch_size] for i in range(full)]
            rem = count_c % batch_size
            if rem:
                extra_indices = torch.randperm(count_c, generator=generator)[:rem]
                extra = shuffled[extra_indices]
                pad_indices = torch.randperm(count_c, generator=generator)[: batch_size - rem]
                pad = shuffled[pad_indices]
                batches.append(torch.cat([extra, pad]))
            class_batches[c] = batches

    # ===================== ANALYTICS =====================
    if analytics:
        sizes = [len(b) for b in overall_batches]
        print(f"Overall batches: {len(overall_batches)} batches (size range: {min(sizes)}–{max(sizes)})")

        print("Class distribution in dataset:")
        for c, cnt in enumerate(class_distribution):
            print(f"  Class {c}: {cnt}/{num_samples} ({cnt / num_samples * 100:.1f}%)")

        batch_counts = np.array([labels[b].sum(dim=0).cpu().numpy() for b in overall_batches])
        avg_counts = batch_counts.mean(axis=0)
        exp_counts = [batch_size * (cnt / num_samples) for cnt in class_distribution]
        print("\nAvg vs expected counts per batch:")
        for c in range(num_classes):
            print(f"  Class {c}: avg={avg_counts[c]:.2f}, expected={exp_counts[c]:.2f}")

        sims = []
        for c in range(num_classes):
            pct_batch = avg_counts[c] / batch_size * 100
            pct_data = class_distribution[c] / num_samples * 100
            sim = (min(pct_batch, pct_data) / max(pct_batch, pct_data) * 100) if pct_data > 0 else 0
            sims.append(sim)
            print(f"  Class {c} similarity: {sim:.1f}%")
        print(f"Overall distribution similarity: {np.mean(sims):.1f}%")

    # ===================== COMPUTE METRICS =====================
    overall_vals = []
    iterator = overall_batches
    if use_tqdm:
        iterator = tqdm(iterator, desc=f"Overall {metric_fn.__name__}")
    for batch_idx in iterator:
        overall_vals.append(metric_fn(X[batch_idx], Y[batch_idx]))

    per_vals = [[] for _ in range(num_classes)]
    for c in range(num_classes):
        # if not c == 1:
        #     continue
        # print(f"Class {c}")
        batches = class_batches[c]
        if not batches:
            continue
        it = tqdm(batches, desc=f"Class {c} {metric_fn.__name__}") if use_tqdm else batches
        for batch_idx in it:
            per_vals[c].append(metric_fn(X[batch_idx], Y[batch_idx]))
        # time.sleep(2)  # Sleep for 2 seconds between class calculations

    # After computing all metrics, decide what to return based on return_raw
    if return_raw:
        # Convert overall_vals to a torch tensor
        overall_tensor = torch.tensor(overall_vals, device=device).float()
        
        # Convert each list in per_vals to a torch tensor
        per_tensors = []
        for vals in per_vals:
            if vals:
                per_tensors.append(torch.tensor(vals, device=device).float())
            else:
                # Create an empty tensor for classes with no values
                per_tensors.append(torch.tensor([], device=device).float())
        
        return overall_tensor, per_tensors
    else:
        ov_tensor = torch.tensor(overall_vals, device=device).float()
        overall_mean = ov_tensor.mean().item()
        overall_std = ov_tensor.std().item() if len(overall_vals) > 1 else 0.0

        per_means, per_stds = [], []
        for vals in per_vals:
            if vals:
                t = torch.tensor(vals, device=device).float()
                per_means.append(t.mean().item())
                per_stds.append(t.std().item() if len(vals) > 1 else 0.0)
            else:
                per_means.append(float('nan'))
                per_stds.append(float('nan'))

        return overall_mean, overall_std, per_means, per_stds
