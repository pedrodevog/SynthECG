# General packages
import torch
from torch import autograd
import numpy as np
import json
import os
# Utils
from utils.utils import find_max_epoch
# Importing custom metrics
from metrics.MMD import run_mmd
from metrics.PSD import run_psd_mmd, run_psd_prd
from metrics.FID import run_fid
from metrics.TXTY import run_TSTR, run_TRTS


def initialize_classifier(path=""):
    num_classes = None
    model = ...     # Initialize your foundation model here
    
    # Loading model weights
    enc_weights = torch.load(path, map_location=torch.device('cpu'))
    model.load_state_dict(enc_weights, strict=False) 

    print(f"Classifier model from {path} == loaded successfully")

    return model.cuda()


def calc_diffusion_step_embedding(diffusion_steps, diffusion_step_embed_dim_in):
    """
    Embed a diffusion step $t$ into a higher dimensional space
    E.g. the embedding vector in the 128-dimensional space is
    [sin(t * 10^(0*4/63)), ... , sin(t * 10^(63*4/63)), cos(t * 10^(0*4/63)), ... , cos(t * 10^(63*4/63))]

    Parameters:
    diffusion_steps (torch.long tensor, shape=(batchsize, 1)):
                                diffusion steps for batch data
    diffusion_step_embed_dim_in (int, default=128):
                                dimensionality of the embedding space for discrete diffusion steps

    Returns:
    the embedding vectors (torch.tensor, shape=(batchsize, diffusion_step_embed_dim_in)):
    """

    assert diffusion_step_embed_dim_in % 2 == 0

    half_dim = diffusion_step_embed_dim_in // 2
    _embed = np.log(10000) / (half_dim - 1)
    _embed = torch.exp(torch.arange(half_dim) * -_embed).cuda()
    _embed = diffusion_steps * _embed
    diffusion_step_embed = torch.cat((torch.sin(_embed),
                                      torch.cos(_embed)), 1)

    return diffusion_step_embed


# Adapted from https://github.com/caogang/wgan-gp/blob/master/gan_toy.py
def calc_gradient_penalty(net_dis, real_data, fake_data, batch_size, lmbda, use_cuda=False):
    # Compute interpolation factors
    alpha = torch.rand(batch_size, 1, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda() if use_cuda else alpha

    # Interpolate between real and fake data.
    interpolates = alpha * real_data + (1 - alpha) * fake_data
    if use_cuda:
        interpolates = interpolates.cuda()
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    # Evaluate discriminator
    disc_interpolates = net_dis(interpolates)

    # Obtain gradients of the discriminator with respect to the inputs
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda() if use_cuda else
                              torch.ones(disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    # Compute MSE between 1.0 and the gradient of the norm penalty to make discriminator
    # to be a 1-Lipschitz function.
    gradient_penalty = lmbda * ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def find_run(config, experiment):
    """
    Search for an existing experiment run with matching configuration.
    This function iterates through numbered run directories to find a matching configuration file.
    If a matching configuration is found, returns that run number. Otherwise, returns the next
    available run number.
    Returns:
        int: The run number of either:
            - An existing run with matching configuration
            - The next available run number if no match is found
    Example:
        If base_path contains:
            run_1/
                config_project_1.json  # different config
            run_2/
                config_project_1.json  # matching config
        find_run() would return 2
    Note:
        - Assumes experiment["ckpt_directory"] and experiment["project"] globals exist
        - Requires matching config comparison with global config variable
    """
    base_path = experiment["ckpt_directory"]
    config_name = f'config_{experiment["project"]}_{experiment["run"]}.json'
    run = 1

    while os.path.exists(os.path.join(base_path, f"run_{run}")):
        config_path = os.path.join(base_path, f"run_{run}", config_name)
        try:
            with open(config_path) as f:
                compare_config = json.load(f)
                config.pop("experiment", None)
                experiment = compare_config.pop("experiment", None)
                if config == compare_config:
                    run_id = experiment["id"]
                    print(f"[CONFIG] Found matching config in run_{run}")
                    return run, run_id
                else:
                    print(f"[WARNING] Config mismatch in run_{run}")
                    # Identify and print the first mismatched key and its values
                    for key in config:
                        if key not in compare_config or config[key] != compare_config[key]:
                            print(f"\tMismatch at key '{key}': config has {config[key]}, compare_config has {compare_config.get(key)}")
                            break
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        run += 1
    
    return run, None


def load_checkpoint(ckpt_dir, ckpt_iter, net, optimizer):
    """
    Load checkpoint model from the directory.
    This function attempts to load a saved model checkpoint and its optimizer state from a specified directory.
    If 'max' is specified as ckpt_iter, it will automatically find and load the checkpoint with the highest iteration number.

    Parameters
    ----------
    ckpt_dir : str
        Directory path containing the checkpoint files
    ckpt_iter : Union[int, str]
        Either an integer specifying the iteration number to load, or 'max' to load the highest iteration
    net : torch.nn.Module
        The neural network model to load the weights into
    optimizer : torch.optim.Optimizer
        The optimizer to load the state into

    Returns
    -------
    Tuple[int, torch.nn.Module, torch.optim.Optimizer]
        A tuple containing:
        - ckpt_iter: The loaded checkpoint iteration (-1 if loading failed)
        - net: The model with loaded weights
        - optimizer: The optimizer with loaded state

    Notes
    -----
    - If loading fails, ckpt_iter will be set to -1 and training will start from initialization
    - Checkpoint files are expected to be .pkl format
    - Checkpoint dict should contain 'model_state_dict' and optionally 'optimizer_state_dict'
    """
    if ckpt_iter == 'max':
        ckpt_iter = find_max_epoch(ckpt_dir)
    if ckpt_iter >= 0:
        try:
            # load checkpoint file
            model_path = os.path.join(ckpt_dir, '{}.pkl'.format(ckpt_iter))
            checkpoint = torch.load(model_path, map_location='cpu')

            # feed model dict and optimizer state
            if isinstance(net, tuple):
                net[0].load_state_dict(checkpoint['modelG_state_dict'])
                net[1].load_state_dict(checkpoint['modelD_state_dict'])
            else:
                net.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint or ('optimizerG_state_dict' in checkpoint and 'optimizerD_state_dict' in checkpoint):
                if isinstance(optimizer, tuple):
                    optimizer[0].load_state_dict(checkpoint['optimizerG_state_dict'])
                    optimizer[1].load_state_dict(checkpoint['optimizerD_state_dict'])
                else:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            print('Successfully loaded model at iteration {}'.format(ckpt_iter))
        except:
            ckpt_iter = -1
            print('No valid checkpoint model found, start training from initialization try.')
    else:
        ckpt_iter = -1
        print('No valid checkpoint model found, start training from initialization.')
    return ckpt_iter, net, optimizer


def save_checkpoint(ckpt_dir, n_iter, net, optimizer):
    """Save model checkpoint and optimizer state.

    This function saves the current state of the model and optimizer to a pickle file.
    The checkpoint can be used later to resume training from this point.

    Args:
        ckpt_dir (str): Directory path where checkpoint will be saved
        n_iter (int): Current iteration number used for checkpoint filename
        net (torch.nn.Module): Neural network model to save
        optimizer (torch.optim.Optimizer): Optimizer whose state to save

    Example:
        >>> save_checkpoint('./checkpoints', 1000, model, optimizer)
        'model at iteration 1000 is saved'
    """
    checkpoint_name = '{}.pkl'.format(n_iter)
    if isinstance(net, tuple) and isinstance(optimizer, tuple):
        torch.save({'modelG_state_dict': net[0].state_dict(),
                    'modelD_state_dict': net[1].state_dict(),
                    'optimizerG_state_dict': optimizer[0].state_dict(),
                    'optimizerD_state_dict': optimizer[1].state_dict()},
                    os.path.join(ckpt_dir, checkpoint_name))
    else:
        torch.save({'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()},
                os.path.join(ckpt_dir, checkpoint_name))
    print(f'Model checkpoint saved successfully at iteration {n_iter} in {ckpt_dir}')


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