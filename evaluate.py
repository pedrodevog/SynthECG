#!/usr/bin/env python
import torch
import os
import json
import numpy as np
import argparse
import wandb
import glob
from tqdm import tqdm

# Importing custom metrics
from metrics.MMD import run_mmd
from metrics.PSD import run_psd_prd, run_psd_mmd
from metrics.FID import run_fid
from metrics.TXTY import run_TRTR

# Import utility functions
from utils.utils import *
from utils.train import *
from utils.evaluate import *

# Model configurations mapping
MODEL_CONFIGS = [
    {
        "name": "SSSD_ECG",
        "config": "configs/config_SSSD_ECG.json"
    },
    {
        "name": "DSAT_ECG",
        "config": "configs/config_DSAT_ECG.json"
    },
    {
        "name": "WaveGAN",
        "config": "configs/config_cond_wavegan_star_ECG.json"
    },
    {
        "name": "Pulse2Pulse",
        "config": "configs/config_cond_pulse2pulse_ECG.json"
    },
]

def find_sample_dirs(config, experiment):
    """
    Find all directories containing generated samples for a specific model and project.
    
    Args:
        config: Configuration dictionary
        experiment: Experiment configuration
        
    Returns:
        dict: Dictionary with keys 'train', 'val', 'test' and their corresponding paths (or None if not found)
    """
    try:
        run, _ = find_run(config, experiment)
        local_path = f"run_{run}"
        ckpt_dir = os.path.join(experiment["ckpt_directory"], local_path)
        ckpt_parts = os.path.normpath(ckpt_dir).split(os.sep)
        ckpt_idx = ckpt_parts.index("checkpoints")
        
        # Expand the ~ symbol properly
        base_path = os.path.expanduser(os.path.join('~/SynthECG', *ckpt_parts[:ckpt_idx]))
        model_name = ckpt_parts[ckpt_idx + 1]
        run_name = ckpt_parts[ckpt_idx + 2]
        
        # Construct output directories
        output_dir_train = os.path.join(base_path, "train", model_name, run_name)
        output_dir_val = os.path.join(base_path, "val", model_name, run_name)
        output_dir_test = os.path.join(base_path, "test", model_name, run_name)
        
        # Check each directory individually and return what exists
        dirs = {}
        for dir_type, output_dir in [("train", output_dir_train), ("val", output_dir_val), ("test", output_dir_test)]:
            if os.path.exists(output_dir):
                dirs[dir_type] = output_dir
                print(f"[INFO] Found {dir_type} directory: {output_dir}")
            else:
                print(f"[WARNING] {dir_type.capitalize()} directory not found: {output_dir}")
                dirs[dir_type] = None
        
        return dirs
        
    except Exception as e:
        print(f"[ERROR] Failed to find sample directories: {e}")
        return {}

def filter_checkpoints(checkpoint_list, checkpoint_filter=None, specific_checkpoints=None):
    """
    Filter checkpoint list based on filter pattern or specific checkpoint selection.
    
    Args:
        checkpoint_list (list): List of checkpoint names
        checkpoint_filter (str): Pattern to filter checkpoints (substring match)
        specific_checkpoints (str): Comma-separated list of specific checkpoints to select
        
    Returns:
        list: Filtered list of checkpoint names
    """
    if not checkpoint_list:
        return []
    
    filtered = checkpoint_list.copy()
    
    # Apply specific checkpoint selection first (highest priority)
    if specific_checkpoints:
        target_checkpoints = [ckpt.strip() for ckpt in specific_checkpoints.split(',')]
        filtered = [ckpt for ckpt in filtered if ckpt in target_checkpoints]
        print(f"[FILTER] Selected specific checkpoints: {target_checkpoints}")
        print(f"[FILTER] Found matching checkpoints: {filtered}")
        return filtered
    
    # Apply pattern filter
    if checkpoint_filter:
        original_count = len(filtered)
        filtered = [ckpt for ckpt in filtered if checkpoint_filter in ckpt]
        print(f"[FILTER] Applied pattern filter '{checkpoint_filter}': {original_count} -> {len(filtered)} checkpoints")
    
    return filtered

def find_sample_files(sample_dir):
    """
    Find all sample files in a directory.
    
    Args:
        sample_dir (str): Path to the directory containing samples
        
    Returns:
        list: List of tuples (data_file, label_file, feature_file, ckpt_name) for each checkpoint
    """
    if sample_dir is None or not os.path.exists(sample_dir):
        print(f"[INFO] Sample directory not accessible: {sample_dir}")
        return []
        
    # Find all data files
    data_files = glob.glob(os.path.join(sample_dir, "data_*.npy"))
    
    if not data_files:
        print(f"[INFO] No sample files found in {sample_dir}")
        return []
    
    sample_files = []
    for data_file in data_files:
        # Construct corresponding label file name
        label_file = data_file.replace("data_", "labels_")
        feature_file = data_file.replace("data_", "features_")
        
        # Check if label file exists
        if os.path.exists(label_file) and os.path.exists(feature_file):
            # Extract checkpoint number from filename
            # Format is data_XXXX.npy where XXXX is the checkpoint number
            ckpt_name = os.path.basename(data_file).replace("data_", "").replace(".npy", "")
            sample_files.append((data_file, label_file, feature_file, ckpt_name))
        else:
            print(f"[WARNING] Missing corresponding files for {data_file}")
    
    # Sort by checkpoint number (if possible)
    try:
        sample_files.sort(key=lambda x: int(x[3]))
    except ValueError:
        sample_files.sort(key=lambda x: x[3])

    if len(sample_files) == 0:
        print(f"[WARNING] No valid sample files found in {sample_dir}")
    elif len(sample_files) > 100:
        print(f"[WARNING] More than 100 sample files found in {sample_dir}. Only the first 100 will be processed.")
        print(f"[WARNING] Dropped {len(sample_files) - 100} files.")
        print(f"[WARNING] Last file dropped: {sample_files[-1]}")
        sample_files = sample_files[:100]
    
    return sample_files

def validate_samples(
        model_name, 
        project_name, 
        data_file_train, 
        label_file_train,
        feature_file_train,
        data_file_eval,
        label_file_eval, 
        feature_file_eval,
        ckpt_name, 
        config,
        n_per_class,
        real_train,
        real_eval,
        real_ref_bank,
        baseline_metrics
    ):
    """
    Validate samples from a single checkpoint with pre-calculated baseline metrics.
    """
    print(f"\n[VALIDATING] {model_name} checkpoint: {ckpt_name}")
    
    # Load generated samples and labels
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_train_data = torch.from_numpy(np.load(data_file_train)).float().to(device)
    model_train_labels = torch.from_numpy(np.load(label_file_train)).float().to(device)
    model_eval_data = torch.from_numpy(np.load(data_file_eval)).float().to(device)
    model_eval_labels = torch.from_numpy(np.load(label_file_eval)).float().to(device)
    model_eval_features = torch.from_numpy(np.load(feature_file_eval)).float().to(device)

    fake_train = (model_train_data, model_train_labels, None)
    fake_eval = (model_eval_data, model_eval_labels, model_eval_features)
    
    print(f"\t[DATA] Model train data: {model_train_data.shape}, labels: {model_train_labels.shape}")
    print(f"\t[DATA] Model eval data: {model_eval_data.shape}, labels: {model_eval_labels.shape}")
    
    # Unpack real data
    train_data, train_labels, train_features = real_train
    eval_data, eval_labels, eval_features = real_eval
    
    # Check if training and evaluation labels are the same as the model labels
    if not torch.equal(model_train_labels, train_labels):
        print("\t[WARNING] Training labels do not match model training labels")
    if not torch.equal(model_eval_labels, eval_labels):
        print("\t[WARNING] Evaluation labels do not match model evaluation labels")
    
    # Run evaluation for generated model samples only
    log_dict = {}
    
    n_classes = eval_labels.shape[1]
    generated_ref_bank = create_reference_bank(model_eval_data, model_eval_labels, n_classes)
    log_dict = log_real_vs_fake(real_ref_bank, generated_ref_bank, log_dict, prefix="train_")
    
    # Calculate metrics only for the generated model data
    evaluate(
        log_dict, 
        real_train,
        real_eval,
        fake_train,
        fake_eval, 
        baseline_metrics
    )

    log_dict["Step"] = int(ckpt_name)
    
    return log_dict

def main():
    parser = argparse.ArgumentParser(description="Validate generated samples and log to wandb")
    parser.add_argument("-m", "--models", nargs="+", choices=[m["name"] for m in MODEL_CONFIGS] + ["all"],
                      default=["all"], help="Models to validate")
    parser.add_argument("-c", "--checkpoint_filter", type=str, default=None,
                      help="Filter checkpoints by pattern (e.g. '1000', '2000', 'final')")
    parser.add_argument("--checkpoint", type=str, default=None,
                      help="Select specific checkpoint(s) to validate (e.g. '630' or '630,1000,2000')")
    parser.add_argument("-n", "--n_per_class", type=int, default=None,
                      help="Number of samples per class to use (None = use all)")
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--skip_missing", action="store_true", help="Skip models with missing directories instead of failing")
    parser.add_argument("--eval_split", choices=["val", "test", "auto"], default="auto",
                      help="Which split to use for evaluation: 'val', 'test', or 'auto' (prefers test, falls back to val)")
    args = parser.parse_args()
    
    # Set models to validate
    if "all" in args.models:
        models_to_validate = MODEL_CONFIGS
    else:
        models_to_validate = [m for m in MODEL_CONFIGS if m["name"] in args.models]
    
    print(f"\n[CONFIG] Models to validate: {[m['name'] for m in models_to_validate]}")
    print(f"[CONFIG] Evaluation split: {args.eval_split}")
    if args.checkpoint_filter:
        print(f"[CONFIG] Checkpoint filter: '{args.checkpoint_filter}'")
    if args.checkpoint:
        print(f"[CONFIG] Specific checkpoints: '{args.checkpoint}'")
    if args.n_per_class:
        print(f"[CONFIG] Samples per class: {args.n_per_class}")
    print(f"[CONFIG] Wandb logging: {'disabled' if args.no_wandb else 'enabled'}")
    print(f"[CONFIG] Skip missing: {'enabled' if args.skip_missing else 'disabled'}")
    
    # Check for conflicting checkpoint options
    if args.checkpoint_filter and args.checkpoint:
        print(f"[WARNING] Both --checkpoint_filter and --checkpoint specified. --checkpoint takes priority.")
    
    # === PRE-LOAD REFERENCE DATA AND CALCULATE BASELINES ONCE FOR ALL MODELS ===
    print("\n[INFO] Loading reference data and calculating baselines for all models...")
    
    # Get the dataset path from the first model's config (assuming all models use the same dataset)
    with open(models_to_validate[0]["config"]) as f:
        first_config = json.load(f)
    
    data_dir = first_config["dataset"]["data_directory"]
    data_dir = os.path.expanduser(data_dir)  # Expand ~ if present
    
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"[ERROR] Data directory not found: {data_dir}")
    
    # Load reference data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data = torch.from_numpy(np.load(os.path.join(data_dir, 'train_data.npy'))).float().to(device)
    train_labels = torch.from_numpy(np.load(os.path.join(data_dir, 'train_labels.npy'))).float().to(device)
    train_features = torch.from_numpy(np.load(os.path.join(data_dir, 'train_features.npy'))).float().to(device)
    val_data = torch.from_numpy(np.load(os.path.join(data_dir, 'val_data.npy'))).float().to(device)
    val_labels = torch.from_numpy(np.load(os.path.join(data_dir, 'val_labels.npy'))).float().to(device)
    val_features = torch.from_numpy(np.load(os.path.join(data_dir, 'val_features.npy'))).float().to(device)
    test_data = torch.from_numpy(np.load(os.path.join(data_dir, 'test_data.npy'))).float().to(device)
    test_labels = torch.from_numpy(np.load(os.path.join(data_dir, 'test_labels.npy'))).float().to(device)
    test_features = torch.from_numpy(np.load(os.path.join(data_dir, 'test_features.npy'))).float().to(device)

    real_train = (train_data, train_labels, train_features)
    real_val = (val_data, val_labels, val_features)
    real_test = (test_data, test_labels, test_features)
    
    # Determine which evaluation split to use globally for baselines
    # This ensures consistency across all models
    if args.eval_split == "val":
        eval_data, eval_labels, eval_features = val_data, val_labels, val_features
        real_eval = real_val
        print("[INFO] Using validation split for baseline calculations")
    elif args.eval_split == "test":
        eval_data, eval_labels, eval_features = test_data, test_labels, test_features
        real_eval = real_test
        print("[INFO] Using test split for baseline calculations")
    else:  # auto - prefer test, fallback to val
        eval_data, eval_labels, eval_features = test_data, test_labels, test_features
        real_eval = real_test
        print("[INFO] Using test split for baseline calculations (auto-selected)")
    
    # Calculate reference bank once
    n_classes = eval_labels.shape[1]
    real_ref_bank = create_reference_bank(eval_data, eval_labels, n_classes)
    
    # Create matched train data for baseline calculations
    matched_train_data, matched_train_features = match_train_val(
        train_data, train_features, train_labels, eval_data, eval_labels
    )
    
    # Prepare data for baseline calculations
    signal_baseline_data = (eval_data, matched_train_data, eval_labels)
    feature_baseline_data = (eval_features, matched_train_features, eval_labels)
    classifier_baseline_data = (train_data, train_labels, eval_data, eval_labels)
    
    # Calculate all baseline metrics once
    baseline_metrics = {}

    # Feature metrics baselines
    baseline_metrics["MMD"] = run_mmd(*signal_baseline_data, analytics=True, return_raw=True)
    baseline_metrics["PSD-PRD"] = run_psd_prd(*signal_baseline_data, analytics=True, return_raw=True)
    baseline_metrics["PSD-MMD"] = run_psd_mmd(*signal_baseline_data, analytics=True, return_raw=True)
    baseline_metrics["FID"] = run_fid(*feature_baseline_data, analytics=True, return_raw=True)
    baseline_metrics["KID"] = run_mmd(*feature_baseline_data, analytics=True, return_raw=True)

    # Classifier baseline
    _, baseline_metrics["TRTR"], _ = run_TRTR(*classifier_baseline_data, analytics=False)
    print(f"[INFO] TRTR calculated: {baseline_metrics['TRTR']}")
    
    print("[INFO] All baseline metrics calculated!")
    
    # Process each model with shared baselines
    for model_info in models_to_validate:
        model_name = model_info["name"]
        
        # Load config to get project info
        with open(model_info["config"]) as f:
            config = json.load(f)
        
        project_name = config["experiment"]["project"]
        run_name = config["experiment"]["run"]
        
        # Find all sample directories for this model
        sample_dirs = find_sample_dirs(config, config["experiment"])
        
        if not sample_dirs or not any(sample_dirs.values()):
            print(f"[WARNING] No sample directories found for {model_name}")
            if args.skip_missing:
                print(f"[INFO] Skipping {model_name} due to --skip_missing flag")
                continue
            else:
                continue
        
        print(f"[INFO] Available directories for {model_name}: {[k for k, v in sample_dirs.items() if v is not None]}")
        
        # Find sample files for available directories
        available_dirs = [(k, v) for k, v in sample_dirs.items() if v is not None]
        
        if len(available_dirs) < 2:
            print(f"[ERROR] Need at least 2 directories (train and val/test) for {model_name}, but only found: {len(available_dirs)}")
            continue
        
        # Select evaluation directory based on user preference
        train_dir = sample_dirs.get('train')
        
        if args.eval_split == "val":
            eval_dir = sample_dirs.get('val')
            eval_split_name = "val"
        elif args.eval_split == "test":
            eval_dir = sample_dirs.get('test')
            eval_split_name = "test"
        else:  # auto
            eval_dir = sample_dirs.get('test') or sample_dirs.get('val')
            eval_split_name = "test" if sample_dirs.get('test') else "val"
        
        if not train_dir or not eval_dir:
            print(f"[ERROR] Missing required directories for {model_name}:")
            print(f"  - Train directory: {'✓' if train_dir else '✗'}")
            print(f"  - {eval_split_name.capitalize()} directory: {'✓' if eval_dir else '✗'}")
            continue
            
        print(f"[INFO] Using train + {eval_split_name} directories for {model_name}")
            
        train_files = find_sample_files(train_dir)
        eval_files = find_sample_files(eval_dir)
        
        if not train_files or not eval_files:
            print(f"[ERROR] Sample files missing in one or more directories for {model_name}")
            continue
        
        # Match files by checkpoint name
        train_dict = {ckpt: (data, label, feat) for data, label, feat, ckpt in train_files}
        eval_dict = {ckpt: (data, label, feat) for data, label, feat, ckpt in eval_files}
        
        common_ckpts = set(train_dict.keys()) & set(eval_dict.keys())
        
        if not common_ckpts:
            print(f"[ERROR] No matching checkpoint files between train and {eval_split_name} directories for {model_name}")
            continue
        
        # Apply checkpoint filtering
        common_ckpts_list = list(common_ckpts)
        filtered_ckpts = filter_checkpoints(
            common_ckpts_list, 
            args.checkpoint_filter, 
            args.checkpoint
        )
        
        if not filtered_ckpts:
            print(f"[ERROR] No checkpoints remain after filtering for {model_name}")
            if args.checkpoint_filter:
                print(f"[ERROR] Filter pattern: '{args.checkpoint_filter}'")
            if args.checkpoint:
                print(f"[ERROR] Specific checkpoints: '{args.checkpoint}'")
            print(f"[ERROR] Available checkpoints: {sorted(common_ckpts_list, key=lambda x: int(x) if x.isdigit() else x)}")
            continue
            
        print(f"[INFO] Found {len(filtered_ckpts)} matching checkpoint pairs for {model_name}")
        if len(filtered_ckpts) != len(common_ckpts_list):
            print(f"[INFO] Filtered from {len(common_ckpts_list)} to {len(filtered_ckpts)} checkpoints")
        
        # Initialize wandb for this model and run
        if not args.no_wandb:
            wandb_config = {
                "model": model_name,
                "project": project_name,
                "n_per_class": args.n_per_class,
                "eval_split": eval_split_name
            }
            if args.checkpoint_filter:
                wandb_config["checkpoint_filter"] = args.checkpoint_filter
            if args.checkpoint:
                wandb_config["specific_checkpoints"] = args.checkpoint
            
            wandb_run = wandb.init(
                project=f"{project_name}",
                name=f"{run_name}_{eval_split_name}",
                config=wandb_config,
            )
        
        # Validate each matching checkpoint pair
        for ckpt_name in tqdm(sorted(filtered_ckpts, key=lambda x: int(x) if x.isdigit() else x), desc=f"Validating {model_name} samples"):
            try:
                data_file_train, label_file_train, feature_file_train = train_dict[ckpt_name]
                data_file_eval, label_file_eval, feature_file_eval = eval_dict[ckpt_name]
                
                log_dict = validate_samples(
                    model_name, 
                    project_name, 
                    data_file_train, 
                    label_file_train,
                    feature_file_train,
                    data_file_eval,
                    label_file_eval, 
                    feature_file_eval,
                    ckpt_name, 
                    config,
                    args.n_per_class,
                    real_train,
                    real_eval,
                    real_ref_bank,
                    baseline_metrics
                )
                
                # Log to wandb
                if not args.no_wandb:
                    wandb.log(log_dict)
                
                print(f"[SUCCESS] Validated checkpoint: {ckpt_name}")
                
            except Exception as e:
                print(f"[ERROR] Failed to validate checkpoint {ckpt_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Finish wandb run
        if not args.no_wandb:
            wandb_run.finish()
        
        print(f"[COMPLETED] Validation for {model_name}")

if __name__ == "__main__":
    main()