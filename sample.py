import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import os
import json
import numpy as np
from utils.utils import generate_four_leads
from utils.train import find_run, initialize_classifier
import tqdm
import argparse

class Args:
    pass

# Model import mapping
MODEL_CONFIGS = [
    {
        "name": "SSSD_ECG",
        "config": "configs/config_SSSD_ECG.json",
        "model_class": "SSSD_ECG",
        "import_path": "models.SSSD_ECG",
        "sample_method": "sample_trained_model"
    },
    {
        "name": "DSAT_ECG",
        "config": "configs/config_DSAT_ECG.json",
        "model_class": "DSAT_ECG",
        "import_path": "models.DSAT_ECG",
        "sample_method": "sample_trained_model"
    },
    {
        "name": "WaveGAN",
        "config": "configs/config_cond_wavegan_star_ECG.json",
        "model_class": "CondWaveGANGenerator",
        "import_path": "models.cond_wavegan_star",
        "sample_method": "sample_trained_model"
    },
    {
        "name": "Pulse2Pulse",
        "config": "configs/config_cond_pulse2pulse_ECG.json",
        "model_class": "CondP2PGenerator",
        "import_path": "models.cond_pulse2pulse",
        "sample_method": "sample_trained_model"
    }
]

def dynamic_import(module, class_name):
    import importlib
    return getattr(importlib.import_module(module), class_name)

def setup(rank, world_size):
    """Initialize the distributed environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    """Clean up distributed environment."""
    dist.destroy_process_group()

def clear_gpu_memory():
    """Clear GPU cache and release memory."""
    import gc
    torch.cuda.empty_cache()
    gc.collect()  # Force Python garbage collection

def main_worker(gpu, ngpus_per_node, args):
    try:
        """Function to be run on each GPU."""
        rank = gpu
        print(f"Running on GPU {gpu}")
        
        # Initialize the distributed environment
        setup(rank, ngpus_per_node)
        
        # Set up device for this process
        torch.cuda.set_device(gpu)
        device = torch.device(f'cuda:{gpu}')

        clear_gpu_memory()
        
        # Load shared classifier (loaded by each process)
        classifier = initialize_classifier(args.classifier_path)
        classifier.eval()
        classifier.to(device)
        
        # Filter models based on user selection
        selected_models = []
        if args.model == "all":
            selected_models = MODEL_CONFIGS
        else:
            # Find the model configuration that matches the requested model name
            for model_info in MODEL_CONFIGS:
                if model_info["name"].lower() == args.model.lower():
                    selected_models = [model_info]
                    break
            
            if not selected_models:
                if rank == 0:
                    print(f"Warning: Model '{args.model}' not found in MODEL_CONFIGS. Available models:")
                    for model_info in MODEL_CONFIGS:
                        print(f"  - {model_info['name']}")
                    print("Exiting.")
                return
        
        # Process each selected model
        for model_index, model_info in enumerate(selected_models):
            # Clear CUDA cache before starting a new model
            clear_gpu_memory()
            
            # Report memory status at the beginning of each model
            if rank == 0:
                print(f"[GPU {gpu}] Memory status before model {model_info['name']}:")
                print(f"  Allocated: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")
                print(f"  Reserved: {torch.cuda.memory_reserved(device) / 1024**3:.2f} GB")
                print(f"  Max Allocated: {torch.cuda.max_memory_allocated(device) / 1024**3:.2f} GB")
            # Load config
            with open(model_info["config"]) as f:
                config = json.load(f)
            experiment = config["experiment"]
            dataset_config = config.get("dataset", {})
            # Find the correct run directory
            run, _ = find_run(config, experiment)
            local_path = f"run_{run}"
            ckpt_dir = os.path.join(experiment["ckpt_directory"], local_path)

            if rank == 0:
                print(f"[{model_info['name']}] Sampling from {ckpt_dir}")

            # Output directory
            ckpt_parts = os.path.normpath(ckpt_dir).split(os.sep)
            try:
                if "checkpoints" in ckpt_parts:
                    ckpt_idx = ckpt_parts.index("checkpoints")
                else:
                    raise ValueError
                base_path = "." if ckpt_idx == 0 else os.path.join(*ckpt_parts[:ckpt_idx])
                model_name = ckpt_parts[ckpt_idx + 1]
                run_name = ckpt_parts[ckpt_idx + 2]
            except (ValueError, IndexError):
                raise ValueError(f"Unexpected ckpt_dir format: {ckpt_dir}")
            output_dir_train = os.path.join("/" + base_path, "train", model_name, run_name)
            output_dir_val = os.path.join("/" + base_path, "val", model_name, run_name)
            output_dir_test = os.path.join("/" + base_path, "test", model_name, run_name)

            if rank == 0:
                os.makedirs(output_dir_train, exist_ok=True)
                os.makedirs(output_dir_val, exist_ok=True)
                os.makedirs(output_dir_test, exist_ok=True)
            
            # Synchronize to make sure directories are created
            dist.barrier()

            # Prepare model
            if model_info["name"] in ["SSSD_ECG", "DSAT_ECG"]:
                diffusion_config = config["diffusion"]
                from utils.utils import calc_diffusion_hyperparams
                diffusion_hyperparams = calc_diffusion_hyperparams(**diffusion_config)
                model_config = config["model"]
                dict_model = {**model_config, **diffusion_hyperparams}
            else:
                model_config = config["model"]
                if model_info["name"] == "WaveGAN":
                    dict_model = {**model_config, **config["generator"]}
                elif model_info["name"] == "Pulse2Pulse":
                    dict_model = {**model_config, **config["generator"]}
                else:
                    dict_model = model_config

            # Import model class
            ModelClass = dynamic_import(model_info["import_path"], model_info["model_class"])

            if not os.path.exists(ckpt_dir):
                if rank == 0:
                    print(f"Warning: Checkpoint directory does not exist: {ckpt_dir}. Skipping {model_info['name']}.")
                continue

            if rank == 0:
                sorted_files = sorted(os.listdir(ckpt_dir))
                print("Full sorted files:", sorted_files)
                print("Index calculation:", args.ckpt_start, args.ckpt_end)
                selected_files = sorted_files[args.ckpt_start:args.ckpt_end]
                print("Selected files:", selected_files)
                for i, file in enumerate(selected_files):
                    print(f"Index {i}: {file}")

            for ckpt_file in sorted(os.listdir(ckpt_dir))[args.ckpt_start:args.ckpt_end]:
                if not ckpt_file.endswith(".pkl"):
                    continue
                ckpt_path = os.path.join(ckpt_dir, ckpt_file)
                
                label_sets = [args.train_labels, args.val_labels, args.test_labels]
                label_names = ["train", "val", "test"]
                output_dirs = [output_dir_train, output_dir_val, output_dir_test]
                
                for labels_np, label_name, output_dir in zip(label_sets, label_names, output_dirs):
                    data_out_path = os.path.join(output_dir, f"data_{ckpt_file.replace('.pkl', '')}.npy")
                    # All ranks check if the combined file exists
                    skip_flag = torch.tensor(int(os.path.exists(data_out_path)), device=device)
                    # Reduce across all ranks: skip only if all ranks see the file exists
                    dist.all_reduce(skip_flag, op=dist.ReduceOp.MIN)
                    if skip_flag.item() == 1:
                        if rank == 0:
                            print(f"[{model_info['name']}] Samples already exist for checkpoint: {ckpt_file} in {data_out_path}. Skipping.")
                        dist.barrier()
                        continue
                    
                    # Partition the data
                    labels_np = np.array(labels_np)
                    num_items = labels_np.shape[0]
                    
                    # Instead of contiguous blocks, each GPU takes every nth item
                    # This way we maintain easier reassembly in original order
                    indices = np.arange(start=gpu, stop=num_items, step=ngpus_per_node)
                    
                    # Skip if this GPU has no data to process
                    if len(indices) == 0:
                        dist.barrier()
                        continue
                    
                    # Process only this GPU's portion but keep track of original indices
                    gpu_labels_np = labels_np[indices]
                    gpu_labels = torch.from_numpy(gpu_labels_np).float().to(device)
                    
                    # Save original indices for reconstruction
                    original_indices = indices
                    
                    if rank == 0:
                        print(f"[{model_info['name']}] GPU {gpu} processing {len(gpu_labels)} samples from checkpoint: {ckpt_path}")
                    
                    # Set up GPU-specific output paths
                    gpu_data_out_path = os.path.join(output_dir, f"data_{ckpt_file.replace('.pkl', '')}_gpu{gpu}.npy")
                    gpu_labels_out_path = os.path.join(output_dir, f"labels_{ckpt_file.replace('.pkl', '')}_gpu{gpu}.npy")
                    gpu_features_out_path = os.path.join(output_dir, f"features_{ckpt_file.replace('.pkl', '')}_gpu{gpu}.npy")
                    
                    # Skip if this GPU portion is already processed
                    if os.path.exists(gpu_data_out_path):
                        if rank == 0:
                            print(f"[{model_info['name']}] GPU {gpu} samples already exist: {gpu_data_out_path}. Skipping.")
                        dist.barrier()
                        continue
                    
                    # Load model
                    net = ModelClass(**dict_model).to(device)
                    state = torch.load(ckpt_path, map_location=device)
                    # For GANs, checkpoint may be a dict with 'modelG_state_dict'
                    if model_info["name"] in ["WaveGAN", "Pulse2Pulse"]:
                        if isinstance(state, dict) and "modelG_state_dict" in state:
                            net.load_state_dict(state["modelG_state_dict"])
                        elif isinstance(state, (list, tuple)):
                            net.load_state_dict(state[0])
                        else:
                            net.load_state_dict(state)
                    else:
                        net.load_state_dict(state["model_state_dict"])
                    net.eval()

                    # Sample from the model
                    with torch.no_grad():
                        sample_fn = getattr(net, model_info["sample_method"])
                        batch_size = 256
                        samples_list = []
                        for i in tqdm.tqdm(range(0, gpu_labels.shape[0], batch_size), 
                                        desc=f"GPU {gpu} sampling {model_info['name']}",
                                        disable=rank != 0):
                            batch_labels = gpu_labels[i:i+batch_size]
                            result = sample_fn(samples=batch_labels.size(0), labels=batch_labels)
                            if isinstance(result, tuple):
                                batch_samples = result[0]
                            else:
                                batch_samples = result
                            samples_list.append(batch_samples.cpu())
                        samples = torch.cat(samples_list, dim=0)
                        samples = generate_four_leads(samples).float().to(device)

                        batch_size = 512
                        features_list = []
                        for i in tqdm.tqdm(range(0, samples.shape[0], batch_size), 
                                        desc=f"GPU {gpu} extracting features {model_info['name']}",
                                        disable=rank != 0):
                            batch_samples = samples[i:i+batch_size]
                            result = classifier(batch_samples)
                            if isinstance(result, tuple):
                                batch_features = result[0]
                            else:
                                batch_features = result
                            features_list.append(batch_features.cpu())
                        features = torch.cat(features_list, dim=0)

                    # Save GPU-specific results with original indices
                    np.save(gpu_data_out_path, samples.cpu().numpy())
                    np.save(gpu_labels_out_path, gpu_labels.cpu().numpy())
                    np.save(gpu_features_out_path, features.cpu().numpy())
                    # Save original indices for proper reordering
                    indices_out_path = os.path.join(output_dir, f"indices_{ckpt_file.replace('.pkl', '')}_gpu{gpu}.npy")
                    np.save(indices_out_path, original_indices)
                    if rank == 0:
                        print(f"[{model_info['name']}] GPU {gpu} saved {samples.shape[0]} {label_name} samples")
                    
                # Wait for all GPUs to finish this checkpoint
                dist.barrier()
                
                # Process 0 combines results
                if rank == 0:
                    for label_name, output_dir in zip(label_names, output_dirs):
                        data_out_path = os.path.join(output_dir, f"data_{ckpt_file.replace('.pkl', '')}.npy")
                        labels_out_path = os.path.join(output_dir, f"labels_{ckpt_file.replace('.pkl', '')}.npy")
                        features_out_path = os.path.join(output_dir, f"features_{ckpt_file.replace('.pkl', '')}.npy")
                        
                        # Skip if combined files already exist
                        if os.path.exists(data_out_path) and os.path.exists(labels_out_path):
                            print(f"[{model_info['name']}] Combined files already exist: {data_out_path}. Skipping combination.")
                            continue
                        
                        # For maintaining original order
                        all_data = []
                        all_labels = []
                        all_features = []
                        all_indices = []
                        
                        # Collect GPU-specific parts with their original indices
                        for g in range(ngpus_per_node):
                            gpu_data_path = os.path.join(output_dir, f"data_{ckpt_file.replace('.pkl', '')}_gpu{g}.npy")
                            gpu_labels_path = os.path.join(output_dir, f"labels_{ckpt_file.replace('.pkl', '')}_gpu{g}.npy")
                            gpu_features_path = os.path.join(output_dir, f"features_{ckpt_file.replace('.pkl', '')}_gpu{g}.npy")
                            gpu_indices_path = os.path.join(output_dir, f"indices_{ckpt_file.replace('.pkl', '')}_gpu{g}.npy")
                            
                            if os.path.exists(gpu_data_path):
                                data = np.load(gpu_data_path)
                                labels = np.load(gpu_labels_path)
                                features = np.load(gpu_features_path)
                                indices = np.load(gpu_indices_path)
                                
                                # Store data with its original indices
                                for i in range(len(indices)):
                                    all_data.append((indices[i], data[i]))
                                    all_labels.append((indices[i], labels[i]))
                                    all_features.append((indices[i], features[i]))
                                    all_indices.append(indices[i])
                        
                        # Sort by original indices to restore order
                        if all_data:
                            # Sort by index
                            all_data.sort(key=lambda x: x[0])
                            all_labels.sort(key=lambda x: x[0])
                            all_features.sort(key=lambda x: x[0])
                            
                            # Extract sorted data
                            final_data = np.array([item[1] for item in all_data])
                            final_labels = np.array([item[1] for item in all_labels])
                            final_features = np.array([item[1] for item in all_features])
                            
                            # Verify the ordering is correct
                            sorted_indices = [item[0] for item in all_data]
                            expected_indices = sorted(all_indices)
                            assert sorted_indices == expected_indices, "Data ordering not preserved correctly"
                            
                            np.save(data_out_path, final_data)
                            np.save(labels_out_path, final_labels)
                            np.save(features_out_path, final_features)
                            
                            print(f"[{model_info['name']}] Combined {final_data.shape[0]} {label_name} samples")
                            
                            # Clean up individual GPU files
                            for g in range(ngpus_per_node):
                                gpu_data_path = os.path.join(output_dir, f"data_{ckpt_file.replace('.pkl', '')}_gpu{g}.npy")
                                if os.path.exists(gpu_data_path):
                                    os.remove(gpu_data_path)
                                    os.remove(os.path.join(output_dir, f"labels_{ckpt_file.replace('.pkl', '')}_gpu{g}.npy"))
                                    os.remove(os.path.join(output_dir, f"features_{ckpt_file.replace('.pkl', '')}_gpu{g}.npy"))
                                    os.remove(os.path.join(output_dir, f"indices_{ckpt_file.replace('.pkl', '')}_gpu{g}.npy"))
                
                # FIX: Added barrier outside the if rank == 0 block so all GPUs wait
                dist.barrier()
    # Clean up        
    finally:
        cleanup()

def get_available_models():
    """Returns a string of available model names for help text."""
    model_names = [model_info["name"] for model_info in MODEL_CONFIGS]
    return ", ".join(model_names) + ", or 'all'"

def main():
    parser = argparse.ArgumentParser(description='Multi-GPU ECG model sampling')
    parser.add_argument('--checkpoint-range', type=str, default='0:101', 
                        help='Range of checkpoints to process (start:end)')
    parser.add_argument('--model', type=str, default='all',
                        help=f'Model to use for sampling. Available options: {get_available_models()}')
    args = parser.parse_args()
    
    # Convert checkpoint range
    ckpt_range = args.checkpoint_range.split(':')
    ckpt_start = int(ckpt_range[0])
    ckpt_end = int(ckpt_range[1])
    
    # Get number of GPUs
    ngpus_per_node = torch.cuda.device_count()
    print(f"Using {ngpus_per_node} GPUs")
    
    # Display selected model
    print(f"Selected model: {args.model}")
    
    # Load validation labels and compute class distribution
    # Extract validation label path from config
    with open(MODEL_CONFIGS[0]["config"]) as f:
        config = json.load(f)
    dataset_config = config.get("dataset", {})
    data_dir = dataset_config.get("data_directory", "data/ptbxl/")
    
    train_label_path = os.path.join(data_dir, "train_labels.npy")
    train_labels = np.load(train_label_path)
    class_counts_train = train_labels.sum(axis=0).astype(int)

    val_label_path = os.path.join(data_dir, "val_labels.npy")
    val_labels = np.load(val_label_path)
    class_counts_val = val_labels.sum(axis=0).astype(int)

    test_label_path = os.path.join(data_dir, "test_labels.npy")
    test_labels = np.load(test_label_path)
    class_counts_test = test_labels.sum(axis=0).astype(int)

    # Print number of samples per dataset per class
    print(f"Train samples per class: {class_counts_train}")
    print(f"Validation samples per class: {class_counts_val}")
    print(f"Test samples per class: {class_counts_test}")

    # Get classifier for evaluation
    classifier_path = config["train"].get("classifier_path", "models/classifier.pth")
    
    # Prepare args for distributed processes
    dist_args = Args()
    dist_args.train_labels = train_labels
    dist_args.val_labels = val_labels
    dist_args.test_labels = test_labels
    dist_args.classifier_path = classifier_path
    dist_args.ckpt_start = ckpt_start
    dist_args.ckpt_end = ckpt_end
    dist_args.model = args.model
    
    # Launch processes
    if ngpus_per_node > 1:
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, dist_args))
    else:
        # Run on single GPU or CPU if no GPUs available
        main_worker(0, 1, dist_args)

if __name__ == "__main__":
    main()