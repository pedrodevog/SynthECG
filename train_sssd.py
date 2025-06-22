import os
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torchinfo
import wandb
from tqdm import tqdm

# Utils
from utils.utils import *
from utils.train import *

# Model
from models.SSSD_ECG import SSSD_ECG

"""
This code is part of the SSSD-ECG project.
Repository: https://github.com/AI4HealthUOL/SSSD-ECG/tree/main
"""

def training_step(net, optimizer, batch, diffusion_hyperparams):
    """
    Performs a single training step for the neural network.
    This function executes one training iteration including forward pass, loss calculation,
    and backpropagation for the given network and batch of data.
    Args:
        net: The neural network model to train
        optimizer: The optimizer used for updating model parameters
        batch: Tuple containing:
            - signal_batch: Input signal_batch data
            - label_batch: Corresponding label_batch/targets
        diffusion_hyperparams: Dictionary containing diffusion model hyperparameters
    Returns:
        loss: The computed loss value for this training step
    Example:
        >>> loss = training_step(model, optimizer, (signal_batch, label_batch), diff_params)
    """
    signal_batch, label_batch = batch
    
    # back-propagation
    optimizer.zero_grad()
    loss = training_loss_label(net, nn.MSELoss(), (signal_batch, label_batch), diffusion_hyperparams)
    loss.backward()
    optimizer.step()

    return loss

def train(
        ckpt_iter,
        ckpt_interval,
        log_interval,
        n_iters,
        learning_rate,
        batch_size,
        classifier_path
    ):
    """
    Train Diffusion Models

    Parameters:
    ckpt_iter (int or 'max'):       the pretrained checkpoint to be loaded; 
                                    automatically selects the maximum iteration if 'max' is selected
    ckpt_interval (int):            number of iterations to save checkpoint
    log_interval (int):             number of iterations to save training log and compute validation
                                    loss, default is 100
    n_iters (int):                  number of iterations to train
    learning_rate (float):          learning rate
    batch_size (int):               batch size
    """

    torch.cuda.set_device(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    project = experiment["project"]
    verbose = False
    n_classes = model_config["label_embed_classes"]
    n_samples = 20

    # Prepare checkpoint directory
    run, run_id = find_run(config, experiment)
    local_path = "run_{}".format(run)
    ckpt_dir = os.path.join(experiment["ckpt_directory"], local_path)
    os.makedirs(ckpt_dir, exist_ok=True)

    # predefine model
    dict_model = {**model_config, **diffusion_hyperparams}
    net = SSSD_ECG(**dict_model).to(device)
    
    # define optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # load checkpoint if exists
    ckpt_iter, net, optimizer = load_checkpoint(ckpt_dir, ckpt_iter, net, optimizer)

    # training
    n_iter = ckpt_iter + 1

    # [WANDB] run initialization
    config_wandb = {}
    config_wandb.update(train_config)
    config_wandb.update(model_config)
    config_wandb.update(diffusion_config)
    
    run = wandb.init(
        project=project,
        name=experiment["run"],
        config=config_wandb,
        fork_from=f"{run_id}?_step={ckpt_iter}"
    )
    run_id = run.id
    experiment["id"] = run_id
    config["experiment"] = experiment
    
    with open(os.path.join(ckpt_dir, f'config_{project}_{experiment["run"]}.json'), 'w') as f:
        json.dump(config, f, indent=4)

    if verbose:
        print(torchinfo.summary(net, input_size=[(1, 8, 1000), (1, n_classes), (1, 1)], col_names=("input_size", "output_size", "num_params", "mult_adds")))

    # map diffusion hyperparameters to gpu
    for key in diffusion_hyperparams:
        if key != "T":
            diffusion_hyperparams[key] = diffusion_hyperparams[key].to(device)

    data_dir = dataset_config["data_directory"]
    if not os.path.exists(data_dir):
        print(f"Data directory not found at {data_dir}.")
        raise FileNotFoundError(f"Data directory not found.")

    signals = np.load(os.path.join(data_dir, 'train_data.npy'))
    labels = np.load(os.path.join(data_dir, 'train_labels.npy'))

    print("data shape: ", signals.shape)
    print("label shape: ", labels.shape)

    test_data = np.load(os.path.join(data_dir, 'test_data.npy'))
    test_labels = np.load(os.path.join(data_dir, 'test_labels.npy'))

    # These are the indices of the 12-lead ECG signals (8 standard + 4 augmented)
    index_8 = torch.tensor([0,2,3,4,5,6,7,11])
    index_4 = torch.tensor([1,8,9,10])
    
    if project == "ECG":
        signals = torch.index_select(torch.tensor(signals), 1, index_8).numpy()

    data = []
    for signal, label in zip(signals, labels):
        data.append([signal, label])

    dataloader = torch.utils.data.DataLoader(
        data, 
        shuffle=True, 
        batch_size=batch_size, 
        drop_last=True
    )
    
    while n_iter < n_iters:
        log_dict = {}
        loss_train_epoch = []
        for i, batch in tqdm(enumerate(dataloader, 0)):
            signal_batch = batch[0].float().to(device)
            label_batch = batch[1].float().to(device)
            loss = training_step(net, optimizer, (signal_batch, label_batch), diffusion_hyperparams)
            loss_train_epoch.append(loss.data.cpu())
            with torch.no_grad():
                if n_iter == 0 and i == 0:
                    log_dict["train/loss"] = loss.item()
                    log_dict["train/loss_avg"] = loss.item()
                    run.log(log_dict)

        with torch.no_grad():
            loss_train_epoch_avg = sum(loss_train_epoch) / float(len(loss_train_epoch))
            
            log_dict["train/loss"] = loss.item()
            log_dict["train/loss_avg"] = loss_train_epoch_avg

            # Log everything to WandB
            run.log(log_dict)

            # save checkpoint
            if n_iter % ckpt_interval == 0:
                save_checkpoint(ckpt_dir, n_iter, net, optimizer)

            # n_iter += 1     # ITERATION
        n_iter += 1     # EPOCH

    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='configs/config_SSSD_ECG.json',
                        help='JSON file for configuration')

    args = parser.parse_args()

    with open(args.config) as f:
        data = f.read()

    global config
    config = json.loads(data)
    display_config(config)
    
    train_config = config["train"]  # training parameters

    global dataset_config
    dataset_config = config["dataset"]  # to load trainset

    global diffusion_config
    diffusion_config = config["diffusion"]  # basic hyperparameters

    global diffusion_hyperparams
    diffusion_hyperparams = calc_diffusion_hyperparams(**diffusion_config)  # dictionary of all diffusion hyperparameters

    global model_config
    model_config = config['model']

    global experiment
    experiment = config['experiment']

    train(**train_config)
