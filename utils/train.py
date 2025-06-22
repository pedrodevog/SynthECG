# General packages
import torch
from torch import autograd
import numpy as np
import json
import os
# Utils
from utils.utils import find_max_epoch


def initialize_classifier(path=""):
    num_classes = None
    model = ...     # Initialize your foundation model here
    
    # Loading model weights
    enc_weights = torch.load(path, map_location=torch.device('cpu'))
    model.load_state_dict(enc_weights, strict=False) 

    print(f"Classifier model from {path} == loaded successfully")

    return model.cuda()


def training_loss_label(net, loss_fn, X, diffusion_hyperparams):
    """
    Compute the training loss of epsilon and epsilon_theta

    Parameters:
    net (torch network):            the wavenet model
    loss_fn (torch loss function):  the loss function, default is nn.MSELoss()
    X (torch.tensor):               training data, shape=(batchsize, 1, length of audio)
    diffusion_hyperparams (dict):   dictionary of diffusion hyperparameters returned by calc_diffusion_hyperparams
                                    note, the tensors need to be cuda tensors

    Returns:
    training loss
    """

    def std_normal(size):
        """
        Generate the standard Gaussian variable of a certain size
        """

        return torch.normal(0, 1, size=size).cuda()


    _dh = diffusion_hyperparams
    T, Alpha_bar = _dh["T"], _dh["Alpha_bar"]

    signal_batch = X[0]
    label_batch = X[1]
    B, C, L = signal_batch.shape  # B is batchsize, C=1, L is audio length
    diffusion_steps = torch.randint(T, size=(B, 1, 1)).cuda()  # randomly sample diffusion steps from 1~T
    z = std_normal(signal_batch.shape)
    transformed_X = torch.sqrt(Alpha_bar[diffusion_steps]) * signal_batch + torch.sqrt(1 - Alpha_bar[diffusion_steps]) * z
    epsilon_theta = net(transformed_X, label_batch, diffusion_steps.view(B, 1))

    return loss_fn(epsilon_theta, z)


def calc_diffusion_hyperparams(T, beta_0, beta_T):
    """
    Compute diffusion process hyperparameters

    Parameters:
    T (int):                    number of diffusion steps
    beta_0 and beta_T (float):  beta schedule start/end value,
                                where any beta_t in the middle is linearly interpolated

    Returns:
    a dictionary of diffusion hyperparameters including:
        T (int), Beta/Alpha/Alpha_bar/Sigma (torch.tensor on cpu, shape=(T, ))
        These cpu tensors are changed to cuda tensors on each individual gpu
    """

    Beta = torch.linspace(beta_0, beta_T, T)  # Linear schedule
    Alpha = 1 - Beta
    Alpha_bar = Alpha + 0
    Beta_tilde = Beta + 0
    for t in range(1, T):
        Alpha_bar[t] *= Alpha_bar[t - 1]  # \bar{\alpha}_t = \prod_{s=1}^t \alpha_s
        Beta_tilde[t] *= (1 - Alpha_bar[t - 1]) / (
                1 - Alpha_bar[t])  # \tilde{\beta}_t = \beta_t * (1-\bar{\alpha}_{t-1})
        # / (1-\bar{\alpha}_t)
    Sigma = torch.sqrt(Beta_tilde)  # \sigma_t^2  = \tilde{\beta}_t

    _dh = {}
    _dh["T"], _dh["Beta"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Sigma"] = T, Beta, Alpha, Alpha_bar, Sigma
    diffusion_hyperparams = _dh
    return diffusion_hyperparams


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
