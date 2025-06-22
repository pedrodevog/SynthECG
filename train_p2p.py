import argparse
import os
from tqdm import tqdm
import numpy as np
import wandb

#Pytorch
import torch
import torch.optim as optim
from torch.autograd import Variable
from torch import autograd
import torchinfo

# Model specific
from models.cond_pulse2pulse import CondP2PGenerator, CondP2PDiscriminator
from utils.utils import *
from utils.train import *

torch.manual_seed(0)
np.random.seed(0)


#====================================
# Run training process
#====================================
def train(
        ckpt_iter,
        ckpt_interval,
        log_interval,
        batch_size,
        learning_rate,
        beta1,
        beta2,
        n_epochs,
        lmbda,
        classifier_path
    ):
    torch.cuda.set_device(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    project = experiment["project"]
    verbose = model_config["verbose"]
    n_classes = generator_config["label_embed_classes"]
    n_channels = model_config["n_channels"]
    n_samples = 20

    # Prepare checkpoint directory
    run, run_id = find_run(config, experiment)
    local_path = "run_{}".format(run)
    ckpt_dir = os.path.join(experiment["ckpt_directory"], local_path)
    os.makedirs(ckpt_dir, exist_ok=True)

    # predefine model
    netG = CondP2PGenerator(**generator_config).to(device)
    netD = CondP2PDiscriminator(**discriminator_config).to(device)
    net = (netG, netD)
    classifier = initialize_classifier(classifier_path)

    # define optimizer
    optimizerG = optim.Adam(netG.parameters(), lr=learning_rate, betas=(beta1, beta2))
    optimizerD = optim.Adam(netD.parameters(), lr=learning_rate, betas=(beta1, beta2))
    optimizer = (optimizerG, optimizerD)

    # load checkpoint if exists
    ckpt_iter, net, optimizer = load_checkpoint(ckpt_dir, ckpt_iter, net, optimizer)

    # training
    n_epoch = ckpt_iter + 1

    # [WANDB] run initialization
    config_wandb = {}
    config_wandb.update(train_config)
    config_wandb.update(generator_config)
    config_wandb.update(discriminator_config)

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

    netG, netD = net
    optimizerG, optimizerD = optimizer

    if verbose:
        print(torchinfo.summary(netG, [(1, 8, 1000), (1, n_classes)], col_names = ("input_size", "output_size", "num_params", "mult_adds")))
        print(torchinfo.summary(netD, (1, 8, 1000), col_names = ("input_size", "output_size", "num_params", "mult_adds")))

    data_dir = dataset_config["data_directory"]
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory {data_dir} not found.")

    signals = np.load(os.path.join(data_dir, 'train_data.npy'))
    labels = np.load(os.path.join(data_dir, 'train_labels.npy'))

    index_8 = torch.tensor([0,2,3,4,5,6,7,11])
    index_4 = torch.tensor([1,8,9,10])
    if project == "ECG":
        signals = torch.index_select(torch.from_numpy(signals), 1, index_8).float().numpy()

    print("signals.shape=", signals.shape)
    print("labels.shape=", labels.shape)

    data = []
    for signal, label in zip(signals, labels):
        data.append([signal, label])

    dataloader = torch.utils.data.DataLoader(
        data, 
        batch_size=batch_size, 
        shuffle=True, 
        drop_last=True
    )

    for epoch in tqdm(range(n_epoch, n_epochs)):

        print('\n')
        print('This is the epoch number', epoch)
        print('\n')
        log_dict = {}

        train_G_flag = False
        D_cost_train_epoch = []
        D_wass_train_epoch = []
        G_cost_epoch = []
    
        for i, batch in tqdm(enumerate(dataloader, 0)):
            
            signal_batch = batch[0].float()
            label_batch = batch[1].float()
            
            signal_batch = {'ecg_signals': signal_batch}

            if (i+1) % 5 == 0:
                train_G_flag = True

            # Set Discriminator parameters to require gradients.
            for p in netD.parameters():
                p.requires_grad = True

            one = torch.tensor(1, dtype=torch.float)
            neg_one = one * -1

            one = one.to(device)
            neg_one = neg_one.to(device)

            #############################
            # (1) Train Discriminator
            #############################

            real_ecgs = signal_batch['ecg_signals'].to(device)
            netD.zero_grad()

            # Noise
            noise = torch.Tensor(batch_size, n_channels, 1000).uniform_(-1, 1)
            noise = noise.to(device)
            noise_Var = Variable(noise, requires_grad=False)

            # a) compute loss contribution from real training data
            D_real = netD(real_ecgs)
            D_real = D_real.mean()  # avg loss
            D_real.backward(neg_one)  # loss * -1

            # b) compute loss contribution from generated data, then backprop.
            fake = autograd.Variable(netG(noise_Var, label_batch.cuda()).data)
            D_fake = netD(fake)
            D_fake = D_fake.mean()
            D_fake.backward(one)

            # c) compute gradient penalty and backprop
            gradient_penalty = calc_gradient_penalty(netD, real_ecgs,
                                                    fake.data, batch_size, lmbda,
                                                    use_cuda=True)
            gradient_penalty.backward(one)

            # Compute cost * Wassertein loss..
            D_cost_train = D_fake - D_real + gradient_penalty
            D_wass_train = D_real - D_fake

            # Update gradient of discriminator.
            optimizerD.step()

            D_cost_train_cpu = D_cost_train.data.cpu()
            D_wass_train_cpu = D_wass_train.data.cpu()

            D_cost_train_epoch.append(D_cost_train_cpu)
            D_wass_train_epoch.append(D_wass_train_cpu)

            #############################
            # (3) Train Generator
            #############################
            if train_G_flag:
                # Prevent discriminator update.
                for p in netD.parameters():
                    p.requires_grad = False

                # Reset generator gradients
                netG.zero_grad()

                # Noise
                noise = torch.Tensor(batch_size, n_channels, 1000).uniform_(-1, 1)
                
                noise = noise.to(device)
                noise_Var = Variable(noise, requires_grad=False)

                fake = netG(noise_Var, label_batch.cuda())     ################ pass labels here!
                G = netD(fake)
                G = G.mean()

                # Update gradients.
                G.backward(neg_one)
                G_cost = -G

                optimizerG.step()

                # Record costs
                G_cost_cpu = G_cost.data.cpu()
                G_cost_epoch.append(G_cost_cpu)
                train_G_flag =False

        with torch.no_grad():
            D_cost_train_epoch_avg = sum(D_cost_train_epoch) / float(len(D_cost_train_epoch))
            D_wass_train_epoch_avg = sum(D_wass_train_epoch) / float(len(D_wass_train_epoch))
            G_cost_epoch_avg = sum(G_cost_epoch) / float(len(G_cost_epoch))

            log_dict['train/D_cost_epoch_avg'] = D_cost_train_epoch_avg.item()
            log_dict['train/D_wass_epoch_avg'] = D_wass_train_epoch_avg.item()
            log_dict['train/G_cost_epoch_avg'] = G_cost_epoch_avg.item()

            # if epoch % log_interval == 0:
            #     generated, generate_label = netG.sample_trained_model(samples=n_samples)
            #     generated_ref_bank = create_reference_bank(generated, generate_label, n_classes)

            #     log_dict = log_real_vs_fake(ref_real_samples, generated_ref_bank, log_dict, prefix="train_")
            #     log_dict = evaluate(log_dict, signals, labels, test_data, test_labels, model_data=generated, model_labels=generate_label, n_samples=n_samples, n_classes=n_classes, classifier=classifier, project=project)

            run.log(log_dict)

            # save checkpoint
            if epoch % ckpt_interval == 0:
                net = (netG, netD)
                optimizer = (optimizerG, optimizerD)
                save_checkpoint(ckpt_dir, epoch, net, optimizer)

            print("Epochs: {}\t\tD_cost: {}\t\t D_wass: {}\t\tG_cost: {}".format(
                        epoch, D_cost_train_epoch_avg, D_wass_train_epoch_avg, G_cost_epoch_avg))


if __name__ == "__main__":
    print("Training process is strted..!")

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='configs/config_cond_pulse2pulse_ECG.json',
                        help='JSON file for configuration')

    args = parser.parse_args()

    with open(args.config) as f:
        data = f.read()

    global config
    config = json.loads(data)
    display_config(config)
    
    train_config = config["train"]  # training parameters

    global dataset_config
    dataset_config = config["dataset"]  # to load dataset

    global model_config
    model_config = config['model']

    global generator_config
    global discriminator_config
    generator_config = {**model_config, **config['generator']}
    discriminator_config = {**model_config, **config['discriminator']}

    global experiment
    experiment = config['experiment']

    train(**train_config)