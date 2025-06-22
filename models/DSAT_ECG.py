import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from utils.train import calc_diffusion_step_embedding

from models.SPADEModel import SpadeDecoderLayerBase


"""
This code is part of the DSAT-ECG project.
Repository: / (acquired via mail)
Paper:      https://www.mdpi.com/1424-8220/23/19/8328
"""

from fairseq.models.transformer.transformer_config import TransformerConfig
from dataclasses import dataclass, field
from typing import Optional
@dataclass
class SpadeConfig(TransformerConfig):
    attention_type: str = field(
        default="full",
        metadata={"help": "type of attention"},
    )

    # relative attention parameters
    use_relative_attention: bool = field(
        default=False,
        metadata={"help": "whether to use relative attention"},
    )
    relative_attention_num_buckets: int = field(
        default=32,
        metadata={"help": "radius of local attention"},
    )
    relative_attention_max_distance: int = field(
        default=128,
        metadata={"help": "radius of local attention"},
    )

    # local attention parameters
    local_radius: int = field(
        default=127,
        metadata={"help": "radius of local attention"},
    )
    s4_every_n_layers: int = field(
        default=1,
        metadata={"help": "use S4 every N layers"},
    )
    s4_local_combine: str = field(
        default="add",
        metadata={"help": "whether to concat/add/stack S4 and local attention"},
    )
    s4_weight: float = field(
        default=0.5,
        metadata={"help": "weight (between 0.0 and 1.0) of S4 when adding with local attention"},
    )

    # S4 parameters
    s4_state_dim: int = field(
        default=64,
        metadata={"help": "state dimension"},
    )
    s4_channels: int = field(
        default=1,
        metadata={"help": "number of channels (heads), default to 1"},
    )
    s4_dt_min: float = field(
        default=0.001,
        metadata={"help": "parameter for time steps"},
    )
    s4_dt_max: float = field(
        default=0.1,
        metadata={"help": "parameter for time steps"},
    )
    s4_lr: Optional[str] = field(
        default=None,
        metadata={"help": "learning rate for the state space parameters, except dt"},
    )
    s4_n_ssm: int = field(
        default=1,
        metadata={"help": "copies of the state space parameters A and B"},
    )
cfg = SpadeConfig()


def swish(x):
    return x * torch.sigmoid(x)


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super(Conv, self).__init__()
        self.padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=self.padding)
        self.conv = nn.utils.weight_norm(self.conv)
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, x):
        out = self.conv(x)
        return out


class ZeroConv1d(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ZeroConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channel, out_channel, kernel_size=1, padding=0)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()

    def forward(self, x):
        out = self.conv(x)
        return out


class Residual_block(nn.Module):
    def __init__(
            self, 
            res_channels, 
            skip_channels,
            diffusion_step_embed_dim_out, 
            in_channels,
#           s4_lmax,
#           s4_d_state,
#           s4_dropout,
#           s4_bidirectional,
#           s4_layernorm,
            label_embed_dim=None
        ):
        super(Residual_block, self).__init__()
        self.res_channels = res_channels

        self.fc_t = nn.Linear(diffusion_step_embed_dim_out, self.res_channels)

        self.spade_1 = SpadeDecoderLayerBase(
            cfg=cfg,
            layer_idx=0,
            no_encoder_attn=True,
            add_bias_kv=False,
            add_zero_attn=False,
            has_relative_attention_bias=False
        )

        self.conv_layer = Conv(self.res_channels, 2 * self.res_channels, kernel_size=3)

        self.spade_2 = SpadeDecoderLayerBase(
            cfg=cfg,
            layer_idx=0,
            no_encoder_attn=True,
            add_bias_kv=False,
            add_zero_attn=False,
            has_relative_attention_bias=False
        )

        self.res_conv = nn.Conv1d(res_channels, res_channels, kernel_size=1)
        self.res_conv = nn.utils.weight_norm(self.res_conv)
        nn.init.kaiming_normal_(self.res_conv.weight)

        self.skip_conv = nn.Conv1d(res_channels, skip_channels, kernel_size=1)
        self.skip_conv = nn.utils.weight_norm(self.skip_conv)
        nn.init.kaiming_normal_(self.skip_conv.weight)

        #the layer-specific fc for label embedding (conditional case)
        self.fc_label = nn.Linear(label_embed_dim, 2 * self.res_channels)  if label_embed_dim is not None else None

    def forward(self, input_data):
        x, label_embed, diffusion_step_embed = input_data
        h = x
        B, C, L = x.shape
        assert C == self.res_channels

        part_t = self.fc_t(diffusion_step_embed)
        part_t = part_t.view([B, self.res_channels, 1])
        h = h + part_t

        h = self.conv_layer(h)

        h,_,_,_ = self.spade_1(h.permute(2,0,1))
        h = h.permute(1,2,0)

        # process label embedding
        if(self.fc_label is not None):
            label_embed = self.fc_label(label_embed).unsqueeze(2) #output B, 2C, 1
            h = h + label_embed


        h,_,_,_ = self.spade_2(h.permute(2,0,1))
        h = h.permute(1,2,0)

        out = torch.tanh(h[:,:self.res_channels,:]) * torch.sigmoid(h[:,self.res_channels:,:])

        res = self.res_conv(out)
        assert x.shape == res.shape
        skip = self.skip_conv(out)

        return (x + res) * math.sqrt(0.5), skip  # normalize for training stability


class Residual_group(nn.Module):
    def __init__(
            self, 
            res_channels, 
            skip_channels, 
            num_res_layers,
            diffusion_step_embed_dim_in,
            diffusion_step_embed_dim_mid,
            diffusion_step_embed_dim_out,
            in_channels,
            label_embed_dim=None
        ):
        super(Residual_group, self).__init__()
        self.num_res_layers = num_res_layers
        self.diffusion_step_embed_dim_in = diffusion_step_embed_dim_in

        self.fc_t1 = nn.Linear(diffusion_step_embed_dim_in, diffusion_step_embed_dim_mid)
        self.fc_t2 = nn.Linear(diffusion_step_embed_dim_mid, diffusion_step_embed_dim_out)

        self.residual_blocks = nn.ModuleList()
        for n in range(self.num_res_layers):
            self.residual_blocks.append(Residual_block(
                res_channels, 
                skip_channels,
                diffusion_step_embed_dim_out=diffusion_step_embed_dim_out,
                in_channels=in_channels,
                label_embed_dim=label_embed_dim
            ))


    def forward(self, input_data):
        noise, label_embed, diffusion_steps = input_data

        diffusion_step_embed = calc_diffusion_step_embedding(diffusion_steps, self.diffusion_step_embed_dim_in)
        diffusion_step_embed = swish(self.fc_t1(diffusion_step_embed))
        diffusion_step_embed = swish(self.fc_t2(diffusion_step_embed))

        h = noise
        skip = 0
        for n in range(self.num_res_layers):
            h, skip_n = self.residual_blocks[n]((h, label_embed, diffusion_step_embed))
            skip += skip_n

        return skip * math.sqrt(1.0 / self.num_res_layers)


class DSAT_ECG(nn.Module):
    def __init__(
            self, 
            in_channels, 
            res_channels, 
            skip_channels, 
            out_channels,
            num_res_layers,
            diffusion_step_embed_dim_in,
            diffusion_step_embed_dim_mid,
            diffusion_step_embed_dim_out,
            T, 
            Alpha, 
            Beta,
            Alpha_bar, 
            Sigma,
            label_embed_classes=0,
            label_embed_dim=128
        ):
        super(DSAT_ECG, self).__init__()

        self.T = T
        self.Beta = Beta
        self.Alpha = Alpha
        self.Alpha_bar = Alpha_bar
        self.Sigma = Sigma
        self.n_channels = in_channels
        self.n_classes = label_embed_classes

        self.init_conv = nn.Sequential(Conv(in_channels, res_channels, kernel_size=1), nn.ReLU())

        # embedding for global conditioning
        self.embedding = nn.Embedding(label_embed_classes, label_embed_dim)# if label_embed_classes>0 is not None else None

        self.residual_layer = Residual_group(
            res_channels=res_channels,
            skip_channels=skip_channels,
            num_res_layers=num_res_layers,
            diffusion_step_embed_dim_in=diffusion_step_embed_dim_in,
            diffusion_step_embed_dim_mid=diffusion_step_embed_dim_mid,
            diffusion_step_embed_dim_out=diffusion_step_embed_dim_out,
            in_channels=in_channels,
            label_embed_dim=label_embed_dim if label_embed_classes > 0 else None
        )

        self.final_conv = nn.Sequential(
            Conv(skip_channels, skip_channels, kernel_size=1),
            nn.ReLU(),
            ZeroConv1d(skip_channels, out_channels)
        )

    def forward(self, noise, label, diffusion_steps):

        label_embed = label @ self.embedding.weight if self.embedding is not None else None

        x = noise
        x = self.init_conv(x)
        x = self.residual_layer((x, label_embed, diffusion_steps))
        y = self.final_conv(x)

        return y
    
    def sample(self, size, labels=None):
        """
        Perform the complete sampling step according to p(x_0|x_T) = \prod_{t=1}^T p_{\theta}(x_{t-1}|x_t)

        Parameters:
        net (torch network):            the model
        size (tuple):                   size of tensor to be generated, 
                                        usually is (number of signals to generate, channels=1, length of signal)
        diffusion_hyperparams (dict):   dictionary of diffusion hyperparameters returned by calc_diffusion_hyperparams
                                        note, the tensors need to be cuda tensors 
        cond: conditioning as integer tensor (optional)
        
        Returns:
        the generated signal(s) in torch.tensor, shape=size
        """

        assert len(self.Alpha) == self.T
        assert len(self.Alpha_bar) == self.T
        assert len(self.Sigma) == self.T
        assert len(size) == 3
        
        x = torch.normal(0, 1, size=size).cuda()
        with torch.no_grad():
            # Use tqdm for progress bar
            for t in tqdm(range(self.T-1, -1, -1), desc="Sampling", leave=True):
                diffusion_steps = (t * torch.ones((size[0], 1))).cuda()  # use the corresponding reverse step
                epsilon_theta = self(x, labels, diffusion_steps)  # predict \epsilon according to \epsilon_\theta
                    
                x = (x - (1-self.Alpha[t])/torch.sqrt(1-self.Alpha_bar[t]) * epsilon_theta) / torch.sqrt(self.Alpha[t])  # update x_{t-1} to \mu_\theta(x_t)
                if t > 0:
                    x = x + self.Sigma[t] * torch.normal(0, 1, size=size).cuda()  # add the variance term to x_{t-1}
        return x
    
    def sample_trained_model(self, samples, labels=None):
        """
        Sample from trained model

        Parameters:
        samples (int): number of samples to generate
        
        Returns:
        the generated signal(s) in torch.tensor, shape=(samples, channels=1, length of signal)
        """
        if labels is None:
            class_labels = torch.arange(self.n_classes)
            labels = torch.eye(self.n_classes)[class_labels.repeat(samples // self.n_classes + 1)][:samples].cuda().float()

        return self.sample((samples, self.n_channels, 1000), labels), labels