__all__ = ['MLP', 'xlstm_block', 'series_decomp', 'Z_generation_model', 'RevIN', 'Padding_Patch_Layer']  

import torch
import torch.nn as nn
from math import ceil
import torch.nn.functional as F

from xlstm import (
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
    FeedForwardConfig,
)


class BasicBlock(nn.Module):
    """A single MLP block: Linear -> GELU -> LayerNorm -> Dropout."""

    def __init__(self, input_dim, output_dim, dropout=0.1):
        super().__init__()
        
        self.block = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        x = self.block(x)
        return x


class MLP(nn.Module):
    """Multi-Layer Perceptron with configurable hidden layers.

    Architecture: BasicBlock(input) -> N × BasicBlock(hidden) -> Linear(output)
    """

    def __init__(self, input_dim, output_dim, hidden_layer, hidden_dim, dropout=0.1):
        super().__init__()
        
        self.mlp = nn.Sequential(
            BasicBlock(input_dim, hidden_dim, dropout),
            *[BasicBlock(hidden_dim, hidden_dim, dropout) for _ in range(hidden_layer)],
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, x):
        x = self.mlp(x)
        return x
    

class moving_avg(nn.Module):
    """Moving average block to highlight the trend of time series."""

    def __init__(self, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # Pad both ends of the time series by repeating boundary values
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """Series decomposition block that separates trend and residual components."""

    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean
    

class xlstm_block(nn.Module):
    """Wrapper around xLSTM block stack supporting both mLSTM-only and mixed mLSTM/sLSTM configurations.

    Args:
        context_length: Sequence length (number of patches or time steps).
        embedding_dim: Model embedding dimension.
        num_blocks: Number of xLSTM layers.
        slstm_at: List of positions for sLSTM blocks; [-1] means no sLSTM.
        dropout: Dropout rate.
    """

    def __init__(self, context_length: int, embedding_dim: int, num_blocks: int, slstm_at: list, dropout: float = 0.2):
        super().__init__()
        
        if -1 in slstm_at:
            # Pure mLSTM configuration (no sLSTM blocks)
            self.cfg = xLSTMBlockStackConfig(
                mlstm_block=mLSTMBlockConfig(
                    mlstm=mLSTMLayerConfig(
                        conv1d_kernel_size=4, 
                        qkv_proj_blocksize=4, 
                        num_heads=4
                    )
                ),
                context_length=context_length,
                num_blocks=num_blocks,
                embedding_dim=embedding_dim,
                dropout=dropout,
            )
        else:
            # Mixed mLSTM + sLSTM configuration
            self.cfg = xLSTMBlockStackConfig(
                mlstm_block=mLSTMBlockConfig(
                    mlstm=mLSTMLayerConfig(
                        conv1d_kernel_size=4, 
                        qkv_proj_blocksize=4, 
                        num_heads=4
                    )
                ),

                slstm_block=sLSTMBlockConfig(
                    slstm=sLSTMLayerConfig(
                        backend="cuda",
                        num_heads=8,
                        conv1d_kernel_size=4,
                        bias_init="powerlaw_blockdependent",
                    ),
                    feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu"),
                ),
                context_length=context_length,
                num_blocks=num_blocks,
                embedding_dim=embedding_dim,
                slstm_at=slstm_at,
                dropout=dropout,
            )
        
        self.xlstm = xLSTMBlockStack(self.cfg)
        
    def forward(self, x):
        y, h = self.xlstm(x)
        return y, h


class Reparameterization_layer(nn.Module):
    """Reparameterization trick layer for variational inference.

    Maps input to mean and log-variance, then samples latent variable z
    using: z = mean + eps * std, where eps ~ N(0, I).
    """

    def __init__(self, x_dim, z_dim, mlp_size: list = [128, 2], dropout=0.2):
        super().__init__()
        mlp_hidden_dim = mlp_size[0]
        mlp_hidden_layers = mlp_size[1]
        
        self.mlp_z = MLP(input_dim=x_dim, output_dim=z_dim * 2, hidden_dim=mlp_hidden_dim, hidden_layer=mlp_hidden_layers, dropout=dropout)
    
    def forward(self, x):
        # Split output into mean and log-variance
        mean, logvar = torch.chunk(self.mlp_z(x), 2, dim=-1)
        # Compute standard deviation from log-variance
        std = torch.exp(0.5 * logvar)
        # Sample epsilon from standard normal distribution
        eps = torch.randn_like(std)
        
        # Reparameterized sample: z = mean + eps * std
        return torch.addcmul(mean, eps, std), mean, logvar


class RevIN(nn.Module):
    """Reversible Instance Normalization for time series.

    Normalizes input during forward pass and denormalizes predictions,
    enabling the model to handle non-stationary time series.

    Args:
        num_features: Number of features or channels.
        eps: Small value added for numerical stability.
        affine: If True, includes learnable affine parameters.
        subtract_last: If True, subtracts last value instead of mean.
    """

    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # Initialize learnable affine parameters: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x


def Padding_Patch_Layer(x, patch_stride, patch_size):
    """Apply zero-padding and unfold the input tensor into patches.

    Args:
        x: Input tensor of shape [bs, d_seq, seq_len].
        patch_stride: Stride between consecutive patches.
        patch_size: Size of each patch.

    Returns:
        Tensor of shape [bs, d_seq, patch_num, patch_size].
    """
    T = x.size(-1)
    N = ceil((T + patch_stride - patch_size) / patch_stride) + 1  # Number of patches

    # Apply zero padding on both sides
    x = F.pad(x, (patch_stride, patch_stride), mode='constant', value=0)
    # Unfold into patches along the last dimension
    x_P = x.unfold(dimension=-1, size=patch_size, step=patch_stride)  # [bs, d_seq, N+, patch_size]
    x_P = x_P[:, :, :N, :]  # Trim to exactly N patches: [bs, d_seq, N, patch_size]
    
    return x_P


class Z_generation_model(nn.Module):
    """Autoregressive latent variable generation model.

    Generates a sequence of latent variables z1:T autoregressively,
    where each zt depends on the corresponding hidden state and the
    previous latent variable z_{t-1}.

    Args:
        h_dim: Dimension of the hidden state input.
        z_size: [N, d_latent] specifying sequence length and latent dimension.
        device: Computing device (CPU or CUDA).
        mlp_size: [hidden_dim, num_hidden_layers] for the reparameterization MLP.
        dropout: Dropout rate.
    """

    def __init__(self, h_dim, z_size: list, device, mlp_size: list = [128, 2], dropout=0.2):
        super().__init__()
        self.N = z_size[0]
        self.d_latent = z_size[1]
        self.hidden = h_dim
        self.device = device
        
        self.reparameterization = Reparameterization_layer(
            x_dim=self.hidden + self.d_latent, z_dim=self.d_latent, mlp_size=mlp_size, dropout=dropout
        )
    
    def forward(self, bs, hidden_state):
        # Initialize output tensors for latent variables and their distributions
        zT = torch.zeros(bs, self.N, self.d_latent).to(self.device)
        z_mean = torch.zeros(bs, self.N, self.d_latent).to(self.device)
        z_logvar = torch.zeros(bs, self.N, self.d_latent).to(self.device)
        
        # Initialize z0 as zeros
        z0 = torch.zeros(bs, self.d_latent).to(self.device)

        # Generate z1 from hidden state h1 and initial z0
        reparameterization_z_input = torch.cat((hidden_state[:, 0, :], z0), dim=-1)  # [bs, d_hidden + d_latent]
        zT[:, 0, :], z_mean[:, 0, :], z_logvar[:, 0, :] = self.reparameterization(reparameterization_z_input)
        
        # Autoregressively generate z2 to zT
        for t in range(1, self.N):
            reparameterization_z_input = torch.cat((hidden_state[:, t, :], zT[:, t - 1, :]), dim=-1)  # [bs, d_hidden + d_latent]
            zT[:, t, :], z_mean[:, t, :], z_logvar[:, t, :] = self.reparameterization(reparameterization_z_input)
        
        return zT, z_mean, z_logvar