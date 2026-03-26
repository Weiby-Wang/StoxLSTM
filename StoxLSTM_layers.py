__all__ = ['MLP', 'xlstm_block', 'series_decomp', 'z_generation_model', 'RevIN', 'Padding_Patch_Layer']

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
    """Multi-layer perceptron with configurable hidden layers.

    Architecture: BasicBlock(input -> hidden) -> [BasicBlock(hidden -> hidden)] * hidden_layer -> Linear(hidden -> output)
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
        # Pad both ends of the time series to preserve sequence length after pooling
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """Series decomposition block: splits a time series into residual and trend components."""

    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class xlstm_block(nn.Module):
    """Wrapper around xLSTMBlockStack with configurable mLSTM/sLSTM composition.

    If slstm_at == [-1], only mLSTM blocks are used (no sLSTM).
    Otherwise, sLSTM blocks are placed at the positions given by slstm_at.
    """

    def __init__(self, context_length: int, embedding_dim: int, num_blocks: int, slstm_at: list, dropout: float = 0.2):
        super().__init__()

        if -1 in slstm_at:
            # mLSTM-only configuration
            self.cfg = xLSTMBlockStackConfig(
                mlstm_block=mLSTMBlockConfig(
                    mlstm=mLSTMLayerConfig(
                        conv1d_kernel_size=4,
                        qkv_proj_blocksize=4,
                        num_heads=4
                    )
                ),
                context_length=context_length,  # sequence length
                num_blocks=num_blocks,           # number of stacked xLSTM blocks
                embedding_dim=embedding_dim,     # model dimension
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
                context_length=context_length,  # sequence length
                num_blocks=num_blocks,           # number of stacked xLSTM blocks
                embedding_dim=embedding_dim,     # model dimension
                slstm_at=slstm_at,               # positions of sLSTM blocks
                dropout=dropout,
            )

        self.xlstm = xLSTMBlockStack(self.cfg)

    def forward(self, x):
        y, h = self.xlstm(x)
        return y, h


class Reparameterization_layer(nn.Module):
    """Reparameterization trick layer for variational inference.

    Maps input x to a sampled latent z = mean + eps * std via:
      [mean, logvar] = MLP(x)
      std = exp(0.5 * logvar)
      z = mean + eps * std,  eps ~ N(0, I)
    """

    def __init__(self, x_dim, z_dim, mlp_size: list = [128, 2], dropout=0.2):
        super().__init__()
        mlp_hidden_dim = mlp_size[0]
        mlp_hidden_layers = mlp_size[1]

        # Output dimension is z_dim*2 to produce both mean and logvar
        self.mlp_z = MLP(input_dim=x_dim, output_dim=z_dim * 2, hidden_dim=mlp_hidden_dim, hidden_layer=mlp_hidden_layers, dropout=dropout)

    def forward(self, x):
        # Split MLP output into mean and log-variance
        mean, logvar = torch.chunk(self.mlp_z(x), 2, dim=-1)
        std = torch.exp(0.5 * logvar)       # standard deviation
        eps = torch.randn_like(std)         # random noise ~ N(0, I)

        # Reparameterized sample: z = mean + eps * std
        return torch.addcmul(mean, eps, std), mean, logvar


class RevIN(nn.Module):
    """Reversible Instance Normalization (RevIN).

    Normalizes input statistics per instance during forward pass,
    and reverses the normalization on model outputs.
    """

    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        """
        Args:
            num_features: number of features (channels) to normalize
            eps: small constant for numerical stability
            affine: if True, learns per-channel affine parameters (weight, bias)
            subtract_last: if True, subtract the last time step instead of the mean
        """
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
        # Learnable per-channel affine scale and shift
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
    """Convert a sequence tensor into overlapping patches.

    Pads both sides of the time axis by patch_stride zeros, then extracts
    patches of size patch_size with the given stride.

    Args:
        x: input tensor [bs, d_seq, seq_len]
        patch_stride: stride between consecutive patches
        patch_size: size of each patch
    Returns:
        x_P: patched tensor [bs, d_seq, N, patch_size]
    """
    T = x.size(-1)
    N = ceil((T + patch_stride - patch_size) / patch_stride) + 1  # total number of patches

    x = F.pad(x, (patch_stride, patch_stride), mode='constant', value=0)
    x_P = x.unfold(dimension=-1, size=patch_size, step=patch_stride)  # [bs, d_seq, N+extra, patch_size]
    x_P = x_P[:, :, :N, :]  # trim to exactly N patches [bs, d_seq, N, patch_size]

    return x_P


class Z_generation_model(nn.Module):
    """Sequential latent variable generation via reparameterization.

    Generates z_{1:T} autoregressively: each z_t is sampled conditioned on
    the hidden state h_t and the previous latent z_{t-1}.
    """

    def __init__(self, h_dim, z_size: list, device, mlp_size: list = [128, 2], dropout=0.2):
        """
        Args:
            h_dim: dimension of the input hidden state
            z_size: [N, d_latent] — number of time steps and latent dimension
            device: torch device
            mlp_size: [hidden_dim, hidden_layers] for the reparameterization MLP
        """
        super().__init__()
        self.N = z_size[0]
        self.d_latent = z_size[1]
        self.hidden = h_dim
        self.device = device

        self.reparameterization = Reparameterization_layer(
            x_dim=self.hidden + self.d_latent,
            z_dim=self.d_latent,
            mlp_size=mlp_size,
            dropout=dropout
        )

    def forward(self, bs, hidden_state):
        """Generate latent sequence z_{1:T} autoregressively.

        Args:
            bs: batch size
            hidden_state: hidden states [bs, N, h_dim]
        Returns:
            zT:      sampled latent sequence [bs, N, d_latent]
            z_mean:  mean of each z_t [bs, N, d_latent]
            z_logvar: log-variance of each z_t [bs, N, d_latent]
        """
        zT = torch.zeros(bs, self.N, self.d_latent).to(self.device)
        z_mean = torch.zeros(bs, self.N, self.d_latent).to(self.device)
        z_logvar = torch.zeros(bs, self.N, self.d_latent).to(self.device)

        # Initialize z_0 as zeros (prior at t=0)
        z0 = torch.zeros(bs, self.d_latent).to(self.device)
        reparameterization_z_input = torch.cat((hidden_state[:, 0, :], z0), dim=-1)  # [bs, h_dim+d_latent]
        zT[:, 0, :], z_mean[:, 0, :], z_logvar[:, 0, :] = self.reparameterization(reparameterization_z_input)

        # Autoregressively generate z_2 through z_T
        for t in range(1, self.N):
            reparameterization_z_input = torch.cat((hidden_state[:, t, :], zT[:, t - 1, :]), dim=-1)
            zT[:, t, :], z_mean[:, t, :], z_logvar[:, t, :] = self.reparameterization(reparameterization_z_input)

        return zT, z_mean, z_logvar
