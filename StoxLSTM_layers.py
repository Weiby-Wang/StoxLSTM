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
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean
    

class xlstm_block(nn.Module):
    def __init__(self, context_length:int, embedding_dim:int, num_blocks:int, slstm_at:list, dropout:float=0.2):
        super().__init__()
        
        if -1 in slstm_at:
            self.cfg = xLSTMBlockStackConfig(
                mlstm_block=mLSTMBlockConfig(
                    mlstm=mLSTMLayerConfig(
                        conv1d_kernel_size=4, 
                        qkv_proj_blocksize=4, 
                        num_heads=4
                    )
                ),
                context_length=context_length, #序列长度
                num_blocks=num_blocks, #层数
                embedding_dim=embedding_dim, #模型维度
                dropout=dropout,
            )
        else:
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
                context_length=context_length, #序列长度
                num_blocks=num_blocks, #层数
                embedding_dim=embedding_dim, #模型维度
                slstm_at=slstm_at,
                dropout=dropout,
            )
        
        self.xlstm = xLSTMBlockStack(self.cfg)
        
    def forward(self, x):
        
        y, h = self.xlstm(x)
        
        return y, h


class Reparameterization_layer(nn.Module):
    def __init__(self, x_dim, z_dim, mlp_size:list=[128, 2], dropout=0.2):
        super().__init__()
        mlp_hidden_dim = mlp_size[0]
        mlp_hidden_layers = mlp_size[1]
        
        self.mlp_z = MLP(input_dim=x_dim, output_dim=z_dim*2, hidden_dim=mlp_hidden_dim, hidden_layer=mlp_hidden_layers, dropout=dropout)
    
    def forward(self, x):
        mean, logvar = torch.chunk(self.mlp_z(x), 2, dim=-1) #得到均值mean和方差的对数log(sigma^2)
        std = torch.exp(0.5 * logvar) #得到标准差
        eps = torch.randn_like(std)
        
        return torch.addcmul(mean, eps, std), mean, logvar


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        if self.subtract_last:
            self.last = x[:,-1,:].unsqueeze(1)
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
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x
    
'''
def Padding_Patch_Layer(x, patch_stride, patch_length):
    #[bs, d_seq, seq_len] => 
    #[bs, d_seq, patch_num, patch_length]
    T = x.size(-1)
    N = ceil((T - patch_length) / patch_stride) + 1 #patch num
    
    replication_padding = nn.ReplicationPad1d((0, patch_stride))
    x_padding = replication_padding(x)
    x_P = x_padding.unfold(dimension=-1, size=patch_length, step=patch_stride) #[bs, d_seq, N+, patch_len]
    x_P = x_P[:, :, :N, :] ##[bs, d_seq, N, patch_len]
    
    return x_P
'''
def Padding_Patch_Layer(x, patch_stride, patch_size):
    #[bs, d_seq, seq_len] => 
    #[bs, d_seq, patch_num, patch_size]
    T = x.size(-1)
    N = ceil((T + patch_stride - patch_size) / patch_stride) + 1 #patch num
    
    #zero_padding = nn.ZeroPad1d((patch_stride, patch_stride))
    #x = zero_padding(x)
    x = F.pad(x, (patch_stride, patch_stride), mode='constant', value=0)
    x_P = x.unfold(dimension=-1, size=patch_size, step=patch_stride) #[bs, d_seq, N+, patch_len]
    x_P = x_P[:, :, :N, :] ##[bs, d_seq, N, patch_len]
    
    return x_P


class Z_generation_model(nn.Module):
    def __init__(self, h_dim, z_size:list, device, mlp_size:list=[128, 2], dropout=0.2):
        super().__init__()
        self.N = z_size[0]
        self.d_latent = z_size[1]
        self.hidden = h_dim
        self.device = device
        
        self.reparameterization = Reparameterization_layer(x_dim=self.hidden+self.d_latent, z_dim=self.d_latent, mlp_size=mlp_size, dropout=dropout)
    
    def forward(self, bs, hidden_state):
        zT = torch.zeros(bs, self.N, self.d_latent).to(self.device) #初始化
        z_mean = torch.zeros(bs, self.N, self.d_latent).to(self.device)
        z_logvar = torch.zeros(bs, self.N, self.d_latent).to(self.device)
        
        z0 = torch.zeros(bs, self.d_latent).to(self.device)
        #z0 = torch.randn(bs, self.d_latent).to(self.device)
        reparameterization_z_input = torch.cat((hidden_state[:, 0, :], z0), dim=-1) #[bs, d_hidden+d_latent]
        zT[:, 0, :], z_mean[:, 0, :], z_logvar[:, 0, :] = self.reparameterization(reparameterization_z_input) #得到z1
        
        for t in range(1, self.N): #得到z2到zT
            reparameterization_z_input = torch.cat((hidden_state[:, t, :], zT[:, t-1, :]), dim=-1) #[bs, d_hidden+d_latent]
            zT[:, t, :], z_mean[:, t, :], z_logvar[:, t, :] = self.reparameterization(reparameterization_z_input) #得到zt
        
        return zT, z_mean, z_logvar