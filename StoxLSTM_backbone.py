# StoxLSTM: A Stochastic Extended Long Short-Term Memory
__all__ = ['StoxLSTM_backbone', 'StoxLSTM_backbone_WO_PCI']

import torch
import torch.nn as nn
from StoxLSTM_layers import MLP, xlstm_block, RevIN, Padding_Patch_Layer, Z_generation_model
from math import ceil


class StoxLSTM_backbone(nn.Module):
    """StoxLSTM backbone with patching and channel independence (PCI).

    This backbone applies patching to the input time series and processes
    each channel independently through the xLSTM-based variational inference
    and generation models.

    Args:
        device: Computing device (CPU or CUDA).
        d_seq: Number of input sequence dimensions (channels).
        d_model: Model embedding dimension.
        d_latent: Latent variable dimension.
        look_back_length: Length of the look-back (history) window.
        prediction_length: Length of the prediction horizon.
        patch_size: Size of each patch.
        patch_stride: Stride between consecutive patches.
        xlstm_h_num_block: Number of xLSTM blocks for the transition model h.
        slstm_h_at: List of positions for sLSTM blocks in h; [-1] for no sLSTM.
        xlstm_g_num_block: Number of xLSTM blocks for the backward filter g.
        slstm_g_at: List of positions for sLSTM blocks in g; [-1] for no sLSTM.
        embed_hidden_dim: Hidden dimension for the embedding MLP.
        embed_hidden_layers_num: Number of hidden layers in the embedding MLP.
        mlp_z_hidden_dim: Hidden dimension for the latent variable MLP.
        mlp_z_hidden_layers_num: Number of hidden layers in the latent variable MLP.
        mlp_proj_down_hidden_dim: Hidden dimension for the projection-down MLP.
        mlp_proj_down_hidden_layers_num: Number of hidden layers in the projection-down MLP.
        mlp_x_p_hidden_dim: Hidden dimension for the patch output MLP.
        mlp_x_p_hidden_layers_num: Number of hidden layers in the patch output MLP.
        mlp_x_hidden_dim: Hidden dimension for the final output MLP.
        mlp_x_hidden_layers_num: Number of hidden layers in the final output MLP.
        revin: Whether to use Reversible Instance Normalization.
        subtract_last: If True, RevIN subtracts last value instead of mean.
        dropout: Dropout rate.
    """

    def __init__(self, device, d_seq: int, d_model: int, d_latent: int, look_back_length: int, prediction_length: int, 
                patch_size: int, patch_stride: int, xlstm_h_num_block: int, slstm_h_at: int, xlstm_g_num_block: int, slstm_g_at: int, 
                embed_hidden_dim: int, embed_hidden_layers_num: int,
                mlp_z_hidden_dim: int, mlp_z_hidden_layers_num: int, mlp_proj_down_hidden_dim: int, mlp_proj_down_hidden_layers_num: int,
                mlp_x_p_hidden_dim: int, mlp_x_p_hidden_layers_num: int, mlp_x_hidden_dim: int, mlp_x_hidden_layers_num: int,
                revin: bool = True, subtract_last: bool = False, dropout: float = 0.2):
        super().__init__()

        # Model size and prediction task parameters
        self.d_seq = d_seq
        self.d_model = d_model
        self.d_hidden = 2 * d_model
        self.d_latent = d_latent
        self.H = prediction_length
        self.L = look_back_length
        self.T = look_back_length + prediction_length
        self.device = device  
        
        # Patching parameters
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.N = ceil((self.T + patch_stride - patch_size) / patch_stride) + 1  # Number of patches
        self.padding_patch_layer = Padding_Patch_Layer
        
        # Reversible Instance Normalization
        self.revin = revin
        if self.revin: 
            self.revin_layer = RevIN(self.d_seq, affine=True, subtract_last=subtract_last)
            
        # Instance Normalization layers for different stages
        self.instance_norm_P = nn.InstanceNorm2d(num_features=d_seq)
        self.instance_norm_embed = nn.InstanceNorm2d(num_features=d_seq)
        self.instance_norm_hT = nn.InstanceNorm2d(num_features=d_seq)
        self.instance_norm_gT = nn.InstanceNorm2d(num_features=d_seq)
        self.instance_norm_xP = nn.InstanceNorm2d(num_features=d_seq)
        
        # xLSTM blocks for transition (h) and backward filtering (g)
        self.xlstm_h = xlstm_block(context_length=self.N, embedding_dim=d_model, num_blocks=xlstm_h_num_block, slstm_at=slstm_h_at, dropout=dropout)
        self.xlstm_g = xlstm_block(context_length=self.N, embedding_dim=3 * d_model, num_blocks=xlstm_g_num_block, slstm_at=slstm_g_at, dropout=dropout)
        
        # Variational state space model components
        self.emb_layer = MLP(input_dim=patch_size, output_dim=d_model, hidden_dim=embed_hidden_dim, hidden_layer=embed_hidden_layers_num, dropout=dropout)
        self.z_generation = Z_generation_model(h_dim=self.d_hidden, z_size=[self.N, d_latent], device=device, mlp_size=[mlp_z_hidden_dim, mlp_z_hidden_layers_num], dropout=dropout)
        self.proj_down = MLP(input_dim=2 * (self.d_hidden + d_model), output_dim=self.d_hidden, hidden_dim=mlp_proj_down_hidden_dim, hidden_layer=mlp_proj_down_hidden_layers_num, dropout=dropout)
        self.mlp_x_P = MLP(input_dim=self.d_hidden + d_latent, output_dim=patch_size, hidden_dim=mlp_x_p_hidden_dim, hidden_layer=mlp_x_p_hidden_layers_num, dropout=dropout)
        self.mlp_x = MLP(input_dim=self.N * patch_size, output_dim=prediction_length, hidden_dim=mlp_x_hidden_dim, hidden_layer=mlp_x_hidden_layers_num, dropout=dropout)  
    
    def inference_model(self, xT_P):
        """Inference model to compute the approximate posterior q(z1:T | x1:T).

        Args:
            xT_P: Full sequence patches [bs, d_seq, N, patch_size].

        Returns:
            meanq: Posterior mean [bs*d_seq, N, d_latent].
            logvarq: Posterior log-variance [bs*d_seq, N, d_latent].
        """
        xT_P = xT_P.view(-1, self.N, self.patch_size)  # [bs*d_seq, N, patch_size]
        
        # Embedding layer: project patches to model dimension
        xT = self.emb_layer(xT_P).view(-1, self.d_seq, self.N, self.d_model)  # [bs, d_seq, N, d_model]
        xT = self.instance_norm_embed(xT)
        xT = xT.view(-1, self.N, self.d_model)  # [bs*d_seq, N, d_model]
        
        # Forward transition process: compute hidden states h1:T
        _, hT = self.xlstm_h(xT)  # [bs*d_seq, N, d_hidden]
        hT = hT.view(-1, self.d_seq, self.N, self.d_hidden)  # [bs, d_seq, N, d_hidden]
        hT = self.instance_norm_hT(hT)
        hT = hT.view(-1, self.N, self.d_hidden)  # [bs*d_seq, N, d_hidden]
        
        # Shift xT by one position (circular shift for alignment)
        xT0 = torch.cat((xT, xT[:, 0:1, :]), dim=1)  # [bs*d_seq, N+1, d_model]
        xT = xT0[:, 1:, :]

        # Backward filtering: combine hidden states with shifted embeddings
        h_x = torch.cat((hT, xT), dim=-1)  # [bs*d_seq, N, d_hidden+d_model]
        _, gT_1 = self.xlstm_g(h_x)  # [bs*d_seq, N (reversed), 2*(d_hidden+d_model)]
        gT = torch.flip(gT_1, dims=[-2])  # Reverse to get forward order [bs*d_seq, N, 2*(d_hidden+d_model)]
        gT = gT.view(-1, self.d_seq, self.N, 2 * (self.d_hidden + self.d_model))
        gT = self.instance_norm_gT(gT).view(-1, self.N, 2 * (self.d_hidden + self.d_model))
        # Project down to hidden dimension
        gT_ = self.proj_down(gT).view(-1, self.d_seq, self.N, self.d_hidden)
        gT_ = self.instance_norm_hT(gT_).view(-1, self.N, self.d_hidden)
        # Generate approximate posterior distribution parameters
        _, meanq, logvarq = self.z_generation(self.bs * self.d_seq, gT_)
        
        return meanq, logvarq
    
    def generation_model(self, xC_P):
        """Generation model to predict x1:T and compute the prior p(z1:T).

        Args:
            xC_P: Context (zero-padded) patches [bs, d_seq, N, patch_size].

        Returns:
            xT_: Predicted output [bs, d_seq, prediction_length].
            meanp: Prior mean [bs*d_seq, N, d_latent].
            logvarp: Prior log-variance [bs*d_seq, N, d_latent].
        """
        xC_P = xC_P.view(-1, self.N, self.patch_size)  # [bs*d_seq, N, patch_size]
        
        # Embedding layer: project patches to model dimension
        xC = self.emb_layer(xC_P).view(-1, self.d_seq, self.N, self.d_model)  # [bs, d_seq, N, d_model]
        xC = self.instance_norm_embed(xC)
        xC = xC.view(-1, self.N, self.d_model)  # [bs*d_seq, N, d_model]
        
        # Forward transition process: compute hidden states
        _, hT = self.xlstm_h(xC)  # [bs*d_seq, N, d_hidden]
        hT = hT.view(-1, self.d_seq, self.N, self.d_hidden)
        hT = self.instance_norm_hT(hT)
        hT = hT.view(-1, self.N, self.d_hidden)
        # Generate prior distribution and sample latent variables
        zT, meanp, logvarp = self.z_generation(self.bs * self.d_seq, hT)  # [bs*d_seq, N, d_latent]
        
        # Emission process: reconstruct patches from hidden states and latent variables
        h_z = torch.cat((hT, zT), dim=-1)  # [bs*d_seq, N, d_hidden+d_latent]
        xT_P = self.mlp_x_P(h_z).view(-1, self.d_seq, self.N, self.patch_size)  # [bs, d_seq, N, patch_size]
        xT_P = self.instance_norm_xP(xT_P)
        xT_P_flatten = xT_P.view(-1, self.d_seq, self.N * self.patch_size)  # [bs, d_seq, N*patch_size]
        # Map flattened patches to prediction length
        xT_ = self.mlp_x(xT_P_flatten)  # [bs, d_seq, prediction_length]
        
        return xT_, meanp, logvarp
    
    def forward(self, xT):
        """Forward pass through the backbone.

        Args:
            xT: Input tensor [bs, d_seq, seq_len].

        Returns:
            xT_: Predictions [bs, d_seq, prediction_length].
            meanq: Posterior mean.
            logvarq: Posterior log-variance.
            meanp: Prior mean.
            logvarp: Prior log-variance.
        """
        self.bs = xT.size(0)
        # Split input into context (look-back) portion
        xC = xT[:, :, :self.L]  # [bs, d_seq, L]
        
        # Apply Reversible Instance Normalization
        if self.revin:
            xC, xT = xC.permute(0, 2, 1), xT.permute(0, 2, 1)  # [bs, seq_len, d_seq]
            xC, xT = self.revin_layer(xC, 'norm'), self.revin_layer(xT, 'norm')
            xC, xT = xC.permute(0, 2, 1), xT.permute(0, 2, 1)  # [bs, d_seq, seq_len]
        
        # Zero-pad context to match full sequence length T
        zeros_padding = torch.zeros(self.bs, self.d_seq, self.H).to(self.device)
        xC_padding = torch.cat((xC, zeros_padding), dim=-1)  # [bs, d_seq, T]
        
        # Create patches from full sequence and zero-padded context
        xT_P = self.padding_patch_layer(xT, self.patch_stride, self.patch_size) 
        xC_P = self.padding_patch_layer(xC_padding, self.patch_stride, self.patch_size)  # [bs, d_seq, N, patch_size]
        xT_P, xC_P = self.instance_norm_P(xT_P), self.instance_norm_P(xC_P)
        
        # Variational inference: compute posterior and prior, generate predictions
        meanq, logvarq = self.inference_model(xT_P)
        xT_, meanp, logvarp = self.generation_model(xC_P)  # [bs, d_seq, prediction_length]
        
        # Reverse the normalization on predictions
        if self.revin:
            xT_ = xT_.permute(0, 2, 1)  # [bs, prediction_length, d_seq]
            xT_ = self.revin_layer(xT_, 'denorm')
            xT_ = xT_.permute(0, 2, 1)  # [bs, d_seq, prediction_length]

        return xT_, meanq, logvarq, meanp, logvarp


class StoxLSTM_backbone_WO_PCI(nn.Module):
    """StoxLSTM backbone without patching and channel independence (WO_PCI).

    This variant processes the raw time series directly without patching,
    treating all channels jointly rather than independently.

    Args:
        device: Computing device (CPU or CUDA).
        d_seq: Number of input sequence dimensions (channels).
        d_model: Model embedding dimension.
        d_latent: Latent variable dimension.
        look_back_length: Length of the look-back (history) window.
        prediction_length: Length of the prediction horizon.
        xlstm_h_num_block: Number of xLSTM blocks for the transition model h.
        slstm_h_at: List of positions for sLSTM blocks in h; [-1] for no sLSTM.
        xlstm_g_num_block: Number of xLSTM blocks for the backward filter g.
        slstm_g_at: List of positions for sLSTM blocks in g; [-1] for no sLSTM.
        embed_hidden_dim: Hidden dimension for the embedding MLP.
        embed_hidden_layers_num: Number of hidden layers in the embedding MLP.
        mlp_z_hidden_dim: Hidden dimension for the latent variable MLP.
        mlp_z_hidden_layers_num: Number of hidden layers in the latent variable MLP.
        mlp_proj_down_hidden_dim: Hidden dimension for the projection-down MLP.
        mlp_proj_down_hidden_layers_num: Number of hidden layers in the projection-down MLP.
        mlp_x_hidden_dim: Hidden dimension for the output MLP.
        mlp_x_hidden_layers_num: Number of hidden layers in the output MLP.
        revin: Whether to use Reversible Instance Normalization.
        subtract_last: If True, RevIN subtracts last value instead of mean.
        dropout: Dropout rate.
    """

    def __init__(self, device, d_seq: int, d_model: int, d_latent: int, look_back_length: int, prediction_length: int, 
                 xlstm_h_num_block: int, slstm_h_at: int, xlstm_g_num_block: int, slstm_g_at: int, 
                 embed_hidden_dim: int, embed_hidden_layers_num: int,
                 mlp_z_hidden_dim: int, mlp_z_hidden_layers_num: int, mlp_proj_down_hidden_dim: int, mlp_proj_down_hidden_layers_num: int,
                 mlp_x_hidden_dim: int, mlp_x_hidden_layers_num: int, revin: bool = True, subtract_last: bool = False, dropout: float = 0.2):
        super().__init__()
        
        self.d_seq = d_seq
        self.d_model = d_model
        self.d_hidden = 2 * d_model
        self.d_latent = d_latent
        self.T = look_back_length + prediction_length
        self.H = prediction_length
        self.L = look_back_length
        self.device = device 
        
        # Reversible Instance Normalization
        self.revin = revin
        if self.revin: 
            self.revin_layer = RevIN(self.d_seq, affine=True, subtract_last=subtract_last)
        
        # xLSTM blocks for transition (h) and backward filtering (g)
        self.xlstm_h = xlstm_block(context_length=self.T, embedding_dim=d_model, num_blocks=xlstm_h_num_block, slstm_at=slstm_h_at)
        self.xlstm_g = xlstm_block(context_length=self.T, embedding_dim=3 * d_model, num_blocks=xlstm_g_num_block, slstm_at=slstm_g_at)
        
        # Variational state space model components
        self.emb_layer = MLP(input_dim=d_seq, output_dim=d_model, hidden_dim=embed_hidden_dim, hidden_layer=embed_hidden_layers_num, dropout=dropout)
        self.z_generation = Z_generation_model(h_dim=self.d_hidden, z_size=[self.T, d_latent], device=device, mlp_size=[mlp_z_hidden_dim, mlp_z_hidden_layers_num], dropout=dropout)
        self.proj_down = MLP(input_dim=2 * (self.d_hidden + d_model), output_dim=self.d_hidden, hidden_dim=mlp_proj_down_hidden_dim, hidden_layer=mlp_proj_down_hidden_layers_num, dropout=dropout)
        self.mlp_x = MLP(input_dim=self.d_hidden + d_latent, output_dim=d_seq, hidden_dim=mlp_x_hidden_dim, hidden_layer=mlp_x_hidden_layers_num, dropout=dropout)
    
    def inference_model(self, xT):
        """Inference model to compute the approximate posterior q(z1:T | x1:T).

        Args:
            xT: Full input sequence [bs, seq_len, d_seq].

        Returns:
            meanq: Posterior mean [bs, seq_len, d_latent].
            logvarq: Posterior log-variance [bs, seq_len, d_latent].
        """
        # Embedding layer: project input features to model dimension
        xT = self.emb_layer(xT)  # [bs, seq_len, d_model]
        
        # Forward transition process: compute hidden states h1:T
        _, hT = self.xlstm_h(xT)  # [bs, seq_len, d_hidden]
        
        # Backward filtering: combine hidden states with embeddings
        h_x = torch.cat((hT, xT), dim=-1)  # [bs, seq_len, d_hidden+d_model]
        _, gT_1 = self.xlstm_g(h_x)  # [bs, seq_len (reversed), 2*(d_hidden+d_model)]
        gT = torch.flip(gT_1, dims=[-2])  # Reverse to get forward order
        
        # Project down and generate approximate posterior parameters
        gT_ = self.proj_down(gT)  # [bs, seq_len, d_hidden]
        _, meanq, logvarq = self.z_generation(self.bs, gT_)  # [bs, seq_len, d_latent]
        
        return meanq, logvarq
    
    def generation_model(self, xC):
        """Generation model to predict x1:T and compute the prior p(z1:T).

        Args:
            xC: Context (look-back) sequence [bs, seq_len=L, d_seq].

        Returns:
            xT_: Predicted output [bs, seq_len=T, d_seq].
            meanp: Prior mean [bs, T, d_latent].
            logvarp: Prior log-variance [bs, T, d_latent].
        """
        # Zero-pad context to match full sequence length T
        zeros_padding = torch.zeros(self.bs, self.H, self.d_seq).to(self.device)
        xC = torch.cat((xC, zeros_padding), dim=1)  # [bs, T, d_seq]
        
        # Embedding layer: project input features to model dimension
        xC = self.emb_layer(xC)  # [bs, T, d_model]
        
        # Forward transition process: compute hidden states
        _, hT = self.xlstm_h(xC)  # [bs, T, d_hidden]
        # Generate prior distribution and sample latent variables
        zT, meanp, logvarp = self.z_generation(self.bs, hT)  # [bs, T, d_latent]
        
        # Emission process: reconstruct from hidden states and latent variables
        h_z = torch.cat((hT, zT), dim=-1)  # [bs, T, d_hidden+d_latent]
        xT_ = self.mlp_x(h_z)  # [bs, T, d_seq]
        
        return xT_, meanp, logvarp
    
    def forward(self, xT):
        """Forward pass through the backbone.

        Args:
            xT: Input tensor [bs, d_seq, seq_len].

        Returns:
            xT_: Predictions [bs, d_seq, prediction_length].
            meanq: Posterior mean.
            logvarq: Posterior log-variance.
            meanp: Prior mean.
            logvarp: Prior log-variance.
        """
        self.bs = xT.size(0)
        # Split input into context (look-back) portion
        xC = xT[:, :, :self.L]  # [bs, d_seq, L]
        
        # Apply Reversible Instance Normalization
        if self.revin:
            xC, xT = xC.permute(0, 2, 1), xT.permute(0, 2, 1)  # [bs, seq_len, d_seq]
            xC, xT = self.revin_layer(xC, 'norm'), self.revin_layer(xT, 'norm')
        
        # Variational inference: compute posterior and prior, generate predictions
        meanq, logvarq = self.inference_model(xT)
        xT_, meanp, logvarp = self.generation_model(xC)  # [bs, T, d_seq]
        
        # Reverse the normalization on predictions
        if self.revin:
            xT_ = self.revin_layer(xT_, 'denorm')
            xT_ = xT_.permute(0, 2, 1)  # [bs, d_seq, prediction_length]

        return xT_, meanq, logvarq, meanp, logvarp