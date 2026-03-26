# StoxLSTM: A Stochastic Extended Long Short-Term Memory
__all__ = ['StoxLSTM_backbone']

# Required imports
import torch
import torch.nn as nn
from StoxLSTM_layers import MLP, xlstm_block, RevIN, Padding_Patch_Layer, Z_generation_model
from math import ceil


class StoxLSTM_backbone(nn.Module):
    """StoxLSTM backbone with patch-based processing and channel independence (CI).

    Applies patching to the input sequence and processes each channel independently.
    Uses a variational inference framework with an inference model (approximate posterior)
    and a generation model (prior + emission).
    """

    def __init__(self, device, d_seq: int, d_model: int, d_latent: int, look_back_length: int, prediction_length: int,
                 patch_size: int, patch_stride: int, xlstm_h_num_block: int, slstm_h_at: int, xlstm_g_num_block: int, slstm_g_at: int,
                 embed_hidden_dim: int, embed_hidden_layers_num: int,
                 mlp_z_hidden_dim: int, mlp_z_hidden_layers_num: int, mlp_proj_down_hidden_dim: int, mlp_proj_down_hidden_layers_num: int,
                 mlp_x_p_hidden_dim: int, mlp_x_p_hidden_layers_num: int, mlp_x_hidden_dim: int, mlp_x_hidden_layers_num: int,
                 revin: bool = True, subtract_last: bool = False, dropout: float = 0.2):
        super().__init__()

        # Model dimensions and sequence lengths
        self.d_seq = d_seq
        self.d_model = d_model
        self.d_hidden = 2 * d_model   # hidden size of xLSTM output (concatenated forward/backward)
        self.d_latent = d_latent
        self.H = prediction_length
        self.L = look_back_length
        self.T = look_back_length + prediction_length
        self.device = device

        # Patching configuration
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.N = ceil((self.T + patch_stride - patch_size) / patch_stride) + 1  # total number of patches
        self.padding_patch_layer = Padding_Patch_Layer

        # Reversible Instance Normalization (RevIN) for distribution shift
        self.revin = revin
        if self.revin:
            self.revin_layer = RevIN(self.d_seq, affine=True, subtract_last=subtract_last)

        # Instance normalization layers for various intermediate representations
        self.instance_norm_P = nn.InstanceNorm2d(num_features=d_seq)
        self.instance_norm_embed = nn.InstanceNorm2d(num_features=d_seq)
        self.instance_norm_hT = nn.InstanceNorm2d(num_features=d_seq)
        self.instance_norm_gT = nn.InstanceNorm2d(num_features=d_seq)
        self.instance_norm_xP = nn.InstanceNorm2d(num_features=d_seq)

        # Forward xLSTM: encodes patches into hidden states h_{1:T}
        self.xlstm_h = xlstm_block(context_length=self.N, embedding_dim=d_model, num_blocks=xlstm_h_num_block, slstm_at=slstm_h_at, dropout=dropout)
        # Backward/smoothing xLSTM: produces smoothed states g_{1:T} for the approximate posterior
        self.xlstm_g = xlstm_block(context_length=self.N, embedding_dim=3 * d_model, num_blocks=xlstm_g_num_block, slstm_at=slstm_g_at, dropout=dropout)

        # Embedding MLP: maps each patch to d_model dimensional space
        self.emb_layer = MLP(input_dim=patch_size, output_dim=d_model, hidden_dim=embed_hidden_dim, hidden_layer=embed_hidden_layers_num, dropout=dropout)
        # Latent variable generation (reparameterization for z_{1:T})
        self.z_generation = Z_generation_model(h_dim=self.d_hidden, z_size=[self.N, d_latent], device=device, mlp_size=[mlp_z_hidden_dim, mlp_z_hidden_layers_num], dropout=dropout)
        # Projection MLP: down-projects smoothed state g to d_hidden
        self.proj_down = MLP(input_dim=2 * (self.d_hidden + d_model), output_dim=self.d_hidden, hidden_dim=mlp_proj_down_hidden_dim, hidden_layer=mlp_proj_down_hidden_layers_num, dropout=dropout)
        # Patch emission MLP: decodes (h, z) to patch-level predictions
        self.mlp_x_P = MLP(input_dim=self.d_hidden + d_latent, output_dim=patch_size, hidden_dim=mlp_x_p_hidden_dim, hidden_layer=mlp_x_p_hidden_layers_num, dropout=dropout)
        # Final projection MLP: maps flattened patches to the prediction horizon
        self.mlp_x = MLP(input_dim=self.N * patch_size, output_dim=prediction_length, hidden_dim=mlp_x_hidden_dim, hidden_layer=mlp_x_hidden_layers_num, dropout=dropout)

    def inference_model(self, xT_P):
        """Inference model: approximates posterior q(z_{1:T} | x_{1:T}).

        Uses both a forward pass (h) and a backward smoothing pass (g) to
        compute the approximate posterior parameters (mean_q, logvar_q).

        Args:
            xT_P: full sequence patches [bs, d_seq, N, patch_size]
        Returns:
            meanq, logvarq: approximate posterior parameters [bs*d_seq, N, d_latent]
        """
        xT_P = xT_P.view(-1, self.N, self.patch_size)  # [bs*d_seq, N, patch_size]

        # Embed each patch into d_model dimensional space
        xT = self.emb_layer(xT_P).view(-1, self.d_seq, self.N, self.d_model)  # [bs, d_seq, N, d_model]
        xT = self.instance_norm_embed(xT)
        xT = xT.view(-1, self.N, self.d_model)  # [bs*d_seq, N, d_model]

        # Forward transition: compute hidden states h_{1:T} via xLSTM
        _, hT = self.xlstm_h(xT)  # hT: [bs*d_seq, N, d_hidden]
        hT = hT.view(-1, self.d_seq, self.N, self.d_hidden)  # [bs, d_seq, N, d_hidden]
        hT = self.instance_norm_hT(hT)
        hT = hT.view(-1, self.N, self.d_hidden)  # [bs*d_seq, N, d_hidden]

        # Shift xT by one step to align x_{t+1} with h_t for the smoothing pass
        xT0 = torch.cat((xT, xT[:, 0:1, :]), dim=1)  # [bs*d_seq, N+1, d_model]
        xT = xT0[:, 1:, :]                             # [bs*d_seq, N, d_model] (shifted)

        # Backward smoothing: run xLSTM on (h_t, x_{t+1}) in reverse to compute g_{1:T}
        h_x = torch.cat((hT, xT), dim=-1)   # [bs*d_seq, N, d_hidden+d_model]
        _, gT_1 = self.xlstm_g(h_x)         # gT_1: [bs*d_seq, N, 2*(d_hidden+d_model)] (reverse time order)
        gT = torch.flip(gT_1, dims=[-2])     # gT:   [bs*d_seq, N, 2*(d_hidden+d_model)] (forward time order)
        gT = gT.view(-1, self.d_seq, self.N, 2 * (self.d_hidden + self.d_model))   # [bs, d_seq, N, ...]
        gT = self.instance_norm_gT(gT).view(-1, self.N, 2 * (self.d_hidden + self.d_model))  # [bs*d_seq, N, ...]
        # Project down to d_hidden for latent generation
        gT_ = self.proj_down(gT).view(-1, self.d_seq, self.N, self.d_hidden)  # [bs, d_seq, N, d_hidden]
        gT_ = self.instance_norm_hT(gT_).view(-1, self.N, self.d_hidden)      # [bs*d_seq, N, d_hidden]

        _, meanq, logvarq = self.z_generation(self.bs * self.d_seq, gT_)

        return meanq, logvarq

    def generation_model(self, xC_P):
        """Generation model: predicts x_{1:T}' and computes prior p(z_{1:T} | c_{1:L}).

        Args:
            xC_P: context-only patches (future part zero-padded) [bs, d_seq, N, patch_size]
        Returns:
            xT_: predicted sequence [bs, d_seq, prediction_length]
            meanp, logvarp: prior parameters [bs*d_seq, N, d_latent]
        """
        xC_P = xC_P.view(-1, self.N, self.patch_size)  # [bs*d_seq, N, patch_size]

        # Embed each patch
        xC = self.emb_layer(xC_P).view(-1, self.d_seq, self.N, self.d_model)  # [bs, d_seq, N, d_model]
        xC = self.instance_norm_embed(xC)
        xC = xC.view(-1, self.N, self.d_model)  # [bs*d_seq, N, d_model]

        # Forward transition using context (zero-padded future)
        _, hT = self.xlstm_h(xC)  # hT: [bs*d_seq, N, d_hidden]
        hT = hT.view(-1, self.d_seq, self.N, self.d_hidden)
        hT = self.instance_norm_hT(hT)
        hT = hT.view(-1, self.N, self.d_hidden)  # [bs*d_seq, N, d_hidden]

        # Sample latent variables z_{1:T} from prior p(z|h)
        zT, meanp, logvarp = self.z_generation(self.bs * self.d_seq, hT)  # zT: [bs*d_seq, N, d_latent]

        # Emission: decode (h, z) -> patch predictions -> final prediction
        h_z = torch.cat((hT, zT), dim=-1)                                          # [bs*d_seq, N, d_hidden+d_latent]
        xT_P = self.mlp_x_P(h_z).view(-1, self.d_seq, self.N, self.patch_size)    # [bs, d_seq, N, patch_size]
        xT_P = self.instance_norm_xP(xT_P)
        xT_P_flatten = xT_P.view(-1, self.d_seq, self.N * self.patch_size)         # [bs, d_seq, N*patch_size]
        xT_ = self.mlp_x(xT_P_flatten)                                             # [bs, d_seq, prediction_length]

        return xT_, meanp, logvarp

    def forward(self, xT):
        """Forward pass.

        Args:
            xT: full input sequence [bs, d_seq, T] where T = look_back_length + prediction_length
        Returns:
            xT_: predicted sequence [bs, d_seq, prediction_length]
            meanq, logvarq: approximate posterior parameters
            meanp, logvarp: prior parameters
        """
        self.bs = xT.size(0)
        xC = xT[:, :, :self.L]  # context (look-back) portion [bs, d_seq, L]

        # Apply RevIN normalization (operates on [bs, seq_len, d_seq] format)
        if self.revin:
            xC, xT = xC.permute(0, 2, 1), xT.permute(0, 2, 1)  # [bs, seq_len, d_seq]
            xC, xT = self.revin_layer(xC, 'norm'), self.revin_layer(xT, 'norm')
            xC, xT = xC.permute(0, 2, 1), xT.permute(0, 2, 1)  # [bs, d_seq, seq_len]

        # Pad context with zeros to produce a causal (context-only) version of length T
        zeros_padding = torch.zeros(self.bs, self.d_seq, self.H).to(self.device)
        xC_padding = torch.cat((xC, zeros_padding), dim=-1)  # [bs, d_seq, T]

        # Create patches from full sequence xT and causal sequence xC_padding
        xT_P = self.padding_patch_layer(xT, self.patch_stride, self.patch_size)
        xC_P = self.padding_patch_layer(xC_padding, self.patch_stride, self.patch_size)  # [bs, d_seq, N, patch_size]
        xT_P, xC_P = self.instance_norm_P(xT_P), self.instance_norm_P(xC_P)

        # Run inference model (posterior) and generation model (prior + prediction)
        meanq, logvarq = self.inference_model(xT_P)
        xT_, meanp, logvarp = self.generation_model(xC_P)  # xT_: [bs, d_seq, prediction_length]

        # Reverse RevIN normalization on predictions
        if self.revin:
            xT_ = xT_.permute(0, 2, 1)          # [bs, prediction_length, d_seq]
            xT_ = self.revin_layer(xT_, 'denorm')
            xT_ = xT_.permute(0, 2, 1)          # [bs, d_seq, prediction_length]

        return xT_, meanq, logvarq, meanp, logvarp


class StoxLSTM_backbone_WO_PCI(nn.Module):
    """StoxLSTM backbone without patching and without channel independence.

    Processes the full sequence directly (no patching); all channels are processed
    jointly. Uses the same variational inference framework as StoxLSTM_backbone.
    """

    def __init__(self, device, d_seq: int, d_model: int, d_latent: int, look_back_length: int, prediction_length: int,
                 xlstm_h_num_block: int, slstm_h_at: int, xlstm_g_num_block: int, slstm_g_at: int,
                 embed_hidden_dim: int, embed_hidden_layers_num: int,
                 mlp_z_hidden_dim: int, mlp_z_hidden_layers_num: int, mlp_proj_down_hidden_dim: int, mlp_proj_down_hidden_layers_num: int,
                 mlp_x_hidden_dim: int, mlp_x_hidden_layers_num: int, revin: bool = True, subtract_last: bool = False, dropout: float = 0.2):
        super().__init__()
        
        # Model dimensions and sequence lengths
        self.d_seq = d_seq
        self.d_model = d_model
        self.d_hidden = 2 * d_model
        self.d_latent = d_latent
        self.T = look_back_length + prediction_length
        self.H = prediction_length
        self.L = look_back_length
        self.device = device

        # Reversible Instance Normalization (RevIN) for distribution shift
        self.revin = revin
        if self.revin:
            self.revin_layer = RevIN(self.d_seq, affine=True, subtract_last=subtract_last)

        # Forward xLSTM: encodes time-step features into hidden states h_{1:T}
        self.xlstm_h = xlstm_block(context_length=self.T, embedding_dim=d_model, num_blocks=xlstm_h_num_block, slstm_at=slstm_h_at)
        # Backward/smoothing xLSTM: produces smoothed states g_{1:T} for the approximate posterior
        self.xlstm_g = xlstm_block(context_length=self.T, embedding_dim=3 * d_model, num_blocks=xlstm_g_num_block, slstm_at=slstm_g_at)

        # Embedding MLP: maps d_seq channel features at each time step to d_model
        self.emb_layer = MLP(input_dim=d_seq, output_dim=d_model, hidden_dim=embed_hidden_dim, hidden_layer=embed_hidden_layers_num, dropout=dropout)
        # Latent variable generation (reparameterization for z_{1:T})
        self.z_generation = Z_generation_model(h_dim=self.d_hidden, z_size=[self.T, d_latent], device=device, mlp_size=[mlp_z_hidden_dim, mlp_z_hidden_layers_num], dropout=dropout)
        # Projection MLP: down-projects smoothed state g to d_hidden for posterior
        self.proj_down = MLP(input_dim=2 * (self.d_hidden + d_model), output_dim=self.d_hidden, hidden_dim=mlp_proj_down_hidden_dim, hidden_layer=mlp_proj_down_hidden_layers_num, dropout=dropout)
        # Emission MLP: decodes (h, z) at each time step to d_seq channel predictions
        self.mlp_x = MLP(input_dim=self.d_hidden + d_latent, output_dim=d_seq, hidden_dim=mlp_x_hidden_dim, hidden_layer=mlp_x_hidden_layers_num, dropout=dropout)

    def inference_model(self, xT):
        """Inference model: approximates posterior q(z_{1:T} | x_{1:T}).

        Args:
            xT: full normalized sequence [bs, T, d_seq]
        Returns:
            meanq, logvarq: approximate posterior parameters [bs, T, d_latent]
        """
        # Embed channel features at each time step
        xT = self.emb_layer(xT)  # [bs, T, d_model]

        # Forward transition: compute hidden states h_{1:T}
        _, hT = self.xlstm_h(xT)  # hT: [bs, T, d_hidden]

        # Backward smoothing: run xLSTM on (h_t, x_t) pairs then reverse for g_{1:T}
        h_x = torch.cat((hT, xT), dim=-1)   # [bs, T, d_hidden+d_model]
        _, gT_1 = self.xlstm_g(h_x)         # gT_1: [bs, T, 2*(d_hidden+d_model)] (reverse time)
        gT = torch.flip(gT_1, dims=[-2])     # gT:   [bs, T, 2*(d_hidden+d_model)] (forward time)

        # Project down to d_hidden and generate latent posterior parameters
        gT_ = self.proj_down(gT)  # [bs, T, d_hidden]
        _, meanq, logvarq = self.z_generation(self.bs, gT_)  # [bs, T, d_latent]

        return meanq, logvarq

    def generation_model(self, xC):
        """Generation model: predicts x_{1:T}' and computes prior p(z_{1:T} | c_{1:L}).

        Args:
            xC: context (look-back) normalized sequence [bs, L, d_seq]
        Returns:
            xT_: predicted full sequence [bs, T, d_seq]
            meanp, logvarp: prior parameters [bs, T, d_latent]
        """
        # Pad context with zeros to reach full length T (future part is unknown)
        zeros_padding = torch.zeros(self.bs, self.H, self.d_seq).to(self.device)
        xC = torch.cat((xC, zeros_padding), dim=1)  # [bs, T, d_seq]

        # Embed channel features at each time step
        xC = self.emb_layer(xC)  # [bs, T, d_model]

        # Forward transition using context (future part zero-padded)
        _, hT = self.xlstm_h(xC)  # hT: [bs, T, d_hidden]

        # Sample latent variables z_{1:T} from prior p(z|h)
        zT, meanp, logvarp = self.z_generation(self.bs, hT)  # zT: [bs, T, d_latent]

        # Emission: decode (h, z) at each time step to channel predictions
        h_z = torch.cat((hT, zT), dim=-1)  # [bs, T, d_hidden+d_latent]
        xT_ = self.mlp_x(h_z)              # [bs, T, d_seq]

        return xT_, meanp, logvarp

    def forward(self, xT):
        """Forward pass.

        Args:
            xT: full input sequence [bs, d_seq, T] where T = look_back_length + prediction_length
        Returns:
            xT_: predicted sequence [bs, d_seq, T]
            meanq, logvarq: approximate posterior parameters
            meanp, logvarp: prior parameters
        """
        self.bs = xT.size(0)
        # xT arrives as [bs, d_seq, T]; extract context portion
        xC = xT[:, :, :self.L]  # [bs, d_seq, L]

        # Always permute to [bs, seq_len, d_seq] for time-step wise processing
        xC = xC.permute(0, 2, 1)  # [bs, L, d_seq]
        xT = xT.permute(0, 2, 1)  # [bs, T, d_seq]

        # Apply RevIN normalization if enabled
        if self.revin:
            xC = self.revin_layer(xC, 'norm')
            xT = self.revin_layer(xT, 'norm')

        # Run inference model (posterior) and generation model (prior + prediction)
        meanq, logvarq = self.inference_model(xT)
        xT_, meanp, logvarp = self.generation_model(xC)  # xT_: [bs, T, d_seq]

        # Reverse RevIN normalization on predictions and convert back to [bs, d_seq, T]
        if self.revin:
            xT_ = self.revin_layer(xT_, 'denorm')
        xT_ = xT_.permute(0, 2, 1)  # [bs, d_seq, T]

        return xT_, meanq, logvarq, meanp, logvarp
