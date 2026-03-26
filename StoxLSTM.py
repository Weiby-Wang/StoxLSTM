__all__ = ['StoxLSTM']

import torch
import torch.nn as nn
from StoxLSTM_backbone import StoxLSTM_backbone, StoxLSTM_backbone_WO_PCI
from StoxLSTM_layers import series_decomp


class Model(nn.Module):
    """StoxLSTM top-level model.

    Supports two operating modes:
    - With decomposition: the input is split into trend and residual components,
      each processed by a separate backbone, then summed.
    - Without decomposition: the full sequence is passed directly to a single backbone.

    Each backbone can optionally use patching + channel independence (patch_and_CI=True)
    or process the full sequence jointly (patch_and_CI=False).
    """

    def __init__(self, configs):
        super().__init__()

        self.decomposition = configs.decomposition
        patch_and_CI = configs.patch_and_CI
        look_back_length, prediction_length = configs.look_back_length, configs.prediction_length
        revin, subtract_last = configs.revin, configs.subtract_last
        kernel_size = configs.kernel_size
        d_seq, d_model, d_latent = configs.d_seq, configs.d_model, configs.d_latent
        patch_size, patch_stride = configs.patch_size, configs.patch_stride
        dropout = configs.fc_dropout
        device = configs.device

        # Helper to build common backbone kwargs for both trend and residual models
        def _backbone_kwargs_with_patch():
            return dict(
                d_seq=d_seq, d_model=d_model, d_latent=d_latent,
                look_back_length=look_back_length, prediction_length=prediction_length,
                patch_size=patch_size, patch_stride=patch_stride,
                xlstm_h_num_block=configs.xlstm_h_num_block, slstm_h_at=configs.slstm_h_at,
                xlstm_g_num_block=configs.xlstm_g_num_block, slstm_g_at=configs.slstm_g_at,
                embed_hidden_dim=configs.embed_hidden_dim, embed_hidden_layers_num=configs.embed_hidden_layers_num,
                mlp_z_hidden_dim=configs.mlp_z_hidden_dim, mlp_z_hidden_layers_num=configs.mlp_z_hidden_layers_num,
                mlp_proj_down_hidden_dim=configs.mlp_proj_down_hidden_dim, mlp_proj_down_hidden_layers_num=configs.mlp_proj_down_hidden_layers_num,
                mlp_x_p_hidden_dim=configs.mlp_x_p_hidden_dim, mlp_x_p_hidden_layers_num=configs.mlp_x_p_hidden_layers_num,
                mlp_x_hidden_dim=configs.mlp_x_hidden_dim, mlp_x_hidden_layers_num=configs.mlp_x_hidden_layers_num,
                device=device, subtract_last=subtract_last, revin=revin, dropout=dropout,
            )

        def _backbone_kwargs_without_patch():
            return dict(
                d_seq=d_seq, d_model=d_model, d_latent=d_latent,
                look_back_length=look_back_length, prediction_length=prediction_length,
                xlstm_h_num_block=configs.xlstm_h_num_block, slstm_h_at=configs.slstm_h_at,
                xlstm_g_num_block=configs.xlstm_g_num_block, slstm_g_at=configs.slstm_g_at,
                embed_hidden_dim=configs.embed_hidden_dim, embed_hidden_layers_num=configs.embed_hidden_layers_num,
                mlp_z_hidden_dim=configs.mlp_z_hidden_dim, mlp_z_hidden_layers_num=configs.mlp_z_hidden_layers_num,
                mlp_proj_down_hidden_dim=configs.mlp_proj_down_hidden_dim, mlp_proj_down_hidden_layers_num=configs.mlp_proj_down_hidden_layers_num,
                mlp_x_hidden_dim=configs.mlp_x_hidden_dim, mlp_x_hidden_layers_num=configs.mlp_x_hidden_layers_num,
                device=device, subtract_last=subtract_last, revin=revin, dropout=dropout,
            )

        if self.decomposition:
            # Series decomposition: separate trend and residual components
            self.decomp_module = series_decomp(kernel_size=kernel_size)
            if patch_and_CI:
                self.model_trend = StoxLSTM_backbone(**_backbone_kwargs_with_patch())
                self.model_res = StoxLSTM_backbone(**_backbone_kwargs_with_patch())
            else:
                print('StoxLSTM without patching and channel independence.')
                self.model_trend = StoxLSTM_backbone_WO_PCI(**_backbone_kwargs_without_patch())
                self.model_res = StoxLSTM_backbone_WO_PCI(**_backbone_kwargs_without_patch())
        else:
            if patch_and_CI:
                self.model = StoxLSTM_backbone(**_backbone_kwargs_with_patch())
            else:
                print('StoxLSTM without patching and channel independence.')
                self.model = StoxLSTM_backbone_WO_PCI(**_backbone_kwargs_without_patch())

    def forward(self, x):
        """Forward pass.

        Args:
            x: input sequence [bs, seq_len, seq_dim]
        Returns:
            x: predicted sequence [bs, prediction_length, seq_dim]
            meanq, logvarq: approximate posterior parameters
            meanp, logvarp: prior parameters
        """
        if self.decomposition:
            # Decompose into residual and trend components
            res_init, trend_init = self.decomp_module(x)

            # Both backbones expect [bs, seq_dim, seq_len]
            res_init, trend_init = res_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)

            res, res_meanq, res_logvarq, res_meanp, res_logvarp = self.model_res(res_init)
            trend, trend_meanq, trend_logvarq, trend_meanp, trend_logvarp = self.model_trend(trend_init)

            # Sum the two predictions
            x = res + trend

            # Concatenate KL terms along the latent dimension
            meanq = torch.cat((res_meanq, trend_meanq), dim=-1)
            logvarq = torch.cat((res_logvarq, trend_logvarq), dim=-1)
            meanp = torch.cat((res_meanp, trend_meanp), dim=-1)   # [bs, N, d_latent*2]
            logvarp = torch.cat((res_logvarp, trend_logvarp), dim=-1)

            x = x.permute(0, 2, 1)  # [bs, seq_len, seq_dim]
        else:
            x = x.permute(0, 2, 1)  # [bs, seq_dim, seq_len]

            x, meanq, logvarq, meanp, logvarp = self.model(x)

            x = x.permute(0, 2, 1)  # [bs, seq_len, seq_dim]

        return x, meanq, logvarq, meanp, logvarp
