import torch
import torch.nn as nn


class MAE_LB_loglikelihood(nn.Module):
    """Loss function combining MAE reconstruction loss with KL divergence
    for variational lower bound optimization using log-likelihood."""

    def __init__(self, length):
        super().__init__()
        self.length = length

    def forward(self, xT_, xT, meanq, logvarq, meanp, logvarp):
        # Slice the prediction and target to the last `length` time steps
        # xT_, xT: [bs, len, seq_dim]
        xT_, xT = xT_[:, -self.length:, :], xT[:, -self.length:, :]

        # Reconstruction loss (Mean Absolute Error)
        MAE = nn.L1Loss()(xT_, xT)

        # KL divergence between approximate posterior q and prior p
        KLD = 0.5 * torch.mean(
            logvarp - logvarq
            + (logvarq.exp() + (meanq - meanp).pow(2)) / (logvarp.exp() + 1e-10)
            - 1
        )

        return MAE + 500 * KLD


class MSE_LB_loglikelihood(nn.Module):
    """Loss function combining MSE reconstruction loss with KL divergence
    for variational lower bound optimization using log-likelihood."""

    def __init__(self, length):
        super().__init__()
        self.length = length

    def forward(self, xT_, xT, meanq, logvarq, meanp, logvarp):
        # Slice the prediction and target to the last `length` time steps
        # xT_, xT: [bs, len, seq_dim]
        xT_, xT = xT_[:, -self.length:, :], xT[:, -self.length:, :]

        # Reconstruction loss (Mean Squared Error)
        MSE = nn.MSELoss()(xT_, xT)

        # KL divergence between approximate posterior q and prior p
        KLD = 0.5 * torch.mean(
            logvarp - logvarq
            + (logvarq.exp() + (meanq - meanp).pow(2)) / (logvarp.exp() + 1e-10)
            - 1
        )

        return MSE + 500 * KLD