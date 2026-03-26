import torch
import torch.nn as nn


class MAE_LB_loglikelihood(nn.Module):
    """Evidence Lower Bound (ELBO) loss using MAE reconstruction term.

    Loss = MAE(prediction, target) + KL_weight * KL(q(z|x) || p(z|c))
    """

    def __init__(self, length):
        super().__init__()
        self.length = length

    def forward(self, xT_, xT, meanq, logvarq, meanp, logvarp):
        # Slice the last `length` timesteps for the prediction window
        xT_, xT = xT_[:, -self.length:, :], xT[:, -self.length:, :]

        MAE = nn.L1Loss()(xT_, xT)

        # KL divergence between approximate posterior q and prior p (both Gaussians)
        KLD = 0.5 * torch.mean(logvarp - logvarq + (logvarq.exp() + (meanq - meanp).pow(2)) / (logvarp.exp() + 1e-10) - 1)

        return MAE + 500 * KLD


class MSE_LB_loglikelihood(nn.Module):
    """Evidence Lower Bound (ELBO) loss using MSE reconstruction term.

    Loss = MSE(prediction, target) + KL_weight * KL(q(z|x) || p(z|c))
    """

    def __init__(self, length):
        super().__init__()
        self.length = length

    def forward(self, xT_, xT, meanq, logvarq, meanp, logvarp):
        # xT_, xT: [bs, len, seq_dim]
        # Slice the last `length` timesteps for the prediction window
        xT_, xT = xT_[:, -self.length:, :], xT[:, -self.length:, :]

        MSE = nn.MSELoss()(xT_, xT)

        # KL divergence between approximate posterior q and prior p (both Gaussians)
        KLD = 0.5 * torch.mean(logvarp - logvarq + (logvarq.exp() + (meanq - meanp).pow(2)) / (logvarp.exp() + 1e-10) - 1)

        return MSE + 500 * KLD
