import torch
import torch.nn as nn

class MAE_LB_loglikelihood(nn.Module):
    def __init__(self, length):
        super().__init__()
        self.length = length

    def forward(self, xT_, xT, meanq, logvarq, meanp, logvarp):
        
        xT_, xT = xT_[:, -self.length:, :], xT[:, -self.length:]
    
        MAE = nn.L1Loss()(xT_, xT) 
        
        KLD = 0.5 * torch.mean(logvarp - logvarq + (logvarq.exp() + (meanq - meanp).pow(2)) / (logvarp.exp()+1e-10) - 1)
        
        return MAE + 500*KLD


class MSE_LB_loglikelihood(nn.Module):
    def __init__(self, length):
        super().__init__()
        self.length = length
    
    def forward(self, xT_, xT, meanq, logvarq, meanp, logvarp):
        #xT_ xT [bs, len, seq_dim]

        xT_, xT = xT_[:, -self.length:, :], xT[:, -self.length:, :]

        MSE = nn.MSELoss()(xT_, xT)

        KLD = 0.5 * torch.mean(logvarp - logvarq + (logvarq.exp() + (meanq - meanp).pow(2)) / (logvarp.exp()+1e-10) - 1)

        return MSE + 500*KLD