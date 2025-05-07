# StoxLSTM: A Stochastic Extend Long Short-Term Memory
__all__ = ['StoxLSTM_backbone']

#加载必要的库
import torch
import torch.nn as nn
from StoxLSTM_layers import MLP, xlstm_block, RevIN, Padding_Patch_Layer, Z_generation_model
from math import ceil


class StoxLSTM_backbone(nn.Module):
    def __init__(self, device, d_seq:int, d_model:int, d_latent:int, look_back_length:int, prediction_length:int, 
                patch_size:int, patch_stride:int, xlstm_h_num_block:int, slstm_h_at:int, xlstm_g_num_block:int, slstm_g_at:int, 
                embed_hidden_dim:int, embed_hidden_layers_num:int,
                mlp_z_hidden_dim:int, mlp_z_hidden_layers_num:int, mlp_proj_down_hidden_dim:int, mlp_proj_down_hidden_layers_num:int,
                mlp_x_p_hidden_dim:int, mlp_x_p_hidden_layers_num:int, mlp_x_hidden_dim:int, mlp_x_hidden_layers_num:int,
                revin:bool=True, subtract_last:bool=False, dropout:float=0.2):
        super().__init__()

        #Model Size and Prediction Task
        self.d_seq = d_seq
        self.d_model = d_model
        self.d_hidden = 2*d_model
        self.d_latent = d_latent
        self.H = prediction_length
        self.L = look_back_length
        self.T = look_back_length + prediction_length
        self.device = device  
        
        #Patching
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.N = ceil((self.T + patch_stride - patch_size) / patch_stride) + 1 #patch num
        self.padding_patch_layer = Padding_Patch_Layer
        
        #RevIN
        self.revin = revin
        if self.revin: 
            self.revin_layer = RevIN(self.d_seq, affine=True, subtract_last=subtract_last)
            
        #Instance Normalization  
        self.instance_norm_P = nn.InstanceNorm2d(num_features=d_seq)
        self.instance_norm_embed = nn.InstanceNorm2d(num_features=d_seq)
        self.instance_norm_hT = nn.InstanceNorm2d(num_features=d_seq)
        self.instance_norm_gT = nn.InstanceNorm2d(num_features=d_seq)
        self.instance_norm_xP = nn.InstanceNorm2d(num_features=d_seq)
        
        #xLSTM
        self.xlstm_h = xlstm_block(context_length=self.N, embedding_dim=d_model, num_blocks=xlstm_h_num_block, slstm_at=slstm_h_at, dropout=dropout)
        self.xlstm_g = xlstm_block(context_length=self.N, embedding_dim=3*d_model, num_blocks=xlstm_g_num_block, slstm_at=slstm_g_at, dropout=dropout)
        
        #State Space Model
        self.emb_layer = MLP(input_dim=patch_size, output_dim=d_model, hidden_dim=embed_hidden_dim, hidden_layer=embed_hidden_layers_num, dropout=dropout)
        self.z_generation = Z_generation_model(h_dim=self.d_hidden, z_size=[self.N, d_latent], device=device, mlp_size=[mlp_z_hidden_dim, mlp_z_hidden_layers_num], dropout=dropout)
        self.proj_down = MLP(input_dim=2*(self.d_hidden+d_model), output_dim=self.d_hidden, hidden_dim=mlp_proj_down_hidden_dim, hidden_layer=mlp_proj_down_hidden_layers_num, dropout=dropout)
        self.mlp_x_P = MLP(input_dim=self.d_hidden+d_latent, output_dim=patch_size, hidden_dim=mlp_x_p_hidden_dim, hidden_layer=mlp_x_p_hidden_layers_num, dropout=dropout)
        self.mlp_x = MLP(input_dim=self.N*patch_size, output_dim=prediction_length, hidden_dim=mlp_x_hidden_dim, hidden_layer=mlp_x_hidden_layers_num, dropout=dropout)  
    
    def inference_model(self, xT_P):
        #推理模型，用于得到z1:T的近似后验概率
        #xT_P [bs, d_seq, N, patch_len]
        xT_P = xT_P.view(-1, self.N, self.patch_size) #[bs*d_seq, N, patch_size]
        
        #embedding layer
        xT = self.emb_layer(xT_P).view(-1, self.d_seq, self.N, self.d_model) #xT [bs, d_seq, N, d_model]
        xT = self.instance_norm_embed(xT)
        xT = xT.view(-1, self.N, self.d_model) #xT [bs*d_seq, N, d_model]
        
        #Transition Process
        #hT
        _, hT = self.xlstm_h(xT) #h1:T [bs*d_seq, N, d_hidden]
        hT = hT.view(-1, self.d_seq, self.N, self.d_hidden) #hT [bs, d_seq, N, d_hidden]
        hT = self.instance_norm_hT(hT)
        hT = hT.view(-1, self.N, self.d_hidden) #hT [bs*d_seq, N, d_hidden]
        
        #移动xT
        xT0 = torch.cat((xT, xT[:, 0:1, :]), dim=1) #[bs*d_seq, N+1, d_model]
        xT = xT0[:, 1:, :]

        #gT滤波
        h_x = torch.cat((hT, xT), dim=-1)  #h_x [bs*d_seq, N, d_hidden+d_model]
        _, gT_1 = self.xlstm_g(h_x) #gT_1 [bs*d_seq, N=N:1, 2*(d_hidden+d_model)]
        gT = torch.flip(gT_1, dims=[-2])  # gT_1 [bs*d_seq, N=1:N, 2*(d_hidden+d_model)]
        gT = gT.view(-1, self.d_seq, self.N, 2*(self.d_hidden+self.d_model))  # [bs, d_seq, N, 2*(d_hidden+d_model)]
        gT = self.instance_norm_gT(gT).view(-1, self.N, 2*(self.d_hidden+self.d_model)) #[bs*d_seq, N, 2*(d_hidden+d_model)]
        gT_ = self.proj_down(gT).view(-1, self.d_seq, self.N, self.d_hidden) #下采样 [bs, d_seq, N, d_hidden)]
        gT_ = self.instance_norm_hT(gT_).view(-1, self.N, self.d_hidden) #[bs*d_seq, N=1:N, d_hidden)]
        _, meanq, logvarq = self.z_generation(self.bs*self.d_seq, gT_)
        
        return meanq, logvarq
    
    def generation_model(self, xC_P):
        #生成模型，用于预测x1:T_，并得到z1:T的先验概率
        #xC_P: [bs, d_seq, N, patch_len]
        xC_P = xC_P.view(-1, self.N, self.patch_size) #[bs*d_seq, N, patch_size]
        
        #embedding Layer
        xC = self.emb_layer(xC_P).view(-1, self.d_seq, self.N, self.d_model) #xC [bs, d_seq, N, d_model]
        xC = self.instance_norm_embed(xC)
        xC = xC.view(-1, self.N, self.d_model) #xC [bs*d_seq, N, d_model]
        
        #Transition Process
        _, hT = self.xlstm_h(xC) #h1:T [bs*d_seq, N, d_hidden]
        hT = hT.view(-1, self.d_seq, self.N, self.d_hidden) #hT [bs, d_seq, N, d_hidden]
        hT = self.instance_norm_hT(hT)
        hT = hT.view(-1, self.N, self.d_hidden) #hT [bs*d_seq, N, d_hidden]
        zT, meanp, logvarp = self.z_generation(self.bs*self.d_seq, hT) #z1:T [bs*d_seq, N, d_latent]
        
        #Emission Process
        h_z = torch.cat((hT, zT), dim=-1) #[bs*d_seq, N, dim=d_hidden+d_latent]
        xT_P = self.mlp_x_P(h_z).view(-1, self.d_seq, self.N, self.patch_size) #[bs, d_seq, N, dim=patch_size]
        xT_P = self.instance_norm_xP(xT_P)
        xT_P_flatten = xT_P.view(-1, self.d_seq, self.N*self.patch_size) #xT_P_flatten [bs, d_seq, N*patch_size]
        xT_ = self.mlp_x(xT_P_flatten) #xT_ [bs, d_seq, prediction_length]
        
        return xT_, meanp, logvarp
    
    def forward(self, xT):
        # x: [bs, d_seq, seq_len]
        #Instance Normalize and Padding
        self.bs = xT.size(0)
        xC = xT[:, :, :self.L] #xC [bs, d_seq, seq_len=L]
        
        #norm
        if self.revin:
            xC, xT = xC.permute(0, 2, 1), xT.permute(0, 2, 1) #[bs, seq_len, d_seq]
            xC, xT = self.revin_layer(xC, 'norm'), self.revin_layer(xT, 'norm')
            xC, xT = xC.permute(0, 2, 1), xT.permute(0, 2, 1) #[bs, d_seq, seq_len]
        
        #padding xC => xT
        zeros_padding = torch.zeros(self.bs, self.d_seq, self.H).to(self.device)
        xC_padding = torch.cat((xC, zeros_padding), dim=-1) #xT => [xC 0...0] x_padding [bs, d_seq, seq_len=T]
        
        #do patching
        xT_P = self.padding_patch_layer(xT, self.patch_stride, self.patch_size) 
        xC_P = self.padding_patch_layer(xC_padding, self.patch_stride, self.patch_size) #xT_P, xC_P: [bs, d_seq, self.N, patch_size]
        xT_P, xC_P = self.instance_norm_P(xT_P), self.instance_norm_P(xC_P)
        
        #Variational Inference
        meanq, logvarq = self.inference_model(xT_P)
        xT_, meanp, logvarp = self.generation_model(xC_P) #xT_ [bs, d_seq, prediction_length]
        
        #denorm
        if self.revin:
            xT_ = xT_.permute(0, 2, 1) #xT_ [bs, prediction_length, d_seq]
            xT_ = self.revin_layer(xT_, 'denorm')
            xT_ = xT_.permute(0, 2, 1) #xT_ [bs, d_seq, prediction_length]

        return xT_, meanq, logvarq, meanp, logvarp
    


class StoxLSTM_backbone_WO_PCI(nn.Module):
    def __init__(self, device, d_seq:int, d_model:int, d_latent:int, look_back_length:int, prediction_length:int, 
                 xlstm_h_num_block:int, slstm_h_at:int, xlstm_g_num_block:int, slstm_g_at:int, 
                 embed_hidden_dim:int, embed_hidden_layers_num:int,
                 mlp_z_hidden_dim:int, mlp_z_hidden_layers_num:int, mlp_proj_down_hidden_dim:int, mlp_proj_down_hidden_layers_num:int,
                 mlp_x_hidden_dim:int, mlp_x_hidden_layers_num:int, revin:bool=True, subtract_last:bool=False, dropout:float=0.2):
        super().__init__()
        
        self.d_seq = d_seq
        self.d_model = d_model
        self.d_hidden = 2*d_model
        self.d_latent = d_latent
        self.T = look_back_length + prediction_length
        self.H = prediction_length
        self.L = look_back_length
        self.device = device 
        
        #RevIN
        self.revin = revin
        if self.revin: 
            self.revin_layer = RevIN(self.d_seq, affine=True, subtract_last=subtract_last)
        
        #xLSTM
        self.xlstm_h = xlstm_block(context_length=self.T, embedding_dim=d_model, num_blocks=xlstm_h_num_block, slstm_at=slstm_h_at)
        self.xlstm_g = xlstm_block(context_length=self.T, embedding_dim=3*d_model, num_blocks=xlstm_g_num_block, slstm_at=slstm_g_at)
        
        #State Space Model
        self.emb_layer = MLP(input_dim=d_seq, output_dim=d_model, hidden_dim=embed_hidden_dim, hidden_layer=embed_hidden_layers_num, dropout=dropout)
        self.z_generation = Z_generation_model(h_dim=self.d_hidden, z_size=[self.T, d_latent], device=device, mlp_size=[mlp_z_hidden_dim, mlp_z_hidden_layers_num], dropout=dropout)
        self.proj_down = MLP(input_dim=2*(self.d_hidden+d_model), output_dim=self.d_hidden, hidden_dim=mlp_proj_down_hidden_dim, hidden_layer=mlp_proj_down_hidden_layers_num, dropout=dropout)
        self.mlp_x = MLP(input_dim=self.d_hidden+d_latent, output_dim=d_seq, hidden_dim=mlp_x_hidden_dim, hidden_layer=mlp_x_hidden_layers_num, dropout=dropout)
    
    def inference_model(self, xT):
        #推理模型，用于得到z1:T的近似后验概率
        #[bs, seq_len, d_seq]
        #embedding layer
        xT = self.emb_layer(xT) #xT [bs, d_seq, d_model]
        
        #Transition Process
        #hT
        _, hT = self.xlstm_h(xT) #h1:T [bs, d_seq, d_hidden]
        
        #gT滤波
        h_x = torch.cat((hT, xT), dim=-1)  #h_x [bs, d_seq, d_hidden+d_model]
        _, gT_1 = self.xlstm_g(h_x) #gT_1 [bs, d_seq=T:1, 2*(d_hidden+d_model)]
        gT = torch.flip(gT_1, dims=[-2])  # gT_1 [bs, d_seq=1:T, 2*(d_hidden+d_model)]
        
        #zT
        gT_ = self.proj_down(gT) #下采样 [bs, d_seq, d_hidden)]
        _, meanq, logvarq = self.z_generation(self.bs, gT_) #[bs, seq_len, dim=d_latent]
        
        return meanq, logvarq
    
    def generation_model(self, xC):
        #生成模型，用于预测x1:T_，并得到z1:T的先验概率
        #[bs, seq_len=C, d_seq]
        #padding xC => xT
        zeros_padding = torch.zeros(self.bs, self.H, self.d_seq).to(self.device)
        xC = torch.cat((xC, zeros_padding), dim=1) #xT => [xC 0...0] x_padding [bs, seq_len=T, d_seq]
        
        #embedding Layer
        xC = self.emb_layer(xC) #xC [bs, seq_len, d_model]
        
        #Transition Process
        _, hT = self.xlstm_h(xC) #h1:T [bs, seq_len, d_hidden]
        zT, meanp, logvarp = self.z_generation(self.bs, hT) #z1:T [bs, seq_len, d_latent]
        
        #Emission Process
        h_z = torch.cat((hT, zT), dim=-1) #[bs, seq_len, dim=d_hidden+d_latent]
        xT_ = self.mlp_x(h_z) #[bs, seq_len, dim=d_seq]
        
        return xT_, meanp, logvarp
    
    def forward(self, xT):
        # x: [bs, d_seq, seq_len]
        #Instance Normalize and Padding
        self.bs = xT.size(0)
        xC = xT[:, :, :self.L] #xC [bs, d_seq, seq_len=L]
        
        #norm
        if self.revin:
            xC, xT = xC.permute(0, 2, 1), xT.permute(0, 2, 1) #[bs, seq_len, d_seq]
            xC, xT = self.revin_layer(xC, 'norm'), self.revin_layer(xT, 'norm') #[bs, seq_len, d_seq]
        
        #Variational Inference
        meanq, logvarq = self.inference_model(xT)
        xT_, meanp, logvarp = self.generation_model(xC) #[bs, prediction_length, d_seq]
        
        #denorm
        if self.revin:
            xT_ = self.revin_layer(xT_, 'denorm')
            xT_ = xT_.permute(0, 2, 1) #xT_ [bs, d_seq, prediction_length]

        return xT_, meanq, logvarq, meanp, logvarp
    
'''
class StoxLSTM_backbone(nn.Module):
    def __init__(self, device, d_seq:int, d_model:int, d_latent:int, look_back_length:int, prediction_length:int, 
                patch_size:int, patch_stride:int, xlstm_h_num_block:int, slstm_h_at:int, xlstm_g_num_block:int, slstm_g_at:int, 
                embed_hidden_dim:int, embed_hidden_layers_num:int,
                mlp_z_hidden_dim:int, mlp_z_hidden_layers_num:int, mlp_proj_down_hidden_dim:int, mlp_proj_down_hidden_layers_num:int,
                mlp_x_p_hidden_dim:int, mlp_x_p_hidden_layers_num:int, mlp_x_hidden_dim:int, mlp_x_hidden_layers_num:int,
                revin:bool=True, subtract_last:bool=False, dropout:float=0.2):
        super().__init__()

        #Model Size and Prediction Task
        self.d_seq = d_seq
        self.d_model = d_model
        self.d_hidden = 2*d_model
        self.d_latent = d_latent
        self.H = prediction_length
        self.L = look_back_length
        self.T = look_back_length + prediction_length
        self.device = device  
        
        #Patching
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.N = ceil((self.T + patch_stride - patch_size) / patch_stride) + 1 #patch num
        self.padding_patch_layer = Padding_Patch_Layer
        
        #RevIN
        self.revin = revin
        if self.revin: 
            self.revin_layer = RevIN(self.d_seq, affine=True, subtract_last=subtract_last)
            
        #Instance Normalization  
        self.instance_norm_P = nn.InstanceNorm2d(num_features=d_seq)
        self.instance_norm_embed = nn.InstanceNorm2d(num_features=d_seq)
        self.instance_norm_hT = nn.InstanceNorm2d(num_features=d_seq)
        self.instance_norm_gT = nn.InstanceNorm2d(num_features=d_seq)
        self.instance_norm_xP = nn.InstanceNorm2d(num_features=d_seq)
        
        #xLSTM
        self.xlstm_h = xlstm_block(context_length=self.N, embedding_dim=d_model, num_blocks=xlstm_h_num_block, slstm_at=slstm_h_at, dropout=dropout)
        self.xlstm_g = xlstm_block(context_length=self.N, embedding_dim=3*d_model, num_blocks=xlstm_g_num_block, slstm_at=slstm_g_at, dropout=dropout)
        
        #State Space Model
        self.emb_layer = MLP(input_dim=patch_size, output_dim=d_model, hidden_dim=embed_hidden_dim, hidden_layer=embed_hidden_layers_num, dropout=dropout)
        self.z_generation = Z_generation_model(h_dim=self.d_hidden, z_size=[self.N, d_latent], device=device, mlp_size=[mlp_z_hidden_dim, mlp_z_hidden_layers_num], dropout=dropout)
        self.proj_down = MLP(input_dim=2*(self.d_hidden+d_model), output_dim=self.d_hidden, hidden_dim=mlp_proj_down_hidden_dim, hidden_layer=mlp_proj_down_hidden_layers_num, dropout=dropout)
        self.mlp_x_P = MLP(input_dim=self.d_hidden+d_latent, output_dim=patch_size, hidden_dim=mlp_x_p_hidden_dim, hidden_layer=mlp_x_p_hidden_layers_num, dropout=dropout)
        self.mlp_x = MLP(input_dim=self.N*patch_size, output_dim=prediction_length, hidden_dim=mlp_x_hidden_dim, hidden_layer=mlp_x_hidden_layers_num, dropout=dropout)  
    
    def inference_model(self, xT_P):
        #推理模型，用于得到z1:T的近似后验概率
        #xT_P [bs, d_seq, N, patch_len]
        xT_P = xT_P.view(-1, self.N, self.patch_size) #[bs*d_seq, N, patch_size]
        
        #embedding layer
        xT = self.emb_layer(xT_P).view(-1, self.d_seq, self.N, self.d_model) #xT [bs, d_seq, N, d_model]
        xT = self.instance_norm_embed(xT)
        xT_ = xT.view(-1, self.N, self.d_model) #xT [bs*d_seq, N, d_model]
        
        #Transition Process
        #hT
        _, hT = self.xlstm_h(xT_) #h1:T [bs*d_seq, N, d_hidden]
        hT = hT.view(-1, self.d_seq, self.N, self.d_hidden) #hT [bs, d_seq, N, d_hidden]
        hT = self.instance_norm_hT(hT)
        hT = hT.view(-1, self.N, self.d_hidden) #hT [bs*d_seq, N, d_hidden]
        
        #移动xT
        xT0 = torch.cat((xT_, xT_[:, 0:1, :]), dim=1) #[bs*d_seq, N+1, d_model]
        xT = xT0[:, 1:, :]

        #gT滤波
        h_x = torch.cat((hT, xT), dim=-1)  #h_x [bs*d_seq, N, d_hidden+d_model]
        _, gT_1 = self.xlstm_g(h_x) #gT_1 [bs*d_seq, N=N:1, 2*(d_hidden+d_model)]
        gT = torch.flip(gT_1, dims=[-2])  # gT_1 [bs*d_seq, N=1:N, 2*(d_hidden+d_model)]
        gT = gT.view(-1, self.d_seq, self.N, 2*(self.d_hidden+self.d_model))  # [bs, d_seq, N, 2*(d_hidden+d_model)]
        gT = self.instance_norm_gT(gT).view(-1, self.N, 2*(self.d_hidden+self.d_model)) #[bs*d_seq, N, 2*(d_hidden+d_model)]
        gT_ = self.proj_down(gT).view(-1, self.d_seq, self.N, self.d_hidden) #下采样 [bs, d_seq, N, d_hidden)]
        gT_ = self.instance_norm_hT(gT_).view(-1, self.N, self.d_hidden) #[bs*d_seq, N=1:N, d_hidden)]
        _, meanq, logvarq = self.z_generation(self.bs*self.d_seq, gT_)
        
        return xT_, gT_, meanq, logvarq
    
    def generation_model(self, xC_P):
        #生成模型，用于预测x1:T_，并得到z1:T的先验概率
        #xC_P: [bs, d_seq, N, patch_len]
        xC_P = xC_P.view(-1, self.N, self.patch_size) #[bs*d_seq, N, patch_size]
        
        #embedding Layer
        xC = self.emb_layer(xC_P).view(-1, self.d_seq, self.N, self.d_model) #xC [bs, d_seq, N, d_model]
        xC = self.instance_norm_embed(xC)
        xC = xC.view(-1, self.N, self.d_model) #xC [bs*d_seq, N, d_model]
        
        #Transition Process
        _, hT = self.xlstm_h(xC) #h1:T [bs*d_seq, N, d_hidden]
        hT = hT.view(-1, self.d_seq, self.N, self.d_hidden) #hT [bs, d_seq, N, d_hidden]
        hT = self.instance_norm_hT(hT)
        hT = hT.view(-1, self.N, self.d_hidden) #hT [bs*d_seq, N, d_hidden]
        zT, meanp, logvarp = self.z_generation(self.bs*self.d_seq, hT) #z1:T [bs*d_seq, N, d_latent]
        
        #Emission Process
        h_z = torch.cat((hT, zT), dim=-1) #[bs*d_seq, N, dim=d_hidden+d_latent]
        xT_P = self.mlp_x_P(h_z).view(-1, self.d_seq, self.N, self.patch_size) #[bs, d_seq, N, dim=patch_size]
        xT_P = self.instance_norm_xP(xT_P)
        xT_P_flatten = xT_P.view(-1, self.d_seq, self.N*self.patch_size) #xT_P_flatten [bs, d_seq, N*patch_size]
        xT_ = self.mlp_x(xT_P_flatten) #xT_ [bs, d_seq, prediction_length]
        
        return xC, xT_P.view(-1, self.N, self.patch_size), hT, xT_, meanp, logvarp
    
    def forward(self, xT):
        # x: [bs, d_seq, seq_len]
        #Instance Normalize and Padding
        self.bs = xT.size(0)
        xC = xT[:, :, :self.L] #xC [bs, d_seq, seq_len=L]
        
        #norm
        if self.revin:
            xC, xT = xC.permute(0, 2, 1), xT.permute(0, 2, 1) #[bs, seq_len, d_seq]
            xC, xT = self.revin_layer(xC, 'norm'), self.revin_layer(xT, 'norm')
            xC, xT = xC.permute(0, 2, 1), xT.permute(0, 2, 1) #[bs, d_seq, seq_len]
        
        #padding xC => xT
        zeros_padding = torch.zeros(self.bs, self.d_seq, self.H).to(self.device)
        xC_padding = torch.cat((xC, zeros_padding), dim=-1) #xT => [xC 0...0] x_padding [bs, d_seq, seq_len=T]
        
        #do patching
        xT_P = self.padding_patch_layer(xT, self.patch_stride, self.patch_size) 
        xC_P = self.padding_patch_layer(xC_padding, self.patch_stride, self.patch_size) #xT_P, xC_P: [bs, d_seq, self.N, patch_size]
        xT_P, xC_P = self.instance_norm_P(xT_P), self.instance_norm_P(xC_P)
        
        #Variational Inference
        xT_P, gT_, meanq, logvarq = self.inference_model(xT_P)
        xC_P, xT_P_flatten, hT, xT_, meanp, logvarp = self.generation_model(xC_P) #xT_ [bs, d_seq, prediction_length]
        
        #denorm
        if self.revin:
            xT_ = xT_.permute(0, 2, 1) #xT_ [bs, prediction_length, d_seq]
            xT_ = self.revin_layer(xT_, 'denorm')
            xT_ = xT_.permute(0, 2, 1) #xT_ [bs, d_seq, prediction_length]

        return xT_P, xC_P, xT_P_flatten, hT, gT_, xT_, meanq, logvarq, meanp, logvarp
    '''