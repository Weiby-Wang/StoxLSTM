import torch
import time
import copy
import matplotlib.pyplot as plt
from torch import nn
import properscoring as ps
import numpy as np
from metrics import metric
from StoxLSTM_layers import series_decomp

class ModelTrainer:
    def __init__(self, args, model, train_dataset, train_loader, val_dataset, val_loader, test_dataset, test_loader, loss_func, optimizer, lr_scheduler, device):
        self.model = model
        self.train_dataset = train_dataset
        self.train_loader = train_loader
        self.val_dataset = val_dataset
        self.val_loader = val_loader
        self.test_dataset = test_dataset
        self.test_loader = test_loader
        self.args = args
        self.device = device
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.scheduler = lr_scheduler
        self.best_model_wts = copy.deepcopy(model.state_dict())


    def find_best_loss(self):
        if self.args.pre_train_wts_load_path:
            self.model.load_state_dict(torch.load(self.args.pre_train_wts_load_path), strict=False)
        val_loss = 0
        val_num = 0
        self.model.eval()
        with torch.no_grad():
            for step, (_, b_y) in enumerate(self.val_loader):
                b_y = b_y.to(self.device)
                xT_, meanq, logvarq, meanp, logvarp = self.model(b_y)
                loss = self.loss_func(xT_, b_y, meanq, logvarq, meanp, logvarp)
                val_loss += loss.item() * b_y.size(0)
                val_num += b_y.size(0)      
        self.best_loss = val_loss / val_num
        print('The best model so far with loss {:.4f}'.format(self.best_loss))


    def train(self):
        print('=' * 15)
        print('Training phase begins.')
        self.find_best_loss()

        num_epochs = self.args.train_epoches
        since = time.time()

        for epoch in range(num_epochs):
            print('-' * 15)
            print(f'Epoch {epoch + 1}/{num_epochs}')
            train_loss, train_num = 0.0, 0
            val_loss, val_num = 0.0, 0
            
            # Training mode
            self.model.train()
            for step, (_, b_y) in enumerate(self.train_loader):
                b_y = b_y.to(self.device)
                xT_, meanq, logvarq, meanp, logvarp = self.model(b_y)
                loss = self.loss_func(xT_, b_y, meanq, logvarq, meanp, logvarp)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item() * b_y.size(0)
                train_num += b_y.size(0)
            self.scheduler.step()

            # Validation mode
            self.model.eval()
            with torch.no_grad():
                for step, (_, b_y) in enumerate(self.val_loader):
                    b_y = b_y.to(self.device)
                    xT_, meanq, logvarq, meanp, logvarp = self.model(b_y)
                    loss = self.loss_func(xT_, b_y, meanq, logvarq, meanp, logvarp)
                    val_loss += loss.item() * b_y.size(0)
                    val_num += b_y.size(0)

            train_loss /= train_num
            val_loss /= val_num
            time_use = time.time() - since
            
            print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            print(f'Learning rate: {self.optimizer.param_groups[0]["lr"]:.6f}')
            print(f'Time use: {time_use:.1f}s')

            if val_loss < self.best_loss and val_loss > 0:
                self.best_loss = val_loss
                self.best_model_wts = copy.deepcopy(self.model.state_dict())
                torch.save(self.best_model_wts, self.args.wts_save_path)
                print(f'The best model so far with loss {self.best_loss:.4f}')


    def test(self):
        print('=' * 15)
        print('Test phase begins.')
        self.model.load_state_dict(torch.load(self.args.wts_load_path))

        #MSE, MAE, VLB
        loss_VLB = self.loss_func

        test_vlb, test_num = 0.0, 0
        preds, trues = [], []

        self.model.eval()
        with torch.no_grad():
            for step, (_, b_y) in enumerate(self.test_loader):
                b_y = b_y.to(self.device)
                xT_, meanq, logvarq, meanp, logvarp = self.model(b_y)
                pred, true = xT_[:, -self.args.prediction_length:, :], b_y[:, -self.args.prediction_length:, :]
                preds.append(pred)
                trues.append(true)
                loss1 = loss_VLB(xT_, b_y, meanq, logvarq, meanp, logvarp)
                test_vlb += loss1.item() * b_y.size(0)
                test_num += b_y.size(0)

        preds = torch.cat(preds, dim=0).cpu().numpy()
        trues = torch.cat(trues, dim=0).cpu().numpy()
        print('test shape:', preds.shape, trues.shape)

        if self.args.data in {'PEMS03', 'PEMS04', 'PEMS07', 'PEMS08'}:
            B, T, C = preds.shape
            preds = self.test_dataset.inverse_transform(preds.reshape(-1, C)).reshape(B, T, C)
            trues = self.test_dataset.inverse_transform(trues.reshape(-1, C)).reshape(B, T, C)

        mae, mse, rmse, mape, mspe = metric(preds, trues)

        print('VLB loss:{:.4f}, mse:{:.4f}, mae:{:.4f}'.format(test_vlb / test_num, mse, mae))
        print('rmse:{:.4f}, mape:{:.4f}, mspe:{:.4f}'.format(rmse, mape, mspe))

        #计算CRPS
        test_crps, test_num = 0.0, 0

        self.model.eval()
        with torch.no_grad():
            for step, (_, b_y) in enumerate(self.test_loader):
                b_y = b_y.to(self.device)
                #真实值
                true = b_y[:, -self.args.prediction_length:, :].permute(0, 2, 1).reshape(-1, self.args.prediction_length) #[bs*seq_dim, H]

                pred = torch.zeros(true.size(0), self.args.prediction_length, 50) #[bs*seq_dim, H, 50]
                for i in range(50):
                    xT_, _, _, _, _ = self.model(b_y)
                    output = xT_.permute(0, 2, 1).reshape(-1, self.args.prediction_length) #[bs*seq_dim, H]
                    pred[:, :, i] = output

                test_crps += np.mean(ps.crps_ensemble(true.cpu().numpy(), pred.cpu().numpy()))
                test_num += 1

        print(f"CRPS:{test_crps / test_num:.4f}")


    def plot_forecasting_results(self):
        print('=' * 15)
        print('Plotting a probabilistic forecasting result figure.')
        self.model.load_state_dict(torch.load(self.args.wts_load_path))

        self.model.eval()
        with torch.no_grad():
            for step, (_, b_y) in enumerate(self.test_loader):
                b_y = b_y.to(self.device)
                break
        
        #Probabilistic Forecasting
        pred_seq_p_dim1 = torch.zeros(self.args.prediction_length, 2000)
        pred_seq_p_dim2 = torch.zeros(self.args.prediction_length, 2000)
        pred_seq_p_dim3 = torch.zeros(self.args.prediction_length, 2000)
        pred_seq_p_dim4 = torch.zeros(self.args.prediction_length, 2000)
        pred_seq_p_dim5 = torch.zeros(self.args.prediction_length, 2000)
        pred_seq_p_dim6 = torch.zeros(self.args.prediction_length, 2000)
        
        with torch.no_grad():
            for i in range(2000):
                xT_, meanq, logvarq, meanp, logvarp = self.model(b_y) #xT_ [bs, H, seq_dim]
                pred_seq_p_dim1[:, i] = xT_[-1, :, -1] #[:, i] [H]
                pred_seq_p_dim2[:, i] = xT_[-1, :, -2] 
                pred_seq_p_dim3[:, i] = xT_[-1, :, -3] 
                pred_seq_p_dim4[:, i] = xT_[-1, :, -4] 
                pred_seq_p_dim5[:, i] = xT_[-1, :, -5] 
                pred_seq_p_dim6[:, i] = xT_[-1, :, -6] 


        pred_seq_p_dim1_np = pred_seq_p_dim1.cpu().detach().numpy()
        pred_seq_p_dim2_np = pred_seq_p_dim2.cpu().detach().numpy()
        pred_seq_p_dim3_np = pred_seq_p_dim3.cpu().detach().numpy()
        pred_seq_p_dim4_np = pred_seq_p_dim4.cpu().detach().numpy()
        pred_seq_p_dim5_np = pred_seq_p_dim5.cpu().detach().numpy()
        pred_seq_p_dim6_np = pred_seq_p_dim6.cpu().detach().numpy()
        
        #time_steps = torch.arange(self.args.look_back_length, self.args.look_back_length + self.args.prediction_length).numpy()
        time_steps = torch.arange(96, 96 + self.args.prediction_length).numpy()
        
        #one sample of confidence interval
        one_sample_dim1 = pred_seq_p_dim1_np[:, 0]        
        one_sample_dim2 = pred_seq_p_dim2_np[:, 0]  
        one_sample_dim3 = pred_seq_p_dim3_np[:, 0]  
        one_sample_dim4 = pred_seq_p_dim4_np[:, 0]  
        one_sample_dim5 = pred_seq_p_dim5_np[:, 0]  
        one_sample_dim6 = pred_seq_p_dim6_np[:, 0]  

        #confidence interval
        lower_bound_dim1 = np.percentile(pred_seq_p_dim1_np, 1, axis=1)
        upper_bound_dim1 = np.percentile(pred_seq_p_dim1_np, 99, axis=1)

        lower_bound_dim2 = np.percentile(pred_seq_p_dim2_np, 1, axis=1)
        upper_bound_dim2 = np.percentile(pred_seq_p_dim2_np, 99, axis=1)

        lower_bound_dim3 = np.percentile(pred_seq_p_dim3_np, 1, axis=1)
        upper_bound_dim3 = np.percentile(pred_seq_p_dim3_np, 99, axis=1)

        lower_bound_dim4 = np.percentile(pred_seq_p_dim4_np, 1, axis=1)
        upper_bound_dim4 = np.percentile(pred_seq_p_dim4_np, 99, axis=1)

        lower_bound_dim5 = np.percentile(pred_seq_p_dim5_np, 1, axis=1)
        upper_bound_dim5 = np.percentile(pred_seq_p_dim5_np, 99, axis=1)
        
        lower_bound_dim6 = np.percentile(pred_seq_p_dim6_np, 1, axis=1)
        upper_bound_dim6 = np.percentile(pred_seq_p_dim6_np, 99, axis=1)
        
        #label_seq
        label_seq_dim1 = b_y[-1, self.args.look_back_length-96:, -1].cpu().numpy()
        label_seq_dim2 = b_y[-1, self.args.look_back_length-96:, -2].cpu().numpy()
        label_seq_dim3 = b_y[-1, self.args.look_back_length-96:, -3].cpu().numpy()
        label_seq_dim4 = b_y[-1, self.args.look_back_length-96:, -4].cpu().numpy()
        label_seq_dim5 = b_y[-1, self.args.look_back_length-96:, -5].cpu().numpy()
        label_seq_dim6 = b_y[-1, self.args.look_back_length-96:, -6].cpu().numpy()
        
        #plot figure
        plt.figure(figsize=(20, 8))
        
        plt.rcParams['xtick.labelsize']  = 18
        plt.rcParams['ytick.labelsize']  = 18 
        plt.rcParams['legend.fontsize']  = 20
        
        plt.subplot(2, 3, 1)
        plt.plot(label_seq_dim1, label='Real sequence', color='blue', linewidth=1)
        plt.plot(time_steps, one_sample_dim1, label='A sampled Forecasting', color='orange', linewidth=1)
        plt.fill_between(time_steps, lower_bound_dim1, upper_bound_dim1, color='green', linewidth=6, alpha=0.6, label='98% Confidence Interval')
        plt.xticks([])
        #plt.legend()
        
        plt.subplot(2, 3, 2)
        plt.plot(label_seq_dim2, label='Real sequence', color='blue', linewidth=1)
        plt.plot(time_steps, one_sample_dim2, label='A sampled Forecasting', color='orange', linewidth=1)
        plt.fill_between(time_steps, lower_bound_dim2, upper_bound_dim2, color='green', linewidth=6, alpha=0.6, label='98% Confidence Interval')     
        plt.xticks([])
        
        plt.subplot(2, 3, 3)
        plt.plot(label_seq_dim3, label='Real sequence', color='blue', linewidth=1)
        plt.plot(time_steps, one_sample_dim3, label='A sampled Forecasting', color='orange', linewidth=1)
        plt.fill_between(time_steps, lower_bound_dim3, upper_bound_dim3, color='green', linewidth=6, alpha=0.6, label='98% Confidence Interval')
        plt.xticks([])
        
        plt.subplot(2, 3, 4)
        plt.plot(label_seq_dim4, label='Real sequence', color='blue', linewidth=1)
        plt.plot(time_steps, one_sample_dim4, label='A sampled Forecasting', color='orange', linewidth=1)
        plt.fill_between(time_steps, lower_bound_dim4, upper_bound_dim4, color='green', linewidth=6, alpha=0.6, label='98% Confidence Interval')
        plt.xticks([])
        
        plt.subplot(2, 3, 5)
        plt.plot(label_seq_dim5, label='Real sequence', color='blue', linewidth=1)
        plt.plot(time_steps, one_sample_dim5, label='A sampled Forecasting', color='orange', linewidth=1)
        plt.fill_between(time_steps, lower_bound_dim5, upper_bound_dim5, color='green',linewidth=6, alpha=0.6, label='98% Confidence Interval')  
        plt.xticks([])   
        
        plt.subplot(2, 3, 6)
        plt.plot(label_seq_dim6, label='Real sequence', color='blue', linewidth=1)
        plt.plot(time_steps, one_sample_dim6, label='A sampled Forecasting', color='orange', linewidth=1)
        plt.fill_between(time_steps, lower_bound_dim6, upper_bound_dim6, color='green', linewidth=6, alpha=0.6, label='98% Confidence Interval')
        plt.xticks([])
        
        plt.tight_layout()
        plt.savefig(self.args.figure_save_path, dpi=300, bbox_inches='tight')
        print('=' * 15)



    # def plot_heatmap(self):
    #     print('=' * 15)
    #     print('Plotting a probabilistic forecasting result figure.')
    #     self.model.load_state_dict(torch.load(self.args.wts_load_path))

    #     self.model.eval()
    #     with torch.no_grad():
    #         for step, (_, b_y) in enumerate(self.test_loader):
    #             b_y = b_y.to(self.device)
    #             break
        
    #     # decomp_module = series_decomp(kernel_size=25)
    #     # res_init, trend_init = decomp_module(b_y.cpu())

    #     # res_T = res_init[-1, :, -1]
    #     # res_c = torch.cat((res_T[:96], torch.zeros(48)), dim=0)

    #     # plt.figure(figsize=(8, 2))
    #     # plt.plot(torch.cat((torch.zeros(28), res_c), dim=0))
    #     # plt.axis('off')
    #     # plt.savefig("figure/res_c")

    #     # plt.figure(figsize=(8, 2))
    #     # plt.plot(torch.cat((torch.zeros(28), res_T), dim=0))
    #     # plt.axis('off')
    #     # plt.savefig("figure/res_t")
    #     xT_P, xC_P, xT_P_flatten, hT, gT_, xT_, meanq, logvarq, meanp, logvarp = self.model(b_y)

    #     self.plot_heatmap_figure(xT_P[-1, : , :], 'figure/xT_P.png')
    #     self.plot_heatmap_figure(xC_P[-1, : , :], 'figure/xC_P.png')
    #     self.plot_heatmap_figure(xT_P_flatten[-1, : , :], 'figure/xT_P_flatten.png')
    #     self.plot_heatmap_figure(meanq[-1, : , :], 'figure/meanq.png')
    #     self.plot_heatmap_figure(logvarq[-1, : , :], 'figure/logvarq.png')
    #     self.plot_heatmap_figure(meanp[-1, : , :], 'figure/meanp.png')
    #     self.plot_heatmap_figure(logvarp[-1, : , :], 'figure/logvarp.png')
    #     self.plot_heatmap_figure(logvarp[-1, : , :], 'figure/logvarp.png')
    #     self.plot_heatmap_figure(hT[-1, : , :], 'figure/hT.png')
    #     self.plot_heatmap_figure(gT_[-1, : , :], 'figure/gT_.png')


    # def plot_heatmap_figure(self, seq, save_path):
    #     # seq [seq_len, seq_dim]
    #     seq_np = seq.cpu().detach().numpy()

    #     plt.figure(figsize=(10, 6))
    #     plt.imshow(seq_np.T, aspect='auto', origin='lower', cmap='viridis')
    #     plt.colorbar(label='Value')
    #     plt.xlabel('seq_len')
    #     plt.ylabel('seq_dim')
    #     plt.title('Tensor Heatmap')
    #     plt.savefig(save_path, dpi=300, bbox_inches='tight')
