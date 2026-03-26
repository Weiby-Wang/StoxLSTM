"""Model trainer for StoxLSTM: handles training, testing, and visualization."""

import torch
import time
import copy
import matplotlib.pyplot as plt
from torch import nn
import properscoring as ps
import numpy as np
from metrics import metric


class ModelTrainer:
    """Manages training, validation, testing, and result visualization.

    Args:
        args: Configuration namespace with all hyperparameters.
        model: The StoxLSTM model instance.
        train_dataset: Training dataset.
        train_loader: Training data loader.
        val_dataset: Validation dataset.
        val_loader: Validation data loader.
        test_dataset: Test dataset.
        test_loader: Test data loader.
        loss_func: Loss function (variational lower bound).
        optimizer: Optimizer instance.
        lr_scheduler: Learning rate scheduler.
        device: Computing device (CPU or CUDA).
    """

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
        """Compute initial best validation loss for early stopping comparison.

        If a pre-trained model checkpoint path is provided, loads weights first.
        """
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
        """Run the full training loop with validation and model checkpointing."""
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
            
            # Training phase
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

            # Validation phase
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

            # Save best model based on validation loss
            if val_loss < self.best_loss and val_loss > 0:
                self.best_loss = val_loss
                self.best_model_wts = copy.deepcopy(self.model.state_dict())
                torch.save(self.best_model_wts, self.args.wts_save_path)
                print(f'The best model so far with loss {self.best_loss:.4f}')

    def test(self):
        """Evaluate the model on the test set with MSE, MAE, RMSE, MAPE, MSPE, and CRPS metrics."""
        print('=' * 15)
        print('Test phase begins.')
        self.model.load_state_dict(torch.load(self.args.wts_load_path))

        # Variational lower bound loss
        loss_VLB = self.loss_func

        test_vlb, test_num = 0.0, 0
        preds, trues = [], []

        # Compute point forecast metrics
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

        # Inverse transform for PEMS datasets to get original scale metrics
        if self.args.data in {'PEMS03', 'PEMS04', 'PEMS07', 'PEMS08'}:
            B, T, C = preds.shape
            preds = self.test_dataset.inverse_transform(preds.reshape(-1, C)).reshape(B, T, C)
            trues = self.test_dataset.inverse_transform(trues.reshape(-1, C)).reshape(B, T, C)

        mae, mse, rmse, mape, mspe = metric(preds, trues)

        print('VLB loss:{:.4f}, mse:{:.4f}, mae:{:.4f}'.format(test_vlb / test_num, mse, mae))
        print('rmse:{:.4f}, mape:{:.4f}, mspe:{:.4f}'.format(rmse, mape, mspe))

        # Compute CRPS (Continuous Ranked Probability Score) for probabilistic evaluation
        test_crps, test_num = 0.0, 0

        self.model.eval()
        with torch.no_grad():
            for step, (_, b_y) in enumerate(self.test_loader):
                b_y = b_y.to(self.device)
                # Ground truth values
                true = b_y[:, -self.args.prediction_length:, :].permute(0, 2, 1).reshape(-1, self.args.prediction_length)  # [bs*seq_dim, H]

                # Generate 50 stochastic forecast samples for ensemble
                pred = torch.zeros(true.size(0), self.args.prediction_length, 50)  # [bs*seq_dim, H, 50]
                for i in range(50):
                    xT_, _, _, _, _ = self.model(b_y)
                    output = xT_.permute(0, 2, 1).reshape(-1, self.args.prediction_length)  # [bs*seq_dim, H]
                    pred[:, :, i] = output

                test_crps += np.mean(ps.crps_ensemble(true.cpu().numpy(), pred.cpu().numpy()))
                test_num += 1

        print(f"CRPS:{test_crps / test_num:.4f}")

    def plot_forecasting_results(self):
        """Generate and save probabilistic forecasting visualization.

        Plots 6 dimensions showing ground truth, a single sampled forecast,
        and 98% confidence intervals from 2000 stochastic samples.
        """
        print('=' * 15)
        print('Plotting a probabilistic forecasting result figure.')
        self.model.load_state_dict(torch.load(self.args.wts_load_path))

        # Get a single test batch for visualization
        self.model.eval()
        with torch.no_grad():
            for step, (_, b_y) in enumerate(self.test_loader):
                b_y = b_y.to(self.device)
                break
        
        # Generate 2000 stochastic forecast samples for probabilistic forecasting
        prediction_length = self.args.prediction_length
        pred_seq_p_dim1 = torch.zeros(prediction_length, 2000)
        pred_seq_p_dim2 = torch.zeros(prediction_length, 2000)
        pred_seq_p_dim3 = torch.zeros(prediction_length, 2000)
        pred_seq_p_dim4 = torch.zeros(prediction_length, 2000)
        pred_seq_p_dim5 = torch.zeros(prediction_length, 2000)
        pred_seq_p_dim6 = torch.zeros(prediction_length, 2000)
        
        with torch.no_grad():
            for i in range(2000):
                xT_, meanq, logvarq, meanp, logvarp = self.model(b_y)  # xT_: [bs, H, seq_dim]
                pred_seq_p_dim1[:, i] = xT_[-1, :, -1]
                pred_seq_p_dim2[:, i] = xT_[-1, :, -2] 
                pred_seq_p_dim3[:, i] = xT_[-1, :, -3] 
                pred_seq_p_dim4[:, i] = xT_[-1, :, -4] 
                pred_seq_p_dim5[:, i] = xT_[-1, :, -5] 
                pred_seq_p_dim6[:, i] = xT_[-1, :, -6] 

        # Convert predictions to numpy for plotting
        pred_seq_p_dim1_np = pred_seq_p_dim1.cpu().detach().numpy()
        pred_seq_p_dim2_np = pred_seq_p_dim2.cpu().detach().numpy()
        pred_seq_p_dim3_np = pred_seq_p_dim3.cpu().detach().numpy()
        pred_seq_p_dim4_np = pred_seq_p_dim4.cpu().detach().numpy()
        pred_seq_p_dim5_np = pred_seq_p_dim5.cpu().detach().numpy()
        pred_seq_p_dim6_np = pred_seq_p_dim6.cpu().detach().numpy()
        
        look_back_length = self.args.look_back_length
        time_steps = torch.arange(look_back_length, look_back_length + prediction_length).numpy()
        
        # Extract a single sample from each dimension
        one_sample_dim1 = pred_seq_p_dim1_np[:, 0]        
        one_sample_dim2 = pred_seq_p_dim2_np[:, 0]  
        one_sample_dim3 = pred_seq_p_dim3_np[:, 0]  
        one_sample_dim4 = pred_seq_p_dim4_np[:, 0]  
        one_sample_dim5 = pred_seq_p_dim5_np[:, 0]  
        one_sample_dim6 = pred_seq_p_dim6_np[:, 0]  

        # Compute 98% confidence intervals (1st to 99th percentile)
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
        
        # Extract ground truth label sequences (last sample in batch, full context + prediction)
        label_seq_dim1 = b_y[-1, :, -1].cpu().numpy()
        label_seq_dim2 = b_y[-1, :, -2].cpu().numpy()
        label_seq_dim3 = b_y[-1, :, -3].cpu().numpy()
        label_seq_dim4 = b_y[-1, :, -4].cpu().numpy()
        label_seq_dim5 = b_y[-1, :, -5].cpu().numpy()
        label_seq_dim6 = b_y[-1, :, -6].cpu().numpy()
        
        # Plot 6 subplots (2 rows × 3 columns)
        plt.figure(figsize=(20, 6))
        
        plt.rcParams['xtick.labelsize']  = 18
        plt.rcParams['ytick.labelsize']  = 18 
        plt.rcParams['legend.fontsize']  = 20

        color_label_seq = (48/255, 104/255, 141/255)          # Blue for ground truth
        color_one_sample = (251/255, 132/255, 2/255)          # Orange for sampled forecast
        color_confidence_interval = (255/255, 202/255, 95/255) # (defined but using green fill)
        
        plt.subplot(2, 3, 1)
        plt.plot(label_seq_dim1, label='Real sequence', color=color_label_seq, linewidth=2)
        plt.plot(time_steps, one_sample_dim1, label='A sampled Forecasting', color=color_one_sample, linewidth=2)
        plt.fill_between(time_steps, lower_bound_dim1, upper_bound_dim1, color="green", linewidth=8, alpha=0.3, label='98% Confidence Interval')
        plt.xticks([])
        
        plt.subplot(2, 3, 2)
        plt.plot(label_seq_dim2, label='Real sequence', color=color_label_seq, linewidth=2)
        plt.plot(time_steps, one_sample_dim2, label='A sampled Forecasting', color=color_one_sample, linewidth=2)
        plt.fill_between(time_steps, lower_bound_dim2, upper_bound_dim2, color="green", linewidth=8, alpha=0.3, label='98% Confidence Interval')     
        plt.xticks([])
        
        plt.subplot(2, 3, 3)
        plt.plot(label_seq_dim3, label='Real sequence', color=color_label_seq, linewidth=2)
        plt.plot(time_steps, one_sample_dim3, label='A sampled Forecasting', color=color_one_sample, linewidth=2)
        plt.fill_between(time_steps, lower_bound_dim3, upper_bound_dim3, color="green", linewidth=8, alpha=0.3, label='98% Confidence Interval')
        plt.xticks([])
        
        plt.subplot(2, 3, 4)
        plt.plot(label_seq_dim4, label='Real sequence', color=color_label_seq, linewidth=2)
        plt.plot(time_steps, one_sample_dim4, label='A sampled Forecasting', color=color_one_sample, linewidth=2)
        plt.fill_between(time_steps, lower_bound_dim4, upper_bound_dim4, color="green", linewidth=8, alpha=0.3, label='98% Confidence Interval')
        plt.xticks([])
        
        plt.subplot(2, 3, 5)
        plt.plot(label_seq_dim5, label='Real sequence', color=color_label_seq, linewidth=2)
        plt.plot(time_steps, one_sample_dim5, label='A sampled Forecasting', color=color_one_sample, linewidth=2)
        plt.fill_between(time_steps, lower_bound_dim5, upper_bound_dim5, color="green",linewidth=8, alpha=0.3, label='98% Confidence Interval')  
        plt.xticks([])   
        
        plt.subplot(2, 3, 6)
        plt.plot(label_seq_dim6, label='Real sequence', color=color_label_seq, linewidth=2)
        plt.plot(time_steps, one_sample_dim6, label='A sampled Forecasting', color=color_one_sample, linewidth=2)
        plt.fill_between(time_steps, lower_bound_dim6, upper_bound_dim6, color="green", linewidth=8, alpha=0.3, label='98% Confidence Interval')
        plt.xticks([])
        
        plt.tight_layout()
        plt.savefig(self.args.figure_save_path, dpi=300, bbox_inches='tight')
        print('=' * 15)
