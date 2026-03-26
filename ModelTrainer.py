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
    """Handles the training, evaluation, and visualization of StoxLSTM models."""

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
        """Compute the initial validation loss (optionally loading pre-trained weights)."""
        if self.args.pre_train_wts_load_path:
            self.model.load_state_dict(torch.load(self.args.pre_train_wts_load_path, weights_only=False), strict=False)
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
        """Train the model, saving the best checkpoint based on validation loss."""
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

            # Save model weights if validation loss improved
            if val_loss < self.best_loss and val_loss > 0:
                self.best_loss = val_loss
                self.best_model_wts = copy.deepcopy(self.model.state_dict())
                torch.save(self.best_model_wts, self.args.wts_save_path)
                print(f'The best model so far with loss {self.best_loss:.4f}')

    def test(self):
        """Evaluate the model on the test set and report deterministic and probabilistic metrics."""
        print('=' * 15)
        print('Test phase begins.')
        self.model.load_state_dict(torch.load(self.args.wts_load_path, weights_only=False))

        loss_VLB = self.loss_func

        test_vlb, test_num = 0.0, 0
        preds, trues = [], []

        self.model.eval()
        with torch.no_grad():
            for step, (_, b_y) in enumerate(self.test_loader):
                b_y = b_y.to(self.device)
                xT_, meanq, logvarq, meanp, logvarp = self.model(b_y)
                # Extract prediction window (last prediction_length steps)
                pred, true = xT_[:, -self.args.prediction_length:, :], b_y[:, -self.args.prediction_length:, :]
                preds.append(pred)
                trues.append(true)
                loss1 = loss_VLB(xT_, b_y, meanq, logvarq, meanp, logvarp)
                test_vlb += loss1.item() * b_y.size(0)
                test_num += b_y.size(0)

        preds = torch.cat(preds, dim=0).cpu().numpy()
        trues = torch.cat(trues, dim=0).cpu().numpy()
        print('test shape:', preds.shape, trues.shape)

        # Inverse-transform PEMS datasets (which require unscaling)
        if self.args.data in {'PEMS03', 'PEMS04', 'PEMS07', 'PEMS08'}:
            B, T, C = preds.shape
            preds = self.test_dataset.inverse_transform(preds.reshape(-1, C)).reshape(B, T, C)
            trues = self.test_dataset.inverse_transform(trues.reshape(-1, C)).reshape(B, T, C)

        mae, mse, rmse, mape, mspe = metric(preds, trues)

        print('VLB loss:{:.4f}, mse:{:.4f}, mae:{:.4f}'.format(test_vlb / test_num, mse, mae))
        print('rmse:{:.4f}, mape:{:.4f}, mspe:{:.4f}'.format(rmse, mape, mspe))

        # Compute Continuous Ranked Probability Score (CRPS) via Monte Carlo sampling
        test_crps, test_num = 0.0, 0

        self.model.eval()
        with torch.no_grad():
            for step, (_, b_y) in enumerate(self.test_loader):
                b_y = b_y.to(self.device)
                # Ground truth: [bs*seq_dim, H]
                true = b_y[:, -self.args.prediction_length:, :].permute(0, 2, 1).reshape(-1, self.args.prediction_length)

                # Draw 50 stochastic samples from the model
                pred = torch.zeros(true.size(0), self.args.prediction_length, 50)
                for i in range(50):
                    xT_, _, _, _, _ = self.model(b_y)
                    output = xT_.permute(0, 2, 1).reshape(-1, self.args.prediction_length)  # [bs*seq_dim, H]
                    pred[:, :, i] = output

                test_crps += np.mean(ps.crps_ensemble(true.cpu().numpy(), pred.cpu().numpy()))
                test_num += 1

        print(f"CRPS:{test_crps / test_num:.4f}")

    def plot_forecasting_results(self):
        """Plot probabilistic forecasting results for 6 channels as confidence intervals.

        Generates 2000 stochastic samples and shows the 1st–99th percentile range
        alongside one sample trace and the ground-truth sequence.
        """
        print('=' * 15)
        print('Plotting a probabilistic forecasting result figure.')
        self.model.load_state_dict(torch.load(self.args.wts_load_path, weights_only=False))

        self.model.eval()
        with torch.no_grad():
            for step, (_, b_y) in enumerate(self.test_loader):
                b_y = b_y.to(self.device)
                break

        num_dims = 6         # number of feature dimensions to plot
        num_samples = 2000   # number of stochastic Monte Carlo samples

        # Collect samples: list of [prediction_length, num_samples] arrays, one per dim
        pred_samples = [torch.zeros(self.args.prediction_length, num_samples) for _ in range(num_dims)]

        with torch.no_grad():
            for i in range(num_samples):
                xT_, _, _, _, _ = self.model(b_y)  # xT_: [bs, H, seq_dim]
                for d in range(num_dims):
                    pred_samples[d][:, i] = xT_[-1, :, -(d + 1)]

        # Convert to numpy for percentile computation
        pred_np = [s.cpu().detach().numpy() for s in pred_samples]

        # Use look_back_length as the offset for the prediction x-axis
        look_back = self.args.look_back_length
        time_steps = torch.arange(look_back, look_back + self.args.prediction_length).numpy()

        # Reference sequences: last `prediction_length` steps from the look-back window
        label_seqs = [b_y[-1, look_back - self.args.prediction_length:, -(d + 1)].cpu().numpy() for d in range(num_dims)]

        # Plot colors
        color_label = (48 / 255, 104 / 255, 141 / 255)   # blue for ground truth
        color_sample = (251 / 255, 132 / 255, 2 / 255)    # orange for one sample
        color_ci = 'green'                                  # green for confidence interval

        plt.figure(figsize=(20, 6))
        plt.rcParams['xtick.labelsize'] = 18
        plt.rcParams['ytick.labelsize'] = 18
        plt.rcParams['legend.fontsize'] = 20

        for d in range(num_dims):
            ax = plt.subplot(2, 3, d + 1)
            lower = np.percentile(pred_np[d], 1, axis=1)
            upper = np.percentile(pred_np[d], 99, axis=1)
            one_sample = pred_np[d][:, 0]

            ax.plot(label_seqs[d], label='Real sequence', color=color_label, linewidth=2)
            ax.plot(time_steps, one_sample, label='A sampled Forecasting', color=color_sample, linewidth=2)
            ax.fill_between(time_steps, lower, upper, color=color_ci, linewidth=8, alpha=0.3, label='98% Confidence Interval')
            ax.set_xticks([])

        plt.tight_layout()
        plt.savefig(self.args.figure_save_path, dpi=300, bbox_inches='tight')
        print('=' * 15)

