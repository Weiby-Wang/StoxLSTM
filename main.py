import argparse
import torch
import torch.nn as nn
from torch.optim import RAdam, lr_scheduler
import numpy as np
import random
import copy
import time
from torch.utils.data import DataLoader
from StoxLSTM import Model
from data_loader import Dataset_Solar, Dataset_General, Dataset_ETT_hour, Dataset_ETT_minute, Dataset_PEMS
from ModelTrainer import ModelTrainer
from Loss_function import MSE_LB_loglikelihood, MAE_LB_loglikelihood


parser = argparse.ArgumentParser(description='Stochastic xLSTM for Time Series Forecasting')

# random seed
parser.add_argument('--random_seed', type=int, default=3407, help='random seed')

# data loader
parser.add_argument('--data', type=str, required=True, default='Electricity', help='dataset type')
parser.add_argument('--root_path', type=str, default='./_dat/', help='root path of the data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--scale', type=int, default=1, help='feature scaling; True 1 False 0')

# forecasting task
parser.add_argument('--look_back_length', type=int, default=96, help='input sequence length')
parser.add_argument('--label_length', type=int, default=96, help='start token length')
parser.add_argument('--prediction_length', type=int, default=96, help='prediction sequence length')

# Series decomposition
parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
parser.add_argument('--kernel_size', type=int, default=25, help='the size of decomposition-kernel')

# RevIN
parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')

# Patch
parser.add_argument('--patch_size', type=int, default=56, help='patch size')
parser.add_argument('--patch_stride', type=int, default=24, help='patch stride')

# StoxLSTM
parser.add_argument('--patch_and_CI', type=int, default=1, help='patching and channel independence; True 1 False 0')
parser.add_argument('--d_seq', type=int, default=370, help='dimension of sequences')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--d_latent', type=int, default=128, help='dimension of latent varables')
parser.add_argument('--fc_dropout', type=float, default=0.1, help='fully connected dropout')
parser.add_argument('--xlstm_h_num_block', type=int, default=2, help='the number of xlstm layers')
parser.add_argument('--slstm_h_at', action='append', type=int, help='list of slstm position; -1 means no sLSTM block')
parser.add_argument('--xlstm_g_num_block', type=int, default=2, help='the number of xlstm layers')
parser.add_argument('--slstm_g_at', action='append', type=int, help='list of slstm position; -1 means no sLSTM block')
parser.add_argument('--embed_hidden_dim', type=int, default=128, help='hidden dimension of the embedding layer')
parser.add_argument('--embed_hidden_layers_num', type=int, default=1, help='the number of hidden layers in the embedding layer')
parser.add_argument('--mlp_z_hidden_dim', type=int, default=128, help='hidden dimension of the mlp z')
parser.add_argument('--mlp_z_hidden_layers_num', type=int, default=2, help='the number of hidden layers in the mlp z')
parser.add_argument('--mlp_proj_down_hidden_dim', type=int, default=128, help='hidden dimension of the project down mlp')
parser.add_argument('--mlp_proj_down_hidden_layers_num', type=int, default=2, help='the number of hidden layers in the project down mlp')
parser.add_argument('--mlp_x_p_hidden_dim', type=int, default=128, help='hidden dimension of the mlp x_p')
parser.add_argument('--mlp_x_p_hidden_layers_num', type=int, default=2, help='the number of hidden layers in the mlp x_p')
parser.add_argument('--mlp_x_hidden_dim', type=int, default=128, help='hidden dimension of the mlp x')
parser.add_argument('--mlp_x_hidden_layers_num', type=int, default=2, help='the number of hidden layers in the mlp x')

# Training and Test
parser.add_argument('--is_training', type=int, default=1, help='is training phase required? 0 means no training, 1 means training')
parser.add_argument('--is_test', type=int, default=1, help='is testing phase required? 0 means no testing, 1 means testing')
parser.add_argument('--plot_result', type=int, default=1, help='plot forecasting results. 0 means no plotting, 1 means plotting')

# optimization
parser.add_argument('--train_epoches', type=int, default=2048, help='train epochs')
parser.add_argument('--batch_size', type=int, default=256, help='batch size of train input data')
parser.add_argument('--learning_rate', type=float, default=3e-4, help='optimizer learning rate')
parser.add_argument('--device', type=str, default='cuda', help='use gpu')
parser.add_argument('--weight_decay', type=float, default=1e-2, help='weight decay of optimizer')

# save figure and model
parser.add_argument('--figure_save_path', type=str, help='path of saving the forecasting figure')
parser.add_argument('--pre_train_wts_load_path', type=str, help='path of the pre-training model weight')
parser.add_argument('--wts_save_path', type=str, required=True, help='path of saving model wts')
parser.add_argument('--wts_load_path', type=str, help='path of loading model wts')

args = parser.parse_args()
print(args)


##定义全局变量
prediction_length = args.prediction_length
look_back_length = args.look_back_length

gpu = args.device
device = torch.device(gpu if torch.cuda.is_available() else "cpu")
print('=' * 15)
print('The computing device is', device)


##设置随机数种子
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False # type: ignore
    torch.backends.cudnn.deterministic = True # type: ignore

set_seed(args.random_seed)


##模型声明
print('=' * 15)
print('Model loading...')
#监督训练
mySLNet = Model(configs=args).to(device)


##创建数据集
print('=' * 15)
print('Data loading...')

data_map = {
    'Electricity': 'electricity.csv',
    'Weather': 'weather.csv',
    'Solar': 'solar.txt',
    'ETTh1': 'ETTh1.csv',
    "ETTh2": 'ETTh2.csv',
    'ETTm1': 'ETTm1.csv',
    'ETTm2': 'ETTm2.csv',
    'Traffic': 'traffic.csv',
    'PEMS03': 'PEMS03.npz',
    'PEMS04': 'PEMS04.npz',
    'PEMS07': 'PEMS07.npz',
    'PEMS08': 'PEMS08.npz',
    'ILI': 'national_illness.csv'
}

dataset_classes = {
    'Electricity': Dataset_General,
    'Weather': Dataset_General,
    'Solar': Dataset_Solar,
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'Trafffic': Dataset_General,
    'PEMS03': Dataset_PEMS,
    'PEMS04': Dataset_PEMS,
    'PEMS07': Dataset_PEMS,
    'PEMS08': Dataset_PEMS,
    'ILI': Dataset_General
}

if args.data in dataset_classes:
    dataset_class = dataset_classes[args.data]
else:
    raise ValueError(f"UnKnown dataset: {args.data}")

batch_size, label_length, root_path  = args.batch_size, args.label_length, args.root_path

train_dataset = dataset_class(root_path=root_path, data_path=data_map[args.data], flag='train', size=[look_back_length, label_length, prediction_length], 
                                      features=args.features, target=args.target, scale=args.scale)
val_dataset = dataset_class(root_path=root_path, data_path=data_map[args.data], flag='val', size=[look_back_length, label_length, prediction_length], 
                                    features=args.features, target=args.target, scale=args.scale)
test_dataset = dataset_class(root_path=root_path, data_path=data_map[args.data], flag='test', size=[look_back_length, label_length, prediction_length], 
                                      features=args.features, target=args.target, scale=args.scale)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=5, shuffle=False)


##创建训练类
loss_func = MSE_LB_loglikelihood(length=args.prediction_length).to(device)

optimizer = RAdam(mySLNet.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2, eta_min=args.learning_rate/200)

trainer = ModelTrainer(args, model=mySLNet, train_dataset=train_dataset, train_loader=train_loader, 
                       val_dataset=val_dataset, val_loader=val_loader, 
                       test_dataset=test_dataset, test_loader=test_loader, 
                       loss_func=loss_func, optimizer=optimizer, lr_scheduler=scheduler, device=device)

if args.is_training:
    trainer.train()

if args.is_test:
    trainer.test()

if args.plot_result:
    trainer.plot_forecasting_results()
    #trainer.plot_heatmap()