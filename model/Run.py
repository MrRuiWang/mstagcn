import os
import sys
import torch
import numpy as np
import torch.nn as nn
import argparse
import configparser
from datetime import datetime
from model.BasicTrainer import Trainer
from lib.TrainInits import init_seed
from lib.TrainInits import print_model_parameters
from model.Model import STGCN_module as Network
from lib.metrics import MAE_torch
from lib.utils import load_graph_data

file_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(file_dir)
sys.path.append(file_dir)

# *************************************************************************#
Mode = 'train'
DEBUG = 'False'
DATASET = 'Manchester'  # PEMS08 or Manchester
DEVICE = 'cuda:0'
MODEL = 'TFGCN'

# get configuration
config_file = './{}.conf'.format(DATASET)
config = configparser.ConfigParser()
config.read(config_file)

# 打印所有的节（sections）
sections = config.sections()
print("所有节（sections）:", sections)
for section in sections:
    items = config.items(section)
    print(f"节 '{section}' 中的所有选项和对应的值:", items)

config_data = config['data']
config_model = config['model']
config_train = config['train']
config_test = config['test']
config_log = config['log']

# parser
args = argparse.ArgumentParser(description='arguments')
args.add_argument('--dataset', default=DATASET, type=str)
args.add_argument('--mode', default=Mode, type=str)
args.add_argument('--device', default=DEVICE, type=str, help='indices of GPUs')
args.add_argument('--debug', default=DEBUG, type=eval)
args.add_argument('--model', default=MODEL, type=str)
args.add_argument('--cuda', default=True, type=bool)

# data
args.add_argument('--times', default=config_data["times"], type=int)
args.add_argument('--len_input', default=config_data["len_input"], type=int)
args.add_argument('--val_ratio', default=config_data["val_ratio"], type=float)
args.add_argument('--test_ratio', default=config_data["test_ratio"], type=float)
args.add_argument('--lag', default=config_data["lag"], type=int)
args.add_argument('--horizon', default=config_data["horizon"], type=int)
args.add_argument('--num_nodes', default=config_data["num_nodes"], type=int)
args.add_argument('--tod', default=config_data["tod"], type=eval)
args.add_argument('--normalizer', default=config_data["normalizer"], type=str)
args.add_argument('--column_wise', default=config_data["column_wise"], type=eval)
args.add_argument('--num_of_weeks', default=config_data["num_of_weeks"], type=int)
args.add_argument('--num_of_months', default=config_data["num_of_months"], type=int)
args.add_argument('--num_of_hours', default=config_data["num_of_hours"], type=int)
args.add_argument('--data_path', default=config_data["data_path"], type=str)
# model
args.add_argument('--input_dim', default=config_model["input_dim"], type=int)
args.add_argument('--output_dim', default=config_model["output_dim"], type=int)
args.add_argument('--embed_dim', default=config_model["embed_dim"], type=int)
args.add_argument('--cheb_k', default=config_model["cheb_order"], type=int)

# Transformer
args.add_argument('--d_ff', default=config_model["d_ff"], type=int)
args.add_argument('--d_v', default=config_model["d_v"], type=int)
args.add_argument('--d_k', default=config_model["d_k"], type=int)
args.add_argument('--n_layers', default=config_model["n_layers"], type=int)
args.add_argument('--n_heads', default=config_model["n_heads"], type=int)
args.add_argument('--d_model', default=config_model["d_model"], type=int)
args.add_argument('--dropout', default=config_model["dropout"], type=int)
# train
args.add_argument('--in_channels', default=config_train["in_channels"], type=int)
args.add_argument('--nb_block', default=config_train["nb_block"], type=int)
args.add_argument('--nb_chev_filter', default=config_train["nb_chev_filter"], type=int)
args.add_argument('--nb_time_filter', default=config_train["nb_time_filter"], type=int)

args.add_argument('--loss_func', default=config_train["loss_func"], type=str)
args.add_argument('--seed', default=config_train["seed"], type=int)
args.add_argument('--batch_size', default=config_train["batch_size"], type=int)
args.add_argument('--epochs', default=config_train["epochs"], type=int)
args.add_argument('--start_epoch', default=config_train["start_epoch"], type=int)
args.add_argument('--lr_init', default=config_train["lr_init"], type=float)
args.add_argument('--lr_decay', default=config_train["lr_decay"], type=eval)
args.add_argument('--lr_decay_rate', default=config_train["lr_decay_rate"], type=float)
args.add_argument('--lr_decay_step', default=config_train["lr_decay_step"], type=str)
args.add_argument('--early_stop', default=config_train["early_stop"], type=eval)
args.add_argument('--early_stop_patience', default=config_train["early_stop_patience"], type=int)
args.add_argument('--grad_norm', default=config_train["grad_norm"], type=eval)
args.add_argument('--max_grad_norm', default=config_train["max_grad_norm"], type=int)
args.add_argument('--teacher_forcing', default=config_train["teacher_forcing"], type=bool)
args.add_argument('--tf_decay_steps', default=2000, type=int, help='teacher forcing decay steps')
args.add_argument('--real_value', default=config_train["real_value"], type=eval,
                  help='use real value for loss calculation')
# test
args.add_argument('--mae_thresh', default=config_test["mae_thresh"], type=eval)
args.add_argument('--mape_thresh', default=config_test["mape_thresh"], type=float)
# log
args.add_argument('--log_dir', default='./', type=str)
args.add_argument('--log_step', default=config_log["log_step"], type=int)
args.add_argument('--plot', default=config_log["plot"], type=eval)

args = args.parse_args()
# init_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.set_device(int(args.device[5]))
else:
    args.device = 'cpu'

# if DATASET == 'Manchester':
#     sensor_ids, sensor_id_to_ind, adj_mx = load_graph_data("../data/Manchester/adj_mat_manchester.pkl")
# else:
#     sensor_ids, sensor_id_to_ind, adj_mx = load_graph_data("../data/PEMS08/adj_mat1023.pkl")
time_strides = args.num_of_hours
model = Network(args, time_strides)
model = model.to(args.device)
for p in model.parameters():

    if p.dim() > 1:
        nn.init.xavier_uniform_(p)
    else:
        nn.init.uniform_(p)
print_model_parameters(model, only_num=False)

if args.data_path == "Manchester":
    data_path = "../data/" + args.data_path + "/" + args.data_path + "DataFinall" + "_r" + str(
        args.num_of_hours) + "_m" + str(
        args.num_of_months) + "_w" + str(args.num_of_weeks) + "_f" + str(args.lag) + ".npz"
else:
    data_path = "../data/" + args.data_path + "/" + args.data_path + "_r" + str(args.num_of_hours) + "_m" + str(
        args.num_of_months) + "_w" + str(args.num_of_weeks) + "_f" + str(args.horizon) + ".npz"
print(data_path)
all_data = np.load(data_path, allow_pickle=True)

print("train:", all_data['train_week'].shape, all_data['train_month'].shape, all_data['train_recent'].shape,
      all_data['train_target'].shape)
print("val:", all_data['val_week'].shape, all_data['val_month'].shape, all_data['val_recent'].shape,
      all_data['val_target'].shape)
print("test:", all_data['test_week'].shape, all_data['test_month'].shape, all_data['test_recent'].shape,
      all_data['test_target'].shape)

train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(
        torch.from_numpy(all_data['train_week'].astype("float32")).type(torch.FloatTensor).to(DEVICE),
        torch.from_numpy(all_data['train_month'].astype("float32")).type(torch.FloatTensor).to(DEVICE),
        torch.from_numpy(all_data['train_recent'].astype("float32")).type(torch.FloatTensor).to(DEVICE),
        torch.from_numpy(all_data['train_target'].astype("float32")).type(torch.FloatTensor).to(DEVICE),
    ),
    batch_size=args.batch_size,
    shuffle=True
)
# validation set data loader
val_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(
        torch.from_numpy(all_data['val_week'].astype("float32")).type(torch.FloatTensor).to(DEVICE),
        torch.from_numpy(all_data['val_month'].astype("float32")).type(torch.FloatTensor).to(DEVICE),
        torch.from_numpy(all_data['val_recent'].astype("float32")).type(torch.FloatTensor).to(DEVICE),
        torch.from_numpy(all_data['val_target'].astype("float32")).type(torch.FloatTensor).to(DEVICE),
    ),
    batch_size=args.batch_size,
    shuffle=False
)

# testing set data loader
test_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(
        torch.from_numpy(all_data['test_week'].astype("float32")).type(torch.FloatTensor).to(DEVICE),
        torch.from_numpy(all_data['test_month'].astype("float32")).type(torch.FloatTensor).to(DEVICE),
        torch.from_numpy(all_data['test_recent'].astype("float32")).type(torch.FloatTensor).to(DEVICE),
        torch.from_numpy(all_data['test_target'].astype("float32")).type(torch.FloatTensor).to(DEVICE),
    ),
    batch_size=args.batch_size,
    shuffle=False
)

if args.loss_func == 'mae':
    loss = torch.nn.L1Loss().to(args.device)
elif args.loss_func == 'mse':
    loss = torch.nn.MSELoss().to(args.device)
else:
    raise ValueError

optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr_init, eps=1.0e-8,
                             weight_decay=0, amsgrad=False)
# learning rate decay
lr_scheduler = None
if args.lr_decay:
    print('Applying learning rate decay.')
    lr_decay_steps = [int(i) for i in list(args.lr_decay_step.split(','))]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                        milestones=lr_decay_steps,
                                                        gamma=args.lr_decay_rate)
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=64)
# config log path
current_time = datetime.now().strftime('%Y%m%d%H%M%S')
current_dir = os.path.dirname(os.path.realpath(__file__))
log_dir = os.path.join(current_dir, 'exper', args.dataset, current_time)
args.log_dir = log_dir
# start training
trainer = Trainer(model, loss, optimizer, train_loader, val_loader, test_loader,
                  args, lr_scheduler=lr_scheduler)
if args.mode == 'train':
    trainer.train()
elif args.mode == 'test':
    model.load_state_dict(torch.load('../pre-trained/{}.pth'.format(args.dataset)))
    print("Load saved model")
    trainer.test(model, trainer.args, test_loader, trainer.logger)
else:
    raise ValueError
os.system("/usr/bin/shutdown")  # 加上之后自动服务器自动关机
