import torch
import torch.nn as nn
import torch.optim as optim

import transformers

import numpy as np
from scipy import stats

from config import DefaultConfig

config = DefaultConfig()


# seed 고정 
def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    

# cost function
def get_criterion(name='mse'):
    if name == 'mse':
        return nn.MSELoss()

# optimizer 정의
def get_optimizer(model, lr, name='adam'):
    if name == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=0.01,
            eps=1e-8
        )
    return optimizer

# scheduler 정의
def get_scheduler(optimizer, train_dataloader, args, name='linear'):
    if name == 'linear':
        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=500,
            num_training_steps=len(train_dataloader)*args.epochs,
            last_epoch=-1
        )
    elif name == 'cosine':
        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=500,
            num_training_steps=len(train_dataloader)*args.epochs,
            last_epoch=-1
        )
    return scheduler

# pearson 정의
def calc_pearson(targets, outputs, device):
    outputs = outputs.cpu().detach().numpy().ravel()
    targets = targets.cpu().detach().numpy().ravel()
    pearsonr = stats.pearsonr(targets, outputs)
    return {"pearsonr": torch.tensor(pearsonr[0], device=device)}