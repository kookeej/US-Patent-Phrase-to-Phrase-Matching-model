import argparse
from tqdm import tqdm
import gc
import numpy as np
import pandas as pd
import copy
from time import sleep
import pickle

import torch
import torch.nn as nn
import torch.optim as optim

from transformers import logging

from config import DefaultConfig
from model import CustomModel
from preprocessing import CustomDataset, pro_dataset
from utils import get_criterion, get_optimizer, get_scheduler, seed_everything, calc_pearson

from colorama import Fore, Style
b_ = Fore.BLUE
y_ = Fore.YELLOW
g_ = Fore.GREEN
r_ = Fore.RED
c_ = Fore.CYAN
sr_ = Style.RESET_ALL



# Settings
config = DefaultConfig()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
seed_everything(config.SEED)
logging.set_verbosity_error()


def train(train_dataloader, valid_dataloader, fold, args):
    gc.collect()
    train_total_loss = []
    valid_total_loss = []
    best_val_loss = np.inf

    train_total_pearson = []
    valid_total_pearson = []

    for epoch in range(args.epochs):
        model.train()
        print(f"{y_}[EPOCH {epoch+1}]{sr_}")
        # 학습 단계 loss/accuracy
        train_loss_value = 0
        train_epoch_loss = []
        train_batch_pearson = 0
        train_epoch_pearson = []

        # 검증 단계 loss/accuracy
        valid_loss_value = 0
        valid_epoch_loss = []
        valid_batch_pearson = 0
        valid_epoch_pearson = []
        train_bar = tqdm(train_dataloader, total=len(train_dataloader))
        for idx, items in enumerate(train_bar):
            if idx == 1: break
            item = {key: val.to(device) for key, val in items.items()}
            optimizer.zero_grad()

            outputs = model(**item)

            loss = criterion(outputs, item['score'])
            pearsonr = calc_pearson(outputs, item['score'], device)

            loss.backward()
            optimizer.step()
            scheduler.step()
            
            train_loss_value += loss.item()
            train_batch_pearson += pearsonr['pearsonr']
            if (idx + 1) % config.TRAIN_LOG_INTERVAL == 0:
                train_bar.set_description("Loss: {:3f}    |    Pearson: {:3f}".\
                    format(train_loss_value/config.TRAIN_LOG_INTERVAL, train_batch_pearson/config.TRAIN_LOG_INTERVAL))
                train_epoch_loss.append(train_loss_value/config.TRAIN_LOG_INTERVAL)
                train_epoch_pearson.append(train_batch_pearson/config.TRAIN_LOG_INTERVAL)
                train_loss_value = 0
                train_batch_pearson = 0

                train_total_loss.append(sum(train_epoch_loss)/len(train_epoch_loss))
                train_total_pearson.append(sum(train_epoch_pearson)/len(train_epoch_pearson))
        print("{}Average Pearson score is {:3f}{}".format(g_, sum(train_epoch_pearson)/len(train_epoch_pearson), sr_))
            
        with torch.no_grad():
            print(f"{b_}---- Validation.... ----{sr_}")
            model.eval()
            valid_bar = tqdm(valid_dataloader, total=len(valid_dataloader))
            for idx, items in enumerate(valid_bar):
                if idx == 1: break
                item = {key: val.to(device) for key,val in items.items()}
                outputs = model(**item)

                # preds = torch.argmax(outputs, dim=-1)
                loss = criterion(outputs, item['score'])
                pearsonr = calc_pearson(item['score'], outputs, device)

                valid_loss_value += loss.item()
                valid_batch_pearson += pearsonr['pearsonr']
                if (idx + 1) % config.VALID_LOG_INTERVAL == 0:
                    valid_bar.set_description("Loss: {:3f}    |    Pearsonr: {:3f}".\
                        format(valid_loss_value/config.VALID_LOG_INTERVAL, valid_batch_pearson/config.VALID_LOG_INTERVAL))
                    valid_epoch_loss.append(valid_loss_value/config.VALID_LOG_INTERVAL)
                    valid_epoch_pearson.append(valid_batch_pearson/config.VALID_LOG_INTERVAL)
                    valid_loss_value = 0
                    valid_batch_pearson = 0

            print("{}Best Loss: {:3f}    |    This epoch Loss: {:3f}     |     This epoch Pearson: {:3f}".format(g_, best_val_loss,
                                                                                                                (sum(valid_epoch_loss)/len(valid_epoch_loss)),
                                                                                                                (sum(valid_epoch_pearson)/len(valid_epoch_pearson))))
            if best_val_loss > (sum(valid_epoch_loss)/len(valid_epoch_loss)):
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), "models/{}_{}.bin".format(config.MODEL_SAVE, fold))
                print(f"{r_}Best Loss Model was Saved!{sr_}")
                best_val_loss = (sum(valid_epoch_loss)/len(valid_epoch_loss))

            valid_total_loss.append(sum(valid_epoch_loss)/len(valid_epoch_loss))
            valid_total_pearson.append(sum(valid_epoch_pearson)/len(valid_epoch_pearson))
        print()
        
        
        
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--learning_rate', type=int, default=1e-5)
    
    args = parser.parse_args()
        
    
    for fold in range(config.FOLDS):
        gc.collect()
        torch.cuda.empty_cache()
        sleep(0.5)
        print(f"{c_}>> FOLD {fold+1} <<{sr_}")
        
        train_dataloader = pickle.load(open(f'data/train_fold{fold}_dataloader.pkl', 'rb'))
        valid_dataloader = pickle.load(open(f'data/valid_fold{fold}_dataloader.pkl', 'rb'))

        model = CustomModel(conf=config.MODEL_CONFIG)
        model.parameters
        model.to(device)

        criterion = get_criterion(name='mse')
        optimizer = get_optimizer(model, lr=args.learning_rate)
        scheduler = get_scheduler(optimizer, train_dataloader, args)

        train(train_dataloader, valid_dataloader, fold, args)
        
        