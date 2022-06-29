import argparse
from tqdm import tqdm
import gc
import pickle

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold


import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from config import DefaultConfig

config = DefaultConfig()

pd.set_option('mode.chained_assignment',  None)


# Dataframe 생성
def make_dataframe(dataset):
    dataset['section'] = dataset['context'].astype(str).str[0]
    dataset['classes'] = dataset['context'].astype(str).str[1:]
    cpctitles = pd.read_csv("data/CPCtitles.csv")
    dataset = dataset.merge(cpctitles, left_on='context', right_on='code')
    del dataset['section_y']
    del dataset['class']
    del dataset['subclass']
    del dataset['group']
    del dataset['main_group']
    del dataset['code']
    dataset = dataset.rename({'section_x': 'section'}, axis=1)
    dataset['title'] = dataset['title'].str.lower()
    dataset['text'] = None
    for i in range(len(dataset)):
        dataset['text'][i] = dataset['anchor'][i] + " [SEP] " + dataset['title'][i]
    dataset.to_csv("data/cpc_train_with_texts.csv", index=False)
    
    return dataset



def make_fold_dataloader(dataset, folds):
    for fold in range(folds):
        print(f"Create FOLD {fold+1} loader...")
    # 모델 로딩
        train_dataset = dataset[dataset.fold != fold].reset_index(drop=True)
        valid_dataset = dataset[dataset.fold == fold].reset_index(drop=True)
        
        # Make fold dataloader
        train_dataloader = pro_dataset(train_dataset, config.TRAIN_BATCH)
        valid_dataloader = pro_dataset(valid_dataset, config.VALID_BATCH)
        
        # Save fold dataloader
        pickle.dump(train_dataloader, open(f'data/train_fold{fold}_dataloader.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(valid_dataloader, open(f'data/valid_fold{fold}_dataloader.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        
   
    
# Tokenizer
def tokenizing(dataset, mode):
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    
    texts = dataset['text'].tolist()
    targets = dataset['target'].tolist()
    length = len(texts)
    score = None

    if mode == "train":
        score = dataset['score'].tolist()

    tokenized = tokenizer(
        texts,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=128,
        return_token_type_ids=False
    )
    
    tokenized2 = tokenizer(
        targets,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=128,
        return_token_type_ids=False
    )
    for key, value in tokenized2.items():
        tokenized[key+"2"] = value
        
    return tokenized, score, length


# Dataset 구성.
class CustomDataset(Dataset):
    def __init__(self, tokenized_dataset, score, length, mode):
        self.tokenized_dataset = tokenized_dataset
        self.mode = mode
        self.length = length
        if self.mode == "train":
            self.score = score

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.tokenized_dataset.items()}
        if self.mode == "train":
            item['score'] = torch.tensor(self.score[idx])
        return item

    def __len__(self):
        return self.length
    
    
def pro_dataset(dataset, batch_size, mode="train"):
    tokenized, labels, length = tokenizing(dataset, mode=mode)
    custom_dataset = CustomDataset(tokenized, labels, length, mode=mode)
    if mode == "train":
        OPT = True
    else:
        OPT = False
    dataloader = DataLoader(
        custom_dataset, 
        batch_size=batch_size,
        shuffle=OPT,
        drop_last=OPT
    )
    return dataloader





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, default='data/train.csv', help="train dataset path")
    parser.add_argument('--max_len', type=int, default=128, help="max token length for tokenizing")

    args = parser.parse_args()
        
    dataset = pd.read_csv(args.train_path)

    print("Dataset size: {}".format(len(dataset)))    
    
    print("Preprocessing dataset...")
    # Make dataframe
    dataset = make_dataframe(dataset)
    
    # Make Kfolds
    skf = StratifiedKFold(n_splits=config.FOLDS, shuffle=True, random_state=config.SEED)
    for fold, (_, val) in enumerate(skf.split(X=dataset, y=dataset.context)):
        dataset.loc[val, 'fold'] = int(fold)
    dataset['fold'] = dataset['fold'].astype(int)
    
    # Make fold train/valid dataloader
    make_fold_dataloader(dataset, config.FOLDS)
    
    print("Data Preprocessing Complete!")     