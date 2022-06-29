import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
import argparse
from time import sleep
from tqdm import tqdm

import torch

from transformers import logging

from config import DefaultConfig
from model import CustomModel
from preprocessing import CustomDataset, pro_dataset

config = DefaultConfig()
logging.set_verbosity_error()

def make_test_format(dataset):
    index = np.arange(len(dataset)).tolist()
    dataset['index'] = index
    dataset['section'] = dataset['context'].astype(str).str[0]
    dataset['classes'] = dataset['context'].astype(str).str[1:]
    cpctitles = pd.read_csv("data/CPCtitles.csv")
    dataset = dataset.merge(cpctitles, left_on='context', right_on='code')
    dataset = dataset.astype({'index':'int'})
    dataset = dataset.sort_values(by='index' ,ascending=True)
    dataset = dataset.reset_index(drop=True)

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
        dataset['text'][i] = dataset['anchor'][i] + "[SEP]" + dataset['title'][i]
        
    return dataset


def inference(test_dataloader, model_path_lst):
    # 예측값 저장 리스트
    preds_lst = []
    for fold in range(config.FOLDS):
        model = CustomModel(config.MODEL_CONFIG)
        model.to(device)
        model.load_state_dict(torch.load(model_path_lst[fold], map_location=device))
        fold_preds_lst = []
        with torch.no_grad():
            print("Inference....")
            model.eval()
            bar = tqdm(enumerate(test_dataloader), total=len(test_dataloader))
            for idx, items in bar:
                sleep(0.1)
                item = {key: val.to(device) for key,val in items.items()}
                outputs = model(**item)
                fold_preds_lst += outputs.tolist()
        preds_lst.append(fold_preds_lst)
    
    return preds_lst
        

if __name__ == '__main__':
    config = DefaultConfig()
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='data/test.csv', help="test datast path")
    parser.add_argument('--save_sub_path', type=str, default='data/{}.csv'.format(config.SAVE_SUB_FILE_NAME), help="submission file path")

    
    args = parser.parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    
    print("Loading Test DataLoader...")
    test_dataset = pd.read_csv(args.dataset_path)
    test_dataset = make_test_format(test_dataset)
    test_dataloader = pro_dataset(test_dataset, config.TEST_BATCH, mode="test")
    
    model_path_lst = [f"models/{config.MODEL_SAVE}_{fold}.bin" for fold in range(config.FOLDS)]
    
    # Inference
    preds_lst = inference(test_dataloader, model_path_lst)
    print("Inference Finish!")
    
    
    preds_sum = [p1+p2+p3+p4+p5 for p1,p2,p3,p4,p5 in zip(preds_lst[0], preds_lst[1], preds_lst[2], preds_lst[3], preds_lst[4])]
    preds_sum = np.array(preds_sum)
    preds_len = len(preds_sum)
    final_lst = preds_sum / preds_len
    final_lst = final_lst.tolist()
    
    
    
    # submission 저장
    print("Save final submission...")
    sub = pd.read_csv("data/sample_submission.csv")
    sub['score'] = final_lst
    sub.to_csv(args.save_sub_path, index=False)
    
    print("Complete!")