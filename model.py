import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import AutoModel

import gc

from config import DefaultConfig

config = DefaultConfig()


# Customized model 생성
class CustomModel(nn.Module):
    def __init__(self, conf):
        super(CustomModel, self).__init__()
        self.model = AutoModel.from_pretrained(config.MODEL_NAME, config=conf)
        self.similarity_fn = nn.CosineSimilarity()
        self.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512)
        )

    def forward(self, input_ids=None, attention_mask=None, 
                input_ids2=None, attention_mask2=None, score=None):
        gc.collect()
        outputs1 = self.model(
            input_ids, attention_mask=attention_mask
        )
        outputs2 = self.model(
            input_ids2, attention_mask=attention_mask2
        )

        outputs1 = self.fc(outputs1[1])  # deberta: 0
        outputs2 = self.fc(outputs2[1])

        score = self.similarity_fn(outputs1, outputs2)
        

        return score