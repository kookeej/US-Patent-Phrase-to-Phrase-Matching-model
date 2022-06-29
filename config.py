import transformers
from transformers import AutoConfig

import re

class DefaultConfig:
    MODEL_NAME = "anferico/bert-for-patents"
    MODEL_CONFIG = AutoConfig.from_pretrained(MODEL_NAME)
    SEED = 42
    FOLDS = 5
    
    OPTION = ""
    MODEL_SAVE = "{}_model".format(OPTION)
    MODEL_PATH = "models/{}.bin".format(MODEL_SAVE)
    SAVE_SUB_FILE_NAME = "submission_{}_{}".format(re.sub('[ \/:*?"<>|]', '', MODEL_NAME), OPTION)
    
    TRAIN_BATCH = 32
    VALID_BATCH = 128
    TEST_BATCH = 64
    
    TRAIN_LOG_INTERVAL = 1
    VALID_LOG_INTERVAL = 1