# 🏆 U.S. Patent Phrase to Phrase Matching
### *Help Identify Similar Phrases in U.S. Patents!😃*
Kaggle의  ***U.S. Patent Phrase to Phrase Matching***모델 코드입니다.     
쉽게 하이퍼 파라미터와 모델을 바꿔가며 실험할 수 있도록 모듈화하였습니다. 실행 방법은 아래와 같습니다.

***

# 1. Preprocessing
CPC titles과 train 데이터셋을 결합시켰습니다.
```python
for i in tqdm(range(0, len(codes1), 256)):
    chunck1 = codes1[i:i+256]     # chunck 단위로 나눔
    tokenized = tokenizer(
        chunck1,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=args.max_len
    )
```
* **미리 데이터로더를 `pickle`을 이용해 직렬화하여 저장**함으로써 실험을 진행할 때마다 데이터 로딩 작업 없이 바로 학습, 추론이 가능하도록 만들었습니다.
```python
# save
pickle.dump(train_dataloader, open('data/train_dataloader.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

# loade
train_dataloader = pickle.load(open('data/train_dataloader.pkl', 'rb'))
```

# 2. Model
* `sentence bert`의 `cross-encoder` 구조를 사용하여 설계했습니다.         
![](https://raw.githubusercontent.com/UKPLab/sentence-transformers/master/docs/img/BiEncoder.png)


***


### 💡 실행 방법

#### 1. Data Preprocessing
default로 모두 미리 설정해놨으며, `train_path`, `test_path`의 데이터셋 경로만 설정해주면 됩니다.
```python
$ python preprocessing.py \
  --train_path =TRAIN_DATASET_PATH
  --test_path  =TEST_DATASET_PATH
  --test_size  =0.1
  --max_len    =MAX_TOKEN_LENGTH  # 256
```

#### 2. Training
```python
$ python train.py \
  --epochs =10
```

#### 3. Inference
```python
$ python inference.py\
  --loader_path   =TEST_DATALOADER_PATH            # test dataloader의 경로(data/test_dataloader.pkl)
  --save_sub_path =SAVE_SUBMISSION_FILE_PATH     # 저장할 submission.csv 파일 경로
```


***
# 📑 Results
**Public score**: 0.85885   
epoch: 6    

Ensemble, Regularzation, Data agmentation을 통해 성능 향상을 노릴 수 있음
