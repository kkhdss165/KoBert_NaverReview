import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd
import gluonnlp as nlp
from tqdm.notebook import tqdm

from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup
from transformers import BertModel

from sklearn.model_selection import train_test_split

from kobert import get_tokenizer
from kobert import get_pytorch_kobert_model

from tqdm import tqdm

from BERTData import BERTDataset, BERTClassifier

import warnings


def changeTo01(x):
  '''
  기존에 부정(1,2)은 0으로 긍정(3,4)은 1로 label을 변경해주었다.
  :param x:
  :return:
  '''
  if x<3:
    return 0
  else:
    return 1

def calc_accuracy(X, Y):
  max_vals, max_indices = torch.max(X, 1)
  train_acc = (max_indices == Y).sum().data.cpu().numpy() / max_indices.size()[0]
  return train_acc

def main():
  device = torch.device("cuda")

  # print(device)

  data = pd.read_csv("./content/naverReview.csv")

  data['star'] = data['star'].apply(changeTo01)

  data_list = []

  # 리뷰, 긍정 or 부정 순으로 데이터 변경
  # (BERT Input에 넣기 위함)
  for review, label in zip(data['review'], data['star']):
    data = []
    data.append(review)
    data.append(label)
    data_list.append(data)

  # 학습데이터 : 테스트데이터 비율을 4 : 1로
  train, test = train_test_split(data_list, test_size=0.2, shuffle=True, random_state=0)

  ## parameter 설정
  max_len = 64  # max seqence length
  batch_size = 64
  warmup_ratio = 0.1
  num_epochs = 5
  max_grad_norm = 1
  log_interval = 200
  learning_rate = 5e-5

  bertmodel, vocab = get_pytorch_kobert_model()
  tokenizer = get_tokenizer()
  tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

  # print(tok.vocab)
  # print(vocab)

  ## DataLoader
  train_dataset = BERTDataset(train, 0, 1, tok, max_len, True, False)
  test_dataset = BERTDataset(test, 0, 1, tok, max_len, True, False)

  # num_workers : how many subprocesses to use for data loading
  train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=2)
  test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=2)

  model = BERTClassifier(bertmodel, dr_rate=0.5).to(device)

  no_decay = ['bias', 'LayerNorm.weight']

  # 최적화해야 할 parameter를 optimizer에게 알려야 함
  optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
  ]

  optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)  # optimizer
  loss_fn = nn.CrossEntropyLoss()  # loss function

  t_total = len(train_loader) * num_epochs
  warmup_step = int(t_total * warmup_ratio)

  scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)

  for e in range(num_epochs):

    train_acc = 0.0
    test_acc = 0.0

    # Train
    model.train()
    for batch_id, (token_ids, valid_length, segment_ids, label) in tqdm(enumerate(train_loader),total=len(train_loader)):
      optimizer.zero_grad()

      token_ids = token_ids.long().to(device)
      segment_ids = segment_ids.long().to(device)
      valid_length = valid_length
      label = label.long().to(device)

      out = model(token_ids, valid_length, segment_ids)
      loss = loss_fn(out, label)
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
      optimizer.step()
      scheduler.step()  # Update learning rate schedule

      train_acc += calc_accuracy(out, label)
      if batch_id % log_interval == 0:
        print("epoch {} batch id {} loss {} train acc {}".format(e + 1, batch_id + 1, loss.data.cpu().numpy(),
                                                                 train_acc / (batch_id + 1)))
    print("epoch {} train acc {}".format(e + 1, train_acc / (batch_id + 1)))

    # Evaluation
    model.eval()
    for batch_id, (token_ids, valid_length, segment_ids, label) in tqdm(enumerate(test_loader), total=len(test_loader)):
      token_ids = token_ids.long().to(device)
      segment_ids = segment_ids.long().to(device)
      valid_length = valid_length
      label = label.long().to(device)
      out = model(token_ids, valid_length, segment_ids)
      test_acc += calc_accuracy(out, label)
    print("epoch {} test acc {}".format(e + 1, test_acc / (batch_id + 1)))

  # 저장
  PATH = './content/'
  torch.save(model.state_dict(), PATH + 'naver_shopping.pt')


if __name__ == '__main__' :
  main()