
# 전체 모델을 돌리기 위한 코드입니다.

from tqdm import tqdm
import os
os.chdir('C:/Users/user/Desktop/박준영/공부/nlp_study/simcse/')
import torch
from util import Args, bring_pretrain, csv_to_pandas
from dataset import bring_dataset
from torch.utils.data import Dataset, DataLoader, random_split
import transformers
from datasets import Dataset
from unsimcsemodel import SimCSEmodel
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoConfig, BatchEncoding
from train import train_model
from tqdm import tqdm


Args=Args()
data = csv_to_pandas(Args.dataset_dir)
tokenizer, model = bring_pretrain(Args.model_name)
train_dataloader, valid_dataloader, test_dataloader = bring_dataset(data, tokenizer, Args)

optimizer = torch.optim.AdamW(params=model.parameters(), lr=Args.learning_rate)

lr_scheduler =transformers.get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=Args.num_warmup_steps,
    # len(train_dataloader) is the number of steps in one epoch
    num_training_steps=len(train_dataloader) * Args.epochs,
)

model = model.to(Args.device)

train_model(model, optimizer, train_dataloader, valid_dataloader, lr_scheduler)