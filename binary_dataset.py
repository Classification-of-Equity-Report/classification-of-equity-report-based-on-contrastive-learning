import torch
import torch.nn as nn
from torch.utils.data import Dataset
from glob import glob
from tqdm import tqdm
from typing import List, Tuple, Union, Dict
import datasets
from transformers import AutoTokenizer
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from pathlib import Path
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class BinaryDataset(Dataset):
    
    def __init__(self, mode, config):
        super().__init__()
        
        self.data_full_path = glob( os.path.join(os.getcwd(), config['data']["data_full_path"]))
        self.tokenizer = AutoTokenizer.from_pretrained(config['transformer']["from_pretrained"])
        self.split_day = config['data']['split_day']
        self.max_length = config['transformer']['max_length']
        self.N = config['data']['N']

        # ! 날짜에 따라 train/test 분리
        if mode == "train":
            self.data_path = [path for path in self.data_full_path if int(path[-12:-4]) < self.split_day]       
        
        elif mode == "test":
            self.data_path = [path for path in self.data_full_path if int(path[-12:-4]) >= self.split_day]
            
        #todo: File.lock() <- prevent from reading the same file
        self.dataset_collector = self.load_datasets()
        self.samples = self.process_datasets()
        
    def load_datasets(self):
        #todo: parallel with concurrent.futures
        
        dataset_collector = []
        
        print("Load datasets...")
        for path in tqdm(self.data_path):
            file = pd.read_pickle(path)
            
            try:
                N_label = file.loc[:,:self.N].dropna().copy()
                file['label'] = N_label.loc[:,self.N].apply(lambda x: 1 if x > 0.05 else 0).copy()
                
                text_label = pd.concat([file['텍스트'], file['label']], axis=1) # file.loc[:,['텍스트', 'label']]
                
                dataset_collector.append(text_label)
            except:
                pass
            
        return pd.concat(dataset_collector, ignore_index=True)
    
    def process_datasets(self):
        
        # make dataset for huggingface🤗
        hf_datasets = datasets.Dataset.from_pandas(self.dataset_collector)
        hf_datasets = hf_datasets.map(lambda x: self.tokenizer(x['텍스트'], 
                                                               max_length=self.max_length,
                                                               truncation=True, padding="max_length"), 
                                      batched=True, remove_columns=['텍스트'])
        
        hf_datasets.set_format("torch", columns = ['label','input_ids', 'token_type_ids', 'attention_mask'])
        ##batch 별로 가져올 때 column은 위와 같음.
        return hf_datasets
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]