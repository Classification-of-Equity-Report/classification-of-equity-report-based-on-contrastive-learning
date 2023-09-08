import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer, logging
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from transformers.optimization import get_linear_schedule_with_warmup
from torch.optim import AdamW
import pytorch_lightning as pl
import math
from torchmetrics import Accuracy


class Fin_Bert(pl.LightningModule):
    def __init__(self, config, length_of_dataset):
        super().__init__()
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config['transformer']['from_pretrained'])
        self.model = AutoModelForSequenceClassification.from_pretrained(config['transformer']['from_pretrained'], 
                                                                     num_labels=config['train']['num_labels'])
        self.lr = config['train']['learning_rate']
        self.total_steps = math.ceil(length_of_dataset / (config['train']['batch_size'] * config['train']['gpu_counts'])) * config['train']['epochs']
        print(self.total_steps)
        self.warmup_steps = config['train']['warmup_steps']
        self._cuda = config['train']['device']
        
    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=self.lr)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.warmup_steps, num_training_steps=self.total_steps)
        sch_config = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [sch_config]
    
    def forward(self, input_ids, token_type_ids, attention_mask, labels):
        # ! make sure that your dataset already tokenized
        inputs = {'input_ids': input_ids, 'token_type_ids': token_type_ids, 'attention_mask': attention_mask}
        # * if you use AutoModelForSequenceClassification, you can get loss, logits *
        # also returns logits when given labels
        output = self.model(**inputs, labels=labels)
        return output
    
    def accuracy(self, preds, labels):
        # Calculate accuracy
        correct = (preds == labels).sum().item()
        total = len(labels)
        accuracy = correct / total
        return accuracy
    
    
    
    def training_step(self, batch, batch_idx):
        labels, input_ids, token_type_ids, attention_mask = batch.values()
           
        output = self.forward(input_ids, token_type_ids, attention_mask, labels)
        
        # Calculate predictions
        train_preds = output.logits.argmax(dim=1)
        
        self.log('train_step_loss', output.loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log('train_accuracy', self.accuracy(train_preds, labels), prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        
        return {"loss": output.loss}
    
    def validation_step(self, batch, batch_idx):
        labels, input_ids, token_type_ids, attention_mask = batch.values()
           
        output = self.forward(input_ids, token_type_ids, attention_mask, labels)
        
        # Calculate predictions
        val_preds = output.logits.argmax(dim=1)
        
        self.log('val_step_loss', output.loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log('val_accuracy', self.accuracy(val_preds, labels), prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        
        return {"loss": output.loss}
    
    def test_step(self, batch, batch_idx):
        labels, input_ids, token_type_ids, attention_mask = batch.values()
        test_loss = self.forward(input_ids, token_type_ids, attention_mask, labels)
        self.log('test_step_loss', test_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        return {"loss": test_loss}