
## import dataset, customized model
from binary_dataset import BinaryDataset
from binary_model import Fin_Bert

from argparse import ArgumentParser
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, Dataset, random_split
from glob import glob
import os
from pathlib import Path
import yaml
from utils import load_args, load_yaml_param_settings, seed_everything
from torchmetrics import Accuracy


##GPU DEVICE설정
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

def train(config, train_data_loader, valid_data_loader, test_data_loader=None):
    
    project_name = 'BERTBinaryClassifier'
    group_name = config['train']['run_name']
    
    ### config 파일 입력 Boolean 값에 따라 달라짐
    if config['transformer']['Baseline'] and not config['transformer']['Sim_'] and not config['transformer']['UnSim_']:
    # Get the baseline weights
        model = Fin_Bert(config, length_of_dataset=len(train_data_loader.dataset))    
    elif config['transformer']['Sim_'] and not config['transformer']['Baseline'] and not config['transformer']['UnSim_']:
    # Get the supervised trained models' weights
       model = Fin_Bert(config, length_of_dataset=len(train_data_loader.dataset))        
    elif config['transformer']['UnSim_'] and not config['transformer']['Baseline'] and not config['transformer']['Sim_']:
    # Get the unsupervised trained models' weights
       model = Fin_Bert(config, length_of_dataset=len(train_data_loader.dataset))        
    else:
        raise ValueError("Invalid model configuration in the config file")

    
    ###Wandb Logger###
    #wandb = WandbLogger()
    #wandb.init(project=project_name, name=None, config=config, group=group_name)
    #wandb_logger = WandbLogger(project=project_name, name=None, config=config)
    #wandb_logger.watch(model, log='all')
    
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,   
        monitor = 'val_step_loss',# track loss by each validation step
        mode = 'min',
        dirpath = config['train']['saving_path'],
        filename = f'{group_name}'+'-{epoch}-{val_loss:.9f}',
    )
    
    trainer = pl.Trainer(logger=False,
                         enable_checkpointing=True,
                         #callbacks=[LearningRateMonitor(logging_interval='step'), checkpoint_callback],
                         max_epochs = config['train']['epochs'],
                         accelerator='gpu',
                         # strategy='ddp', ## 병렬처리
                         devices= config['train']['gpu_counts'] if torch.cuda.is_available() else None,
                         precision = config['train']['precision'],
                         )
    
    trainer.fit(model, 
                train_dataloaders =train_data_loader, 
                val_dataloaders = valid_data_loader)
    
    if test_data_loader is not None:
        trainer.test(test_dataloaders=test_data_loader)

    #when logging finish wandb logging
    #wandb.finish()

if __name__ == "__main__":
    
    args = load_args()
    config = load_yaml_param_settings(args.config)
    ## make train, valid, test dataset
    
    ## using the config files' valid_ratio, split the train dataset into train, valid dataset    

    ##seed fix
    seed_everything(config['train']['seed'])
     
    train_dataset = BinaryDataset(mode="train", config=config)
    valid_size = int(len(train_dataset) * config['train']['valid_ratio'])
    train_dataset, valid_dataset = random_split(train_dataset, [len(train_dataset)-valid_size, valid_size])

    
    
    
    ##test가 여기서 또 진행이 되는건가?? -> 현재는 None임
    # test_dataset = BinaryDataset(mode="test", config=config)
    
    # Create data loaders
    train_data_loader = DataLoader(train_dataset, batch_size=config['train']['batch_size'], shuffle=True)
    val_data_loader = DataLoader(valid_dataset, batch_size=config['train']['batch_size'])
    # test_data_loader = DataLoader(test_dataset, batch_size=config['train']['batch_size']) if test_dataset else None
    
    
    # Train the model 
    train(config, train_data_loader, val_data_loader) #test_data_loader)
    
    
    
    
    
    
    
    


    