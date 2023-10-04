import os
import sys
import numpy as np
import pandas as pd
import torch
from trainer import Trainer
from easydict import EasyDict
from model.meta import PoolFormer
from dataloader import DisDataset

train_df = pd.read_csv(r"train_df.csv")
valid_df = pd.read_csv(r"valid_df.csv")
save_path = os.getcwd()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

args = EasyDict(
    {
     # Path settings
     'train_dir':'trainset',
     'valid_dir':'validset',
     'save_dict' : 'save_model',
     'train_df':train_df,
     'valid_df':valid_df,
     
     # Model parameter settings
     'CODER':'poolformer_m36',
     'drop_path_rate':0.2,
     'model_class': PoolFormer,
     'weight':None,
     'pretrained':False,
     
     # Training parameter settings
     ## Base Parameter
     'img_size':224,
     'test_size':224,
     'BATCH_SIZE':100,
     'epochs':200,
     'optimizer':'Lamb',
     'lr':3e-5,
     'weight_decay':1e-3,
     'Dataset' : DisDataset,
     'aware':False,
     'fold_num':1,
     'bagging_num':4,
     'aug':False,
     #scheduler 
     'scheduler':'cos',
     ## Scheduler (OnecycleLR)
     'warm_epoch':5,
     'max_lr':1e-3,

     ### Cosine Annealing
     'min_lr':5e-6,
     'tmax':145,

     ## etc.
     'patience':3,
     'clipping':None,

     # Hardware settings
     'amp':True,
     'multi_gpu':False,
     'logging':False,
     'num_workers':4,
     'seed':42,
     'device':device,

    })

if __name__ == '__main__': 
    # seed_everything(np.random.randint(1, 5000))
    print(args.CODER + " train..")
    trainer = Trainer(args)
    # trainer = Mixup_trainer(args)
    trainer.fit()