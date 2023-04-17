import os
import glob
import time
import datetime
import random
import hashlib
import argparse
from copy import deepcopy
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm

from utils.utils import argument_setting, select_model, CLIreporter, save_exp_result, checkpoint_save, checkpoint_load
from dataloaders.dataloaders import make_dataset
from dataloaders.preprocessing import preprocessing_cat, preprocessing_num

def set_random_seed(SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

def argument_setting():
    parser = argparse.ArgumentParser()
    # Options for dataset and data type, split ratio, CV, resize, augmentation
    parser.add_argument("--in_channels", default=1, type=int, help='')

    # Options for dataset and data type, split ratio, CV, resize, augmentation
    parser.add_argument("--dataset", type=str, choices=['UKB','ABCD'], required=True, help='Selelct dataset')
    parser.add_argument("--data_type", nargs='+', type=str, help='Select data type(sMRI, dMRI)',
                        choices=['fmriprep', 'freesurfer', 'freesurfer_256', 'freesurfer_crop_resize128','T1_MNI_resize128',
                                 'FA_crop_resize128', 'FA_MNI_resize128', 'FA_wm_MNI_resize128',
                                 'FA_unwarpped_nii', 'FA_warpped_nii',
                                 'MD_unwarpped_nii', 'MD_warpped_nii', 'RD_unwarpped_nii', 'RD_warpped_nii',
                                 'T1_MNI_resize_areamode', 'FA_MNI_resize_areamode'])
    parser.add_argument("--balanced_split", type=str, help='')
    parser.add_argument("--N", default=None, type=int, help='')
    parser.add_argument("--tissue", default=None, type=str, help='Select tissue mask(Cortical grey matter, \
                        Sub-cortical grey matter, White matter, CSF, Pathological tissue)',
                        choices=['cgm', 'scgm', 'wm', 'csf', 'pt'])
    parser.add_argument("--metric", default='', type=str, help='')
    parser.add_argument("--val_size", default=0.1, type=float, help='')
    parser.add_argument("--test_size", default=0.1, type=float, help='')
    parser.add_argument("--cv", default=None, type=int, choices=[1,2,3,4,5], help="option for 5-fold CV. 1~5.")
    parser.add_argument("--resize", nargs="*", default=(96, 96, 96), type=int, help='')
    parser.add_argument("--transform", nargs="*", default=[], type=str, choices=['crop'],
                        help="option for additional transform - [crop] are available")
    parser.add_argument("--augmentation", nargs="*", default=[], type=str, choices=['shift','flip'],
                        help="Data augmentation - [shift, flip] are available")

    # Hyperparameters for model training
    parser.add_argument("--lr", default=0.01, type=float, help='')
    parser.add_argument("--lr_adjust", default=0.01, type=float, help='')
    parser.add_argument("--epoch_FC", type=int, default=0, help='Option for training only FC layer')
    parser.add_argument("--optim", default='Adam', type=str, choices=['Adam','SGD','RAdam','AdamW'], help='')
    parser.add_argument("--weight_decay", default=0.01, type=float, help='')
    parser.add_argument("--scheduler", default='', type=str, help='') 
    parser.add_argument("--early_stopping", default=None, type=int, help='')
    parser.add_argument("--num_workers", default=3, type=int, help='')
    parser.add_argument("--train_batch_size", default=16, type=int, help='')
    parser.add_argument("--val_batch_size", default=16, type=int, help='')
    parser.add_argument("--test_batch_size", default=1, type=int, help='')

    # Options for experiment setting
    parser.add_argument("--gpus", nargs='+', type=int, help='')
    parser.add_argument("--sbatch", type=str, choices=['True', 'False'])
    parser.add_argument("--cat_target", nargs='+', default=[], type=str, help='')
    parser.add_argument("--num_target", nargs='+', default=[], type=str, help='')
    parser.add_argument("--num_normalize", type=str, default=True, help='')
    parser.add_argument("--confusion_matrix",  nargs='*', default=[], type=str, help='')
    parser.add_argument("--filter", nargs="*", default=[], type=str,
                        help='options for filter data by phenotype. usage: --filter abcd_site:10 sex:1')
    parser.add_argument("--mode", default='pretraining', type=str,  choices=['pretraining','finetuning','transfer'],
                        help='Option for learning from scratch')
    parser.add_argument("--load", default='', type=str, help='Load model weight that mathces {your_exp_dir}/result/*{load}*')
    parser.add_argument("--unfrozen_layers", default='0', type=str, help='Select the number of layers that would be unfrozen')
    parser.add_argument("--init_unfrozen", default='', type=str, help='Initializes unfrozen layers')
    parser.add_argument("--debug", default='', type=str, help='')
    parser.add_argument("--phenotype", default='total', type=str, help='')

        
    args = parser.parse_args()
    print("Categorical target labels are {} and Numerical target labels are {}".format(args.cat_target, args.num_target))

    if not args.cat_target:
        args.cat_target = []
    elif not args.num_target:
        args.num_target = []
    elif not args.cat_target and args.num_target:
        raise ValueError('YOU SHOULD SELECT THE TARGET!')

    return args

args = argument_setting()
seed = 1234
set_random_seed(seed)
partition, subject_data = make_dataset(args)

for n in [0,1,2,3]:
    testloader = torch.utils.data.DataLoader(partition['test'],
                                            batch_size=args.train_batch_size,
                                            shuffle=False,
                                            pin_memory=False,
                                            num_workers=n)   
    ts = time.time()
    for i, data in enumerate(testloader):
        d = data
        if i == 2:
            break
    te = time.time()
    print(f'Num worker={n:2d},\tTime spent: {te-ts:.4f} s')

    
for n in [0,1,2,3]:
    testloader = torch.utils.data.DataLoader(partition['test'],
                                            batch_size=args.train_batch_size,
                                            shuffle=False,
                                            pin_memory=True,
                                            num_workers=n)   
    ts = time.time()
    for i, data in enumerate(testloader):
        d = data
        if i == 2:
            break
    te = time.time()
    print(f'Num worker={n:2d},\tTime spent: {te-ts:.4f} s,\tpin_memory=True,')    

    
for n in [0,1,2,3]:
    testloader = torch.utils.data.DataLoader(partition['test'],
                                            batch_size=args.train_batch_size,
                                            shuffle=False,
                                            pin_memory=False,
                                            num_workers=n,
                                            persistent_workers=(n>0))   
    ts = time.time()
    for i, data in enumerate(testloader):
        d = data
        if i == 2:
            break
    te = time.time()
    print(f'Num worker={n:2d},\tTime spent: {te-ts:.4f} s,\tpersistent_worker=True')

    
for n in [0,1,2,3]:
    testloader = torch.utils.data.DataLoader(partition['test'],
                                            batch_size=args.train_batch_size,
                                            shuffle=False,
                                            pin_memory=True,
                                            num_workers=n,
                                            persistent_workers=(n>0))   
    ts = time.time()
    for i, data in enumerate(testloader):
        d = data
        if i == 2:
            break
    te = time.time()
    print(f'Num worker={n:2d},\tTime spent: {te-ts:.4f} s,\tpin_memory=True,\tpersistent_worker=True')    

    