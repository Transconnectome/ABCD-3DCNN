import os
from os import listdir
from os.path import isfile, join
import time
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, utils
import torch.optim as optim


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

import random
import copy
from copy import deepcopy
import argparse
import datetime
import hashlib

from envs.inference_engine import inference_engine
from util.utils import save_exp_result, set_random_seed
from dataloaders.dataloaders import check_study_sample,loading_images, loading_phenotype, combining_image_target, partition_dataset_finetuning,  partition_dataset_pretrain


parser = argparse.ArgumentParser()

#########################
#### data parameters ####
#########################
parser.add_argument('--include_synthetic', action='store_true')
parser.set_defaults(include_synthetic=False)
parser.add_argument("--study_sample",default='UKB',type=str, required=False,help='')
parser.add_argument("--train_size",default=0.8,type=float,required=False,help='')
parser.add_argument("--val_size",default=0.1,type=float,required=False,help='')
parser.add_argument("--test_size",default=0.1,type=float,required=False,help='')
parser.add_argument("--img_size",default=[96, 96, 96] ,type=int,nargs="*",required=False,help='')


#########################
### batch size params ###
#########################
parser.add_argument("--batch_size",default=16,type=int,required=False,help='Total batch size. This batch size would be divided by the number of (DDP) proccesses.')


#########################
#### task parameters ####
#########################
parser.add_argument("--cat_target", type=str, nargs='*', required=False, help='')
parser.add_argument("--num_target", type=str, nargs='*', required=False, help='')


##################
## model params ##
##################
parser.add_argument("--model",required=True,type=str,help='',choices=['swin_small_3D', 'swin_base_3D', 'swin_large_3D','vit_base_patch16_3D', 'vit_large_patch16_3D', 'vit_huge_patch16_3D'])
parser.add_argument("--attention_drop",default=0.5,type=float,required=False,help='dropout rate of encoder attention layer')
parser.add_argument("--projection_drop",default=0.5,type=float,required=False,help='dropout rate of encoder projection layer')
parser.add_argument("--path_drop",default=0.0,type=float,required=False,help='dropout rate of encoder attention block')
# Swin specific parameters
parser.add_argument("--window_size",default=8,type=int,required=False,help='The size of window.')
# ViT specific parameters 
parser.add_argument("--model_patch_size",default=16,type=int,required=False,help='The size of model patch used for patch emebdding.')
parser.add_argument("--use_rel_pos_bias",action='store_true',help='Use relative positional bias for positional encoding')
parser.set_defaults(use_rel_pos_bias=False)
parser.add_argument("--use_sincos_pos",action='store_true',help='Use relative positional bias for positional encoding')
parser.set_defaults(use_sincos_pos=False)
    
##########################
#### other parameters ####
##########################
parser.add_argument("--torchscript",action='store_true', help = 'if you want to activate kernel fusion activate this option')
parser.set_defaults(torchscript=False)
parser.add_argument("--exp_name",type=str,required=True,help='')
parser.set_defaults(load_imagenet_pretrained=False)
parser.add_argument("--checkpoint_dir", type=str, default=None,required=True)
parser.add_argument("--use_gpu", action='store_true')
parser.set_defaults(use_gpu=False)




####global args
args = parser.parse_args()


if not args.cat_target:
    args.cat_target = []
    print("This experiment predicts {}.".format(args.num_target))
elif not args.num_target:
    args.num_target = []
    print("This experiment predicts {}.".format(args.cat_target))
elif not args.cat_target and args.num_target:
       raise ValueError('YOU SHOULD SELECT THE TARGET!')




if __name__ == "__main__":

    ## ========= Settingfor data ========= ##
    current_dir = os.getcwd()
    image_dir, phenotype_dir = check_study_sample(study_sample=args.study_sample)
    image_files, _ = loading_images(image_dir, args, study_sample=args.study_sample)
    subject_data, target_list, num_classes = loading_phenotype(phenotype_dir, args, study_sample=args.study_sample)
    
    ## data preprocesing categorical variable and numerical variables
    imageFiles_labels = combining_image_target(subject_data, image_files, target_list, study_sample=args.study_sample)

    # partitioning dataset and preprocessing (change the range of categorical variables and standardize numerical variables )
    partition = partition_dataset_finetuning(imageFiles_labels, args)
        

    ## ====================================== ##



    ## ========= Run Experiment and saving result ========= ##
    # seed number
    seed = 1234

    ######init_distributed_mode(args)
    set_random_seed(seed)
    save_dir = current_dir + '/result'
    
    time_hash = datetime.datetime.now().time()
    hash_key = hashlib.sha1(str(time_hash).encode()).hexdigest()[:6]
    args.exp_name = args.exp_name + f'_{hash_key}'

    # Run MAE Experiment
    torch.backends.cudnn.benchmark = True
    setting, result = inference_engine(partition, num_classes, save_dir, deepcopy(args))
    save_exp_result(save_dir, setting, result)