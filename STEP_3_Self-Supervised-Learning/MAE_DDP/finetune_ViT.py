######### The Code is referred from https://github.com/facebookresearch/mae
######### DDP reference: https://towardsdatascience.com/distribute-your-pytorch-model-in-less-than-20-lines-of-code-61a786e6e7b0

## ======= load module ======= ##
import model.model_ViT as ViT
from util.utils import CLIreporter, save_exp_result, checkpoint_save, checkpoint_load, set_random_seed
from dataloaders.dataloaders import check_study_sample,loading_images, loading_phenotype, combining_image_target, partition_dataset_finetuning,  partition_dataset_pretrain
from dataloaders.preprocessing import preprocessing_cat, preprocessing_num
from envs.finetuning_experiments import *
from util.distributed_parallel import *
import hashlib
import datetime


import os
from os import listdir
from os.path import isfile, join
from tqdm.auto import tqdm ##progress
import time
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, utils
import torch.optim as optim
from torchsummary import summary


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

import random
import copy
from copy import deepcopy
import argparse

import warnings
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser()

#########################
#### data parameters ####
#########################
parser.add_argument("--study_sample",default='UKB',type=str,required=False,help='')
parser.add_argument("--val_size",default=0.1,type=float,required=False,help='')
parser.add_argument("--test_size",default=0.1,type=float,required=False,help='')
parser.add_argument("--img_size",default=[96, 96, 96] ,type=int,nargs="*",required=False,help='')
parser.add_argument("--mixup",default=None,type=float,required=False,help='')


#############################
#### finetune parameters ####
#############################
parser.add_argument('--global_pool', action='store_true')
parser.set_defaults(global_pool=True)
parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')


#########################
#### task parameters ####
#########################
parser.add_argument("--cat_target", type=str, nargs='*', required=False, help='')
parser.add_argument("--num_target", type=str,nargs='*', required=False, help='')


#########################
### batch size params ###
#########################
parser.add_argument("--batch_size",default=16,type=int,required=False,help='Total batch size. This batch size would be divided by the number of (DDP) proccesses.')
parser.add_argument("--accumulation_steps",default=1,type=int,required=False,help='mini batch size == accumulation_steps * args.train_batch_size')


#########################
## ViT specific params #
#########################
parser.add_argument("--model",required=True,type=str,help='',choices=['vit_base_patch16_3D','vit_large_patch16_3D','vit_huge_patch14_3D','vit_base_patch16_3D','vit_large_patch16_3D','vit_huge_patch14_3D'])
parser.add_argument("--attention_drop",default=0.5,type=float,required=False,help='dropout rate of encoder attention layer')
parser.add_argument("--projection_drop",default=0.5,type=float,required=False,help='dropout rate of encoder projection layer')
parser.add_argument("--path_drop",default=0.0,type=float,required=False,help='dropout rate of encoder attention block')
parser.add_argument("--use_rel_pos_bias",action='store_true',help='Use relative positional bias for positional encoding')
parser.set_defaults(use_rel_pos_bias=False)
parser.add_argument("--use_sincos_pos",action='store_true',help='Use relative positional bias for positional encoding')
parser.set_defaults(use_sincos_pos=False)
#parser.add_argument("--mask_ratio",required=False,default=0.75,type=float,help='the ratio of random masking')
#parser.add_argument("--norm_pix_loss",action='store_true',help='Use (per-patch) normalized pixels as targets for computing loss')
#parser.set_defaults(norm_pix_loss=False)

##########################
#### optim parameters ####
##########################
parser.add_argument("--optim",type=str,required=True,help='', choices=['Adam','AdamW','SGD', 'LARS', 'LAMB'])
parser.add_argument("--lr", default=0.01,type=float,required=False,help='')
parser.add_argument("--weight_decay",default=0.05,type=float,required=False,help='')
parser.add_argument("--epoch",type=int,required=True,help='')
parser.add_argument('--gradient_clipping', action='store_true')
parser.set_defaults(gradient_accumulation=False)
    
##########################
#### other parameters ####
##########################
parser.add_argument("--in_channels",default=1,type=int,required=False,help='')
parser.add_argument("--exp_name",type=str,required=True,help='')
parser.add_argument('--pretrained_model', type=str, default=None, required=False)
parser.add_argument("--checkpoint_dir", type=str, default=None,required=False)
parser.add_argument("--load_imagenet_pretrained", action='store_true', help = 'load imagenet pretrained model')
parser.set_defaults(load_imagenet_pretrained=False)
parser.add_argument("--resume", action='store_true', help = 'if you add this option in the command line like --resume, args.resume would change to be True')
parser.set_defaults(resume=False)
    
#########################
#### dist parameters ####
#########################
parser.add_argument("--sbatch", action='store_true')
parser.set_defaults(sbatch=False)
parser.add_argument("--world_size", type=int,  default = -1)
parser.add_argument("--rank", type=int, default=-1)
parser.add_argument("--local_rank", type=int, default=-1)



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
    image_files = loading_images(image_dir, args, study_sample=args.study_sample)
    subject_data, target_list, num_classes = loading_phenotype(phenotype_dir, args, study_sample=args.study_sample)
    
    ## data preprocesing categorical variable and numerical variables
    imageFiles_labels = combining_image_target(subject_data, image_files, target_list, study_sample=args.study_sample)

    # partitioning dataset and preprocessing (change the range of categorical variables and standardize numerical variables )
    partition = partition_dataset_finetuning(imageFiles_labels, args)
    ## ====================================== ##


    ## ========= Run Experiment and saving result ========= ##
    # seed number
    seed = 1234

    # initialize Distributed Data Parallel and divide batch size by the number of (DDP) proccesses
    init_distributed(args)
    args.batch_size = args.batch_size // args.world_size 

    ######init_distributed_mode(args)
    set_random_seed(seed)
    save_dir = current_dir + '/result'
    
    time_hash = datetime.datetime.now().time()
    hash_key = hashlib.sha1(str(time_hash).encode()).hexdigest()[:6]
    args.exp_name = args.exp_name + f'_{hash_key}'


    # Run MAE Experiment
    torch.backends.cudnn.benchmark = True
    setting, result = ViT_experiment(partition, num_classes, save_dir, deepcopy(args))
    save_exp_result(save_dir, setting, result)