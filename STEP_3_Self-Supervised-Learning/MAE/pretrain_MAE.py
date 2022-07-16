######### The Code is referred from https://github.com/facebookresearch/mae


## ======= load module ======= ##
import model.model_MAE as MAE
from envs.pretraining_experiments import MAE_experiment
from util.utils import CLIreporter, save_exp_result, checkpoint_save, checkpoint_load, set_random_seed
from dataloaders.dataloaders import loading_images,  partition_dataset_pretrain
from dataloaders.preprocessing import preprocessing_cat, preprocessing_num
from envs.pretraining_experiments import *
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
parser.add_argument("--val_size",default=0.1,type=float,required=False,help='')
parser.add_argument("--test_size",default=0.1,type=float,required=False,help='')
parser.add_argument("--img_size",default=96,type=int,nargs="*",required=False,help='')

 #########################
 ### batch size params ###
 #########################
parser.add_argument("--train_batch_size",default=16,type=int,required=False,help='')
parser.add_argument("--accumulation_steps",default=1,type=int,required=False,help='mini batch size == accumulation_steps * args.train_batch_size')

#########################
## MAE specific params #
#########################
parser.add_argument("--model",required=True,type=str,help='',choices=['mae_vit_base_patch16_3D','mae_vit_large_patch16_3D','mae_vit_huge_patch14_3D','mae_vit_base_patch16_3D','mae_vit_large_patch16_3D','mae_vit_huge_patch14_3D'])
parser.add_argument("--mask_ratio",required=False,default=0.75,type=float,help='the ratio of random masking')
parser.add_argument("--norm_pix_loss",action='store_true',help='Use (per-patch) normalized pixels as targets for computing loss')
parser.set_defaults(norm_pix_loss=False)

##########################
#### optim parameters ####
##########################
parser.add_argument("--optim",type=str,required=True,help='', choices=['Adam','SGD', 'LARS', 'LAMB'])
parser.add_argument("--lr", default=0.01,type=float,required=False,help='')
parser.add_argument("--weight_decay",default=0.05,type=float,required=False,help='')
parser.add_argument("--epoch",type=int,required=True,help='')
    
##########################
#### other parameters ####
##########################
parser.add_argument("--in_channels",default=1,type=int,required=False,help='')
parser.add_argument("--exp_name",type=str,required=True,help='')
parser.add_argument("--checkpoint_dir", type=str, default=None,required=False)
parser.add_argument("--resume", action='store_true', help = 'if you add this option in the command line like --resume, args.resume would change to be True')
parser.set_defaults(resume=False)
    
#########################
#### dist parameters ####
#########################
parser.add_argument("--gpus", type=int, nargs='*',required=False)
parser.add_argument("--sbatch", action='store_true')
parser.set_defaults(sbatch=False)


####global args
args = parser.parse_args()



if __name__ == "__main__":

    ## ========= Setting ========= ##
    current_dir = os.getcwd()
    image_dir = '/scratch/connectome/3DCNN/data/2.UKB/1.sMRI_fs_cropped'
    phenotype_dir = '/scratch/connectome/3DCNN/data/2.UKB/2.demo_qc/UKB_phenotype.csv'
    #image_dir = '/master_ssd/3DCNN/data/1.ABCD/2.sMRI_freesurfer'
    #phenotype_dir = '/master_ssd/3DCNN/data/1.ABCD/4.demo_qc/ABCD_phenotype_total.csv'    
    image_files = loading_images(image_dir, args)
    #subject_data, target_list = loading_phenotype(phenotype_dir, args)
    os.chdir(image_dir)
    ## ====================================== ##

    ## ========= data preprocesing categorical variable and numerical variables ========= ##
    #imageFiles_labels = combining_image_target(subject_data, image_files, target_list)

    # partitioning dataset and preprocessing (change the range of categorical variables and standardize numerical variables )
    partition = partition_dataset_pretrain(image_files,args)
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
    setting, result = MAE_experiment(partition, save_dir, deepcopy(args))
    save_exp_result(save_dir, setting, result)