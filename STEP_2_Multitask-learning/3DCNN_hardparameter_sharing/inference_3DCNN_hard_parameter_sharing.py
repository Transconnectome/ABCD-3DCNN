## ======= load module ======= ##
import models.simple3d as simple3d #model script
import models.vgg3d as vgg3d #model script
import models.resnet3d as resnet3d #model script
import models.densenet3d as densenet3d #model script
from utils.utils import CLIreporter, save_exp_result, checkpoint_save, checkpoint_load
from utils.lr_scheduler import *
from dataloaders.dataloaders import check_study_sample, loading_images, loading_phenotype, combining_image_target, partition_dataset
from dataloaders.preprocessing import preprocessing_cat, preprocessing_num
from envs.experiments import train, validate, test 
import hashlib
import datetime




import os
from os import listdir
from os.path import isfile, join
from tqdm.auto import tqdm ##progress
import time
import math
import random
import argparse

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
from copy import deepcopy

import warnings
warnings.filterwarnings("ignore")

def argument_setting():
    parser = argparse.ArgumentParser()

    #parser.add_argument("--GPU_NUM",default=1,type=int,required=True,help='')
    parser.add_argument("--study_sample",default='UKB',type=str,required=False,help='')
    parser.add_argument("--model",required=True,type=str,help='',choices=['simple3D','vgg3D11','vgg3D13','vgg3D16','vgg3D19','resnet3D50','resnet3D101','resnet3D152', 'densenet3D121', 'densenet3D169','densenet201','densenet264'])
    parser.add_argument("--val_size",default=0.1,type=float,required=False,help='')
    parser.add_argument("--test_size",default=0.1,type=float,required=False,help='')
    parser.add_argument("--resize",default=[96, 96, 96],type=int,nargs="*",required=False,help='')
    parser.add_argument("--batch_size",default=16,type=int,required=False,help='')
    parser.add_argument("--in_channels",default=1,type=int,required=False,help='')
    parser.add_argument("--exp_name",type=str,required=True,help='')
    parser.add_argument("--cat_target", type=str, nargs='*', required=False, help='')
    parser.add_argument("--num_target", type=str,nargs='*', required=False, help='')
    parser.add_argument("--confusion_matrix", type=str, nargs='*',required=False, help='')
    parser.add_argument("--gpus", type=int,nargs='*', required=False, help='')
    parser.add_argument("--sbatch", type=str, required=False, choices=['True', 'False'])
    parser.add_argument("--checkpoint_dir", type=str, default=None,required=False)
    parser.add_argument('--get_predicted_score', action='store_true', help='save the result of inference in the result file')
    parser.set_defaults(get_predicted_score=False)

    args = parser.parse_args()
    print("Categorical target labels are {} and Numerical target labels are {}".format(args.cat_target, args.num_target))

    if not args.cat_target:
        args.cat_target = []
    elif not args.num_target:
        args.num_target = []
    elif not args.cat_target and args.num_target:
        raise ValueError('YOU SHOULD SELECT THE TARGET!')

    return args

## ========= Experiment =============== ##
def inference_engine(partition, subject_data, save_dir, args): #in_channels,out_dim
    targets = args.cat_target + args.num_target

    # Simple CNN
    if args.model == 'simple3D':
        assert args.resize == [96, 96, 96]
        net = simple3d.simple3D(subject_data, args)
    # VGGNet
    elif args.model == 'vgg3D11':
        assert args.resize == [96, 96, 96]
        net = vgg3d.vgg3D11(subject_data, args)
    elif args.model == 'vgg3D13':
        assert args.resize == [96, 96, 96]
        net = vgg3d.vgg3D13(subject_data, args)
    elif args.model == 'vgg3D16':
        assert args.resize == [96, 96, 96]
        net = vgg3d.vgg3D16(subject_data, args)
    elif args.model == 'vgg3D19':
        assert args.resize == [96, 96, 96]
        net = vgg3d.vgg3D19(subject_data, args)
    # ResNet
    elif args.model == 'resnet3D50':
        net = resnet3d.resnet3D50(subject_data, args)
    elif args.model == 'resnet3D101':
        net = resnet3d.resnet3D101(subject_data, args)
    elif args.model == 'resnet3D152':
        net = resnet3d.resnet3D152(subject_data, args)
    # DenseNet
    elif args.model == 'densenet3D121':
        net = densenet3d.densenet3D121(subject_data, args)
    elif args.model == 'densenet3D161':
        net = densenet3d.densenet3D161(subject_data, args) 
    elif args.model == 'densenet3D169':
        net = densenet3d.densenet3D169(subject_data, args) 
    elif args.model == 'densenet3D201':
        net = densenet3d.densenet3D201(subject_data, args)


    # test
    net.to('cpu')
    torch.cuda.empty_cache()

    assert args.checkpoint_dir
    net = checkpoint_load(net, args.checkpoint_dir)
    if args.sbatch == 'True':
        net.cuda()
    else:
        net.to(f'cuda:{args.gpus[0]}')
    test_result, confusion_matrices = test(net, partition, args)

    # summarize results
    result = {}
    result.update(test_result)
    
    
    if confusion_matrices != None:
        result['confusion_matrices'] = confusion_matrices

    return vars(args), result
## ==================================== ##



if __name__ == "__main__":

    ## ========= Setting ========= ##
    args = argument_setting()
    current_dir = os.getcwd()
    image_dir, phenotype_dir = check_study_sample(study_sample=args.study_sample)
    image_files = loading_images(image_dir=image_dir, args=args, study_sample=args.study_sample)
    subject_data, target_list, num_classes = loading_phenotype(phenotype_dir=phenotype_dir, args=args, study_sample=args.study_sample)
    
    ## data preprocesing categorical variable and numerical variables
    imageFiles_labels = combining_image_target(subject_data, image_files, target_list, study_sample=args.study_sample)

    # partitioning dataset and preprocessing (change the range of categorical variables and standardize numerical variables )
    partition = partition_dataset(imageFiles_labels,args)
    ## ====================================== ##


    ## ========= Run Experiment and saving result ========= ##
    # seed number
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    save_dir = current_dir + '/result'
    
    time_hash = datetime.datetime.now().time()
    hash_key = hashlib.sha1(str(time_hash).encode()).hexdigest()[:6]
    args.exp_name = args.exp_name + f'_{hash_key}'


    # Run Experiment
    setting, result = inference_engine(partition, subject_data, save_dir, deepcopy(args))

    # Save result
    save_exp_result(save_dir, setting, result)
    ## ====================================== ##
