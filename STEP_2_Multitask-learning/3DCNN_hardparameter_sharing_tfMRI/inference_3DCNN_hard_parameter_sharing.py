## ======= load module ======= ##
import models.simple3d as simple3d #model script
import models.vgg3d as vgg3d #model script
import models.resnet3d as resnet3d #model script
import models.densenet3d as densenet3d #model script
import models.efficientnet3d as efficientnet3d
from utils.utils import set_random_seed, CLIreporter, save_exp_result, checkpoint_save, checkpoint_load
from utils.lr_scheduler import *
from dataloaders.dataloaders import check_study_sample, loading_images, loading_phenotype, combining_image_target, partition_dataset, matching_partition_dataset, undersampling_ALLset, partition_dataset_predefined
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
    parser.add_argument("--model",required=True,type=str,help='',choices=[
                                                                        'densenet3D121', 'densenet3D169','densenet201','densenet264',
                                                                        'densenet3D121_cbam', 'densenet3D169_cbam','densenet3D201_cbam','densenet3D264_cbam', 
                                                                        'flipout_densenet3D121', 'flipout_densenet3D169','flipout_densenet3D201','flipout_densenet3D264', 
                                                                        'variational_densenet3D121', 'variational_densenet3D169','variational_densenet3D201','variational_densenet3D264',   
                                                                        'efficientnet3D-b0','efficientnet3D-b1','efficientnet3D-b2','efficientnet3D-b3','efficientnet3D-b4','efficientnet3D-b5','efficientnet3D-b6','efficientnet3D-b7'
                                                                        ])
    parser.add_argument("--train_size",default=0.8,type=float,required=False,help='')
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
    parser.add_argument("--checkpoint_dir", type=str, default=None,required=False)
    parser.add_argument('--get_predicted_score', action='store_true', help='save the result of inference in the result file')
    parser.set_defaults(get_predicted_score=False)
    parser.add_argument('--matching_baseline_2years', action='store_true', help='save the result of inference in the result file')
    parser.set_defaults(matching_baseline_2years=False)
    parser.add_argument('--matching_baseline_gps', action='store_true', help='save the result of inference in the result file')
    parser.set_defaults(matching_baseline_gps=False)
    parser.add_argument("--undersampling_dataset_target", type=str, default=None, required=False, help='')
    parser.add_argument('--partitioned_dataset_number', default=None, type=int, required=False)

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
    # DenseNet
    if args.model == 'densenet3D121':
        import models.densenet3d as densenet3d #model script
        from envs.experiments import train, validate, test 
        net = densenet3d.densenet3D121(subject_data, args)
    elif args.model == 'densenet3D169':
        import models.densenet3d as densenet3d #model script
        from envs.experiments import train, validate, test 
        net = densenet3d.densenet3D169(subject_data, args) 
    elif args.model == 'densenet3D201':
        import models.densenet3d as densenet3d #model script
        from envs.experiments import train, validate, test 
        net = densenet3d.densenet3D201(subject_data, args)
    # DenseNet with CBAM module
    if args.model == 'densenet3D121_cbam':
        import models.densenet3d_cbam as densenet3d_cbam #model script
        from envs.experiments import train, validate, test 
        net = densenet3d_cbam.densenet3D121_cbam(subject_data, args)
    elif args.model == 'densenet3D169_cbam':
        import models.densenet3d_cbam as densenet3d_cbam #model script
        from envs.experiments import train, validate, test 
        net = densenet3d_cbam.densenet3D169_cbam(subject_data, args) 
    elif args.model == 'densenet3D201_cbam':
        import models.densenet3d_cbam as densenet3d_cbam #model script
        from envs.experiments import train, validate, test 
        net = densenet3d_cbam.densenet3D201_cbam(subject_data, args)
    # Bayesian (variational) DenseNet
    elif args.model == 'variational_densenet3D121':
        import models.variational_densenet3d as variational_densenet3d
        from envs.bayesian_experiments import train, validate, test 
        net = variational_densenet3d.densenet3D121(subject_data, args)
    elif args.model == 'variational_densenet3D169':
        import models.flipout_densenet3d as variational_densenet3d
        from envs.bayesian_experiments import train, validate, test 
        net = variational_densenet3d.densenet3D169(subject_data, args) 
    elif args.model == 'variational_densenet3D201':
        import models.flipout_densenet3d as variational_densenet3d
        from envs.bayesian_experiments import train, validate, test 
        net = variational_densenet3d.densenet3D201(subject_data, args)
    # Bayesian (flipout) DenseNet
    elif args.model == 'flipout_densenet3D121':
        import models.flipout_densenet3d as flipout_densenet3d
        from envs.bayesian_experiments import train, validate, test 
        net = flipout_densenet3d.densenet3D121(subject_data, args)
    elif args.model == 'flipout_densenet3D169':
        import models.flipout_densenet3d as flipout_densenet3d
        from envs.bayesian_experiments import train, validate, test 
        net = flipout_densenet3d.densenet3D169(subject_data, args) 
    elif args.model == 'flipout_densenet3D201':
        import models.flipout_densenet3d as flipout_densenet3d
        from envs.bayesian_experiments import train, validate, test 
        net = flipout_densenet3d.densenet3D201(subject_data, args)

    # test
    net.to('cpu')
    torch.cuda.empty_cache()

    assert args.checkpoint_dir
    net = checkpoint_load(net, args.checkpoint_dir)

    # setting DataParallel
    devices = []
    for d in range(torch.cuda.device_count()):
        devices.append(d)
    net = nn.DataParallel(net, device_ids = devices)
    net.cuda()

    if args.model.find('flipout_') != -1: 
        test_result, confusion_matrices, pred_score = test(net, partition, args, num_monete_carlo=50)
    else: 
        test_result, confusion_matrices, pred_score = test(net, partition, args)
    

    # summarize results
    result = {}
    result.update(test_result)
    if args.get_predicted_score:
        result.update(pred_score)
    
    
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

    if args.undersampling_dataset_target: 
        subject_data, target_list = loading_phenotype(phenotype_dir=phenotype_dir, args=args, study_sample=args.study_sample, undersampling_dataset_target=args.undersampling_dataset_target)
        ## data preprocesing categorical variable and numerical variables
        imageFiles_labels = combining_image_target(subject_data, image_files, target_list, undersampling_dataset_target=args.undersampling_dataset_target, study_sample=args.study_sample)
        partition = undersampling_ALLset(imageFiles_labels, target_list, undersampling_dataset_target=args.undersampling_dataset_target ,args=args)
    elif args.partitioned_dataset_number is not None: 
        subject_data, target_list = loading_phenotype(phenotype_dir=phenotype_dir, args=args, study_sample=args.study_sample, partitioned_dataset_number=args.partitioned_dataset_number)
        ## data preprocesing categorical variable and numerical variables
        imageFiles_labels = combining_image_target(subject_data, image_files, target_list, partitioned_dataset_number=args.partitioned_dataset_number, study_sample=args.study_sample)
        partition = partition_dataset_predefined(imageFiles_labels, target_list, partitioned_dataset_number=args.partitioned_dataset_number, args=args)
    else:
        subject_data, target_list = loading_phenotype(phenotype_dir=phenotype_dir, args=args, study_sample=args.study_sample)
        ## data preprocesing categorical variable and numerical variables
        imageFiles_labels = combining_image_target(subject_data, image_files, target_list, study_sample=args.study_sample)
        partition = partition_dataset(imageFiles_labels, target_list, args=args)
    ## ====================================== ##


    ## ========= Run Experiment and saving result ========= ##
    # seed number
    seed = 1234
    set_random_seed(seed)
    save_dir = current_dir + '/result'
    
    time_hash = datetime.datetime.now().time()
    hash_key = hashlib.sha1(str(time_hash).encode()).hexdigest()[:6]
    args.exp_name = args.exp_name + f'_{hash_key}'

    setting, result = inference_engine(partition, subject_data, save_dir, deepcopy(args))
    # Save result
    save_exp_result(save_dir, setting, result)
    ## ====================================== ##
