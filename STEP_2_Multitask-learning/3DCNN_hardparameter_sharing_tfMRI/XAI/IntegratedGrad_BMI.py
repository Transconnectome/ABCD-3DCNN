### load library
#from importlib.metadata import files
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from copy import deepcopy # Add Deepcopy for args

import numpy as np
import pandas as pd
import seaborn as sns # visualization
import matplotlib.pyplot as plt # graph
import sklearn

import os
import glob
import sys
import argparse
import time
from tqdm.auto import tqdm # process bar
import random
import datetime
import hashlib
import json
from typing import Dict

from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader


import models_wrapper as densenet3d #model script
import models.efficientnet3d as efficientnet3d

from utils.utils import set_random_seed, checkpoint_load, makedir
from dataloaders.dataloaders import check_study_sample, loading_images, loading_phenotype, combining_image_target, partition_dataset, matching_partition_dataset, undersampling_ALLset, partition_dataset_predefined

#from captum.attr import IntegratedGradients, NoiseTunnel
from captum.attr import IntegratedGradients
from XAI.custom_noise_tunnel import NoiseTunnel


import warnings
warnings.filterwarnings("ignore")

## arguments
def argument_setting():
    parser = argparse.ArgumentParser()

    #parser.add_argument("--GPU_NUM",default=1,type=int,required=True,help='')
    parser.add_argument("--study_sample",default='UKB',type=str,required=False,help='')
    parser.add_argument("--model",required=True,type=str,help='',choices=[
                                                                        'densenet3D121', 'densenet3D169','densenet201','densenet264', 
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
    parser.add_argument('--partitioned_dataset_number', default=0, type=int, required=False)

    args = parser.parse_args()
    print("Categorical target labels are {} and Numerical target labels are {}".format(args.cat_target, args.num_target))

    if not args.cat_target:
        args.cat_target = []
    elif not args.num_target:
        args.num_target = []
    elif not args.cat_target and args.num_target:
        raise ValueError('YOU SHOULD SELECT THE TARGET!')

    return args


def get_config(config_dir):
    with open(config_dir, 'r') as file:
        config = json.load(file)
    print("Configuration of IntegratedGrad is as follow.{}".format(config))
    return config


def get_subject_list(dataset_subjid):
    subject_list = []
    for i, subj in enumerate(dataset_subjid):
        _, subj = os.path.split(subj)
        if subj.find('.nii.gz') != -1: 
            subj = subj.replace('.nii.gz','')
        elif subj.find('.npy') != -1: 
            subj = subj.replace('.npy','')
        subject_list.append(subj) 
    return subject_list

def save_attribute(attr_outputs: np.ndarray, subject_list: list, target_name, save_dir, args): 
    if args.study_sample.find('female') != -1: 
        attr_save_dir = os.path.join(*[save_dir, target_name, 'female'])
    elif args.study_sample.find('male') != -1: 
        attr_save_dir = os.path.join(*[save_dir, target_name, 'male'])   
    else:  
        if args.partitioned_dataset_number is not None: 
            attr_save_dir = os.path.join(*[save_dir, target_name, 'partition%s' % args.partitioned_dataset_number]) 
        else:
            attr_save_dir = os.path.join(save_dir, target_name)
    makedir(attr_save_dir)
    print(len(attr_outputs), len(subject_list))
    assert len(attr_outputs) == len(subject_list)
    for i, subject_id in enumerate(subject_list):
        file_name = os.path.join(attr_save_dir, subject_id + '.npy')
        np.save(file_name, attr_outputs[i])
    



def XAI_engine(interpreter, dataset, ig_config, args):
    testloader = torch.utils.data.DataLoader(dataset,
                                            batch_size=args.batch_size,
                                            shuffle=False,
                                            num_workers=2)

    

    attr_outputs = {}

    if args.cat_target:
        for cat_target in args.cat_target:
            attr_outputs[cat_target] = torch.tensor([])

    if args.num_target:
        for num_target in args.num_target:
            attr_outputs[num_target] = torch.tensor([])


    for idx, data in enumerate(tqdm(testloader,0)):
        image, targets = data       
        image = image.cuda()

        if args.cat_target:
            for cat_target in args.cat_target: 
                attr = interpreter.attribute(image.cuda(), target=targets[cat_target].long().cuda(), internal_batch_size=args.batch_size, nt_samples=ig_config['nt_samples'], nt_samples_batch_size=ig_config['nt_samples_batch_size'], stdevs=ig_config['stdevs'], nt_type=ig_config['nt_type'])
                attr_outputs[cat_target] = torch.cat([attr_outputs[cat_target], attr.detach().cpu()])
        if args.num_target:
            for num_target in args.num_target:
                attr = interpreter.attribute(image.cuda(), internal_batch_size=args.batch_size, nt_samples=ig_config['nt_samples'], nt_samples_batch_size=ig_config['nt_samples_batch_size'], stdevs=ig_config['stdevs'], nt_type=ig_config['nt_type'])
                attr_outputs[num_target] = torch.cat([attr_outputs[num_target], attr.detach().cpu()])
        



    if args.cat_target:
        for cat_target in args.cat_target: 
            attr_outputs[cat_target] = attr_outputs[cat_target].squeeze(1).numpy() # remove channel dimension and change into np.ndarray
    if args.num_target:
        for num_target in args.num_target:
            attr_outputs[num_target] = attr_outputs[num_target].squeeze(1).numpy()  # remove channel dimension and change into np.ndarray

    return attr_outputs



def XAI_experiments(partition, subject_data, save_dir, config_dir, args):
    targets = args.cat_target + args.num_target
    # DenseNet
    if args.model == 'densenet3D121':
        import models_wrapper as densenet3d #model script
        from envs.experiments import train, validate, test 
        net = densenet3d.densenet3D121(subject_data, args)
    elif args.model == 'densenet3D169':
        import models_wrapper as densenet3d #model script
        from envs.experiments import train, validate, test 
        net = densenet3d.densenet3D169(subject_data, args) 
    elif args.model == 'densenet3D201':
        import models_wrapper as densenet3d #model script
        from envs.experiments import train, validate, test 
        net = densenet3d.densenet3D201(subject_data, args)
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

    # load checkpoint and attach module to GPU 
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
    net.eval()

    ig_config = get_config(config_dir=config_dir)
    
    # setting for Integrated Grad
    interpreter = IntegratedGradients(net)
    interpreter = NoiseTunnel(interpreter) # if you don't want to use noise tunneling, plz deactivate this line

    # getting feature map 
    dataset = partition['test']
    subject_list = get_subject_list(dataset.image_files)
    attr_outputs = XAI_engine(interpreter=interpreter, dataset=dataset, ig_config=ig_config, args=args)

    
    if args.cat_target:
        for cat_target in args.cat_target: 
            save_attribute(attr_outputs=attr_outputs[cat_target], subject_list=subject_list, target_name=cat_target, save_dir=save_dir, args=args)
    if args.num_target:
        for num_target in args.num_target:
            save_attribute(attr_outputs=attr_outputs[num_target], subject_list=subject_list, target_name=num_target, save_dir=save_dir, args=args)
    


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
    save_dir = current_dir + '/result' + '/attribute'
    
    time_hash = datetime.datetime.now().time()
    hash_key = hashlib.sha1(str(time_hash).encode()).hexdigest()[:6]
    args.exp_name = args.exp_name + f'_{hash_key}'

    config_dir = current_dir + '/config.json'
    

    # Run Experiment
    XAI_experiments(partition, subject_data, save_dir, config_dir, args)



