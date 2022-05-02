## ======= load module ======= ##
import models.resnet3d as resnet3d #model script
import models.densenet3d as densenet3d #model script
from utils.utils import argument_setting_finetuning, CLIreporter, save_exp_result, checkpoint_save, checkpoint_load
from dataloaders.dataloaders import loading_images, loading_phenotype, combining_image_target, partition_dataset_finetuning
from dataloaders.preprocessing import preprocessing_cat, preprocessing_num
from envs.finetuning_experiments import *
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

import warnings
warnings.filterwarnings("ignore")




if __name__ == "__main__":

    ## ========= Setting ========= ##
    args = argument_setting_finetuning()
    current_dir = os.getcwd()
    image_dir = '/master_ssd/3DCNN/data/2.UKB/1.sMRI_fs_cropped'
    phenotype_dir = '/master_ssd/3DCNN/data/2.UKB/2.demo_qc/UKB_phenotype.csv'
    #image_dir = '/master_ssd/3DCNN/data/1.ABCD/2.sMRI_freesurfer'
    #phenotype_dir = '/master_ssd/3DCNN/data/1.ABCD/4.demo_qc/ABCD_phenotype_total.csv'    
    image_files = loading_images(image_dir, args)
    subject_data, target_list = loading_phenotype(phenotype_dir, args)
    os.chdir(image_dir)
    ## ====================================== ##

    ## ========= data preprocesing categorical variable and numerical variables ========= ##
    imageFiles_labels = combining_image_target(subject_data, image_files, target_list)

    # partitioning dataset and preprocessing (change the range of categorical variables and standardize numerical variables )
    partition = partition_dataset_finetuning(imageFiles_labels,args)
    ## ====================================== ##


    ## ========= Run Experiment and saving result ========= ##
    # seed number
    seed = 1234
    np.random.seed(seed)
    torch.manual_seed(seed)
    save_dir = current_dir + '/result'
    
    time_hash = datetime.datetime.now().time()
    hash_key = hashlib.sha1(str(time_hash).encode()).hexdigest()[:6]
    args.exp_name = args.exp_name + f'_{hash_key}'


    # Run fine-tuning Experiment
    setting, result = finetuning_experiment(partition, subject_data, save_dir, deepcopy(args))

    # Save result
    save_exp_result(save_dir, setting, result)
    ## ====================================== ##
