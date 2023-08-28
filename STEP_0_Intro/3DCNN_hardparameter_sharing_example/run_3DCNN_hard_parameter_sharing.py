## ======= load module ======= ##
from utils.utils import set_random_seed,save_exp_result
from dataloaders.dataloaders import check_study_sample, loading_images, loading_phenotype, combining_image_target, partition_dataset, partition_dataset_predefined
from envs.experiments import experiment 
import hashlib
import datetime

import os
from os import listdir
from os.path import isfile, join
from tqdm.auto import tqdm ##progress
import argparse
from copy import deepcopy

import warnings
warnings.filterwarnings("ignore")


def argument_setting():
    parser = argparse.ArgumentParser()
    # arguments for dataset 
    parser.add_argument("--study_sample",default='UKB',type=str,required=False,help='')
    parser.add_argument('--partitioned_dataset_number', default=None, type=int, required=False)
    parser.add_argument("--train_size",default=0.8,type=float,required=False,help='')
    parser.add_argument("--val_size",default=0.1,type=float,required=False,help='')
    parser.add_argument("--test_size",default=0.1,type=float,required=False,help='')
    parser.add_argument("--resize",default=[96, 96, 96],type=int,nargs="*",required=False,help='')
    # arguments for model 
    parser.add_argument("--in_channels",default=1,type=int,required=False,help='')
    parser.add_argument("--model",required=True,type=str,help='',choices=[
                                                                        'resnet3D50', 'resnet3D101','resnet3D152', 
                                                                        'densenet3D121', 'densenet3D169','densenet3D201','densenet3D264', 
                                                                        'efficientnet3D-b0','efficientnet3D-b1','efficientnet3D-b2','efficientnet3D-b3','efficientnet3D-b4','efficientnet3D-b5','efficientnet3D-b6','efficientnet3D-b7',
                                                                        'swinV1_tiny_3D','swinV1_small_3D','swinV1_base_3D','swinV1_large_3D',
                                                                        'swinV2_tiny_3D','swinV2_small_3D','swinV2_base_3D','swinV2_large_3D',
                                                                        ])
    # arguments for training 
    parser.add_argument("--batch_size",default=16,type=int,required=False,help='')
    parser.add_argument("--optim",type=str,required=True,help='', choices=['Adam','AdamW','SGD', 'SGDW'])
    parser.add_argument("--lr", default=0.01,type=float,required=False,help='')
    parser.add_argument("--weight_decay",default=0.001,type=float,required=False,help='')
    parser.add_argument("--beta",default=1.0,type=float,required=False,help='')
    parser.add_argument("--epoch",type=int,required=True,help='')
    parser.add_argument("--cat_target", type=str, nargs='*', required=False, help='')
    parser.add_argument("--num_target", type=str,nargs='*', required=False, help='')
    parser.add_argument('--accumulation_steps', default=1, type=int, required=False)
    # arguments for resuming training
    parser.add_argument("--checkpoint_dir", type=str, default=None,required=False)
    # arguments for others 
    parser.add_argument("--seed",default=1234,type=int,required=False,help='')
    parser.add_argument("--exp_name",type=str,required=True,help='')
    parser.add_argument("--confusion_matrix", type=str, nargs='*',required=False, help='')
    parser.add_argument('--get_predicted_score', action='store_true', help='save the result of inference in the result file')
    parser.set_defaults(get_predicted_score=False)
    parser.add_argument('--gradient_clipping', action='store_true')
    parser.set_defaults(gradient_clipping=False)

    args = parser.parse_args()
    print("Categorical target labels are {} and Numerical target labels are {}".format(args.cat_target, args.num_target))

    if not args.cat_target:
        args.cat_target = []
    elif not args.num_target:
        args.num_target = []
    elif not args.cat_target and args.num_target:
        raise ValueError('YOU SHOULD SELECT THE TARGET!')

    return args



if __name__ == "__main__":
    ## ========= Setting ========= ##
    # arguments
    args = argument_setting()
    current_dir = os.getcwd()
    # seed number 
    set_random_seed(args.seed)
    # others 
    save_dir = current_dir + '/result'
    time_hash = datetime.datetime.now().time()
    hash_key = hashlib.sha1(str(time_hash).encode()).hexdigest()[:6]
    args.exp_name = args.exp_name + f'_{hash_key}'
    ## ====================================== ##

    ## ========= DataLoading and Preparing Dataset ========= ##
    image_dir, phenotype_dir = check_study_sample(study_sample=args.study_sample)
    image_files = loading_images(image_dir=image_dir, args=args, study_sample=args.study_sample)

    if args.partitioned_dataset_number is not None: 
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
    # Run Experiment
    setting, result = experiment(partition, subject_data, save_dir, deepcopy(args))

    # Save result
    save_exp_result(save_dir, setting, result)
    ## ====================================== ##
