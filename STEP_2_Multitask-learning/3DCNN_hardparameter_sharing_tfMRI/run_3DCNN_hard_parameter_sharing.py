## ======= load module ======= ##
import os
from tqdm.auto import tqdm ##progress
import argparse
from copy import deepcopy

import hashlib
import datetime

from utils.utils import set_random_seed, save_exp_result
from envs.experiments import experiment
from dataloaders.dataloaders import check_study_sample, loading_images, loading_phenotype, combining_image_target, partition_dataset, undersampling_ALLset, partition_dataset_predefined

import warnings
warnings.filterwarnings("ignore")


def argument_setting():
    parser = argparse.ArgumentParser()
    # parameters for data load and split 
    parser.add_argument("--study_sample",default='UKB',type=str,required=False,help='') 
    parser.add_argument("--train_size",default=0.8,type=float,required=False,help='')
    parser.add_argument("--val_size",default=0.1,type=float,required=False,help='')
    parser.add_argument("--test_size",default=0.1,type=float,required=False,help='')
    parser.add_argument("--cat_target", type=str, nargs='*', required=False, help='')
    parser.add_argument("--num_target", type=str,nargs='*', required=False, help='')
    parser.add_argument('--scaling_num_target', action='store_true', help='activate this option if you want to standardize continuous target')
    parser.set_defaults(scaling_num_target=False)
    parser.add_argument('--partitioned_dataset_number', default=None, type=int, required=False)
    # parameters for training
    parser.add_argument("--exp_name",type=str,required=True,help='')
    parser.add_argument("--batch_size",default=16,type=int,required=False,help='')
    parser.add_argument('--accumulation_steps', default=1, type=int, required=False)
    parser.add_argument("--in_channels",default=1,type=int,required=False,help='')
    parser.add_argument("--optim",type=str,required=True,help='', choices=['Adam','AdamW','SGD', 'SGDW'])
    parser.add_argument("--lr", default=0.01,type=float,required=False,help='')
    parser.add_argument("--weight_decay",default=0.001,type=float,required=False,help='')
    parser.add_argument("--epoch",type=int,required=True,help='')
    parser.add_argument("--model",required=True,type=str,help='',choices=[
                                                                        'densenet3D121', 'densenet3D169','densenet3D201','densenet3D264', 
                                                                        'densenet3D121_cbam', 'densenet3D169_cbam','densenet3D201_cbam','densenet3D264_cbam', 
                                                                        'flipout_densenet3D121', 'flipout_densenet3D169','flipout_densenet3D201','flipout_densenet3D264', 
                                                                        'variational_densenet3D121', 'variational_densenet3D169','variational_densenet3D201','variational_densenet3D264', 
                                                                        'efficientnet3D-b0','efficientnet3D-b1','efficientnet3D-b2','efficientnet3D-b3','efficientnet3D-b4','efficientnet3D-b5','efficientnet3D-b6','efficientnet3D-b7'
                                                                        ])
    parser.add_argument('--gradient_clipping', action='store_true')
    parser.set_defaults(gradient_accumulation=False)
    # parameters for mixup variant agumentation
    parser.add_argument("--beta",default=1.0,type=float,required=False,help='')
    parser.add_argument('--mixup', type=float, default=0, help='')
    parser.add_argument('--cutmix', type=float, default=0, help='')
    parser.add_argument('--c_mixup', type=float, default=0, help='')
    parser.add_argument('--manifold_mixup', type=float, default=0, help='')
    # parameters for inference
    parser.add_argument("--confusion_matrix", type=str, nargs='*',required=False, help='')
    parser.add_argument('--get_predicted_score', action='store_true', help='save the result of inference in the result file')
    parser.set_defaults(get_predicted_score=False)
    # parameters for bayesian neural networks
    parser.add_argument('--moped', action='store_true', help="activate 'Model Priors with Empirical Bayes using Deterministic DNN'")
    parser.set_defaults(moped=False)
    # parameters for others
    parser.add_argument("--seed",default=1234,type=int,required=False,help='')
    parser.add_argument("--gpus", type=int,nargs='*', required=False, help='')
    parser.add_argument("--checkpoint_dir", type=str, default=None,required=False)

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
    args = argument_setting()
    current_dir = os.getcwd()
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
    # seed number
    set_random_seed(args.seed)
    save_dir = current_dir + '/result'
    
    time_hash = datetime.datetime.now().time()
    hash_key = hashlib.sha1(str(time_hash).encode()).hexdigest()[:6]
    args.exp_name = args.exp_name + f'_{hash_key}'


    # Run Experiment
    setting, result = experiment(partition, subject_data, save_dir, deepcopy(args))

    # Save result
    save_exp_result(save_dir, setting, result)
    ## ====================================== ##
