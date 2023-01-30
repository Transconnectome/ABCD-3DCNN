## ======= load module ======= ##
from utils.utils import argument_setting, select_model, save_exp_result, checkpoint_load #  
from dataloaders.dataloaders import make_dataset
from dataloaders.preprocessing import preprocessing_cat, preprocessing_num
from envs.experiments import test

import os
import glob
from tqdm.auto import tqdm ##progress
import random
from copy import deepcopy

import torch
import torch.nn as nn

import numpy as np

import warnings
warnings.filterwarnings("ignore")
    
## ========= Experiment =============== ##
def experiment(partition, subject_data, save_dir, args): #in_channels,out_dim
    
    # selecting a model
    net = select_model(subject_data, args) #  
    
    # loading pretrained model if transfer option is given
    if args.load:
        print("*** Model setting for test *** \n")
        model_dir = glob.glob(f'/scratch/connectome/jubin/result/model/*{args.load}*')[0]
        print(f"Loaded {args.load}")
        net = checkpoint_load(net, model_dir)
    elif args.load == '':
        print("Warning: Invalid model selection")
        sys.exit()
    
    # setting a DataParallel and model on GPU
    if args.sbatch == "True":
        devices = []
        for d in range(torch.cuda.device_count()):
            devices.append(d)
        net = nn.DataParallel(net, device_ids = devices)
    else:
        if not args.gpus:
            raise ValueError("GPU DEVICE IDS SHOULD BE ASSIGNED")
        else:
            net = nn.DataParallel(net, device_ids=args.gpus)
            
    if args.sbatch == 'True':
        net.cuda()
    else:
        net.to(f'cuda:{args.gpus[0]}')
        
    test_acc, confusion_matrices = test(net, partition, args)
    
    result = {'test_acc':test_acc}
    
    print(f"Test result: {test_acc} for {args.load}") 
    
    if confusion_matrices != None:
        result['confusion_matrices'] = confusion_matrices
        
    return vars(args), result
## ==================================== ##

def seed_all(SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    

if __name__ == "__main__":
    cwd = os.getcwd()

    ## ========= Setting ========= ##
    args = argument_setting()
    # seed number
    args.seed = 1234
    seed_all(args.seed)
    
    if args.transfer in ['age','MAE']:
        assert 96 in args.resize, "age(MSE/MAE) transfer model's resize should be 96"
    elif args.transfer == 'sex':
        assert 80 in args.resize, "sex transfer model's resize should be 80"
    
    save_dir = os.getcwd() + '/result'
    partition, subject_data = make_dataset(args)  

    ## ========= Run Experiment and saving result ========= ## 

    # Run Experiment
    print(f"*** Test for {args.exp_name} Start ***")
    setting, result = experiment(partition, subject_data, save_dir, deepcopy(args))
    
    # Save result
    save_exp_result(save_dir, setting, result)
    print("*** Experiment Done ***\n")
    ## ====================================== ##

