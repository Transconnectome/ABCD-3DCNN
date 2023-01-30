## ======= load module ======= ##
from utils.utils import argument_setting, select_model, CLIreporter, save_exp_result, checkpoint_save, checkpoint_load #  
#from utils.optimizer import CosineAnnealingWarmUpRestarts
from dataloaders.dataloaders import make_dataset
from dataloaders.preprocessing import preprocessing_cat, preprocessing_num
from envs.experiments import train, validate, test
from envs.transfer import setting_transfer
import hashlib
import datetime

import os
from os import listdir
from os.path import isfile, join
import glob
from tqdm.auto import tqdm ##progress
import time
import math
import random
from copy import deepcopy

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


import warnings
warnings.filterwarnings("ignore")


## ========= Helper Functions =============== ##

def setup_results(args):
    train_losses = {}
    train_accs = {}
    val_losses = {}
    val_accs = {}
    
    targets = args.cat_target + args.num_target
    for target_name in targets:
        train_losses[target_name] = []
        train_accs[target_name] = []
        val_losses[target_name] = []
        val_accs[target_name] = []

    result = {}
    result['train_losses'] = train_losses
    result['train_accs'] = train_accs
    result['val_losses'] = val_losses
    result['val_accs'] = val_accs
    
    return result

    
def set_optimizer(args, net):
    if args.optim == 'SGD':
        optimizer = optim.SGD(params = filter(lambda p: p.requires_grad, net.parameters()),
                              lr=args.lr, momentum=0.9)
    elif args.optim == 'Adam':
        optimizer = optim.Adam(params = filter(lambda p: p.requires_grad, net.parameters()),
                               lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim =='RAdam':
        optimizer = optim.RAdam(params = filter(lambda p: p.requires_grad, net.parameters()),
                                lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay)
    elif args.optim == 'AdamW':
        optimizer = optim.AdamW(params = filter(lambda p: p.requires_grad, net.parameters()),
                                lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay)
    else:
        raise ValueError('In-valid optimizer choice')
        
    return optimizer
    
    
def set_lr_scheduler(args, optimizer, len_dataloader):
    if args.scheduler == '':
        scheduler = None
    elif args.scheduler == 'on':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,'max', patience=10, factor=0.1, min_lr=1e-9)
    elif args.scheduler.lower() == 'cos':
#             scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=5, T_mult=2, eta_max=0.1, T_up=2, gamma=1)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2, eta_min=0)
    elif 'step' in args.scheduler:
        if len(args.scheduler.split('_')) != 2:
            step_size = 80
        else:
            step_size = int(args.scheduler.split('_')[1])
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size, gamma=0.1)
    elif args.scheduler.lower() == 'onecycle':
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=args.epoch)
    else:
        raise Exception("Invalid scheduler option")
        
    return scheduler
    
    
def run_experiment(args, net, partition, result, save_dir, mode):
    if mode == 'FC':
        epoch_exp = args.epoch_FC
        num_unfrozen = 0
    else:
        epoch_exp = args.epoch
        num_unfrozen = args.unfrozen_layer
    
    targets = args.cat_target + args.num_target
    
    setting_transfer(args, net.module, num_unfrozen = num_unfrozen)
    optimizer = set_optimizer(args, net)
    scheduler = set_lr_scheduler(args, optimizer, len(partition['train']))

    best_val_loss = float('inf')
    patience = 0

    for epoch in tqdm(range(epoch_exp)):
        ts = time.time()
        net, train_loss, train_acc = train(net,partition,optimizer,args)
        val_loss, val_acc = validate(net,partition,scheduler,args)
        te = time.time()

        ## sorting the results
        train_loss_sum = 0
        val_loss_sum = 0
        val_acc_sum = 0
        for target_name in targets:
            result['train_losses'][target_name].append(train_loss[target_name])
            result['train_accs'][target_name].append(train_acc[target_name])
            result['val_losses'][target_name].append(val_loss[target_name])
            result['val_accs'][target_name].append(val_acc[target_name])
            val_loss_sum += val_loss[target_name]
            val_acc_sum += val_acc[target_name]
            train_loss_sum += train_loss[target_name]
            
        val_loss_total = val_loss_sum/len(targets)
        val_acc_total = val_acc_sum/len(targets)
        train_loss_total = train_loss_sum/len(targets)
        
        if val_loss_total < best_val_loss:
            best_val_loss = val_loss_total
            best_train_loss = train_loss_total
            patience = 0
        else:
            patience += 1

        ## visualize the result
        CLIreporter(targets, train_loss, train_acc, val_loss, val_acc)
        print('Epoch {}. Current learning rate {}. Took {:2.2f} sec'.format(epoch+1,optimizer.param_groups[0]['lr'],te-ts))

        ## saving the checkpoint and results
        checkpoint_dir = checkpoint_save(net, save_dir, epoch, val_acc, result['val_accs'], args)                     
        if epoch%10 == 0:
            save_exp_result(save_dir, vars(args).copy(), result)

        ## Early-Stopping
        if args.early_stopping != None and patience == args.early_stopping:
            print(f"*** Validation Loss patience reached {args.early_stopping} epochs. Early Stopping Experiment ***")
            break
    
    if mode == 'FC':
        result['best_val_loss_FC'] = best_val_loss
        result['best_train_loss_FC'] = best_train_loss
    else:
        result['best_val_loss'] = best_val_loss
        result['best_train_loss'] = best_train_loss
    return result, checkpoint_dir


## ========= Experiment =============== ##
def experiment(partition, subject_data, save_dir, args): #in_channels,out_dim
    
    # selecting a model
    net = select_model(subject_data, args) #  
    
    # loading pretrained model if transfer option is given
    if (args.transfer != "") and (args.load == ""):
        print("*** Model setting for transfer learning *** \n")
        net = checkpoint_load(net, args.transfer)
    elif args.load:
        print("*** Model setting for transfer learning & fine tuning *** \n")
        model_dir = glob.glob(f'/scratch/connectome/jubin/result/model/*{args.load}*')[0]
        print(f"Loaded {model_dir[:-4]}")
        net = checkpoint_load(net, model_dir)
    else:
        print("*** Model setting for learning from scratch ***")
    
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
            
    net.to(f'cuda:{net.device_ids[0]}')
    
    # setting for results' DataFrame
    result = setup_results(args)
    
    # training a model
    print("*** Start training a model *** \n")
    
    if args.epoch_FC != 0:
        assert args.epoch > 0, "Epoch_FC should be used with epoch option. ex) epoch_FC=100, epoch=200"
        print("*** Transfer Learning - Training FC layers *** \n")
        
        result, _ = run_experiment(args, net, partition, result, save_dir, 'FC')
                
        print(f"Adjust learning rate for Training unfrozen layers from {args.lr} to {args.lr*args.lr_adjust}")
        args.lr *= args.lr_adjust
        result['lr_adjusted'] = args.lr
            
    print("*** Training unfrozen layers *** \n")
    
    result, checkpoint_dir = run_experiment(args, net, partition, result, save_dir, 'ALL')
                    
    # testing a model
    print("\n*** Start testing a model *** \n")
    net.to('cpu')
    torch.cuda.empty_cache()

    net = checkpoint_load(net, checkpoint_dir)
    if args.sbatch == 'True':
        net.cuda()
    else:
        net.to(f'cuda:{args.gpus[0]}')
    test_acc, confusion_matrices = test(net, partition, args)
    result['test_acc'] = test_acc
    print(f"Test result: {test_acc} for {args.exp_name}") 
    
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
    time_hash = datetime.datetime.now().time()
    hash_key = hashlib.sha1(str(time_hash).encode()).hexdigest()[:6]
    args.exp_name = args.exp_name + f'_{hash_key}'

    # Run Experiment
    print(f"*** Experiment {args.exp_name} Start ***")
    setting, result = experiment(partition, subject_data, save_dir, deepcopy(args))
    print("setting was",args)
    # Save result
    save_exp_result(save_dir, setting, result)
    print("*** Experiment Done ***\n")
    ## ====================================== ##

