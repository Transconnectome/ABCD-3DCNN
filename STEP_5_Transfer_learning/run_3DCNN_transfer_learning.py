## ======= load module ======= ##
from utils.utils import argument_setting, select_model, CLIreporter, save_exp_result, checkpoint_save, checkpoint_load  #  
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

def set_optimizer(args, net):
    if args.optim == 'SGD':
        optimizer = optim.SGD(
            params = filter(lambda p: p.requires_grad, net.parameters()),
            lr = args.lr,
            momentum = 0.9)
    elif args.optim == 'Adam':
        optimizer = optim.Adam(
            params = filter(lambda p: p.requires_grad, net.parameters()),
            lr = args.lr,
            weight_decay = args.weight_decay)
    elif args.optim =='RAdam':
        optimizer = optim.RAdam(
            params = filter(lambda p: p.requires_grad, net.parameters()),
            lr = args.lr,
            weight_decay = args.weight_decay,
            betas=(0.9, 0.999), eps=1e-08, )
    else:
        raise ValueError('In-valid optimizer choice')
        
    return optimizer
    
def set_lr_scheduler(args, optimizer):
    if args.scheduler != None:
        if args.scheduler == 'on':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,'max', patience=10, factor=0.1, min_lr=1e-9)
        elif args.scheduler == 'cos':
#             scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=5, T_mult=2, eta_max=0.1, T_up=2, gamma=1)
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=1, eta_min=0)
    else:
        scheduler = None
        
    return scheduler
    


## ========= Experiment =============== ##
def experiment(partition, subject_data, save_dir, args): #in_channels,out_dim
    
    
    # selecting a model
    net = select_model(subject_data, args) #  
    
    
    # loading pretrained model if transfer option is given
    if (args.transfer != None) and (args.load == ""):
        print("*** Model setting for transfer learning *** \n")
        net = checkpoint_load(net, args.transfer)
    elif args.load:
        print("*** Model setting for transfer learning & fine tuning *** \n")
        model_dir = glob.glob(f'/scratch/connectome/jubin/result/model/*_{args.load}*.pth')[0]
        net = checkpoint_load(net, model_dir)
    
    
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
    train_losses = {}
    train_accs = {}
    val_losses = {}
    val_accs = {}
    test_accs = {}
    
    targets = args.cat_target + args.num_target
    for target_name in targets:
        train_losses[target_name] = []
        train_accs[target_name] = []
        val_losses[target_name] = []
        val_accs[target_name] = []
        test_accs[target_name] = []

    result = {}
    result['train_losses'] = train_losses
    result['train_accs'] = train_accs
    result['val_losses'] = val_losses
    result['val_accs'] = val_accs
    result['test_accs'] = test_accs
    
    
    # training a model
    print("*** Start training a model *** \n")
    
    if (args.unfrozen_layer > 0) and (args.load == ""):
        print("*** Transfer Learning - Training FC layers *** \n")
        
        setting_transfer(net.module, num_unfreezed = 0)
        optimizer = set_optimizer(args, net)
        scheduler = set_lr_scheduler(args, optimizer)

        for epoch in tqdm(range(args.epoch_FC)):
            ts = time.time()
            net, train_loss, train_acc = train(net,partition,optimizer,args)
            val_loss, val_acc = validate(net,partition,scheduler,args)
            te = time.time()
                        
            if args.transfer == 'simclr':
                test_acc, confusion_matrices = test(net, partition, args)
                print(f"Test result for epoch {epoch}: ",test_acc)

            ## sorting the results
            for target_name in targets:
                train_losses[target_name].append(train_loss[target_name])
                train_accs[target_name].append(train_acc[target_name])
                val_losses[target_name].append(val_loss[target_name])
                val_accs[target_name].append(val_acc[target_name])
                if args.transfer == 'simclr':
                    test_accs[target_name].append(test_acc[target_name])


            ## visualize the result
            CLIreporter(targets, train_loss, train_acc, val_loss, val_acc)
            print('Epoch {}. Current learning rate {}. Took {:2.2f} sec'.format(epoch+1,optimizer.param_groups[0]['lr'],te-ts))
            
            ## saving the checkpoint
            checkpoint_dir = checkpoint_save(net, save_dir, epoch, val_acc, val_accs, args)

            
            if epoch%10 == 0:
                save_exp_result(save_dir, vars(args).copy(), result)
            
        print("Adjust learning rate for Training unfrozen layers")
        print(f"From {args.lr} to {args.lr/100}")    
        args.lr /= 100
        result['lr_adjusted'] = args.lr

            
    print("*** Training unfrozen layers *** \n")
    
    setting_transfer(net.module, num_unfreezed = args.unfrozen_layer)
    optimizer = set_optimizer(args, net)
    scheduler = set_lr_scheduler(args, optimizer)
    
    for epoch in tqdm(range(args.epoch)):
        ts = time.time()
        net, train_loss, train_acc = train(net,partition,optimizer,args)
        val_loss, val_acc = validate(net,partition,scheduler,args)
        te = time.time()
        
        if args.transfer == 'simclr':
            test_acc, confusion_matrices = test(net, partition, args)
            print(f"Test result for epoch {epoch}: ",test_acc)
            
        ## sorting the results
        for target_name in targets:
            train_losses[target_name].append(train_loss[target_name])
            train_accs[target_name].append(train_acc[target_name])
            val_losses[target_name].append(val_loss[target_name])
            val_accs[target_name].append(val_acc[target_name])
            if args.transfer == 'simclr' and (epoch+1) < args.epoch:
                test_accs[target_name].append(test_acc[target_name])            

        ## visualize the result
        CLIreporter(targets, train_loss, train_acc, val_loss, val_acc)
        print('Epoch {}. Current learning rate {}. Took {:2.2f} sec'.format(epoch+1,optimizer.param_groups[0]['lr'],te-ts))

        ## saving the checkpoint
        checkpoint_dir = checkpoint_save(net, save_dir, epoch, val_acc, val_accs, args)
        
        if epoch%10 == 0:
            save_exp_result(save_dir, vars(args).copy(), result)

        
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
    print("Test result: ",test_acc)
    
    # summarizing results
    
    result['test_acc'] = test_acc
    result['train_acc'] = train_acc
    result['val_acc'] = val_acc
    
    if confusion_matrices != None:
        result['confusion_matrices'] = confusion_matrices

        
    return vars(args), result
## ==================================== ##


if __name__ == "__main__":
    cwd = os.getcwd()

    ## ========= Setting ========= ##
    args = argument_setting()
    
    if args.transfer:
        args.resize = (96, 96, 96) if args.transfer == 'age' else (80, 80, 80)
        
    save_dir = os.getcwd() + '/result' #  
    partition, subject_data = make_dataset(args) #  

    ## ========= Run Experiment and saving result ========= ##
    # seed number
    seed = 1234
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    time_hash = datetime.datetime.now().time()
    hash_key = hashlib.sha1(str(time_hash).encode()).hexdigest()[:6]
    args.exp_name = args.exp_name + f'_{hash_key}'

    # Run Experiment
    print(f"*** Experiment {args.exp_name} Start ***")
    setting, result = experiment(partition, subject_data, save_dir, deepcopy(args))

    # Save result
    save_exp_result(save_dir, setting, result)
    print("*** Experiment Done ***\n")
    ## ====================================== ##

