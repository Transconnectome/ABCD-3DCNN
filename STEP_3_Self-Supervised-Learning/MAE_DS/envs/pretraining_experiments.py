import numpy as np
import os
from model.layers.drop_path import drop_path

import torch
import torch.nn as nn
import torch.optim as optim


from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix

import deepspeed

## ======= load module ======= ##
import model.model_MAE as MAE

from util.utils import CLIreporter, save_exp_result, checkpoint_save, checkpoint_load, saving_outputs, set_random_seed
from util.optimizers import LAMB, LARS 
from util.lr_sched import CosineAnnealingWarmUpRestarts


import time
from tqdm import tqdm
import copy

def MAE_train(net, trainloader, optimizer, args):

    net.train()

    losses = []
    
    optimizer.zero_grad()
    for i, images in enumerate(trainloader,0):
        images = images.to(net.device)
        """
        if loss is calculated inside the model class, the output from the model forward method would be [loss] * number of devices. In other words, len(net(images)) == n_gpus
        """
        #loss, pred, mask  = net(images)
        with torch.cuda.amp.autocast():
            pred, target, mask = net(images)
            loss = (pred - target) ** 2
            loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
            loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        losses.append(loss.item())

        net.backward(loss)
        net.step()

    return net, np.mean(losses) 
        

def MAE_validation(net, valloader, args):

    net.eval()
    
    losses = []
    
    with torch.no_grad():
        for i, images in enumerate(valloader,0):
            images = images.to(net.device)
            with torch.cuda.amp.autocast():
                pred, target, mask = net(images)    # if 0 in mask indicate the input for the encoder (unremoved patches), and 1 in mask indicate the non-input for the encoder (removed patches)
                loss = (pred - target) ** 2
                loss = loss.mean(dim=-1)    # [N, L], mean loss per patch
                loss = (loss * mask).sum() / mask.sum() # mean loss on removed patches
            losses.append(loss.item())
                
            # saving example images 
            if i == 0:
                saving_outputs(net, pred, mask, target, '/scratch/connectome/dhkdgmlghks/3DCNN_test/MAE_DS')
                
    return net, np.mean(losses)


def MAE_experiment(partition, save_dir, args): #in_channels,out_dim

    # setting network 
    net = MAE.__dict__[args.model](img_size = args.img_size, attn_drop=args.attention_drop, drop=args.projection_drop, drop_path=args.path_drop, norm_pix_loss=args.norm_pix_loss, mask_ratio = args.mask_ratio)
    checkpoint_dir = args.checkpoint_dir


    net, optimizer, trainloader, _ = deepspeed.initialize(args=args, model=net, model_parameters=net.parameters(), training_data = partition['train'])
    _, _, valloader, _ = deepspeed.initialize(args=args, model=net, model_parameters=net.parameters(), training_data = partition['val'])

    # setting learning rate scheduler 
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min', patience=10)
    #scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=1, gamma=0.5)
    #scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=2, eta_min=0)
    #scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=150, T_mult=2, eta_max=args.lr,  T_up=5, gamma=0.5)

    # setting AMP scaler 
   # scaler = torch.cuda.amp.GradScaler()

    # loading last checkpoint if resume training
    if args.resume == True:
        if args.checkpoint_dir != None:
            net,  last_epoch = checkpoint_load(net, checkpoint_dir,  mode='pretrain')
            print('Training start from epoch {} and learning rate {}.'.format(last_epoch, optimizer.param_groups[0]['lr']))
        else: 
            raise ValueError('IF YOU WANT TO RESUME TRAINING FROM PREVIOUS STATE, YOU SHOULD SET THE FILE PATH AS AN OPTION. PLZ CHECK --checkpoint_dir OPTION')
    else:
        last_epoch = 0 


    # setting DataParallel
    #if args.sbatch == True:
    #    devices = []
    #    for d in range(torch.cuda.device_count()):
    #        devices.append(d)
    #    net = nn.DataParallel(net, device_ids = devices)
    #else:
    #    if not args.gpus:
    #        raise ValueError("GPU DEVICE IDS SHOULD BE ASSIGNED")
    #    else:
    #        net = nn.DataParallel(net, device_ids=args.gpus)

    # attach network and optimizer to cuda device
    #net.to(f'cuda:{net.device_ids[0]}')

    #for state in optimizer.state.values():
    #    for k, v in state.items():
    #        if isinstance(v, torch.Tensor):
    #            state[k] = v.to(f'cuda:{net.device_ids[0]}')

    # setting for results' data frame
    train_losses = []
    val_losses = []

    # training
    for epoch in tqdm(range(last_epoch, last_epoch + args.epoch)):
        ts = time.time()
        net, train_loss = MAE_train(net, trainloader, optimizer,  args)
        net, val_loss = MAE_validation(net, valloader, args)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        te = time.time()

        # visualize the result
        print('Epoch {}. Train Loss: {:2.2f}. Validation Loss: {:2.2f}. Current learning rate {}. Took {:2.2f} sec'.format(epoch+1, train_loss, val_loss, optimizer.param_groups[0]['lr'],te-ts))
        torch.cuda.empty_cache()

        # saving the checkpoint
        checkpoint_dir = checkpoint_save(net, save_dir, epoch, args, mode='pretrain')


    # summarize results
    result = {}
    result['train_losses'] = train_losses
    result['validation_losses'] = val_losses

    return vars(args), result
        

## ==================================== ##