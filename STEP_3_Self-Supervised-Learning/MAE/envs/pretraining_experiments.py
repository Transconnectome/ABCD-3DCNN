import numpy as np
import os

import torch
import torch.nn as nn
import torch.optim as optim


from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix


## ======= load module ======= ##
import model.model_MAE as MAE

from util.utils import CLIreporter, save_exp_result, checkpoint_save, checkpoint_load, set_random_seed
from util.optimizers import LAMB, LARS 


import time
from tqdm import tqdm
import copy

def MAE_train(net, partition, optimizer, args):
    """
    Flow of data augmentation is as follows. 
    intensity_crop_resize (together image set1 and image set2) at CPU -> cuda -> transform (sepreately applied to image set1 and image set2) at GPU. 
    By applying scale intensity, crop, and resize, it can resolve CPU -> GPU bottleneck problem which occur when too many augmentation techniques are sequentially operated at CPU.
    That's because we can apply augmentation technique to all of samples in mini-batches simulateously by tensor operation at GPU.
    However, GPUs have limitations in their RAM memory. Thus, too large matrix couldn't be attached. 
    So, in this code, crop and resizing operatiins are done at CPU, afterward other augmentations are applied at GPU.
    
    This strategy dramatically reduce training time by resolving the CPU ->GPU bottleneck problem
    """

    scaler = torch.cuda.amp.GradScaler()

    trainloader = torch.utils.data.DataLoader(partition['train'],
                                             batch_size=args.train_batch_size,
                                             shuffle=True,
                                             drop_last = True,
                                             pin_memory=True,
                                             num_workers=len(net.device_ids)*9)

    net.train()

    losses = []
    
    optimizer.zero_grad()
    for i, images in enumerate(trainloader,0):
        images = images.to(f'cuda:{net.device_ids[0]}')
        """
        if loss is calculated inside the model class, the output from the model forward method would be [loss] * number of devices. In other words, len(net(images)) == n_gpus
        """
        #loss, pred, mask  = net(images)
        pred, target, mask = net(images)
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        losses.append(loss.item())
        
        assert args.accumulation_steps >= 1
        if args.accumulation_steps == 1:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        elif args.accumulation_steps > 1:           # gradient accumulation
            loss = loss / args.accumulation_steps
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()    
            

    return net, np.mean(losses) 
        


def MAE_experiment(partition, save_dir, args): #in_channels,out_dim

    # setting network 
    net = MAE.__dict__[args.model](img_size = args.img_size, norm_pix_loss=False, mask_ratio = args.mask_ratio)
    checkpoint_dir = args.checkpoint_dir


    # setting optimizer 
    if args.optim == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)
    elif args.optim == 'Adam':
        optimizer = optim.Adam(net.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    elif args.optim == 'LARS':
        optimizer = LARS(net.parameters(), lr=args.lr, momentum=0.9)
    elif args.optim == 'LAMB':
        optimizer = LAMB(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)        
    else:
        raise ValueError('In-valid optimizer choice')

    # setting learning rate scheduler 
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min', patience=10)
    #scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=1, gamma=0.5)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=0)

    # loading last checkpoint if resume training
    if args.resume == 'True':
        if args.checkpoint_dir != None:
            net, optimizer, scheduler, last_epoch, optimizer.param_groups[0]['lr'] = checkpoint_load(net, checkpoint_dir, optimizer, scheduler, args, mode='pretrain')
            print('Training start from epoch {} and learning rate {}.'.format(last_epoch, optimizer.param_groups[0]['lr']))
        else: 
            raise ValueError('IF YOU WANT TO RESUME TRAINING FROM PREVIOUS STATE, YOU SHOULD SET THE FILE PATH AS AN OPTION. PLZ CHECK --checkpoint_dir OPTION')
    else:
        last_epoch = 0 


    # setting DataParallel
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

    # attach network and optimizer to cuda device
    net.to(f'cuda:{net.device_ids[0]}')

    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(f'cuda:{net.device_ids[0]}')

    # setting for results' data frame
    train_losses = []

    # training
    for epoch in tqdm(range(last_epoch, last_epoch + args.epoch)):
        ts = time.time()
        net, loss = MAE_train(net, partition, optimizer, args)
        train_losses.append(loss)
        #scheduler.step(loss)
        scheduler.step()
        te = time.time()

        # visualize the result
        print('Epoch {}. Loss: {:2.2f}. Current learning rate {}. Took {:2.2f} sec'.format(epoch+1, loss, optimizer.param_groups[0]['lr'],te-ts))

        # saving the checkpoint
        checkpoint_dir = checkpoint_save(net, optimizer, save_dir, epoch, scheduler, args, mode='simCLR')


    # summarize results
    result = {}
    result['train_losses'] = train_losses

    return vars(args), result
        

## ==================================== ##