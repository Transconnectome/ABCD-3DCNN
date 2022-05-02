import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim


from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix
from models.prediction_model import prediction_model

## ======= load module ======= ##

from models.prediction_model import prediction_model #model script
from models.simCLR import simCLR #model script

from envs.loss_functions import calculating_loss, calculating_acc, ContrastiveLoss
from utils.utils import argument_setting_simCLR, CLIreporter, save_exp_result, checkpoint_save, checkpoint_load
from utils.LARS_optimizer import LARS 


import time
from tqdm import tqdm
import copy

def simCLR_train(net, partition, optimizer, args):
    scaler = torch.cuda.amp.GradScaler()

    trainloader = torch.utils.data.DataLoader(partition['train'],
                                             batch_size=args.train_batch_size,
                                             shuffle=True,
                                             drop_last = True,
                                             pin_memory=True,
                                             num_workers=24)

    net.train()

    loss_function = ContrastiveLoss(args.train_batch_size)
    losses = []
    
    optimizer.zero_grad()
    for i, images in enumerate(trainloader,0):
        image1 = images[0]
        image2 = images[1]

        image1 = image1.to(f'cuda:{net.device_ids[0]}')
        image2 = image2.to(f'cuda:{net.device_ids[0]}')

        z1 = net(image1)
        z2 = net(image2)
        
        loss = loss_function(z1, z2)
        losses.append(loss.item())

        loss = loss / args.accumulation_steps
        scaler.scale(loss).backward()
        
        if (i+1) % args.accumulation_steps == 0: # gradient accumulation
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
    return net, np.mean(losses) 
        


def simCLR_experiment(partition, save_dir, args): #in_channels,out_dim

    # setting network 
    net = simCLR(args)
    checkpoint_dir = args.checkpoint_dir


    # setting optimizer 
    if args.optim == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)
    elif args.optim == 'Adam':
        optimizer = optim.Adam(net.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    elif args.optim == 'LARS':
        optimizer = LARS(net.parameters(), lr=args.lr, momentum=0.9)
    else:
        raise ValueError('In-valid optimizer choice')


    # loading last checkpoint if resume training
    if args.resume == 'True':
        if args.checkpoint_dir != None:
            net, optimizer, last_epoch = checkpoint_load(net, checkpoint_dir, optimizer, mode='simCLR')
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

    # attach network to cuda device
    net.to(f'cuda:{net.device_ids[0]}')

    # setting for results' data frame
    train_losses = []

    # training
    for epoch in tqdm(range(last_epoch, last_epoch + args.epoch)):
        ts = time.time()
        net, loss = simCLR_train(net, partition, optimizer, args)
        train_losses.append(loss)
        te = time.time()

        # visualize the result
        print('Epoch {}. Loss: {:2.2f}. Current learning rate {}. Took {:2.2f} sec'.format(epoch+1, loss, optimizer.param_groups[0]['lr'],te-ts))

        # saving the checkpoint
        checkpoint_dir = checkpoint_save(net, optimizer, save_dir, epoch, args, mode='simCLR')

    # summarize results
    result = {}
    result['train_losses'] = train_losses

    return vars(args), result
        

## ==================================== ##