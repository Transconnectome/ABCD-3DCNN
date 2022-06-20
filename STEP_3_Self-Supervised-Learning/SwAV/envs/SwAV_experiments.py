import numpy as np
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

#import apex 
#from apex.parallel.LARC import LARC

from sklearn.metrics import confusion_matrix
from models.prediction_model import prediction_model

## ======= load module ======= ##

from models.prediction_model import prediction_model #model script
from models.SwAV import SwAV, distributed_sinkhorn #model script

from envs.loss_functions import calculating_loss, calculating_acc, ContrastiveLoss
from utils.utils import argument_setting_SwAV, CLIreporter, save_exp_result, checkpoint_save, checkpoint_load, get_queue_path
from utils.optimizers import LAMB, LARS 

from dataloaders.data_augmentation import applying_augmentation

import time
from tqdm import tqdm
import copy

def SwAV_train(net, trainloader, optimizer, epoch, queue, args):
    """
    Flow of data augmentation is as follows. 
    intensity_crop_resize (together image set1 and image set2) at CPU -> cuda -> transform (sepreately applied to image set1 and image set2) at GPU. 
    By applying scale intensity, crop, and resize, it can resolve CPU -> GPU bottleneck problem which occur when too many augmentation techniques are sequentially operated at CPU.
    That's because we can apply augmentation technique to all of samples in mini-batches simulateously by tensor operation at GPU.
    However, GPUs have limitations in their RAM memory. Thus, too large matrix couldn't be attached. 
    So, in this code, crop and resizing operatiins are done at CPU, afterward other augmentations are applied at GPU.
    
    This strategy dramatically reduce training time by resolving the CPU ->GPU bottleneck problem
    """

    # setting
    scaler = torch.cuda.amp.GradScaler()
    net.train()
    use_the_queue = False
    losses = []

    for it, images in enumerate(trainloader,0):
        # normalize the prototypes 
        with torch.no_grad(): 
            w = net.module.prototypes.prototypes.weight.data.clone()
            w = nn.functional.normalize(w, dim=1, p=2)
            net.module.prototypes.prototypes.weight.copy_(w)
        
         
        # ============ multi-resolutions forward passes ... ============
        views_standard = images['standard']
        views_low = images['low_resolution']

        embedding_standard, output_standard = net(views_standard, mode='augmentation')
        embedding_low, output_low = net(views_low)
        batch_size = args.train_batch_size

        # ============ swav loss ... ============
        loss = 0
        for i in range(args.nmb_standard_views):
            with torch.no_grad():
                out_standard = output_standard[batch_size * i : batch_size * (i+1)].detach()

                # time to use the queue 
                if queue is not None:
                    if use_the_queue or not torch.all(queue[i, -1, :] == 0):
                        use_the_queue = True
                        out_standard = torch.cat((torch.mm(
                            queue[i],
                            net.module.prototypes.prototypes.weight.t()
                            #net.prototypes.prototypes.weight.t()
                        ), out_standard))
                    # fill the queue 
                    queue[i, batch_size:] = queue[i, :-batch_size].clone()
                    queue[i, :batch_size] = embedding_standard[batch_size * i: batch_size * (i+1)]

                # get assignments 
                q = distributed_sinkhorn(out_standard, args)[-batch_size:]
                
            # cluster assignment prediction 


            # crop_id = 0 일 때에는 view1와 low resolution crop들이 들어가고, crop_id = 1일 때에는 view0와 low resolution crop들이 들어감. 
            # 즉, standard view에서 얻은 embedding은 swap을 해주고, low resolution crop들은 무조건 넣기
            subloss = 0 
            x_standard = output_standard[i] / args.temperature
            subloss -= torch.mean(torch.sum(q * F.log_softmax(x_standard), dim=1))
            for v in range(args.nmb_low_views):
                x_low = output_low[batch_size * v : batch_size * (v + 1)] / args.temperature
                subloss -= torch.mean(torch.sum(q * F.log_softmax(x_low, dim=1), dim=1))
            loss += subloss / (np.sum(args.nmb_low_views) + 1)
        loss /= args.nmb_standard_views
        losses.append(loss.item())
        
        # ============ backward and optim step ... ============
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        # cancel gradients for the prototypes
        iteration = epoch * len(trainloader) + it
        if iteration < args.freeze_prototypes_niters:
            for name, p in net.module.named_parameters():
                if 'prototypes' in name: 
                    p.grad = None

        scaler.step(optimizer)
        scaler.update()

    return net, np.mean(losses), queue 
        


def SwAV_experiment(partition, save_dir, args): #in_channels,out_dim

    # setting data loader 
    ######sampler = torch.utils.data.distributed.DistributedSampler(partition['train'])
    trainloader = torch.utils.data.DataLoader(partition['train'],
                                              ######sampler = sampler,
                                              batch_size=args.train_batch_size,
                                              shuffle=True,
                                              drop_last = True,
                                              pin_memory=True,
                                              num_workers=24)
    
    # setting network 
    net = SwAV(args)
    checkpoint_dir = args.checkpoint_dir


    # setting optimizer 
    if args.optim == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    elif args.optim == 'Adam':
        optimizer = optim.Adam(net.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    elif args.optim == 'LARS':
        optimizer = LARS(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=.0005)
    elif args.optim == 'LAMB':
        optimizer = LAMB(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        """
    elif args.optim == 'LARC':
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
        optimizer = LARC(optimizer, trust_coefficient = 0.001, clip = False)
        """
    else:
        raise ValueError('In-valid optimizer choice')

    # setting learning rate scheduler 
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min', patience=10)
    #scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=1, gamma=0.5)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2, eta_min=0)

    # loading last checkpoint if resume training
    if args.resume == 'True':
        if args.checkpoint_dir != None:
            net, optimizer, scheduler, last_epoch, optimizer.param_groups[0]['lr'] = checkpoint_load(net, checkpoint_dir, optimizer, scheduler, args, mode='simCLR')
            print('Training start from epoch {} and learning rate {}.'.format(last_epoch, optimizer.param_groups[0]['lr']))
        else: 
            raise ValueError('IF YOU WANT TO RESUME TRAINING FROM PREVIOUS STATE, YOU SHOULD SET THE FILE PATH AS AN OPTION. PLZ CHECK --checkpoint_dir OPTION')
    else:
        last_epoch = 0 


    # synchronize batch norm layers 
    ######net = nn.SyncBatchNorm.convert_sync_batchnorm(net)

    # attach network to cuda device
    net.cuda()


    # wrap model 
    ######net = nn.Parallel.DistributedDataParallel(net, device_ids = args.gpu_to_work_on)
    net = nn.DataParallel(net)

    # bulid the queue 
    queue = None 
    queue_path = get_queue_path(save_dir, args)
    if os.path.isfile(queue_path):
        queue = torch.load(queue_path)['queue']
    # the queue needs to be divisible by the batch size 
    ######args.queue_length -= args.queue_length % (args.batch_size * args.world_size) # the number of stored features in the dictionary.  
    args.queue_length -= args.queue_length % args.train_batch_size

    # setting for results' data frame
    train_losses = []

    # training
    for epoch in tqdm(range(last_epoch, last_epoch + args.epoch)):
        ts = time.time()

        # set sampler
        ######trainloader.sampler.set_epoch(epoch)

        # optionally starts a queue 
        if args.queue_length > 0 and epoch >= args.epoch_queue_starts and queue is None:
            queue = torch.zeros(
                len(args.crops_for_assign),
                    args.queue_length,
                    ######args.queue_length // args.world_size,
                    args.feat_dim 

            ).cuda()

        # train the network
        net, loss, queue = SwAV_train(net, trainloader, optimizer, epoch, queue,args)
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