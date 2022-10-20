import numpy as np
import os


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.distributed import DistributedSampler 


from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix


## ======= load module ======= ##
import model.simMIM as simMIM

from util.utils import CLIreporter, save_exp_result, checkpoint_save, checkpoint_load, saving_outputs, set_random_seed, load_imagenet_pretrained_weight
from util.optimizers import LAMB, LARS 
from util.lr_sched import CosineAnnealingWarmUpRestarts

import time
from tqdm import tqdm
import copy

def simMIM_train(net, partition, optimizer, scaler, epoch, args):
    train_sampler = DistributedSampler(partition['train'], shuffle=True)    

    trainloader = torch.utils.data.DataLoader(partition['train'],
                                             batch_size=args.batch_size,
                                             sampler=train_sampler, 
                                             shuffle=False,
                                             drop_last = False,
                                             num_workers=16)

    net.train()
    trainloader.sampler.set_epoch(epoch)

    losses = []
    
    optimizer.zero_grad()
    for i, data in enumerate(trainloader,0):
        #images = images.to(f'cuda:{net.device_ids[0]}')
        
        images, mask = data[0], data[1]
        images = images.cuda()
        mask = mask.cuda()
        
        """
        if loss is calculated inside the model class, the output from the model forward method would be [loss] * number of devices. In other words, len(net(images)) == n_gpus
        """
        #loss, pred, mask  = net(images)
        with torch.cuda.amp.autocast():
            loss, _, _ = net(images, mask)
        losses.append(loss.item())

        assert args.accumulation_steps >= 1
        if args.accumulation_steps == 1:
            scaler.scale(loss).backward()
            # gradient clipping 
            if args.gradient_clipping == True:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1, error_if_nonfinite=False)   # max_norm=1 from https://arxiv.org/pdf/2010.11929.pdf
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        elif args.accumulation_steps > 1:           # gradient accumulation
            loss = loss / args.accumulation_steps
            scaler.scale(loss).backward()
            if  (i + 1) % args.accumulation_steps == 0:
                # gradient clipping 
                if args.gradient_clipping == True:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1, error_if_nonfinite=False)   # max_norm=1 from https://arxiv.org/pdf/2010.11929.pdf
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()    
            

    return net, np.mean(losses) 
        

def simMIM_validation(net, partition, epoch, args):
    val_sampler = DistributedSampler(partition['val'])  
    valloader = torch.utils.data.DataLoader(partition['val'],
                                            sampler=val_sampler, 
                                            batch_size=args.batch_size,
                                            shuffle=False,
                                            num_workers=16)

    net.eval()
    valloader.sampler.set_epoch(epoch)
    
    losses = []
    
    with torch.no_grad():
        for i, data in enumerate(valloader,0):
            #images = images.to(f'cuda:{net.device_ids[0]}')
            images, mask = data
            images = images.cuda()
            mask = mask.cuda()
            with torch.cuda.amp.autocast():
                loss, images_rec, mask_rec = net(images, mask)
                if i == 0:
                    np.save('/scratch/connectome/dhkdgmlghks/3DCNN_test/simMIM/img_%s.npy' % args.model, images.detach().cpu().numpy())  
                    np.save('/scratch/connectome/dhkdgmlghks/3DCNN_test/simMIM/img_rec_%s.npy' % args.model, images_rec.detach().cpu().numpy())
                    np.save('/scratch/connectome/dhkdgmlghks/3DCNN_test/simMIM/mask_%s.npy' % args.model, mask_rec.detach().cpu().numpy())
            losses.append(loss.item())
                
    return net, np.mean(losses)


def simMIM_experiment(partition, save_dir, args): #in_channels,out_dim
    if args.load_imagenet_pretrained:
        pretrained_weight = load_imagenet_pretrained_weight(args)       # This line return directory of imagenet_pretrained_weight
        pretrained2d = True
    else: 
        pretrained_weight = None 
        pretrained2d = False 

    if args.pretrained_model is not None: 
        pretrained_weight = args.pretrained_model
        pretrained2d = False 
         

    # setting network 
    if args.model.find('swin') != -1:
        net = simMIM.__dict__[args.model](pretrained=pretrained_weight, pretrained2d=pretrained2d, window_size=args.window_size, drop_rate=args.projection_drop, num_classes=0)
        # change an attribute of mask generator
        partition['train'].transform.update_config(net.patch_size)
        partition['val'].transform.update_config(net.patch_size)
    elif args.model.find('vit') != -1:
        assert args.model_patch_size == args.mask_patch_size
        net = simMIM.__dict__[args.model](pretrained=pretrained_weight,img_size=args.img_size, patch_size=args.model_patch_size, attn_drop=args.attention_drop, drop=args.projection_drop, drop_path=args.path_drop, use_rel_pos_bias=args.use_rel_pos_bias, use_sincos_pos=args.use_sincos_pos, num_classes=0)
        # change an attribute of mask generator
        partition['train'].transform.update_config(net.patch_size)
        partition['val'].transform.update_config(net.patch_size)
        print('The size of Patch is %i and the size of Mask Patch is %i' % (args.model_patch_size, args.mask_patch_size))

    if args.torchscript:
        torch._C._jit_set_autocast_mode(True)
        net = torch.jit.script(net)
    checkpoint_dir = args.checkpoint_dir


    # setting optimizer 
    if args.optim == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=0, momentum=0.9)
    elif args.optim == 'Adam':
        optimizer = optim.Adam(net.parameters(),lr=0,weight_decay=args.weight_decay)
    elif args.optim == 'LARS':
        optimizer = LARS(net.parameters(), lr=0, momentum=0.9)
    elif args.optim == 'LAMB':
        optimizer = LAMB(net.parameters(), lr=0, weight_decay=args.weight_decay)        
    elif args.optim == 'AdamW':
        optimizer = optim.AdamW(net.parameters(), lr=0, weight_decay=args.weight_decay,betas=(0.9, 0.95))
    else:
        raise ValueError('In-valid optimizer choice')

    # setting learning rate scheduler 
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min', patience=10)
    #scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=1, gamma=0.5)
    #scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=2, eta_min=0)
    scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=100, T_mult=2, eta_max=args.lr,  T_up=10, gamma=0.5)
    
    #for one cycle scheduler 
    """
    steps_per_epoch = len(torch.utils.data.DataLoader(partition['train'],
                                             batch_size=args.batch_size,
                                             sampler=DistributedSampler(partition['train'], shuffle=True), 
                                             shuffle=False,
                                             drop_last = False,
                                             num_workers=16))
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, epochs=args.epoch, steps_per_epoch=steps_per_epoch)
    """
  
    # setting AMP gradient scaler 
    scaler = torch.cuda.amp.GradScaler()

    # loading last checkpoint if resume training
    if args.resume == True:
        if args.checkpoint_dir != None:
            net, optimizer, scheduler, last_epoch, optimizer.param_groups[0]['lr'], scaler = checkpoint_load(net=net, checkpoint_dir=checkpoint_dir, optimizer=optimizer, scheduler=scheduler, scaler=scaler, mode='pretrain')
            print('Training start from epoch {} and learning rate {}.'.format(last_epoch, optimizer.param_groups[0]['lr']))
        else: 
            raise ValueError('IF YOU WANT TO RESUME TRAINING FROM PREVIOUS STATE, YOU SHOULD SET THE FILE PATH AS AN OPTION. PLZ CHECK --checkpoint_dir OPTION')
    else:
        last_epoch = 0 
    
    # attach network to cuda device. This line should come before wrapping the model with DDP 
    net.cuda()

    # setting DataParallel
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids = [args.gpu], find_unused_parameters=True)
    
    # attach optimizer to cuda device.
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()

    # setting for results' data frame
    train_losses = []
    val_losses = []
    

    # training
    for epoch in tqdm(range(last_epoch, last_epoch + args.epoch)):
        ts = time.time()
        net, train_loss = simMIM_train(net, partition, optimizer, scaler, epoch, args)
        net, val_loss = simMIM_validation(net, partition, epoch, args)
        
        # store result per epoch 
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        #scheduler.step(loss)
        scheduler.step()
        te = time.time()

        # visualize the result and saving the checkpoint
        # saving model. When use DDP, if you do not indicate device ids, the number of saved checkpoint would be the same as the number of process.
        if args.gpu == 0:
            print('Epoch {}. Train Loss: {:2.2f}. Validation Loss: {:2.2f}. Current learning rate {}. Took {:2.2f} sec'.format(epoch+1, train_loss, val_loss, optimizer.param_groups[0]['lr'],te-ts))
            checkpoint_dir = checkpoint_save(net, optimizer, save_dir, epoch, scheduler, scaler, args, mode='pretrain')
        
        torch.cuda.empty_cache()
            

    # summarize results
    result = {}
    result['train_losses'] = train_losses
    result['validation_losses'] = val_losses

    return vars(args), result
        

## ==================================== ##