import numpy as np
import os


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.distributed import DistributedSampler 


from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix


## ======= load module ======= ##
import model.model_Swin as Swin
import model.model_SwinV2 as SwinV2
from model.model_Swin import SwinTransformer3D
from model.model_SwinV2 import SwinTransformer3D_v2
from functools import partial

from util.utils import CLIreporter, save_exp_result, set_checkpoint_dir, checkpoint_save, checkpoint_load, saving_outputs, set_random_seed, load_imagenet_pretrained_weight, freeze_backbone
from util.optimizers import LAMB, LARS 
from util.lr_sched import CosineAnnealingWarmUpRestarts
from util.loss_functions  import loss_forward, mixup_loss, calculating_eval_metrics
from util.augmentation import mixup_data


import time
from tqdm import tqdm
import copy

def Swin_train(net, partition, optimizer, scaler, epoch, num_classes, args):
    train_sampler = DistributedSampler(partition['train'], shuffle=True)    

    trainloader = torch.utils.data.DataLoader(partition['train'],
                                             batch_size=args.batch_size,
                                             sampler=train_sampler,
                                             shuffle=False, 
                                             drop_last=True,
                                             num_workers=16)
    trainloader.sampler.set_epoch(epoch)
    net.train()
    

    losses = []
    if args.mixup:
        loss_fn = mixup_loss(num_classes)
    else:
        loss_fn = loss_forward(num_classes)

    eval_metrics = calculating_eval_metrics(num_classes=num_classes)
    
    optimizer.zero_grad()
    for i, data in enumerate(trainloader,0):
        images, labels = data
        if args.num_target: 
            labels = labels.float() 
        images = images.cuda()
        labels = labels.cuda() 
        """
        if loss is calculated inside the model class, the output from the model forward method would be [loss] * number of devices. In other words, len(net(images)) == n_gpus
        """
        #loss, pred, mask  = net(images)
        if args.mixup:
            mixed_images, labels_a, labels_b, lam = mixup_data(images, labels)
            with torch.cuda.amp.autocast():
                pred = net(mixed_images)
                loss = loss_fn(pred, labels_a, labels_b, lam)
        else:
            with torch.cuda.amp.autocast():
                pred = net(images)
                loss = loss_fn(pred, labels)
        losses.append(loss.item())
        eval_metrics.store(pred, labels)

        assert args.accumulation_steps >= 1
        if args.accumulation_steps == 1:
            #scaler.scale(loss).sum().backward()    # pytorch 2.xx
            scaler.scale(loss).backward()   # pytorch 1.xx 
            # gradient clipping 
            if args.gradient_clipping == True:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5, error_if_nonfinite=False)   # max_norm=1 from https://arxiv.org/pdf/2010.11929.pdf
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        elif args.accumulation_steps > 1:           # gradient accumulation
            loss = loss / args.accumulation_steps
            #scaler.scale(loss).sum().backward()    # pytorch 2.xx
            scaler.scale(loss).backward()   # pytorch 1.xx 
            if  (i + 1) % args.accumulation_steps == 0:
                # gradient clipping 
                if args.gradient_clipping == True:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5, error_if_nonfinite=False)   # max_norm=1 from https://arxiv.org/pdf/2010.11929.pdf
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()


    return net, np.mean(losses), eval_metrics.get_result()
        

def Swin_validation(net, partition, epoch, num_classes, args):
    val_sampler = DistributedSampler(partition['val'])  
    valloader = torch.utils.data.DataLoader(partition['val'],
                                            sampler=val_sampler, 
                                            batch_size=args.batch_size,
                                            shuffle=False,
                                            drop_last=True,
                                            num_workers=16)

    net.eval()
    valloader.sampler.set_epoch(epoch)
    
    losses = []
    loss_fn = loss_forward(num_classes)

    eval_metrics = calculating_eval_metrics(num_classes=num_classes)    
    with torch.no_grad():
        for i, data in enumerate(valloader,0):
            #images = images.to(f'cuda:{net.device_ids[0]}')
            images, labels = data
            if args.num_target: 
                labels = labels.float() 
            labels = labels.cuda()
            images = images.cuda()

            with torch.cuda.amp.autocast():
                pred = net(images)
                loss = loss_fn(pred, labels)
            losses.append(loss.item())
            eval_metrics.store(pred, labels)
                           
    return net, np.mean(losses), eval_metrics.get_result() 



def Swin_experiment(partition, num_classes, save_dir, args): #in_channels,out_dim
    if args.load_imagenet_pretrained:
        pretrained_weight = load_imagenet_pretrained_weight(args)       # This line return directory of imagenet_pretrained_weight
        pretrained2d = True
        simMIM_pretrained = False
    else: 
        pretrained_weight = None 
        pretrained2d = False 
        simMIM_pretrained = False

    if args.pretrained_model is not None: 
        pretrained_weight = args.pretrained_model
        pretrained2d = False 
        simMIM_pretrained = True 

        
    # setting network 
    if args.model.find('V2') != -1: 
        net = SwinV2.__dict__[args.model](pretrained=pretrained_weight, pretrained2d=pretrained2d, simMIM_pretrained=simMIM_pretrained, drop_rate=args.projection_drop, num_classes=num_classes)
    else: 
        net = Swin.__dict__[args.model](pretrained=pretrained_weight, pretrained2d=pretrained2d, simMIM_pretrained=simMIM_pretrained, drop_rate=args.projection_drop, num_classes=num_classes)
    if args.torchscript:
        torch._C._jit_set_autocast_mode(True)
        net = torch.jit.script(net)
    checkpoint_dir = set_checkpoint_dir(save_dir=save_dir, args=args)


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
    """
    #for one cycle scheduler 
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

    # loading pre-trained model or last checkpoint 
    if args.resume == False: # loading pre-trained model
        last_epoch = 0 
    elif args.resume == True:  # loading last checkpoint 
        if args.checkpoint_dir != None:
            net, optimizer, scheduler, last_epoch, optimizer.param_groups[0]['lr'], scaler = checkpoint_load(net=net, checkpoint_dir=args.checkpoint_dir, optimizer=optimizer, scheduler=scheduler, scaler=scaler, mode='pretrain')
            print('Training start from epoch {} and learning rate {}.'.format(last_epoch, optimizer.param_groups[0]['lr']))
        else: 
            raise ValueError('IF YOU WANT TO RESUME TRAINING FROM PREVIOUS STATE, YOU SHOULD SET THE FILE PATH AS AN OPTION. PLZ CHECK --checkpoint_dir OPTION')
   
    # attach network to cuda device. This line should come before wrapping the model with DDP 
    net.cuda()

    # setting DataParallel
    if args.backbone_freeze:
        net = freeze_backbone(net)
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids = [args.gpu], find_unused_parameters=True)
    # pytorch 2.0
    #net = torch.compile(net)
    
    # attach optimizer to cuda device.
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()

    # setting for results' data frame
    train_losses = []
    val_losses = []

    previous_performance = {}
    previous_performance['ACC'] = [0.0]
    previous_performance['abs_loss'] = [100000.0]
    previous_performance['mse_loss'] = [100000.0]
    if num_classes == 2:
        previous_performance['AUROC'] = [0.0]


    #### training
    for epoch in tqdm(range(last_epoch, last_epoch + args.epoch)):
        ts = time.time()
        net, train_loss, train_performance = Swin_train(net, partition, optimizer, scaler, epoch, num_classes, args)
        net, val_loss, val_performance = Swin_validation(net, partition, epoch, num_classes, args)
        

        
        #scheduler.step(loss)
        scheduler.step()
        te = time.time()

        # visualize the result and saving the checkpoint
        # saving model. When use DDP, if you do not indicate device ids, the number of saved checkpoint would be the same as the number of process.
        if args.gpu == 0:
            # store result per epoch 
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            print('Epoch {}. Train Loss: {:2.2f}. Validation Loss: {:2.2f}. \n Training Prediction Performance: {}. \n Validation Prediction Performance: {}. \n Current learning rate {}. Took {:2.2f} sec'.format(epoch+1, train_loss, val_loss, train_performance, val_performance, optimizer.param_groups[0]['lr'],te-ts))
            if 'ACC' or 'AUROC' in val_performance.keys():
                if args.metric == 'ACC':
                    previous_performance['ACC'].append(val_performance['ACC'])
                    if val_performance['ACC'] > max(previous_performance['ACC'][:-1]):
                        checkpoint_save(net, optimizer, checkpoint_dir, epoch, scheduler, scaler, args, val_performance,mode='finetune')
                elif args.metric == 'AUROC': 
                    previous_performance['AUROC'].append(val_performance['AUROC'])
                    if val_performance['AUROC'] > max(previous_performance['AUROC'][:-1]):
                        checkpoint_save(net, optimizer, checkpoint_dir, epoch, scheduler, scaler, args, val_performance,mode='finetune')
            
            if 'abs_loss' or 'mse_loss' in val_performance.keys():
                if args.metric == 'abs_loss': 
                    previous_performance['abs_loss'].append(val_performance['abs_loss'])
                    if val_performance['abs_loss'] < min(previous_performance['abs_loss'][:-1]):
                        checkpoint_save(net, optimizer, checkpoint_dir, epoch, scheduler, scaler, args, val_performance,mode='finetune')
                elif args.metric == 'mse_loss':
                    previous_performance['mse_loss'].append(val_performance['mse_loss'])
                    if val_performance['mse_loss'] < min(previous_performance['mse_loss'][:-1]):
                        checkpoint_save(net, optimizer, checkpoint_dir, epoch, scheduler, scaler, args, val_performance,mode='finetune')

        torch.cuda.empty_cache()
    
    # summarize results
    result = {}
    result['train_losses'] = train_losses
    result['validation_losses'] = val_losses

    return vars(args), result, checkpoint_dir
        

## ==================================== ##