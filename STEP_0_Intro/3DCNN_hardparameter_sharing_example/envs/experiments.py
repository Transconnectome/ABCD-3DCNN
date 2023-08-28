from tqdm.auto import tqdm ##progress
import time
import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from utils.optimizer import SGDW

from envs.loss_functions import calculating_loss, calculating_eval_metrics
from utils.lr_scheduler import *
from utils.early_stopping import * 
from utils.utils import CLIreporter, checkpoint_save, checkpoint_load
from utils.utils import combine_pred_subjid, combine_emb_subjid
from tqdm import tqdm

### ========= Train,Validate, and Test ========= ###
'''The process of calcuating loss and accuracy metrics is as follows.
   1) sequentially calculate loss and accuracy metrics of target labels with for loop.
   2) store the result information with dictionary type.
   3) return the dictionary, which form as {'cat_target':value, 'num_target:value}
   This process is intended to easily deal with loss values from each target labels.'''


'''All of the loss from predictions are summated and this loss value is used for backpropagation.'''
# define training step
def train(net,partition,optimizer, args):
    '''GradScaler is for calculating gradient with float 16 type'''
    scaler = torch.cuda.amp.GradScaler()

    trainloader = torch.utils.data.DataLoader(partition['train'],
                                             batch_size=args.batch_size,
                                             shuffle=True,
                                             num_workers=12)

    net.train()
    loss_fn = calculating_loss(device=f'cuda:{net.device_ids[0]}', args=args)
    eval_metrices_log = calculating_eval_metrics(args=args)
    
    optimizer.zero_grad()
    for i, data in enumerate(trainloader,0):
        image, targets = data
        image = image.to(f'cuda:{net.device_ids[0]}')
        # feed forward network with floating point 16
        with torch.cuda.amp.autocast():
            output = net(image)
        loss, train_loss = loss_fn(targets, output)
        eval_metrices_log.store(output, targets)
        if args.accumulation_steps:
            loss = loss / args.accumulation_steps
            # pytorch 2.0
            #scaler.scale(loss).sum().backward()
            scaler.scale(loss).backward()
            if  (i + 1) % args.accumulation_steps == 0:
                # gradient clipping 
                if args.gradient_clipping == True:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5, error_if_nonfinite=False)   
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
    # calculating total loss and acc of separate mini-batch
    if args.cat_target:
        for cat_target in args.cat_target:
            train_loss[cat_target] = np.mean(train_loss[cat_target])
    if args.num_target:
        for num_target in args.num_target:
            train_loss[num_target] = np.mean(train_loss[num_target])

    return net, train_loss, eval_metrices_log.get_result()


# define validation step
def validate(net,partition, scheduler,args):
    valloader = torch.utils.data.DataLoader(partition['val'],
                                           batch_size=args.batch_size,
                                           shuffle=True,
                                           num_workers=12)

    net.eval()

    loss_fn = calculating_loss(device=f'cuda:{net.device_ids[0]}', args=args)
    eval_metrices_log = calculating_eval_metrics(args=args)

    with torch.no_grad():
        for i, data in enumerate(valloader,0):
            image, targets = data
            image = image.to(f'cuda:{net.device_ids[0]}')
            with torch.cuda.amp.autocast():
                output = net(image)
            loss, val_loss = loss_fn(targets, output)
            eval_metrices_log.store(output, targets)

    if args.cat_target:
        for cat_target in args.cat_target:
            val_loss[cat_target] = np.mean(val_loss[cat_target])
    if args.num_target:
        for num_target in args.num_target:
            val_loss[num_target] = np.mean(val_loss[num_target])

    # learning rate scheduler
    #scheduler.step(sum(val_acc.values())) #if you want to use ReduceLROnPlateau lr scheduler, activate this line and deactivate the below line 
    scheduler.step()

    return val_loss, eval_metrices_log.get_result()



# define test step
def test(net,partition,args):
    # flag for data shuffle 
    data_shuffle = False 

    assert data_shuffle == False 
    testloader = torch.utils.data.DataLoader(partition['test'],
                                            batch_size=args.batch_size,
                                            shuffle=data_shuffle,
                                            num_workers=12)

    net.eval()

    device = 'cuda:0'

    loss_fn = calculating_loss(device=device, args=args)
    eval_metrices_log = calculating_eval_metrics(args=args)

    with torch.no_grad():
        for i, data in enumerate(tqdm(testloader),0):
            image, targets = data
            image = image.to(device)

            with torch.cuda.amp.autocast():
                output = net(image)
            loss, test_loss = loss_fn(targets, output)
            eval_metrices_log.store(output, targets)
    

    if args.get_predicted_score:
        if args.cat_target:
            for cat_target in args.cat_target: 
                eval_metrices_log.pred_score[cat_target] = eval_metrices_log.pred[cat_target].squeeze(-1).tolist()
                eval_metrices_log.pred_score[cat_target] = combine_pred_subjid(eval_metrices_log.pred_score[cat_target], partition['test'].image_files)
                eval_metrices_log.pred_score["predicted_%s" % cat_target] = eval_metrices_log.pred_score.pop(cat_target)
        if args.num_target:
            for num_target in args.num_target: 
                eval_metrices_log.pred_score[num_target] = eval_metrices_log.pred[num_target].squeeze(-1).tolist()
                eval_metrices_log.pred_score[num_target] = combine_pred_subjid(eval_metrices_log.pred_score[num_target], partition['test'].image_files)
                eval_metrices_log.pred_score["predicted_%s" % num_target] = eval_metrices_log.pred_score.pop(num_target)
        return eval_metrices_log.get_result(), None, eval_metrices_log.pred_score
    else: 
        return eval_metrices_log.get_result(), None, None
## ============================================ ##



## ========= Experiment =============== ##
def experiment(partition, subject_data, save_dir, args): #in_channels,out_dim
    targets = args.cat_target + args.num_target

    # ResNet 
    if args.model == 'resnet3D50':
        import models.resnet3d as resnet3d #model script
        net = resnet3d.resnet3D50(subject_data, args)
    elif args.model == 'resnet3D101':
        import models.resnet3d as resnet3d #model script
        net = resnet3d.resnet3D101(subject_data, args)
    elif args.model == 'resnet3D152':
        import models.resnet3d as resnet3d #model script
        net = resnet3d.resnet3D152(subject_data, args)
    # DenseNet
    elif args.model == 'densenet3D121':
        import models.densenet3d as densenet3d #model script
        net = densenet3d.densenet3D121(subject_data, args)
    elif args.model == 'densenet3D169':
        import models.densenet3d as densenet3d #model script
        net = densenet3d.densenet3D169(subject_data, args) 
    elif args.model == 'densenet3D201':
        import models.densenet3d as densenet3d #model script
        net = densenet3d.densenet3D201(subject_data, args)
    # EfficientNet V1 
    elif args.model.find('efficientnet3D') != -1: 
        import models.efficientnet3d as efficientnet3d
        net = efficientnet3d.efficientnet3D(subject_data,args)
    # Swin Transformer V1 & V2
    elif args.model.find('swinV1') != -1: 
        import models.swinV1 as swinv1 
        net = swinv1.__dict__[args.model](subject_data=subject_data, args=args)
    elif args.model.find('swinV2') != -1: 
        import models.swinV2 as swinv2
        net = swinv2.__dict__[args.model](subject_data=subject_data, args=args, img_size=args.resize)


    # load checkpoint
    if args.checkpoint_dir is not None: 
        net = checkpoint_load(net, args.checkpoint_dir)


    # setting optimizer 
    if args.optim == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)
    elif args.optim == 'SGDW':
        optimizer = SGDW(net.parameters(), lr=0, momentum=0.9)
    elif args.optim == 'Adam':
        optimizer = optim.Adam(net.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    elif args.optim == 'AdamW': 
        optimizer = optim.AdamW(net.parameters(), lr=0, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    else:
        raise ValueError('In-valid optimizer choice')

    # setting learning rate schedluer
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,'max', patience=20) #if you want to use this scheduler, you should activate the line 134 of envs/experiments.py
    #scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=30, T_mult=2, eta_max=args.lr, T_up=5, gamma=0.5)
    scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=args.epoch, T_mult=2, eta_max=args.lr, T_up=5, gamma=0.5)
    
    
    # applying early stopping 
    early_stopping = EarlyStopping(patience=30)


    # setting DataParallel
    devices = []
    for d in range(torch.cuda.device_count()):
        devices.append(d)
    net = nn.DataParallel(net, device_ids = devices)
    # pytorch 2.0 
    #net = torch.compile(net)
    

    # attach network and optimizer to cuda device
    net.cuda()


    """
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(f'cuda:{net.device_ids[0]}')
    """
    
    # setting for results' data frame
    train_losses = {}
    train_accs = {}
    val_losses = {}
    val_accs = {}

    for target_name in targets:
        train_losses[target_name] = []
        train_accs[target_name] = [-10000.0]
        #train_accs[target_name] = [10000.0]
        val_losses[target_name] = []
        val_accs[target_name] = [-10000.0]
        #val_accs[target_name] = [10000.0]
        
    for epoch in tqdm(range(args.epoch)):
        ts = time.time()
        net, train_loss, train_acc = train(net,partition,optimizer, args)
        torch.cuda.empty_cache()
        val_loss, val_acc = validate(net,partition,scheduler,args)
        te = time.time()

         # sorting the results
        if args.cat_target: 
            for cat_target in args.cat_target: 
                train_losses[cat_target].append(train_loss[cat_target])
                train_accs[cat_target].append(train_acc[cat_target]['ACC'])
                val_losses[cat_target].append(val_loss[cat_target])
                val_accs[cat_target].append(val_acc[cat_target]['ACC'])
                early_stopping(val_acc[cat_target]['ACC'])
        if args.num_target: 
            for num_target in args.num_target: 
                train_losses[num_target].append(train_loss[num_target])
                train_accs[num_target].append(train_acc[num_target]['r_square'])
                #train_accs[num_target].append(train_acc[num_target]['abs_loss'])
                val_losses[num_target].append(val_loss[num_target])
                val_accs[num_target].append(val_acc[num_target]['r_square'])
                #val_accs[num_target].append(val_acc[num_target]['abs_loss'])
                early_stopping(val_acc[num_target]['r_square'])
                #early_stopping(val_acc[num_target]['abs_loss'])            

        # visualize the result
        CLIreporter(targets, train_loss, train_acc, val_loss, val_acc)
        print('Epoch {}. Current learning rate {}. Took {:2.2f} sec'.format(epoch+1,optimizer.param_groups[0]['lr'],te-ts))

        # saving the checkpoint
        #if train_acc[targets[0]] > 0.9:
        checkpoint_dir = checkpoint_save(net, save_dir, epoch, val_acc, val_accs, args)

        # early stopping 
        #if early_stopping.early_stop: 
        #    break

    # test
    net.to('cpu')
    torch.cuda.empty_cache()

    net = checkpoint_load(net, checkpoint_dir)

    # setting DataParallel
    devices = []
    for d in range(torch.cuda.device_count()):
        devices.append(d)
    net = nn.DataParallel(net, device_ids = devices)
    net.cuda()

    test_acc, confusion_matrices, predicted_score = test(net, partition, args)

    # summarize results
    result = {}
    result['train_losses'] = train_losses
    result['train_accs'] = train_accs
    result['val_losses'] = val_losses
    result['val_accs'] = val_accs

    result['train_acc'] = train_acc
    result['val_acc'] = val_acc
    result['test_acc'] = test_acc 
    if args.get_predicted_score: 
        result['predicted_score'] = predicted_score  
    
    if confusion_matrices != None:
        result['confusion_matrices'] = confusion_matrices

    return vars(args), result
## ==================================== ##




def extract_embedding(net, partition_dataset, args):
    # flag for data shuffle 
    data_shuffle = False 

    assert data_shuffle == False 
    dataloader = torch.utils.data.DataLoader(partition_dataset,
                                            batch_size=args.batch_size,
                                            shuffle=data_shuffle,
                                            num_workers=24)

    net.eval()
    if hasattr(net, 'module'):
        device = net.device_ids[0]
    else: 
        device = f'cuda:{args.gpus[0]}'

    embeddings = torch.tensor([])
    with torch.no_grad():
        for i, data in enumerate(tqdm(dataloader),0):
            image, _ = data
            image = image.to(device)
            conv_emb = net._hook_embeddings(image).detach().cpu()
            embeddings = torch.cat([embeddings, conv_emb]) 

    embeddings = combine_emb_subjid(embeddings, partition_dataset.image_files)

    return embeddings