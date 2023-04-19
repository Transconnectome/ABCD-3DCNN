import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, roc_auc_score

from envs.loss_functions import calculating_loss, calculating_eval_metrics, MixUp_loss, CutMix_loss, C_MixUp_loss
from utils.utils import combine_pred_subjid, combine_emb_subjid, freeze_conv, unfreeze_conv
from tqdm import tqdm
import os 

## ========= Reference ========= #
#(ref) pytorch-bayesian (training and evaluation): https://github.com/IntelLabs/bayesian-torch/bayesian_torch/examples/main_bayesian_flipout_imagenet.py


### ========= Train,Validate, and Test ========= ###
'''The process of calcuating loss and accuracy metrics is as follows.
   1) sequentially calculate loss and accuracy metrics of target labels with for loop.
   2) store the result information with dictionary type.
   3) return the dictionary, which form as {'cat_target':value, 'num_target:value}
   This process is intended to easily deal with loss values from each target labels.'''


'''All of the loss from predictions are summated and this loss value is used for backpropagation.'''

# define training step
def train(net,partition,optimizer, global_steps, args):
    '''GradScaler is for calculating gradient with float 16 type'''
    scaler = torch.cuda.amp.GradScaler()

    trainloader = torch.utils.data.DataLoader(partition['train'],
                                             batch_size=args.batch_size,
                                             shuffle=True,
                                             num_workers=12)

    net.train()

    if args.conv_unfreeze_iter:
        if global_steps < args.conv_unfreeze_iter:
            net = freeze_conv(net)
        elif global_steps == args.conv_unfreeze_iter:
            net = unfreeze_conv(net, global_steps, args)
            print('From now, convolution layers will be updated.')
        else:
            net = unfreeze_conv(net, global_steps, args)
    if args.mixup != 0.0:
        loss_fn = MixUp_loss(args=args, device=f'cuda:{net.device_ids[0]}', beta=args.beta) # 
    elif args.cutmix != 0.0 : 
        loss_fn = CutMix_loss(args=args, device=f'cuda:{net.device_ids[0]}', beta=args.beta)
    elif args.c_mixup != 0.0: 
        loss_fn = C_MixUp_loss(args=args, device=f'cuda:{net.device_ids[0]}', beta=args.beta, cutmix=True)
    elif args.manifold_mixup != 0.0:
        loss_fn = MixUp_loss(args=args, device=f'cuda:{net.device_ids[0]}', beta=args.beta) # 
    else:
        loss_fn = calculating_loss(device=f'cuda:{net.device_ids[0]}', args=args)
    eval_metrices_log = calculating_eval_metrics(args=args)


    optimizer.zero_grad()
    for i, data in enumerate(trainloader,0):
        image, targets = data
        image = image.to(f'cuda:{net.device_ids[0]}')

        if args.mixup != 0.0:
            r = np.random.rand(1) 
            if r < args.mixup: 
                image, label_a, label_b, lam = loss_fn.mixup_data(image, targets) 
                with torch.cuda.amp.autocast():
                    output, kl = net(image)
                    scaled_kl = (kl.data[0] / image.shape[0])
                loss, train_loss = loss_fn(output, label_a, label_b, lam, kl=scaled_kl)
            else: 
                with torch.cuda.amp.autocast():
                    output, kl = net(image)
                    scaled_kl = (kl.data[0] / image.shape[0])
                loss, train_loss = loss_fn(output, targets, kl=scaled_kl)
        elif args.cutmix != 0.0: 
            r = np.random.rand(1) 
            if r < args.cutmix: 
                image, label_a, label_b, lam = loss_fn.generate_mixed_sample(image, targets) 
                with torch.cuda.amp.autocast():
                    output, kl = net(image)
                    scaled_kl = (kl.data[0] / image.shape[0])
                    loss, train_loss = loss_fn(output, label_a, label_b, lam, kl=scaled_kl)
            else: 
                with torch.cuda.amp.autocast():
                    output, kl = net(image)
                    scaled_kl = (kl.data[0] / image.shape[0])
                    loss, train_loss = loss_fn(output, targets, kl=scaled_kl)
        elif args.c_mixup!= 0.0: 
            r = np.random.rand(1) 
            if r < args.c_mixup: 
                image, label_a, label_b, lam = loss_fn.mixup_data(image, targets) 
                with torch.cuda.amp.autocast():
                    output, kl = net(image)
                    scaled_kl = (kl.data[0] / image.shape[0])
                loss, train_loss = loss_fn(output, label_a, label_b, lam, kl=scaled_kl)
            else: 
                with torch.cuda.amp.autocast():
                    output, kl = net(image)
                    scaled_kl = (kl.data[0] / image.shape[0])
                loss, train_loss = loss_fn(output, targets, kl=scaled_kl)            
        elif args.manifold_mixup != 0.0:
            """
            ref: https://github.com/vikasverma1077/manifold_mixup/tree/870ef77caaa5092144d82c56f26b07b29eefabec
            """
            r = np.random.rand(1) 
            if r < args.manifold_mixup: 
                y = targets[loss_fn.target].to(f'cuda:{net.device_ids[0]}')
                with torch.cuda.amp.autocast():
                    lam = loss_fn.get_lambda(tensor=True).to(f'cuda:{net.device_ids[0]}')  
                    output, label_a, label_b = net(image, y=y,target=loss_fn.target, manifold_mixup=True, lam=lam.tile((len(net.device_ids),))) # because of Data Parallel, replicate lambda. the size of lambda = (# of device,)
                    loss, train_loss = loss_fn(output, label_a, label_b, lam, kl=scaled_kl)    
            else: 
                with torch.cuda.amp.autocast():
                    output, kl = net(image)
                    scaled_kl = (kl.data[0] / image.shape[0])
                loss, train_loss = loss_fn(targets, output, kl=scaled_kl)
        else: 
            with torch.cuda.amp.autocast():
                output, kl = net(image)
                scaled_kl = (kl.data[0] / image.shape[0])
            loss, train_loss = loss_fn(targets, output, kl=scaled_kl)
        
        eval_metrices_log.store(output, targets)
        if args.accumulation_steps:
            loss = loss / args.accumulation_steps
            scaler.scale(loss).backward()
            if  (i + 1) % args.accumulation_steps == 0:
                # gradient clipping 
                if args.gradient_clipping == True:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1, error_if_nonfinite=False)   
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                global_steps += 1 
                # decide whther to freeze layers or not 
                if args.conv_unfreeze_iter:
                    if global_steps < args.conv_unfreeze_iter:
                        net = freeze_conv(net)
                    elif global_steps == args.conv_unfreeze_iter:
                        net = unfreeze_conv(net, global_steps, args)
                        print('From now, convolution layers will be updated.')
                    else:
                        net = unfreeze_conv(net, global_steps, args)
            


    # calculating total loss and acc of separate mini-batch
    if args.cat_target:
        for cat_target in args.cat_target:
            train_loss[cat_target] = np.mean(train_loss[cat_target])

    if args.num_target:
        for num_target in args.num_target:
            train_loss[num_target] = np.mean(train_loss[num_target])


    return net, train_loss, eval_metrices_log.get_result(),  global_steps


# define validation step
def validate(net,partition,scheduler,args, num_monte_carlo=20):
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
            
            output_mc = {}
            for k in targets.keys(): 
                output_mc[k] = []
            scaled_kls = [] 
            with torch.cuda.amp.autocast():
                for _ in range(num_monte_carlo): 
                    output, kl = net(image)             # original output size = (B, num_classes)
                    for k in output.keys(): 
                        output_mc[k].append(output[k].unsqueeze(1)) # to get results from several monte carlo sampling, each sampling results are stored as the shape of (B,n um_monte_carlo, num_classes) 
                    scaled_kl = (kl.data[0] / image.shape[0])
                    scaled_kls.append(scaled_kl)
            for k in output_mc.keys(): 
                output_mc[k] = torch.mean(torch.cat(output_mc[k], axis=1), axis=1)
            loss, val_loss = loss_fn(targets, output_mc, kl=torch.mean(torch.tensor(scaled_kls)))
            """
            with torch.cuda.amp.autocast():
                output, kl = net(image)
                scaled_kl = (kl.data[0] / image.shape[0])
            loss, val_loss = loss_fn(targets, output, kl=scaled_kl)
            """
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
def test(net,partition,args, num_monete_carlo=None):
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

            output_mc = {}
            output_mc_mean = {}
            for k in targets.keys(): 
                output_mc[k] = []
                output_mc_mean[k] = []
            scaled_kls = [] 
            with torch.cuda.amp.autocast():
                if num_monete_carlo:
                    for _ in range(num_monete_carlo): 
                        output, kl = net(image)             # original output size = (B, num_classes)
                        for k in output.keys(): 
                            output_mc[k].append(output[k].unsqueeze(1)) # to get results from several monte carlo sampling, each sampling results are stored as the shape of (B,n um_monte_carlo, num_classes) 
                        scaled_kl = (kl.data[0] / image.shape[0])
                        scaled_kls.append(scaled_kl)
                    for k in output_mc.keys(): 
                        output_mc[k] = torch.cat(output_mc[k], axis=1)
                        output_mc_mean[k] = torch.mean(output_mc[k], axis=1)
                    loss, test_loss = loss_fn(targets, output_mc_mean, kl=torch.mean(torch.tensor(scaled_kls)))
                    eval_metrices_log.store(output_mc_mean, targets, pred_score=output_mc)
                    
                else: 
                    output, kl = net(image)
                    scaled_kl = (kl.data[0] / image.shape[0])
                    loss, test_loss = loss_fn(targets, output, kl=scaled_kl)
                    eval_metrices_log.store(output, targets, pred_score=output)

    

    if args.get_predicted_score:
        if args.cat_target:
            for cat_target in args.cat_target: 
                eval_metrices_log.pred_score[cat_target] = eval_metrices_log.pred_score[cat_target].squeeze(-1).tolist()
                eval_metrices_log.pred_score[cat_target] = combine_pred_subjid(eval_metrices_log.pred_score[cat_target], partition['test'].image_files)
                eval_metrices_log.pred_score["predicted_%s" % cat_target] = eval_metrices_log.pred_score.pop(cat_target)
        if args.num_target:
            for num_target in args.num_target: 
                eval_metrices_log.pred_score[num_target] = eval_metrices_log.pred_score[num_target].squeeze(-1).tolist()
                eval_metrices_log.pred_score[num_target] = combine_pred_subjid(eval_metrices_log.pred_score[num_target], partition['test'].image_files)
                eval_metrices_log.pred_score["predicted_%s" % num_target] = eval_metrices_log.pred_score.pop(num_target)
        return eval_metrices_log.get_result(), None, eval_metrices_log.pred_score
    else: 
        return eval_metrices_log.get_result(), None, None
    


## ============================================ ##



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
        device = 'cuda:0'



    embeddings = torch.tensor([])
    with torch.no_grad():
        for i, data in enumerate(tqdm(dataloader),0):
            image, _ = data
            image = image.to(device)
            conv_emb = net._hook_embeddings(image).detach().cpu()
            embeddings = torch.cat([embeddings, conv_emb]) 

    embeddings = combine_emb_subjid(embeddings, partition_dataset.image_files)

    return embeddings