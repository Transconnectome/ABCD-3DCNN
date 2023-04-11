import random
from collections import defaultdict

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix

from envs.loss_functions import calculating_loss_acc, calc_acc_auroc, calc_R2, calc_MAE_MSE_R2

### ========= Train,Validate, and Test ========= ###
'''The process of calcuating loss and accuracy metrics is as follows.
   1) sequentially calculate loss and accuracy metrics of target labels with for loop.
   2) store the result information with dictionary type.
   3) return the dictionary, which form as {'cat_target':value, 'num_target:value}
   This process is intended to easily deal with loss values from each target labels.'''


'''All of the loss from predictions are summated and this loss value is used for backpropagation.'''    
# define training step
def train(net, trainloader, optimizer, scaler, args):
    net.train()
    train_loss = defaultdict(list)
    train_acc =  defaultdict(list)
    
    for i, data in enumerate(trainloader,0):
        image, targets = data if len(data) == 2 else (data, [])
        image = image.to(f'cuda:{net.device_ids[0]}')
        with torch.cuda.amp.autocast():
            output = net(image)
            loss = calculating_loss_acc(targets, output, train_loss, train_acc, net, args)
            loss = loss / args.accumulation_steps if args.accumulation_steps else loss
        
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if args.accumulation_steps:
            if ((i + 1) % args.accumulation_steps == 0) or (i == (len(trainloader)-1)):
                scaler.step(optimizer)
                scaler.update()
        else:
            scaler.step(optimizer)
            scaler.update()

    # calculating total loss and acc of separate mini-batch
    for target in train_loss:
        train_loss[target] = np.mean(train_loss[target])
        train_acc[target] = np.mean(train_acc[target])

    return net, train_loss, train_acc


# define validation step
def validate(net, valloader, scheduler, args):
    val_loss = defaultdict(list)
    val_acc = defaultdict(list)

    net.eval()
    with torch.no_grad():
        for i, data in enumerate(valloader,0):
            image, targets = data if len(data) == 2 else (data, [])
            image = image.to(f'cuda:{net.device_ids[0]}')
            with torch.cuda.amp.autocast():
                output = net(image)
                loss = calculating_loss_acc(targets, output, val_loss, val_acc, net, args)

    for target in val_loss:
        val_loss[target] = np.mean(val_loss[target])
        val_acc[target] = np.mean(val_acc[target])

    # learning rate scheduler
    if scheduler:
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(sum(val_acc.values()))
        else:
            scheduler.step()

    return val_loss, val_acc


def calc_confusion_matrix(confusion_matrices, curr_target, output, y_true):
    _, predicted = torch.max(output.data,1)
    tn, fp, fn, tp = confusion_matrix(y_true.numpy(), predicted.numpy()).ravel()
    confusion_matrices[curr_target]['True Positive'] = int(tp)
    confusion_matrices[curr_target]['True Negative'] = int(tn)
    confusion_matrices[curr_target]['False Positive'] = int(fp)
    confusion_matrices[curr_target]['False Negative'] = int(fn) 
    
    
# define test step
def test(net, partition, args):
    def seed_worker(worker_id):
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
    g = torch.Generator()
    g.manual_seed(args.seed)

    testloader = torch.utils.data.DataLoader(partition['test'],
                                         batch_size=args.test_batch_size,
                                         shuffle=False,
                                         num_workers=args.num_workers,
                                         persistent_workers=True,
                                         worker_init_fn=seed_worker,
                                         generator=g)
    net.eval()
    if hasattr(net, 'module'):
        device = net.device_ids[0]
    else: 
        device = 'cuda:0' if args.sbatch =='True' else f'cuda:{args.gpus[0]}'
    
    outputs = defaultdict(list)
    y_true = defaultdict(list)
    test_loss = defaultdict(list) 
    test_acc = defaultdict(list)
    confusion_matrices = defaultdict(defaultdict)

    with torch.no_grad():
        for i, data in enumerate(tqdm(testloader),0):
            image, targets = data if len(data) == 2 else (data, [])
            image = image.to('cuda')
            output = net(image)
            if 'MM' not in args.model or args.mode != 'pretraining':
                for curr_target in output:
                    outputs[curr_target].append(output[curr_target].cpu())
                    y_true[curr_target].append(targets[curr_target].cpu())
            else:
                loss = calculating_loss_acc(targets, outputs, test_loss, test_acc, net, args)
    
    # caculating ACC and R2 at once  
    if 'MM' in args.model and args.mode == 'pretraining':
        test_acc = np.mean(test_loss[args.metric])
        return test_acc, None

    for curr_target in outputs:
        outputs[curr_target] = torch.cat(outputs[curr_target])
        y_true[curr_target] = torch.cat(y_true[curr_target])
            
        acc_func = calc_acc_auroc if curr_target in args.cat_target else calc_MAE_MSE_R2
        curr_acc = acc_func(outputs[curr_target], y_true[curr_target], args, None)
        test_acc[curr_target].append(curr_acc)
        
        if curr_target in args.confusion_matrix:
            calc_confusion_matrix(confusion_matrices, curr_target,
                                  outputs[curr_target], y_true[curr_target])

    return test_acc, confusion_matrices

## ============================================ ##
