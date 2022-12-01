import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, roc_auc_score

from envs.loss_functions import calculating_loss, calculating_acc
from utils.utils import combine_pred_subjid
from tqdm import tqdm
import os 

### ========= Train,Validate, and Test ========= ###
'''The process of calcuating loss and accuracy metrics is as follows.
   1) sequentially calculate loss and accuracy metrics of target labels with for loop.
   2) store the result information with dictionary type.
   3) return the dictionary, which form as {'cat_target':value, 'num_target:value}
   This process is intended to easily deal with loss values from each target labels.'''


'''All of the loss from predictions are summated and this loss value is used for backpropagation.'''

# define training step
def train(net,partition,optimizer,args):
    '''GradScaler is for calculating gradient with float 16 type'''
    scaler = torch.cuda.amp.GradScaler()

    trainloader = torch.utils.data.DataLoader(partition['train'],
                                             batch_size=args.batch_size,
                                             shuffle=True,
                                             num_workers=24)

    net.train()

    correct = {}
    total = {}


    train_loss = {}
    train_acc = {}  # collecting r-square of contiunuous varibale directly
    

    if args.cat_target:
        for cat_target in args.cat_target:
            correct[cat_target] = 0
            total[cat_target] = 0
            train_acc[cat_target] = []
            train_loss[cat_target] = []

    if args.num_target:
        for num_target in args.num_target:
            train_acc[num_target] = []
            train_loss[num_target] = []


    optimizer.zero_grad()
    for i, data in enumerate(trainloader,0):
        image, targets = data
        image = image.to(f'cuda:{net.device_ids[0]}')
        # feed forward network with floating point 16
        with torch.cuda.amp.autocast():
            output = net(image)
        loss, train_loss = calculating_loss(targets, output, train_loss,net, args)
        train_acc = calculating_acc(targets, output, correct, total, train_acc, net, args)
        if args.accumulation_steps:
            loss = loss / args.accumulation_steps
            scaler.scale(loss).backward()
            if  (i + 1) % args.accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()


    # calculating total loss and acc of separate mini-batch
    if args.cat_target:
        for cat_target in args.cat_target:
            train_acc[cat_target] = np.mean(train_acc[cat_target])
            train_loss[cat_target] = np.mean(train_loss[cat_target])

    if args.num_target:
        for num_target in args.num_target:
            train_acc[num_target] = np.mean(train_acc[num_target])
            train_loss[num_target] = np.mean(train_loss[num_target])


    return net, train_loss, train_acc


# define validation step
def validate(net,partition,scheduler,args):
    valloader = torch.utils.data.DataLoader(partition['val'],
                                           batch_size=args.batch_size,
                                           shuffle=True,
                                           num_workers=24)

    net.eval()

    correct = {}
    total = {}


    val_loss = {}
    val_acc = {}  # collecting r-square of contiunuous varibale directly


    if args.cat_target:
        for cat_target in args.cat_target:
            correct[cat_target] = 0
            total[cat_target] = 0
            val_acc[cat_target] = []
            val_loss[cat_target] = []

    if args.num_target:
        for num_target in args.num_target:
            val_acc[num_target] = []
            val_loss[num_target] = []


    with torch.no_grad():
        for i, data in enumerate(valloader,0):
            image, targets = data
            image = image.to(f'cuda:{net.device_ids[0]}')

            with torch.cuda.amp.autocast():
                output = net(image)
            loss, val_loss = calculating_loss(targets, output, val_loss,net, args)
            val_acc = calculating_acc(targets, output, correct, total, val_acc, net, args)

    if args.cat_target:
        for cat_target in args.cat_target:
            val_acc[cat_target] = np.mean(val_acc[cat_target])
            val_loss[cat_target] = np.mean(val_loss[cat_target])

    if args.num_target:
        for num_target in args.num_target:
            val_acc[num_target] = np.mean(val_acc[num_target])
            val_loss[num_target] = np.mean(val_loss[num_target])


    # learning rate scheduler
    #scheduler.step(sum(val_acc.values())) #if you want to use ReduceLROnPlateau lr scheduler, activate this line and deactivate the below line 
    scheduler.step()

    return val_loss, val_acc



# define test step
def test(net,partition,args):
    # flag for data shuffle 
    data_shuffle = False 

    assert data_shuffle == False 
    testloader = torch.utils.data.DataLoader(partition['test'],
                                            batch_size=args.batch_size,
                                            shuffle=data_shuffle,
                                            num_workers=24)

    net.eval()
    if hasattr(net, 'module'):
        device = net.device_ids[0]
    else: 
        if args.sbatch =='True':
            device = 'cuda:0'
        else:
            device = f'cuda:{args.gpus[0]}'
    #correct = {}
    #y_true = {}
    
    outputs = {}
    y_true = {}
    test_result = {}
    confusion_matrices = {}


    if args.cat_target:
        for cat_target in args.cat_target:
            outputs[cat_target] = torch.tensor([])
            y_true[cat_target] = torch.tensor([])
            test_result[cat_target] = {'test_loss':[], 'test_ACC':[], 'test_AUROC': []}

    if args.num_target:
        for num_target in args.num_target:
            outputs[num_target] = torch.tensor([])
            y_true[num_target] = torch.tensor([])
            test_result[num_target] = {'test_abs_loss': [], 'test_mse_loss':[], 'test_r2':[]}
    
    with torch.no_grad():
       for i, data in enumerate(tqdm(testloader),0):
           image, targets = data
           image = image.to(device)
           output = net(image)

           if args.cat_target:
               for cat_target in args.cat_target:
                   outputs[cat_target] = torch.cat((outputs[cat_target], output[cat_target].cpu()))
                   y_true[cat_target] = torch.cat((y_true[cat_target], targets[cat_target].cpu()))
           if args.num_target:
               for num_target in args.num_target:
                   outputs[num_target] = torch.cat((outputs[num_target], output[num_target].cpu()))
                   y_true[num_target] = torch.cat((y_true[num_target], targets[num_target].cpu()))

        #test_acc, correct, total = calculating_acc(targets, output, correct, y_true, test_acc, net, args, test_mode=True)

    
    # caculating ACC and R2 at once  
    if args.cat_target:
        for cat_target in args.cat_target:
            loss = torch.nn.functional.cross_entropy(outputs[cat_target].data, y_true[cat_target].long())
            test_result[cat_target]['test_loss'].append(loss.item())

            _, predicted = torch.max(outputs[cat_target].data,1)
            correct = (predicted == y_true[cat_target]).sum().item()
            total = y_true[cat_target].size(0)
            test_result[cat_target]['test_ACC'].append(100 * (correct / total))

            auroc = roc_auc_score(y_true[cat_target].detach().cpu(), outputs[cat_target].data[:, 1].detach().cpu())
            test_result[cat_target]['test_AUROC'].append(auroc.item())

            if args.confusion_matrix:
                for label_cm in args.confusion_matrix: 
                    if len(np.unique(y_true[cat_target].numpy())) == 2:
                        confusion_matrices[label_cm] = {}
                        confusion_matrices[label_cm]['True Positive'] = 0
                        confusion_matrices[label_cm]['True Negative'] = 0
                        confusion_matrices[label_cm]['False Positive'] = 0
                        confusion_matrices[label_cm]['False Negative'] = 0
                        if label_cm == cat_target:
                            tn, fp, fn, tp = confusion_matrix(y_true[cat_target].numpy(), predicted.numpy()).ravel()
                            confusion_matrices[label_cm]['True Positive'] = int(tp)
                            confusion_matrices[label_cm]['True Negative'] = int(tn)
                            confusion_matrices[label_cm]['False Positive'] = int(fp)
                            confusion_matrices[label_cm]['False Negative'] = int(fn)                       


    if args.num_target:
        for num_target in args.num_target:
            predicted =  outputs[num_target].float()
            abs_loss = torch.nn.functional.l1_loss(predicted, y_true[num_target].float().unsqueeze(1))
            mse_loss = torch.nn.functional.mse_loss(predicted, y_true[num_target].float().unsqueeze(1))
            y_var = torch.var(y_true[num_target])
            r_square = 1 - (mse_loss / y_var)
            test_result[num_target]['test_abs_loss'].append(abs_loss.item())
            test_result[num_target]['test_mse_loss'].append(mse_loss.item())
            test_result[num_target]['test_r2'].append(r_square.item())
            confusion_matrices = None


    #if args.cat_target:
    #    for cat_target in args.cat_target:
    #        test_acc[cat_target] = np.mean(test_acc[cat_target])

    #if args.num_target:
    #    for num_target in args.num_target:
    #        test_acc[num_target] = np.mean(test_acc[num_target])

    if args.get_predicted_score:
        if args.cat_target:
            for cat_target in args.cat_target: 
                outputs[cat_target] = outputs[cat_target].squeeze(-1).tolist()
                outputs[cat_target] = combine_pred_subjid(outputs[cat_target], partition['test'].image_files)
                outputs["predicted_%s" % cat_target] = outputs.pop(cat_target)
        if args.num_target:
            for num_target in args.num_target: 
                outputs[num_target] = outputs[num_target].squeeze(-1).tolist()
                outputs[num_target] = combine_pred_subjid(outputs[num_target], partition['test'].image_files)
                outputs["predicted_%s" % num_target] = outputs.pop(num_target)
        test_result.update(outputs)


    return test_result, confusion_matrices


## ============================================ ##
