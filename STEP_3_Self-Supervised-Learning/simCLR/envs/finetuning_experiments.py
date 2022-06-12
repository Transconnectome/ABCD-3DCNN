import numpy as np
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix
from models.prediction_model import prediction_model

## ======= load module ======= ##

from models.prediction_model import prediction_model #model script

from envs.loss_functions import calculating_loss, calculating_acc
from utils.utils import CLIreporter, load_pretrained_model, save_exp_result, checkpoint_save, checkpoint_load, load_pretrained_model, freezing_layers

import time
from tqdm import tqdm
import copy

### ========= Train,Validate, and Test ========= ###
'''The process of calcuating loss and accuracy metrics is as follows.
   1) sequentially calculate loss and accuracy metrics of target labels with for loop.
   2) store the result information with dictionary type.
   3) return the dictionary, which form as {'cat_target':value, 'num_target:value}
   This process is intended to easily deal with loss values from each target labels.'''


'''All of the loss from predictions are summated and this loss value is used for backpropagation.'''

# define training step
def train(net, partition, optimizer, args):
    '''GradScaler is for calculating gradient with float 16 type'''
    scaler = torch.cuda.amp.GradScaler()

    trainloader = torch.utils.data.DataLoader(partition['train'],
                                             batch_size=args.train_batch_size,
                                             shuffle=True,
                                             pin_memory=True,
                                             num_workers=len(net.device_ids)*4)

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


    for i, data in enumerate(trainloader,0):
        optimizer.zero_grad()
        image, targets = data
        image = image.to(f'cuda:{net.device_ids[0]}', non_blocking=True)
        output = net(image)

        loss, train_loss = calculating_loss(targets, output, train_loss,net, args)
        train_acc = calculating_acc(targets, output, correct, total, train_acc, net, args)
        
        scaler.scale(loss).backward()# multi-head model sum all the loss from predicting each target variable and back propagation
        scaler.step(optimizer)
        scaler.update()


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
                                           batch_size=args.val_batch_size,
                                           shuffle=True,
                                           pin_memory=True,
                                           num_workers=len(net.device_ids)*4)

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
            image = image.to(f'cuda:{net.device_ids[0]}', non_blocking=True)
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
    scheduler.step(sum(val_acc.values()))


    return val_loss, val_acc



# define test step
def test(net,partition,args):
    testloader = torch.utils.data.DataLoader(partition['test'],
                                            batch_size=args.test_batch_size,
                                            shuffle=False,
                                            pin_memory=True,
                                            num_workers=4)

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
    test_acc = {}
    confusion_matrices = {}


    if args.cat_target:
        for cat_target in args.cat_target:
            outputs[cat_target] = torch.tensor([])
            y_true[cat_target] = torch.tensor([])
            test_acc[cat_target] = []

    if args.num_target:
        for num_target in args.num_target:
            outputs[num_target] = torch.tensor([])
            y_true[num_target] = torch.tensor([])
            test_acc[num_target] = []
            
            
    with torch.no_grad():
        for i, data in enumerate(testloader,0):
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
            _, predicted = torch.max(outputs[cat_target].data,1)
            correct = (predicted == y_true[cat_target]).sum().item()
            total = y_true[cat_target].size(0)
            test_acc[cat_target].append(100 * (correct / total))

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
            criterion = nn.MSELoss()
            loss = criterion(predicted, y_true[num_target].float().unsqueeze(1))
            y_var = torch.var(y_true[num_target])
            r_square = 1 - (loss / y_var)
            test_acc[num_target].append(r_square.item())


    return test_acc, confusion_matrices

## ============================================ ##

def finetuning_experiment(partition, subject_data, save_dir, args): #in_channels,out_dim
    targets = args.cat_target + args.num_target
    
    # setting network
    net = prediction_model(subject_data, args)
    checkpoint_dir = args.checkpoint_dir
    pretrained_model_dir = args.pretrained_model_dir

    # setting optimizer 
    if args.optim == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    elif args.optim == 'Adam':
        optimizer = optim.Adam(net.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    else:
        raise ValueError('In-valid optimizer choice')

    # learning rate schedluer
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,'max', patience=4)

    # loading pre-trained SSL backbone model or resume training 
    if args.resume == 'False':
        if args.pretrained_model_dir != None:
            net = load_pretrained_model(net, pretrained_model_dir)
            net.backbone = freezing_layers(net.backbone)
            last_epoch = 0
        else:
            raise ValueError('IF YOU WANT TO FINETUNING WITH PRE-TRAINED MODEL, YOU SHOULD SET THE FILE PATH AS AN OPTION. PLZ CHECK --pretrained_model_dir OPTION')
    else:
        if args.checkpoint_dir != None:
            net, optimizer, scheduler, last_epoch, optimizer.param_groups[0]['lr'] = checkpoint_load(net, checkpoint_dir, optimizer, mode='finetuing')
            print('Training start from epoch {} and learning rate {}.'.format(last_epoch, optimizer.param_groups[0]['lr']))
        else:
            raise ValueError('IF YOU WANT TO RESUME TRAINING FROM PREVIOUS STATE, YOU SHOULD SET THE FILE PATH AS AN OPTION. PLZ CHECK --checkpoint_dir OPTION')
        
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
    train_losses = {}
    train_accs = {}
    val_losses = {}
    val_accs = {}

    for target_name in targets:
        train_losses[target_name] = []
        train_accs[target_name] = []
        val_losses[target_name] = []
        val_accs[target_name] = []
        
    # training
    for epoch in tqdm(range(last_epoch, args.epoch + last_epoch)):
        ts = time.time()
        net, train_loss, train_acc = train(net,partition,optimizer,args)
        val_loss, val_acc = validate(net,partition,scheduler,args)
        te = time.time()

         # sorting the results
        for target_name in targets:
            train_losses[target_name].append(train_loss[target_name])
            train_accs[target_name].append(train_acc[target_name])
            val_losses[target_name].append(val_loss[target_name])
            val_accs[target_name].append(val_acc[target_name])

        # visualize the result
        CLIreporter(targets, train_loss, train_acc, val_loss, val_acc)
        print('Epoch {}. Current learning rate {}. Took {:2.2f} sec'.format(epoch+1,optimizer.param_groups[0]['lr'],te-ts))

        # saving the checkpoint
        checkpoint_dir = checkpoint_save(net, optimizer, save_dir, epoch, args, current_result=val_acc, previous_result=val_accs, mode='prediction')

    # test
    net.to('cpu')
    torch.cuda.empty_cache()

    net, _, _ = checkpoint_load(net, checkpoint_dir, optimizer, mode ='eval')
    if args.sbatch == 'True':
        net.cuda()
    else:
        net.to(f'cuda:{args.gpus[0]}')
    test_acc, confusion_matrices = test(net, partition, args)

    # summarize results
    result = {}
    result['train_losses'] = train_losses
    result['train_accs'] = train_accs
    result['val_losses'] = val_losses
    result['val_accs'] = val_accs

    result['train_acc'] = train_acc
    result['val_acc'] = val_acc
    result['test_acc'] = test_acc

    result['confusion_matrices'] = confusion_matrices

    return vars(args), result
## ==================================== ##

## ============================================ ##

