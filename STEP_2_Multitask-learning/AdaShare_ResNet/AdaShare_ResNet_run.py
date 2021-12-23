## =================================== ##
## ======= run_ResNet3d_multi ======= ##
## =================================== ##

## ======= load module ======= ##
# custom script
import AdaShare_ResNet #model script
from data_preprocessing import * #data preprocessing script
from split_dataset import * #data split script 
from CLI_reporter import * 
from checkpoint import *
from ArgsSetting import *

# modules 
import glob
import os
from os import listdir
from os.path import isfile, join
from tqdm.auto import tqdm ##progress
import time
import math
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision import transforms, utils
import torch.optim as optim


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

import random
import hashlib
import json

from nilearn import plotting
import nibabel as nib

#from sklearn.metrics import confusion_matrix, roc_auc_score

import monai ##monai: medical open network for AI
from monai.data import CSVSaver, ImageDataset, DistributedWeightedRandomSampler
from monai.transforms import AddChannel, Compose, RandRotate90, Resize, ScaleIntensity, Flip, ToTensor
from monai.utils import set_determinism
from monai.apps import CrossValidation

import argparse 
from copy import deepcopy

import warnings
warnings.filterwarnings("ignore")
## =================================== ##

## ========= Argument Setting ========= ##
args = ArgsSetting()

## ==================================== ##


## ========= Data Preprocessing: image ========= ##
### get image file names (subject ID + '.npy') as list
base_dir = '/home/ubuntu/HEEHWAN/AdaShare_ResNet'
data_dir = '/home/ubuntu/HEEHWAN/preprocessed_masked'
os.chdir(data_dir)
image_files = glob.glob('*.npy')
image_files = sorted(image_files)
image_files = image_files[:30]
print("Loading image file names as list is completed")
## ====================================== ##


## ========= get subject ID and target variables ========= ##
if not args.cat_target:
    args.cat_target = []
elif not args.num_target:
    args.num_target = []
elif not args.cat_target and args.num_target:
    raise ValueError('YOU SHOULD SELECT THE TARGET!')

targets = args.cat_target + args.num_target

col_list = targets + ['subjectkey']

### get subject ID and target variables
subject_data = pd.read_csv('/home/ubuntu/HEEHWAN/ABCD_phenotype_total.csv')
subject_data = subject_data.loc[:,col_list]
subject_data = subject_data.sort_values(by='subjectkey')
subject_data = subject_data.dropna(axis = 0)
subject_data = subject_data.reset_index(drop=True) # removing subject have NA values 
## ====================================== ##


## ========= split train, val, test ========= ##
if args.cat_target:
    preprocessing_cat(subject_data, args)

if args.num_target:
    preprocessing_num(subject_data, args)

imageFiles_labels = combining_image_target(image_files, subject_data, args)

partition = partition(imageFiles_labels,args)
## ====================================== ##

### ========= Train,Validate, and Test ========= ###
'''The process of calcuating loss and accuracy metrics is as follows.
   1) sequentially calculate loss and accuracy metrics of target labels with for loop.
   2) store the result information with dictionary type.
   3) return the dictionary, which form as {'cat_target':value, 'num_target:value}
   This process is intended to easily deal with loss values from each target labels.'''


'''All of the loss from predictions are summated and this loss value is used for backpropagation.'''

def calculating_loss_acc(labels, output, cat_target, num_target, correct, total, loss_dict, acc_dict,net):
    '''define calculating loss and accuracy function used during training and validation step'''


    loss = 0.0
    for cat_label in cat_target:
        label = labels[cat_label]
        label = label.to(f'cuda:{net.backbone.device_ids[0]}')
        tmp_output = output[cat_label]

        criterion = nn.CrossEntropyLoss()

        tmp_loss = criterion(tmp_output.float(),label.long())
        loss += tmp_loss                             # loss is for aggregating all the losses from predicting each target variable

        # restoring train loss and accuracy for each task (target variable)
        loss_dict[cat_label] += tmp_loss.item()     # train_loss is for restoring loss from predicting each target variable
        _, predicted = torch.max(tmp_output.data,1)
        correct[cat_label] += (predicted == label).sum().item()
        total[cat_label] += label.size(0)

    for num_label in num_target:
        y_true = labels[num_label]
        y_true = y_true.to(f'cuda:{net.backbone.device_ids[0]}')
        tmp_output = output[num_label]

        criterion =nn.MSELoss()

        tmp_loss = criterion(tmp_output.float(),y_true.float().unsqueeze(1))
        loss +=tmp_loss

        # restoring train loss and accuracy for each task (target variable)
        loss_dict[num_label] += tmp_loss.item()     # train_loss is for restoring loss from predicting each target variable
        acc_dict[num_label] += tmp_loss.item() # RMSE for evaluating continuous variable

    return loss, correct, total, loss_dict,acc_dict


def calculating_acc(labels, output, cat_target, num_target, correct, total, acc_dict, net):
    '''define calculating accuracy function used during test step'''

    for cat_label in cat_target:
        label = labels[cat_label]
        label = label.to(f'cuda:{net.backbone.device_ids[0]}')
        tmp_output = output[cat_label]


        _, predicted = torch.max(tmp_output.data,1)
        correct[cat_label] += (predicted == label).sum().item()
        total[cat_label] += label.size(0)


    for num_label in num_target:
        y_true = labels[num_label]
        y_true = y_true.to(f'cuda:{net.backbone.device_ids[0]}')
        tmp_output = output[num_label]

        criterion =nn.MSELoss()

        tmp_loss = criterion(tmp_output.float(),y_true.float().unsqueeze(1))


        # restoring train loss and accuracy for each task (target variable)
        acc_dict[num_label] += tmp_loss.item() # RMSE for evaluating continuous variable
    return correct, total, acc_dict


# define training step
def warmup(net,partition,optimizer,input_dict):
    """input_dict['mode'] == fix_policy means that model is on the learning phase. 
    warmup function is used only in warming up phase."""

    '''GradScaler is for calculating gradient with float 16 type'''
    scaler = torch.cuda.amp.GradScaler()

    trainloader = torch.utils.data.DataLoader(partition['train_warmup'],
                                             batch_size=args.train_batch_size,
                                             shuffle=True,
                                             num_workers=24)

    net.train()

    correct = {}
    total = {}


    train_warmup_loss = {}
    train_warmup_acc = {}  # collecting r-square of contiunuous varibale directly
    

    if args.cat_target:
        for cat_label in args.cat_target:
            correct[cat_label] = 0
            total[cat_label] = 0
            train_warmup_loss[cat_label] = 0.0

    if args.num_target:
        for num_label in args.num_target:
            train_warmup_acc[num_label] = 0.0
            train_warmup_loss[num_label] = 0.0


    for i, data in enumerate(trainloader,0):
        optimizer.zero_grad()

        image, labels = data
        image = image.to(f'cuda:{net.backbone.device_ids[0]}')
        output, policys = net(image, **input_dict)

        loss, correct, total, train_warmup_loss,train_warmup_acc = calculating_loss_acc(labels, output, args.cat_target, args.num_target, correct, total, train_warmup_loss, train_warmup_acc,net)

        scaler.scale(loss).backward()# multi-head model sum all the loss from predicting each target variable and back propagation
        scaler.step(optimizer)
        scaler.update()

    # calculating total loss and acc of separate mini-batch
    if args.cat_target:
        for cat_label in args.cat_target:
            train_warmup_acc[cat_label] = 100 * correct[cat_label] / total[cat_label]
            train_warmup_loss[cat_label] = train_warmup_loss[cat_label] / len(trainloader)

    if args.num_target:
        for num_label in args.num_target:
            train_warmup_acc[num_label] = train_warmup_acc[num_label] / len(trainloader)
            train_warmup_loss[num_label] = train_warmup_loss[num_label] / len(trainloader)


    return net, train_warmup_loss, train_warmup_acc, policys


# define training step
def train(net,partition,optimizer,input_dict):
    """input_dict['mode'] == fix_policy means that model is on the learning phase. 
    train function is used only in learning phase, it should be distinguished."""
    
    '''GradScaler is for calculating gradient with float 16 type'''
    scaler = torch.cuda.amp.GradScaler()

    trainloader = torch.utils.data.DataLoader(partition['train'],
                                             batch_size=args.train_batch_size,
                                             shuffle=True,
                                             num_workers=24)

    net.train()

    correct = {}
    total = {}


    train_loss = {}
    train_acc = {}  # collecting r-square of contiunuous varibale directly
    

    if args.cat_target:
        for cat_label in args.cat_target:
            correct[cat_label] = 0
            total[cat_label] = 0
            train_loss[cat_label] = 0.0

    if args.num_target:
        for num_label in args.num_target:
            train_acc[num_label] = 0.0
            train_loss[num_label] = 0.0


    for i, data in enumerate(trainloader,0):
        optimizer.zero_grad()

        image, labels = data
        image = image.to(f'cuda:{net.backbone.device_ids[0]}')

        output = net(image, **input_dict)                     
        loss, correct, total, train_loss,train_acc = calculating_loss_acc(labels, output, args.cat_target, args.num_target, correct, total, train_loss, train_acc,net)

        scaler.scale(loss).backward()# multi-head model sum all the loss from predicting each target variable and back propagation
        scaler.step(optimizer)
        scaler.update()

    # calculating total loss and acc of separate mini-batch
    if args.cat_target:
        for cat_label in args.cat_target:
            train_acc[cat_label] = 100 * correct[cat_label] / total[cat_label]
            train_loss[cat_label] = train_loss[cat_label] / len(trainloader)

    if args.num_target:
        for num_label in args.num_target:
            train_acc[num_label] = train_acc[num_label] / len(trainloader)
            train_loss[num_label] = train_loss[num_label] / len(trainloader)


    return net, train_loss, train_acc




# define validation step
def validate(net,partition,scheduler,input_dict):
    """input_dict['mode'] == fix_policy means that model is on the learning phase. 
    validation function is used in both warming up phase and learning phase, it should be distinguished."""
    valloader = torch.utils.data.DataLoader(partition['val'],
                                           batch_size=args.val_batch_size,
                                           shuffle=False,
                                           num_workers=24)

    net.eval()

    correct = {}
    total = {}


    val_loss = {}
    val_acc = {}  # collecting r-square of contiunuous varibale directly

    if args.cat_target:
        for cat_label in args.cat_target:
            correct[cat_label] = 0
            total[cat_label] = 0
            val_loss[cat_label] = 0.0

    if args.num_target:
        for num_label in args.num_target:
            val_acc[num_label] = 0.0
            val_loss[num_label] = 0.0



    # change network setting regarding whether it's on warming up phase or learning phase 
    with torch.no_grad():
        for i, data in enumerate(valloader,0):
            image, labels = data
            image = image.to(f'cuda:{net.backbone.device_ids[0]}')

            if input_dict['mode'] == 'fix_policy':
                output = net(image, **input_dict)  
            else:
                output, policys = net(image, **input_dict)
                

            loss, correct, total, val_loss,val_acc = calculating_loss_acc(labels, output, args.cat_target, args.num_target, correct, total, val_loss, val_acc,net)


    if args.cat_target:
        for cat_label in args.cat_target:
            val_acc[cat_label] = 100 * correct[cat_label] / total[cat_label]
            val_loss[cat_label] = val_loss[cat_label] / len(valloader)

    if args.num_target:
        for num_label in args.num_target:
            val_acc[num_label] = val_acc[num_label] / len(valloader)
            val_loss[num_label] = val_loss[num_label] / len(valloader)


    # learning rate scheduler
    scheduler.step(sum(val_loss.values()))


    return val_loss, val_acc




# define test step
def test(net,partition,input_dict, args):
    """input_dict['mode'] == fix_policy means that model is on the learning phase. 
    test function is used only in learning phase."""

    testloader = torch.utils.data.DataLoader(partition['test'],
                                            batch_size=args.test_batch_size,
                                            shuffle=False,
                                            num_workers=24)

    net.eval()

    correct = {}
    total = {}

    test_acc = {}


    if args.cat_target:
        for cat_label in args.cat_target:
            correct[cat_label] = 0
            total[cat_label] = 0

    if args.num_target:
        for num_label in args.num_target:
            test_acc[num_label] = 0.0


    for i, data in enumerate(testloader,0):
            image, labels = data
            image = image.to(f'cuda:{net.backbone.device_ids[0]}')
            output = net(image, **input_dict)

            correct, total, test_acc = calculating_acc(labels,output, args.cat_target, args.num_target, correct, total, test_acc,net)



    if args.cat_target:
        for cat_label in args.cat_target:
            test_acc[cat_label] = 100 * correct[cat_label] / total[cat_label]

    if args.num_target:
        for num_label in args.num_target:
            test_acc[num_label] = test_acc[num_label] / len(testloader)


    return test_acc
## ============================================ ##


## ========= Warming up Experiment =============== ##
def experiment_warmup(partition, subject_data, args) -> dict: #in_channels,out_dim

    if args.model == 'resnet3D50':
        net = AdaShare_ResNet.resnet3D50(subject_data, args)
    elif args.model == 'resnet3D101':
        net = AdaShare_ResNet.resnet3D101(subject_data, args)
    elif args.model == 'resnet3D152':
        net = AdaShare_ResNet.resnet3D152(subject_data, args)


    if args.optim == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.5)
    elif args.optim == 'Adam':
        optimizer = optim.Adam(net.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    else:
        raise ValueError('In-valid optimizer choice')

    
    #net = nn.DataParallel(net, device_ids=[1,2])
    net.to(f'cuda:{net.backbone.device_ids[0]}')
    

    # learning rate schedluer
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min')

    # setting for results' data frame
    train_warmup_losses = {}
    train_warmup_accs = {}
    val_warmup_losses = {}
    val_warmup_accs = {}

    for target_name in targets:
        train_warmup_losses[target_name] = []
        train_warmup_accs[target_name] = []
        val_warmup_losses[target_name] = []
        val_warmup_accs[target_name] = []
        

    # setting for hyperparameter of NAS 
    input_dict_train = {'temperature': 5, 'is_policy': True, 'mode': 'train'}
    input_dict_val = {'temperature': 5, 'is_policy': True, 'mode': 'eval'}

    for epoch in tqdm(range(int(args.epoch*args.warmup_itr_ratio))):
        ts = time.time()
        net, train_warmup_loss, train_warmup_acc, policys = warmup(net,partition,optimizer,input_dict_train)
        val_warmup_loss, val_warmup_acc = validate(net,partition,scheduler,input_dict_val)
        te = time.time()
        print('Epoch {}. Took {:2.2f} sec'.format(epoch, te-ts))

        # sorting the results
        for target_name in targets:
            train_warmup_losses[target_name].append(train_warmup_loss[target_name])
            train_warmup_accs[target_name].append(train_warmup_acc[target_name])
            val_warmup_losses[target_name].append(val_warmup_loss[target_name])
            val_warmup_accs[target_name].append(val_warmup_acc[target_name])

        # visualize the result
        CLIreporter(targets, train_warmup_loss, train_warmup_acc, val_warmup_loss, val_warmup_acc)

        # checkpoint saving(only if current model shows better mean validation accuracy across tasks than previous epochs)
        torch.save({'model_state_dict': net.state_dict(),'optimizer_state_dict': optimizer.state_dict()}, str(base_dir+'/checkpoint/AdaShare_ResNet.pt'))            

    # summarize results
    result = {}
    result['train_losses'] = train_warmup_losses
    result['train_accs'] = train_warmup_accs
    result['val_losses'] = val_warmup_losses
    result['val_accs'] = val_warmup_accs

    result['train_acc'] = train_warmup_acc
    result['val_acc'] = val_warmup_acc

    result.update(policys) # policys.shape = (layers,2); policys[:,1] = the possibility that the layer block is important to predict a specific task. If policy[:,1] < 0.5, the layer block is dropped in learning phase.

    
    return vars(args), result
## ==================================== ##



## ========= Learning Experiment =============== ##
def experiment_learning(partition, subject_data, policys, args): #in_channels,out_dim


    if args.model == 'resnet3D50':
        net = AdaShare_ResNet.resnet3D50(subject_data, args)
    elif args.model == 'resnet3D101':
        net = AdaShare_ResNet.resnet3D101(subject_data, args)
    elif args.model == 'resnet3D152':
        net = AdaShare_ResNet.resnet3D152(subject_data, args)


    if args.optim == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.5)
    elif args.optim == 'Adam':
        optimizer = optim.Adam(net.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    else:
        raise ValueError('In-valid optimizer choice')

    # load the model trained with warming up training set
    checkpoint = torch.load(str(base_dir+'/checkpoint/AdaShare_ResNet.pt'))
    net.backbone.load_state_dict(checkpoint['model_state_dict'],strict= False)
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


    # load the model on the cuda device
    net.to(f'cuda:{net.backbone.device_ids[0]}')

    # load estimated policys from warmingup phase
    setattr(net, 'policys', policys.to(f'cuda:{net.backbone.device_ids[0]}'))

    # learning rate schedluer
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min')

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
        

    # setting for hyperparameter of NAS 
    input_dict_train = {'temperature': 5, 'is_policy': True, 'mode': 'fix_policy'}
    input_dict_val = {'temperature': 5, 'is_policy': True, 'mode': 'fix_policy'}

    for epoch in tqdm(range(args.epoch-int(args.epoch*args.warmup_itr_ratio))):
        ts = time.time()
        net, train_loss, train_acc = train(net,partition,optimizer,input_dict_train)
        val_loss, val_acc = validate(net,partition,scheduler,input_dict_val)
        te = time.time()
        print('Epoch {}. Took {:2.2f} sec'.format(epoch, te-ts))

        # sorting the results
        for target_name in targets:
            train_losses[target_name].append(train_loss[target_name])
            train_accs[target_name].append(train_acc[target_name])
            val_losses[target_name].append(val_loss[target_name])
            val_accs[target_name].append(val_acc[target_name])

        # visualize the result
        CLIreporter(targets, train_loss, train_acc, val_loss, val_acc)

        # checkpoint saving(only if current model shows better mean validation accuracy across tasks than previous epochs)
        checkpoint_saving(net, optimizer, val_accs, base_dir)

    # load the best model and do inference
    checkpoint = torch.load(str(base_dir+'/checkpoint/AdaShare_ResNet.pt'))
    net.load_state_dict(checkpoint['model_state_dict'],strict= False)


    # test
    test_acc = test(net, partition, input_dict_val, args)

    # summarize results
    result = {}
    result['train_losses'] = train_losses
    result['train_accs'] = train_accs
    result['val_losses'] = val_losses
    result['val_accs'] = val_accs

    result['train_acc'] = train_acc
    result['val_acc'] = val_acc
    result['test_acc'] = test_acc

    return vars(args), result
## ==================================== ##


## ========= Run Experiment and saving result ========= ##
# define result-saving function
def save_exp_result(setting, result):
    exp_name = setting['exp_name']
    del setting['epoch']
    del setting['test_batch_size']

    hash_key = hashlib.sha1(str(setting).encode()).hexdigest()[:6]
    result.update(setting)
    if result['curriculum learning'] == 'Warming Up Phase':
        filename = str(base_dir + '/results/{}-{}_WarmingUp.json').format(exp_name, hash_key)
    else:
        filename = str(base_dir + '/results/{}-{}_Learning.json').format(exp_name, hash_key)

    with open(filename, 'w') as f:
        json.dump(result, f)

# seed number
seed = 1234
np.random.seed(seed)
torch.manual_seed(seed)


# Run wraming up Experiment and save result
setting, result = experiment_warmup(partition, subject_data,deepcopy(args))
result['curriculum learning'] = 'Warming Up Phase'
save_exp_result(setting,result)

print('===================== Warming Up Phase Done ===================== ')
print('===================== Now We Start Learning Phase ===================== ')

# based on the results from warming up phase, fix policy for block dropping in further learning phase 
policys = []
for t_id in range(len(targets)):
    policy = np.array(result['%s_policy' % targets[t_id]])
    l_policy = np.argmax(policy, axis=1)                        # estimated policys are values range from 0 to 1. However, for layer block dropping in learning phase, input policy values need to be 0 or 1.  
    l_policy = np.stack((1 - l_policy, l_policy), axis=1)        # If the second position of the stacked tuple (1-l_policy, l_policy) is 1 (= l_policy is 1), layer block of position i, namely np.stack((1 - l_policy, l_policy), dim=1)[i], would be skipped. (  
    policys.append(l_policy)
policys = torch.from_numpy(np.array(policys))

CLIblockdropping(targets, policys.tolist())


setting, result = experiment_learning(partition, subject_data, policys, deepcopy(args))
result['curriculum learning'] = 'Learning Phase'
save_exp_result(setting,result)
