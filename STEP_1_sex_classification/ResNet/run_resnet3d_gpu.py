## =================================== ##
## ======= ResNet ======= ##
## =================================== ##

'''
__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']
'''

## ======= load module ======= ##
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

from sklearn.metrics import confusion_matrix, roc_auc_score

import monai ##monai: medical open network for AI
from monai.data import CSVSaver, ImageDataset, DistributedWeightedRandomSampler
from monai.transforms import AddChannel, Compose, RandRotate90, Resize, ScaleIntensity, Flip, ToTensor
from monai.utils import set_determinism
from monai.apps import CrossValidation

import argparse
from copy import deepcopy ## add deepcopy for args

import warnings
warnings.filterwarnings("ignore")
import resnet3d
## =================================== ##

## ========= Argument Setting ========= ##
parser = argparse.ArgumentParser()

#parser.add_argument("--GPU_NUM",default=1,type=int,required=True,help='')
parser.add_argument("--model",required=True,type=str,choices=['resnet50', 'resnet101', 'resnet152'],help='')
parser.add_argument("--val_size",default=0.1,type=float,required=False,help='')
parser.add_argument("--test_size",default=0.1,type=float,required=False,help='')
parser.add_argument("--resize",default=(96,96,96),required=False,help='')
parser.add_argument("--train_batch_size",default=32,type=int,required=False,help='')
parser.add_argument("--val_batch_size",default=8,type=int,required=False,help='')
parser.add_argument("--test_batch_size",default=1,type=int,required=False,help='')
parser.add_argument("--optim",type=str,required=True,help='', choices=['Adam','SGD'])
parser.add_argument("--lr", default=0.01,type=float,required=False,help='')
parser.add_argument("--momentum", default=0.5,type=float,required=False,help='')
parser.add_argument("--weight_decay",default=0.001,type=float,required=False,help='')
parser.add_argument("--epoch",type=int,required=True,help='')
parser.add_argument("--exp_name",type=str,required=True,help='')
parser.add_argument("--gpu_ids",type=int,nargs='*',required=True,help='')

args = parser.parse_args()
## ==================================== ##

## ========= GPU Setting ========= ##
#GPU_NUM = args.GPU_NUM # enter the number you want to use GPU
#GPU_NUM=1 # ***
#device = 'cpu'
#device = 'cuda:1'
#device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')

#torch.cuda.set_device(device)
#print('Experiment is performed on GPU {}'.format(torch.cuda.current_device()))
## ==================================== ##

## ========= Data Preprocessing: image ========= ##
### get the current working directory
#currentPath = os.getcwd()
#print(currentPath)

### get image file names (subject ID + '.npy') as list
base_dir = '/home/connectome/dhkdgmlghks/3DCNN_test'
data_dir = '/home/connectome/dhkdgmlghks/3DCNN_test/preprocessed_masked'
os.chdir(data_dir)
image_files = glob.glob('*.npy')
image_files = sorted(image_files)
#image_files = image_files[:100]
print("Loading image file names as list is completed")
## ====================================== ##

## ========= Data Preprocessing: target ========= ##
### get subject ID and target variables
target = 'sex'

subject_dir = '/home/connectome/dhkdgmlghks/3DCNN_test'
os.chdir(subject_dir)
subject_data = pd.read_csv('ABCD_phenotype_total.csv')
subject_data = subject_data.loc[:,['subjectkey',target]]
subject_data = subject_data.sort_values(by='subjectkey')
subject_data = subject_data.dropna(axis = 0)
subject_data = subject_data.reset_index(drop=True) # removing subject have NA values in sex
print("Loading subject list is completed")
## ====================================== ##

## ========= Data Preprocessing ========= ##
os.chdir(data_dir) # if I do not change directory here, image data is not loaded
# get subject ID and target variables as sorted list

imageFiles_labels = []

for subjectID in tqdm(image_files):
    subjectID = subjectID[:-4] #removing '.npy' for comparing
    #print(subjectID)
    for i in range(len(subject_data)):
        if subjectID == subject_data['subjectkey'][i]:
            if subject_data['sex'][i] == 1:
                imageFiles_labels.append((subjectID+'.npy',0))
            elif subject_data['sex'][i] == 2:
                imageFiles_labels.append((subjectID+'.npy',1))
            else:
                print('NaN value for {}'.format(subjectID))
                continue
## ====================================== ##

## ========= Split train, val, test========= ##
# defining train,val, test set splitting function
def partition(imageFiles_labels,args):
    random.shuffle(imageFiles_labels)

    images = []
    labels = []
    for imageFile_label in imageFiles_labels:
        image, label = imageFile_label
        images.append(image)
        labels.append(label)

    resize = args.resize
    train_transform = Compose([ScaleIntensity(),
                               AddChannel(),
                               Resize(resize),
                              ToTensor()])

    val_transform = Compose([ScaleIntensity(),
                               AddChannel(),
                               Resize(resize),
                              ToTensor()])

    test_transform = Compose([ScaleIntensity(),
                               AddChannel(),
                               Resize(resize),
                              ToTensor()])

    num_total = len(images)
    num_train = int(num_total*(1 - args.val_size - args.test_size))
    #print(num_train)
    num_val = int(num_total*args.val_size)
    #print(num_val)
    num_test = int(num_total*args.test_size)
    #print(num_test)

    images_train = images[:num_train]
    labels_train = labels[:num_train]

    images_val = images[num_train:num_train+num_val]
    labels_val = labels[num_train:num_train+num_val]

    images_test = images[num_total-num_test:]
    labels_test = labels[num_total-num_test:]

    train_set = ImageDataset(image_files=images_train,labels=labels_train,transform=train_transform)
    val_set = ImageDataset(image_files=images_val,labels=labels_val,transform=val_transform)
    test_set = ImageDataset(image_files=images_test,labels=labels_test,transform=test_transform)

    partition = {}
    partition['train'] = train_set
    partition['val'] = val_set
    partition['test'] = test_set

    return partition

# split dataset
partition = partition(imageFiles_labels,args)
print("Splitting data to train, val, test is completed")
## ====================================== ##

## ========= Train,Validate, and Test ========= ##
# define training step
def train(net,partition,optimizer,criterion,args):
    trainloader = torch.utils.data.DataLoader(partition['train'],
                                             batch_size=args.train_batch_size,
                                             shuffle=True,
                                             num_workers=2)

    net.train()

    correct = 0
    total = 0
    train_loss = 0.0


    for i, data in enumerate(trainloader,0):
        optimizer.zero_grad() #this code makes {train gradient=0}
        image, label = data
        image = image.to(f'cuda:{net.device_ids[0]}')
        label = label.to(f'cuda:{net.device_ids[0]}')
        output = net(image)

        loss = criterion(output,label)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(net(image).data,1)
        total += label.size(0)
        correct += (predicted == label).sum().item()

    train_loss = train_loss / len(trainloader)
    train_acc = 100 * correct / total

    return net, train_loss, train_acc


# define validation step
def validate(net,partition,criterion, scheduler,args):
    valloader = torch.utils.data.DataLoader(partition['val'],
                                           batch_size=args.val_batch_size,
                                           shuffle=False,
                                           num_workers=2)

    net.eval()

    correct = 0
    total = 0
    val_loss = 0.0

    with torch.no_grad():

        for i, data in enumerate(valloader,0):
            image, label = data
            image = image.to(f'cuda:{net.device_ids[0]}')
            label = label.to(f'cuda:{net.device_ids[0]}')
            output = net(image)

            loss = criterion(output,label)

            val_loss += loss.item()
            _, predicted = torch.max(output.data,1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

        val_loss = val_loss / len(valloader)
        val_acc = 100 * correct / total

    scheduler.step(val_acc)
    return val_loss, val_acc


# define test step
def test(net,partition,args):
    testloader = torch.utils.data.DataLoader(partition['test'],
                                            batch_size=args.test_batch_size,
                                            shuffle=False,
                                            num_workers=2)

    net.eval()

    correct = 0
    total = 0

    cmt = {}
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    
    subj_predicted = {}
    subj_predicted['label'] = []
    subj_predicted['pred'] = []
    
    for i, data in enumerate(testloader,0):
        image, label = data
        image = image.to(f'cuda:{net.device_ids[0]}')
        label = label.to(f'cuda:{net.device_ids[0]}')
        output = net(image)

        _, predicted = torch.max(output.data,1)
        total += label.size(0)
        correct += (predicted == label).sum().item()
        
        # calculate confusion_matrix
        result_cmt = confusion_matrix(label.cpu(), predicted.cpu())

        if len(result_cmt) == 1:
            if label.item() ==1:
                true_positive += 1
            else:
                true_negative += 1
        else:

            tn, fp, fn, tp = result_cmt.ravel()
            true_positive += int(tp)
            true_negative += int(tn)
            false_positive += int(fp)
            false_negative += int(fn)
        
        
        cmt['true_positive'] = true_positive
        cmt['true_negative'] = true_negative
        cmt['false_positive'] = false_positive
        cmt['false_negative'] = false_negative

        # subj_predicted
        subj_predicted['label'].append(label.cpu().tolist()[0])
        subj_predicted['pred'].append(output.data.cpu().tolist()[0])
        #print(subj_predicted)
           
    test_acc = 100 * correct / total
    
    return test_acc, cmt, subj_predicted
## ============================================ ##

## ========= Experiment =============== ##
def experiment(partition, args): #in_channels,out_dim
    
    if args.model == 'resnet50':
        net = resnet3d.resnet3D50()
    elif args.model == 'resnet101':
        net = resnet3d.resnet3D101()
    elif args.model == 'resnet152':
        net = resnet3d.resnet3D152()

    net = torch.nn.DataParallel(net,device_ids=args.gpu_ids)
    net = net.to(f'cuda:{net.device_ids[0]}') #net = net.to(device)
        
    criterion = nn.CrossEntropyLoss()
    if args.optim == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.optim == 'Adam':
        optimizer = optim.Adam(net.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    else:
        raise ValueError('In-valid optimizer choice')
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []


    for epoch in tqdm(range(args.epoch)):
        ts = time.time()
        net, train_loss, train_acc = train(net,partition,optimizer,criterion,args)
        val_loss, val_acc = validate(net,partition,criterion, scheduler,args)
        test_acc = test(net,partition,args)
        te = time.time()

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print('Epoch {}, ACC(train/val): {:2.2f}/{:2.2f}, Loss(train/val): {:2.2f}/{:2.2f}. Current learning rate {}.Took {:2.2f} sec'.format(epoch,train_acc,val_acc,train_loss,val_loss,optimizer.param_groups[0]['lr'],te-ts))


    test_acc, cmt, subj_predicted = test(net,partition,args)

    result = {}
    result['train_losses'] = train_losses
    result['train_accs'] = train_accs
    result['val_losses'] = val_losses
    result['val_accs'] = val_accs
    result['train_acc'] = train_acc
    result['val_acc'] = val_acc
    result['test_acc'] = test_acc

    return vars(args), result, cmt, subj_predicted
## ==================================== ##

## ========= Run Experiment and saving result ========= ##
# define result-saving function
def save_exp_result(setting, result, cmt, subj_predicted):
    exp_name = setting['exp_name']
    
    del setting['epoch']
    del setting['test_batch_size']

    hash_key = hashlib.sha1(str(setting).encode()).hexdigest()[:6]
    filename = '/scratch/3DCNN/ResNet_results/{}-{}.json'.format(exp_name, hash_key)
    result.update(setting)
    result.update(cmt)
    result.update(subj_predicted)
    
    with open(filename, 'w') as f:
        json.dump(result, f)

# seed number
seed = 1234
np.random.seed(seed)
torch.manual_seed(seed)

# Run Experiment and save result
setting, result, cmt, subj_predicted = experiment(partition, deepcopy(args))
save_exp_result(setting,result, cmt, subj_predicted)
## ==================================================== ##
