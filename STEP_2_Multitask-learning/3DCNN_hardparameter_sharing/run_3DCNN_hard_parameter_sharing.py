## ======= load module ======= ##
import models.simple3d as simple3d
import models.vgg3d as vgg3d #model script
import models.resnet3d as resnet3d #model script
import models.densenet3d as densenet3d #model script
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
from torchsummary import summary


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

import random
import hashlib
import json

#from nilearn import plotting
#import nibabel as nib

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
parser = argparse.ArgumentParser()


#parser.add_argument("--GPU_NUM",default=1,type=int,required=True,help='')
parser.add_argument("--model",required=True,type=str,help='',choices=['simple3D','vgg3D11','vgg3D13','vgg3D16','vgg3D19','resnet3D50','resnet3D101','resnet3D152', 'densenet3D121', 'densenet3D169','densenet201','densenet264'])
parser.add_argument("--val_size",default=0.1,type=float,required=False,help='')
parser.add_argument("--test_size",default=0.1,type=float,required=False,help='')
parser.add_argument("--resize",default=[96, 96, 96],type=int,nargs="*",required=False,help='')
parser.add_argument("--train_batch_size",default=16,type=int,required=False,help='')
parser.add_argument("--val_batch_size",default=16,type=int,required=False,help='')
parser.add_argument("--test_batch_size",default=1,type=int,required=False,help='')
parser.add_argument("--in_channels",default=1,type=int,required=False,help='')
parser.add_argument("--optim",type=str,required=True,help='', choices=['Adam','SGD'])
parser.add_argument("--lr", default=0.01,type=float,required=False,help='')
parser.add_argument("--weight_decay",default=0.001,type=float,required=False,help='')
parser.add_argument("--epoch",type=int,required=True,help='')
parser.add_argument("--exp_name",type=str,required=True,help='')
parser.add_argument("--cat_target", type=str, nargs='*', required=False, help='')
parser.add_argument("--num_target", type=str,nargs='*', required=False, help='')
parser.add_argument("--gpus", type=int,nargs='*', required=False, help='')


args = parser.parse_args()

print("Categorical target labels are {} and Numerical target labels are {}".format(args.cat_target, args.num_target))

## ==================================== ##

#get_ipython().system('nvidia-smi')

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
base_dir = '/master_ssd/3DCNN/data/1.sMRI_fmriprep/preprocessed_masked'
data_dir = '/master_ssd/3DCNN/data/1.sMRI_fmriprep/preprocessed_masked'
os.chdir(data_dir)
image_files = glob.glob('*.npy')
image_files = sorted(image_files)
#image_files = image_files[:100]
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

##
### get subject ID and target variables
subject_data = pd.read_csv('/home/connectome/dhkdgmlghks/3DCNN_test/ABCD_phenotype_total.csv')
subject_data = subject_data.loc[:,col_list]
subject_data = subject_data.sort_values(by='subjectkey')
subject_data = subject_data.dropna(axis = 0)
subject_data = subject_data.reset_index(drop=True) # removing subject have NA values in sex
## ====================================== ##

## ========= define functinos for data preprocesing categorical variable and numerical variables ========= ##
## categorical
def preprocessing_cat():
    for cat in args.cat_target:
        if not 0 in list(subject_data.loc[:,cat]):
            subject_data[cat] = subject_data[cat] - 1
        else:
            continue

## numeric
def preprocessing_num():
    for num in args.num_target:
        mean = np.mean(subject_data[num],axis=0)
        std = np.std(subject_data[num],axis=0)
        subject_data[num] = (subject_data[num]-mean)/std

## combine categorical + numeric
def combining_image_target():
    imageFiles_labels = []

    for subjectID in tqdm(image_files):
        subjectID = subjectID[:-4] #removing '.npy' for comparing
        #print(subjectID)
        for i in range(len(subject_data)):
            if subjectID == subject_data['subjectkey'][i]:
                imageFile_label = {}
                imageFile_label['subjectkey'] = subjectID+'.npy'

                # combine all target variables in dictionary type.
                for j in range(len(col_list)-1):
                    imageFile_label[subject_data.columns[j]] = subject_data[subject_data.columns[j]][i]


                imageFiles_labels.append(imageFile_label)

    return imageFiles_labels
## ====================================== ##

## ========= divide into train, val, test ========= ##
def case_control_count(labels, dataset_type):
    if args.cat_target:
        for cat_target in args.cat_target:
            target_labels = []

            for label in labels:
                target_labels.append(label[cat_target])
            
            n_control = target_labels.count(0)
            n_case = target_labels.count(1)
            print('In {} dataset, {} contains {} CASE and {} CONTROL'.format(dataset_type, cat_target,n_case, n_control))

# defining train,val, test set splitting function
def partition(imageFiles_labels,args):
    random.shuffle(imageFiles_labels)

    images = []
    labels = []
    targets = args.cat_target + args.num_target

    for imageFile_label in imageFiles_labels:
        image = imageFile_label['subjectkey']
        label = {}

        for label_name in targets[:len(targets)]:
            label[label_name]=imageFile_label[label_name]

        images.append(image)
        labels.append(label)


    resize = tuple(args.resize)
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

    # number of total / train,val, test
    num_total = len(images)
    num_train = int(num_total*(1 - args.val_size - args.test_size))
    num_val = int(num_total*args.val_size)
    num_test = int(num_total*args.test_size)

    # image and label information of train
    images_train = images[:num_train]
    labels_train = labels[:num_train]

    # image and label information of valid
    images_val = images[num_train:num_train+num_val]
    labels_val = labels[num_train:num_train+num_val]

    # image and label information of test
    images_test = images[num_total-num_test:]
    labels_test = labels[num_total-num_test:]

    train_set = ImageDataset(image_files=images_train,labels=labels_train,transform=train_transform)
    val_set = ImageDataset(image_files=images_val,labels=labels_val,transform=val_transform)
    test_set = ImageDataset(image_files=images_test,labels=labels_test,transform=test_transform)

    partition = {}
    partition['train'] = train_set
    partition['val'] = val_set
    partition['test'] = test_set

    case_control_count(labels_train, 'train')
    case_control_count(labels_val, 'validation')
    case_control_count(labels_test, 'test')

    return partition
## ====================================== ##

## ========= split train, val, test ========= ##
if args.cat_target:
    preprocessing_cat()

if args.num_target:
    preprocessing_num()

imageFiles_labels = combining_image_target()

partition = partition(imageFiles_labels,args)

## ====================================== ##

### ========= Train,Validate, and Test ========= ###
'''The process of calcuating loss and accuracy metrics is as follows.
   1) sequentially calculate loss and accuracy metrics of target labels with for loop.
   2) store the result information with dictionary type.
   3) return the dictionary, which form as {'cat_target':value, 'num_target:value}
   This process is intended to easily deal with loss values from each target labels.'''


'''All of the loss from predictions are summated and this loss value is used for backpropagation.'''
def calculating_loss_acc(targets, output, cat_target, num_target, correct, total, loss_dict, acc_dict,net):
    '''define calculating loss and accuracy function used during training and validation step'''
    loss = 0.0

    for cat_label in cat_target:
        label = targets[cat_label]
        label = label.to(f'cuda:{net.device_ids[0]}')
        tmp_output = output[cat_label]

        criterion = nn.CrossEntropyLoss()

        tmp_loss = criterion(tmp_output.float(),label.long())
        loss += tmp_loss                         # loss is for aggregating all the losses from predicting each target variable

        # restoring train loss and accuracy for each task (target variable)
        loss_dict[cat_label] += tmp_loss.item()     # train_loss is for restoring loss from predicting each target variable
        _, predicted = torch.max(tmp_output.data,1)
        correct[cat_label] += (predicted == label).sum().item()
        total[cat_label] += label.size(0)
        #print(label.size(0))

    for num_label in num_target:
        y_true = targets[num_label]
        y_true = y_true.to(f'cuda:{net.device_ids[0]}')
        tmp_output = output[num_label]

        criterion =nn.MSELoss()

        tmp_loss = criterion(tmp_output.float(),y_true.float().unsqueeze(1))
        loss += tmp_loss
                #print(tmp_loss)

        # restoring train loss and accuracy for each task (target variable)
        loss_dict[num_label] += tmp_loss.item()     # train_loss is for restoring loss from predicting each target variable
        acc_dict[num_label] += tmp_loss.item() # RMSE for evaluating continuous variable

    return loss, correct, total, loss_dict,acc_dict


def calculating_acc(targets, output, cat_target, num_target, correct, total, acc_dict, net):
    '''define calculating accuracy function used during test step'''

    for cat_label in cat_target:
        label = targets[cat_label]
        label = label.to(f'cuda:{net.device_ids[0]}')
        tmp_output = output[cat_label]


        _, predicted = torch.max(tmp_output.data,1)
        correct[cat_label] += (predicted == label).sum().item()
        total[cat_label] += label.size(0)


    for num_label in num_target:
        y_true = targets[num_label]
        y_true = y_true.to(f'cuda:{net.device_ids[0]}')
        tmp_output = output[num_label]

        criterion =nn.MSELoss()

        tmp_loss = criterion(tmp_output.float(),y_true.float().unsqueeze(1))


        # restoring train loss and accuracy for each task (target variable)
        acc_dict[num_label] += tmp_loss.item() # RMSE for evaluating continuous variable
    return correct, total, acc_dict

# define training step
def train(net,partition,optimizer,args):
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

        image, targets = data
        image = image.to(f'cuda:{net.device_ids[0]}')
        output = net(image)
        loss, correct, total, train_loss,train_acc = calculating_loss_acc(targets,output, args.cat_target, args.num_target, correct, total, train_loss, train_acc,net)

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
def validate(net,partition,scheduler,args):
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


    with torch.no_grad():
        for i, data in enumerate(valloader,0):
            image, targets = data
            image = image.to(f'cuda:{net.device_ids[0]}')
            output = net(image)

            loss, correct, total, val_loss,val_acc = calculating_loss_acc(targets,output, args.cat_target, args.num_target, correct, total, val_loss, val_acc,net)


    if args.cat_target:
        for cat_label in args.cat_target:
            val_acc[cat_label] = 100 * correct[cat_label] / total[cat_label]
            val_loss[cat_label] = val_loss[cat_label] / len(valloader)

    if args.num_target:
        for num_label in args.num_target:
            val_acc[num_label] = val_acc[num_label] / len(valloader)
            val_loss[num_label] = val_loss[num_label] / len(valloader)


    # learning rate scheduler
    scheduler.step(sum(val_acc.values()))


    return val_loss, val_acc




# define test step
def test(net,partition,args):
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
            image, targets = data
            image = image.to(f'cuda:{net.device_ids[0]}')
            output = net(image)

            correct, total, test_acc = calculating_acc(targets,output, args.cat_target, args.num_target, correct, total, test_acc,net)



    if args.cat_target:
        for cat_label in args.cat_target:
            test_acc[cat_label] = 100 * correct[cat_label] / total[cat_label]

    if args.num_target:
        for num_label in args.num_target:
            test_acc[num_label] = test_acc[num_label] / len(testloader)


    return test_acc
## ============================================ ##

## ========= Experiment =============== ##
def CLIreporter(targets, train_loss, train_acc, val_loss, val_acc):
    '''command line interface reporter per every epoch during experiments'''
    var_column = []
    visual_report = {}
    visual_report['Loss (train/val)'] = []
    visual_report['MSE or ACC (train/val)'] = []
    for label_name in targets:
        var_column.append(label_name)

        loss_value = '{:2.2f} / {:2.2f}'.format(train_loss[label_name],val_loss[label_name])
        acc_value = '{:2.2f} / {:2.2f}'.format(train_acc[label_name],val_acc[label_name])
        visual_report['Loss (train/val)'].append(loss_value)
        visual_report['MSE or ACC (train/val)'].append(acc_value)

    print(pd.DataFrame(visual_report, index=var_column))
## ============================================ ##

## ========= Experiment =============== ##
def experiment(partition, subject_data, args): #in_channels,out_dim
    targets = args.cat_target + args.num_target

    # Simple CNN
    if args.model == 'simple3D':
        assert args.resize == [96, 96, 96]
        net = simple3d.simple3d(subject_data, args)
    # VGGNet
    elif args.model == 'vgg3D11':
        assert args.resize == [96, 96, 96]
        net = vgg3d.vgg3D11(subject_data, args)
    elif args.model == 'vgg3D13':
        assert args.resize == [96, 96, 96]
        net = vgg3d.vgg3D13(subject_data, args)
    elif args.model == 'vgg3D16':
        assert args.resize == [96, 96, 96]
        net = vgg3d.vgg3D16(subject_data, args)
    elif args.model == 'vgg3D19':
        assert args.resize == [96, 96, 96]
        net = vgg3d.vgg3D19(subject_data, args)
    # ResNet
    elif args.model == 'resnet3D50':
        net = resnet3d.resnet3D50(subject_data, args)
    elif args.model == 'resnet3D101':
        net = resnet3d.resnet3D101(subject_data, args)
    elif args.model == 'restnet3D152':
        net = resnet3d.resnet3D152(subject_data, args)
    # DenseNet
    elif args.model == 'densenet3D121':
        net = densenet3d.densenet3D121(subject_data, args)
    elif args.model == 'densenet3D161':
        net = densenet3d.densenet3D161(subject_data, args) 
    elif args.model == 'densenet3D169':
        net = densenet3d.densenet3D169(subject_data, args) 
    elif args.model == 'densenet3D201':
        net = densenet3d.densenet3D201(subject_data, args)     

    net = nn.DataParallel(net, device_ids=args.gpus)
    net.to(f'cuda:{net.device_ids[0]}')

    if args.optim == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.5)
    elif args.optim == 'Adam':
        optimizer = optim.Adam(net.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    else:
        raise ValueError('In-valid optimizer choice')

    # learning rate schedluer
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,'max', patience=15)

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
        

    for epoch in tqdm(range(args.epoch)):
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

        # test
    test_acc = test(net, partition, args)

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
    filename = '/home/connectome/dhkdgmlghks/3DCNN_test/{}-{}.json'.format(exp_name, hash_key)
    result.update(setting)

    with open(filename, 'w') as f:
        json.dump(result, f)

# seed number
seed = 1234
np.random.seed(seed)
torch.manual_seed(seed)



# Run Experiment and save result
setting, result = experiment(partition, subject_data,deepcopy(args))
#save_exp_result(setting,result)