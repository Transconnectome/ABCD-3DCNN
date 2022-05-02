### load library
#from importlib.metadata import files
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from copy import deepcopy # Add Deepcopy for args

import numpy as np
import pandas as pd
import nibabel as nib
import seaborn as sns # visualization
import matplotlib.pyplot as plt # graph
import sklearn

import os
import glob
import sys
import argparse
import time
from tqdm.auto import tqdm # process bar
import random

import monai
from monai.data import CSVSaver, ImageDataset, DistributedWeightedRandomSampler
from monai.transforms import AddChannel, Compose, RandRotate90, Resize, ScaleIntensity, Flip, ToTensor
from monai.utils import set_determinism
from monai.apps import CrossValidation

from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader

import densenet3d

## arguments
parser = argparse.ArgumentParser()
parser.add_argument("--mask_size",default=7,type=float,required=False,help='')
parser.add_argument("--stride",default=5,type=float,required=False,help='')
parser.add_argument("--val_size",default=0.1,type=float,required=False,help='')
parser.add_argument("--test_size",default=0.1,type=float,required=False,help='')
parser.add_argument("--resize",default=[80, 80, 80],type=int,nargs="*",required=False,help='')
parser.add_argument("--target", type=str, default='sex', required=True, help='')
parser.add_argument("--gpus", type=int, default=[7],nargs='*', required=False, help='')
parser.add_argument("--batch_size", type=int, default=256, required=False, help='')
parser.add_argument("--save_dir", default='/scratch/connectome/dhkdgmlghks/UKB_interpretation/sex/OcclusionSensitivity',type=str, required=True)
parser.add_argument("--sbatch", type=str, default='False', required=False, help='')
args = parser.parse_args()




###### 1) load image data ######
# getting image file names (subject ID + '.npy') as list 
base_dir = '/master_ssd/3DCNN/data/2.UKB/1.sMRI_fs_cropped'
data_dir = '/master_ssd/3DCNN/data/2.UKB/1.sMRI_fs_cropped'
os.chdir(data_dir)
image_files = glob.glob('*.nii.gz')
image_files = sorted(image_files)
#image_files = image_files[:100]


###### 2) load subjectkey and target variable ######
# getting subject ID and target variables 
col_list = [args.target] + ['eid']

subject_data = pd.read_csv('/master_ssd/3DCNN/data/2.UKB/2.demo_qc/UKB_phenotype.csv')
subject_data = subject_data.loc[:,col_list]
subject_data = subject_data.sort_values(by='eid')
subject_data = subject_data.dropna(axis = 0) 
subject_data = subject_data.reset_index(drop=True) # removing subject have NA values in sex

imageFiles_labels = []
    
subj= []
if type(subject_data['eid'][0]) == np.str_ or type(subject_data['eid'][0]) == str:
    for i in range(len(image_files)):
        subj.append(str(image_files[i][:-12]))
elif type(subject_data['eid'][0]) == np.int_ or type(subject_data['eid'][0]) == int:
    for i in range(len(image_files)):
        subj.append(int(image_files[i][:-12]))
    
image_list = pd.DataFrame({'eid':subj, 'image_files': image_files})
subject_data = pd.merge(subject_data, image_list, how='inner', on='eid')

col_list = col_list + ['image_files']
    
for i in tqdm(range(len(subject_data))):
    imageFile_label = {}
    for j, col in enumerate(col_list):
        imageFile_label[col] = subject_data[col][i]
    imageFiles_labels.append(imageFile_label)


### 3) split data into train / val / test
def partition_dataset(imageFiles_labels,args):
    #random.shuffle(imageFiles_labels)

    images = []
    labels = []
    targets = args.target 

    for imageFile_label in imageFiles_labels:
        image = imageFile_label['image_files']
        label = imageFile_label[args.target]

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
    images_test = images[num_train+num_val:]
    labels_test = labels[num_train+num_val:]

    train_set = ImageDataset(image_files=images_train,labels=labels_train,transform=train_transform)
    val_set = ImageDataset(image_files=images_val,labels=labels_val,transform=val_transform)
    test_set = ImageDataset(image_files=images_test,labels=labels_test,transform=test_transform)

    partition = {}
    partition['train'] = train_set
    partition['val'] = val_set
    partition['test'] = test_set

    
    return partition, images_test 

partition, subject_list = partition_dataset(imageFiles_labels,args)


### 4) load pre-trained model
'''네트워크를 선언한다.
   이미 저장되어 있는 학습된 parameter를 앞서 선언한 네트워크에 얹어준다.'''
net = densenet3d.densenet121()


model_state = torch.load("/scratch/connectome/dhkdgmlghks/UKB_sex_densenet3D121_6cbde7.pth", map_location='cpu')
model_state['FClayers.weight'] = model_state.pop('FClayers.0.0.weight')
model_state['FClayers.bias'] = model_state.pop('FClayers.0.0.bias')
net.load_state_dict(model_state)
if args.sbatch == 'True':
    net = nn.DataParallel(net)
else:
    net = nn.DataParallel(net, device_ids = args.gpus)

net.to(f'cuda:{net.device_ids[0]}')

### 5) CAM class setting 
occ_sens = monai.visualize.OcclusionSensitivity(nn_module=net, mask_size = args.mask_size, n_batch=args.batch_size, stride=args.stride)


### 6) GradCAM calculation and saving results as numpy file 
for idx, subj in enumerate(subject_list):
    subj = subj[:-7]
    subject_list[idx] = subj

testloader = torch.utils.data.DataLoader(partition['test'],
                                         batch_size=1,
                                         shuffle=False,
                                         num_workers=2)

net.eval()

for idx, data in tqdm(enumerate(testloader,0)):
    image, label = data 
    image = image.to(f'cuda:{net.device_ids[0]}')
    label = label.to(f'cuda:{net.device_ids[0]}')

    criterion = nn.Softmax(dim=1)
    pred_prob = net(image)
    pred_prob = criterion(pred_prob)
    _, predicted = torch.max(pred_prob.data,1)

    occ_map, _ = occ_sens(image)
    occ_map = occ_map[..., predicted.item()]  # predicted label에 대한 occlusion map을 얻는 과정
    occ_map = occ_map.cpu().numpy()
    occ_map = occ_map.reshape(tuple(args.resize))

    # only case when predictino is right
    if predicted == label:
        # if prediction and label are male
        if label.item() == 0:
            sex_score = pred_prob.data[0][predicted.item()]
            if sex_score >= 0.75:
                save_dir = os.path.join(args.save_dir, 'male_upper_0.75')
                file_path = os.path.join(save_dir, subject_list[idx]) 
                np.save(file_path, occ_map)
            else:
                save_dir = os.path.join(args.save_dir, 'male_lower_0.75')
                file_path = os.path.join(save_dir, subject_list[idx])
                np.save(file_path, occ_map)
                
        # elif prediction and label are female
        elif label.item() == 1:
            sex_score = pred_prob.data[0][predicted.item()]
            if sex_score >= 0.75:
                save_dir = os.path.join(args.save_dir, 'female_upper_0.75')
                file_path = os.path.join(save_dir, subject_list[idx]) 
                np.save(file_path, occ_map)
            else:
                save_dir = os.path.join(args.save_dir, 'female_lower_0.75')
                file_path = os.path.join(save_dir, subject_list[idx])
                np.save(file_path, occ_map)





