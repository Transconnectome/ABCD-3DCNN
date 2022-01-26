import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import numpy as np
import time
from copy import deepcopy # Add Deepcopy for args
import seaborn as sns
import matplotlib.pyplot as plt
import nibabel as nib
import glob
import os
from tqdm.auto import tqdm
from nilearn import plotting
import matplotlib.pyplot as plt
import pandas as pd
import random
import hashlib
import json
from os import listdir
from os.path import isfile, join
import monai
from monai.data import CSVSaver, ImageDataset, DistributedWeightedRandomSampler
from monai.transforms import AddChannel, Compose, RandRotate90, Resize, ScaleIntensity, Flip, ToTensor
from monai.utils import set_determinism
from monai.apps import CrossValidation


## ========= Argument Setting ========= ##
parser = argparse.ArgumentParser()

parser.add_argument("--GPU_NUM",default=1,type=int,required=True,help='')
parser.add_argument("--model",required=True,type=str,help='')
parser.add_argument("--val_size",default=0.1,type=float,required=False,help='')
parser.add_argument("--test_size",default=0.1,type=float,required=False,help='')
parser.add_argument("--resize",default=(96,96,96),required=False,help='')
parser.add_argument("--train_batch_size",default=16,type=int,required=False,help='')
parser.add_argument("--val_batch_size",default=8,type=int,required=False,help='')
parser.add_argument("--test_batch_size",default=8,type=int,required=False,help='')
parser.add_argument("--in_channels",default=1,type=int,required=False,help='')
parser.add_argument("--out_dim",default=2,type=int,required=False,help='')
parser.add_argument("--optim",type=str,required=True,help='')
parser.add_argument("--lr",default=1e-5,type=float,required=False,help='')
parser.add_argument("--l2",default=0.00001,type=float,required=False,help='')
parser.add_argument("--epoch",type=int,required=True,help='')
parser.add_argument("--exp_name",type=str,required=True,help='')

args = parser.parse_args()
## ==================================== ##
print(args.epoch)


## ========= GPU Setting ========= ##
GPU_NUM = args.GPU_NUM # 원하는 GPU 번호 입력
device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device)
print('Experiment is performed on GPU {}'.format(torch.cuda.current_device()))
## ==================================== ##


## ========= Data Preprocessing ========= ##
# getting image file names (subject ID + '.npy') as list
base_dir = '/home/connectome/dhkdgmlghks/docker/share/preprocessed_masked'
data_dir = '/home/connectome/dhkdgmlghks/docker/share/preprocessed_masked'
os.chdir(data_dir)
image_files = glob.glob('*.npy')
image_files = sorted(image_files)



# getting subject ID and target variables
target = 'sex'

subject_data = pd.read_csv('/home/connectome/dhkdgmlghks/docker/share/data/ABCD_phenotype_total.csv')
subject_data = subject_data.loc[:,['subjectkey',target]]
subject_data = subject_data.sort_values(by='subjectkey')
subject_data = subject_data.dropna(axis = 0)
subject_data = subject_data.reset_index(drop=True) # removing subject have NA values in sex


# getting subject ID and target variables as sorted list
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


#defining train,val, test set splitting function
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
## ====================================== ##


## ========= Model define ============= ##
# model 1
class CNN3D_1(nn.Module):
    def __init__(self,in_channels,out_dim):
        super(CNN3D_1,self).__init__()
        self.conv1 = nn.Conv3d(in_channels,8,kernel_size=3,stride=(1,1,1),padding=1)
        self.conv2 = nn.Conv3d(8,16,kernel_size=3,stride=(1,1,1),padding=1)
        self.conv3 = nn.Conv3d(16,32,kernel_size=3,stride=(1,1,1),padding=1)
        self.conv4 = nn.Conv3d(32,64,kernel_size=3,stride=(1,1,1),padding=1)
        self.conv5 = nn.Conv3d(64,128,kernel_size=3,stride=(1,1,1),padding=1)
        self.conv6 = nn.Conv3d(128,256,kernel_size=3,stride=(1,1,1),padding=1)

        self.batchnorm1 = nn.BatchNorm3d(8)
        self.batchnorm2 = nn.BatchNorm3d(16)
        self.batchnorm3 = nn.BatchNorm3d(64)




        self.maxpool = nn.MaxPool3d(kernel_size=(2,2,2),stride=(2,2,2)) # 두가지 선택지, 1) padding = 0, kernel = 2; 2) padding = 1, kernel = 3
        self.act = nn.ReLU()

        self.classifier = nn.Sequential(nn.Linear(6**3*256,25),
                                        nn.BatchNorm1d(25),
                                        nn.Sigmoid(),
                                        nn.Dropout(),
                                        nn.Linear(25,out_dim))


    def forward(self,x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.batchnorm1(x)
        x = self.act(x)
        #print(x.size())
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.batchnorm2(x)
        x = self.act(x)
        #print(x.size())
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.maxpool(x)
        x = self.batchnorm3(x)
        x = self.act(x)
        #print(x.size())
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.maxpool(x)
        x = self.act(x)
        #print(x.size())
        x = x.view(x.size(0),-1)
        #print(x.size())
        x = self.classifier(x)

        return x

# model 2
class CNN3D_2(nn.Module):
    def __init__(self,in_channels,out_dim):
        super(CNN3D_2,self).__init__()
        self.conv1 = nn.Conv3d(in_channels,8,kernel_size=3,stride=(1,1,1),padding=1)
        self.conv2 = nn.Conv3d(8,16,kernel_size=3,stride=(1,1,1),padding=1)
        self.conv3 = nn.Conv3d(16,32,kernel_size=3,stride=(1,1,1),padding=1)
        self.conv4 = nn.Conv3d(32,64,kernel_size=3,stride=(1,1,1),padding=1)
        self.conv5 = nn.Conv3d(64,128,kernel_size=3,stride=(1,1,1),padding=1)
        self.conv6 = nn.Conv3d(128,256,kernel_size=3,stride=(1,1,1),padding=1)

        self.batchnorm1 = nn.BatchNorm3d(8)
        self.batchnorm2 = nn.BatchNorm3d(16)
        self.batchnorm3 = nn.BatchNorm3d(64)




        self.maxpool = nn.MaxPool3d(kernel_size=(1,2,2),stride=(1,2,2)) # 두가지 선택지, 1) padding = 0, kernel = 2; 2) padding = 1, kernel = 3
        self.act = nn.ReLU()

        self.classifier = nn.Sequential(nn.Linear(256*96*6*6,19),
                                        nn.BatchNorm1d(19),
                                        nn.Sigmoid(),
                                        nn.Dropout(),
                                        nn.Linear(19,out_dim))


    def forward(self,x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.batchnorm1(x)
        #print(x.size())
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.batchnorm2(x)
        #print(x.size())
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.maxpool(x)
        x = self.batchnorm3(x)
        #print(x.size())
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.maxpool(x)
        #print(x.size())
        x = x.view(x.size(0),-1)
        #print(x.size())
        x = self.classifier(x)

        return x
## ==================================== ##


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


    if args.optim == 'SAM':
        for i, data in enumerate(trainloader,0):
            optimizer.zero_grad()
            image, label = data
            image = image.cuda()
            label = label.cuda()

            def closure():
                loss = criterion(net(image),label)
                loss.backward()
                return loss

            loss = criterion(net(image),label)
            loss.backward()
            optimizer.step(closure)
            optimizer.zero_grad()

            train_loss += loss.item()
            _, predicted = torch.max(net(image).data,1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

    else:
        for i, data in enumerate(trainloader,0):
            optimizer.zero_grad()
            image, label = data
            image = image.cuda()
            label = label.cuda()
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
def validate(net,partition,criterion,args):
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
            image = image.cuda()
            label = label.cuda()
            output = net(image)

            loss = criterion(output,label)

            val_loss += loss.item()
            _, predicted = torch.max(output.data,1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

        val_loss = val_loss / len(valloader)
        val_acc = 100 * correct / total

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

    for i, data in enumerate(testloader,0):
        image, label = data
        image = image.cuda()
        label = label.cuda()
        output = net(image)

        _, predicted = torch.max(output.data,1)
        total += label.size(0)
        correct += (predicted == label).sum().item()

    test_acc = 100 * correct / total

    return test_acc
## ============================================ ##


## ========= defining 'SAM' optimizer =============== ##
class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]["e_w"] = e_w

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.sub_(self.state[p]["e_w"])  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm
## ================================================== ##


## ========= Experiment =============== ##
def experiment(partition,args): #in_channels,out_dim
    if args.model == 'CNN3D_1':
        net = CNN3D_1(in_channels=args.in_channels,
                    out_dim=args.out_dim)

        net.cuda()

        criterion = nn.CrossEntropyLoss()
        if args.optim == 'SGD':
            optimizer = optim.SGD(net.parameters(), lr=args.lr,weight_decay=args.l2)
        elif args.optim == 'RMSprop':
            optimizer = optim.RMSprop(net.parameters(),lr=args.lr,weight_decay=args.l2)
        elif args.optim == 'Adam':
            optimizer = optim.Adam(net.parameters(),lr=args.lr,weight_decay=args.l2)
        elif args.optim == 'SAM':
            base_optimizer = torch.optim.SGD
            optimizer = SAM(net.parameters(),base_optimizer,lr=args.lr,momentum=0.9)
        else:
            raise ValueError('In-valid optimizer choice')

        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []


        for epoch in tqdm(range(args.epoch)):
            ts = time.time()
            net, train_loss, train_acc = train(net,partition,optimizer,criterion,args)
            val_loss, val_acc = validate(net,partition,criterion,args)
            te = time.time()

            train_losses.append(train_loss)
            train_accs.append(train_acc)
            val_losses.append(val_loss)
            val_accs.append(val_acc)

            print('Epoch {}, ACC(train/val): {:2.2f}/{:2.2f}, Loss(train/val): {:2.2f}/{:2.2f}. Took {:2.2f} sec'.format(epoch,train_acc,val_acc,train_loss,val_loss,te-ts))


        test_acc = test(net,partition,args)

        result = {}
        result['train_losses'] = train_losses
        result['train_accs'] = train_accs
        result['val_losses'] = val_losses
        result['val_accs'] = val_accs
        result['train_acc'] = train_acc
        result['val_acc'] = val_acc
        result['test_acc'] = test_acc

        return vars(args), result

    else:
        net = CNN3D_2(in_channels=args.in_channels,
                    out_dim=args.out_dim)

        net.cuda()

        criterion = nn.CrossEntropyLoss()
        if args.optim == 'SGD':
            optimizer = optim.SGD(net.parameters(), lr=args.lr,weight_decay=args.l2)
        elif args.optim == 'RMSprop':
            optimizer = optim.RMSprop(net.parameters(),lr=args.lr,weight_decay=args.l2)
        elif args.optim == 'Adam':
            optimizer = optim.Adam(net.parameters(),lr=args.lr,weight_decay=args.l2)
        elif args.optim == 'SAM':
            base_optimizer = torch.optim.SGD
            optimizer = SAM(net.parameters(),base_optimizer,lr=args.lr,momentum=0.9)
        else:
            raise ValueError('In-valid optimizer choice')

        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []


        for epoch in tqdm(range(args.epoch)):
            ts = time.time()
            net, train_loss, train_acc = train(net,partition,optimizer,criterion,args)
            val_loss, val_acc = validate(net,partition,criterion,args)
            test_acc = test(net,partition,args)
            te = time.time()

            train_losses.append(train_loss)
            train_accs.append(train_acc)
            val_losses.append(val_loss)
            val_accs.append(val_acc)

            print('Epoch {}, ACC(train/val): {:2.2f}/{:2.2f}, Loss(train/val): {:2.2f}/{:2.2f}. Took {:2.2f} sec'.format(epoch,train_acc,val_acc,train_loss,val_loss,te-ts))


        test_acc = test(net,partition,args)

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
    filename = '/scratch/3DCNN/simpleCNN_results/{}-{}.json'.format(exp_name, hash_key)
    result.update(setting)
    
    with open(filename, 'w') as f:
        json.dump(result, f)

# seed number
seed = 1234
np.random.seed(seed)
torch.manual_seed(seed)

# Run Experiment and save result
setting, result = experiment(partition, deepcopy(args))
save_exp_result(setting,result)
## ==================================================== ##
