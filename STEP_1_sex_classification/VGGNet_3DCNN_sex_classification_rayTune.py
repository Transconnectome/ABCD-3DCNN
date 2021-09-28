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
from monai.transforms import AddChannel, Compose, RandRotate90, Resize,NormalizeIntensity, Flip, ToTensor
from monai.utils import set_determinism
from monai.apps import CrossValidation

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from functools import partial


## ========= Argument Setting ========= ##
parser = argparse.ArgumentParser()

#parser.add_argument("--GPU_NUM",default=1,type=int,required=True,help='')
parser.add_argument("--val_size",default=0.1,type=float,required=False,help='')
parser.add_argument("--test_size",default=0.1,type=float,required=False,help='')
parser.add_argument("--resize",default=(96,96,96),required=False,help='')
parser.add_argument("--train_batch_size",default=32,type=int,required=False,help='')
parser.add_argument("--val_batch_size",default=16,type=int,required=False,help='')
parser.add_argument("--test_batch_size",default=16,type=int,required=False,help='')
parser.add_argument("--in_channels",default=1,type=int,required=False,help='')
parser.add_argument("--out_dim",default=2,type=int,required=False,help='')
parser.add_argument("--optim",type=str,required=True,help='')
#parser.add_argument("--lr",default=1e-5,type=float,required=False,help='')
parser.add_argument("--l2",default=5*1e-4,type=float,required=False,help='')
parser.add_argument("--epoch",type=int,required=True,help='')
parser.add_argument("--exp_name",type=str,required=True,help='')

args = parser.parse_args()
## ==================================== ##



## ========= GPU Setting ========= ##
#GPU_NUM = args.GPU_NUM # 원하는 GPU 번호 입력
#device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
#torch.cuda.set_device(device)
#print('Experiment is performed on GPU {}'.format(torch.cuda.current_device()))
## ==================================== ##


## ========= Data Preprocessing ========= ##
# getting image file names (subject ID + '.npy') as list
base_dir = '/home/connectome/dhkdgmlghks/docker/share/preprocessed_masked/'
data_dir = '/home/connectome/dhkdgmlghks/docker/share/preprocessed_masked/'
os.chdir(data_dir)
image_files = glob.glob('*.npy')
image_files = sorted(image_files)
#image_files = image_files[:500]
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
                imageFiles_labels.append((data_dir+subjectID+'.npy',0))
            elif subject_data['sex'][i] == 2:
                imageFiles_labels.append((data_dir+subjectID+'.npy',1))
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
    train_transform = Compose([NormalizeIntensity(),
                               AddChannel(),
                               Resize(resize),
                              ToTensor()])

    val_transform = Compose([NormalizeIntensity(),
                               AddChannel(),
                               Resize(resize),
                              ToTensor()])

    test_transform = Compose([NormalizeIntensity(),
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
# model configuration
cfg = {
    'VGG11': [64,'M',128,'M',256,256,'M',512,512,'M',512,512,'M'],
    'VGG13': [64,64,'M',128,128,'M',256,256,'M',512,512,'M',512,512,'M'],
    'VGG16': [64,64,'M',128,128,'M',256,256,256,'M',512,512,512,'M',512,512,512,'M'],
    'VGG19': [64,64,'M',128,128,'M',256,256,256,256,'M',512,512,512,512,'M',512,512,512,512,'M'],
}

# model definition
class VGG3D(nn.Module):
    def __init__(self,model_code,in_channels,out_dim,layer1,layer2):
        super(VGG3D,self).__init__()

        self.layers = self._make_layers(model_code,in_channels)
        self.classifier = nn.Sequential(nn.Linear(3**3*512,layer1),
                                        nn.ReLU(),
                                        nn.Dropout(),
                                        nn.Linear(layer1,layer2),
                                        nn.ReLU(),
                                        nn.Dropout(),
                                        nn.Linear(layer2,out_dim),
                                        nn.Softmax(dim=1))

    def forward(self,x):
        x = self.layers(x)
        x = x.view(x.size(0),-1)
        x = self.classifier(x)

        return x


    def _make_layers(self,model_code,in_channels):
        layers = []

        for x in cfg[model_code]:
            if x == 'M':
                layers += [nn.MaxPool3d(kernel_size=(2,2,2),stride=(2,2,2))]
            else:
                layers += [nn.Conv3d(in_channels=in_channels,
                                    out_channels=x,
                                    kernel_size=3,
                                    stride=(1,1,1),
                                    padding=1)]
                layers += [nn.BatchNorm3d(x)]
                layers += [nn.ReLU()]
                in_channels = x

        return nn.Sequential(*layers)





## ==================================== ##


## ========= Train,Validate, and Test ========= ##

# define train and validate
def train_validate(config,partition,checkpoint_dir,args):

    # first setting network and attach to cuda
    #device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')

    net = VGG3D(model_code=args.model_code,
                in_channels=args.in_channels,
                out_dim=args.out_dim,
                layer1=config['layer1'],
                layer2=config['layer2'])

    net = nn.DataParallel(net,device_ids=[7,6,5,4,3,2,1,0])
    #net.to(device)
    net.to(f'cuda:{net.device_ids[0]}')
    print('Run {} with FC layer1 {} and FC layer2 {}'.format(args.model_code,config['layer1'],config['layer2']))


    criterion = nn.CrossEntropyLoss()

    if args.optim == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=config['lr'],weight_decay=args.l2,momentum=0.9)
    elif args.optim == 'RMSprop':
        optimizer = optim.RMSprop(net.parameters(),lr=config['lr'],weight_decay=args.l2)
    elif args.optim == 'Adam':
        optimizer = optim.Adam(net.parameters(),lr=config['lr'],weight_decay=args.l2)
    elif args.optim == 'SAM':
        base_optimizer = torch.optim.SGD
        optimizer = SAM(net.parameters(),base_optimizer,lr=config['lr'],momentum=0.9)
    else:
        raise ValueError('In-valid optimizer choice')


    # for hyper parameter tuning
    if checkpoint_dir:
        model_state, optimizer_state = torch.load(os.path.join(checkpoint_dir,"checkpoint"))
        net.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    for epoch in range(args.epoch):
        ts = time.time()
        # training stage
        scaler = torch.cuda.amp.GradScaler()
        trainloader = torch.utils.data.DataLoader(partition['train'],
                                                 batch_size=args.train_batch_size,
                                                 shuffle=True,
                                                 num_workers=32)

        net.train()

        train_loss = 0.0


        if args.optim == 'SAM':
            for i, data in enumerate(trainloader,0):
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


        else:
            for i, data in enumerate(trainloader,0):
                optimizer.zero_grad()

                with torch.cuda.amp.autocast():
                    image, label = data
                    image = image.to(f'cuda:{net.device_ids[0]}')
                    label = label.to(f'cuda:{net.device_ids[0]}')
                    #image = image.cuda()
                    #label = label.cuda()
                    output = net(image)

                    loss = criterion(output,label)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                train_loss += loss.item()

            train_loss = train_loss / len(trainloader)


        # validation stage
        valloader = torch.utils.data.DataLoader(partition['val'],
                                               batch_size=args.val_batch_size,
                                               shuffle=False,
                                               num_workers=32)

        net.eval()

        correct = 0
        total = 0
        val_loss = 0.0

        with torch.no_grad():

            for i, data in enumerate(valloader,0):
                image, label = data
                image = image.to(f'cuda:{net.device_ids[0]}')
                label = label.to(f'cuda:{net.device_ids[0]}')
                #image = image.cuda()
                #label = label.cuda()
                output = net(image)

                loss = criterion(output,label)

                val_loss += loss.item()
                _, predicted = torch.max(output.data,1)
                total += label.size(0)
                correct += (predicted == label).sum().item()

            val_loss = val_loss / len(valloader)
            val_acc = 100 * correct / total

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir,"checkpoint")
            torch.save((net.state_dict(),optimizer.state_dict()),path)

        tune.report(loss=val_loss,accuracy=val_acc)


        te = time.time()

        print('Epoch {}, ACC(val): {:2.2f}, Loss(train/val): {:2.2f}/{:2.2f}. Took {:2.2f} sec'.format(epoch,val_acc,train_loss,val_loss,te-ts))




# define test step
def test(best_trial,partition,args):
    net = VGG3D(model_code=args.model_code,
                in_channels=args.in_channels,
                out_dim=args.out_dim,
                layer1=best_trial.config['layer1'],
                layer2=best_trial.config['layer2'])

    #net.cuda()
    net = nn.DataParallel(net,device_ids=[7,6,5,4,3,2,1,0])
    #net.to(device)
    net.to(f'cuda:{net.device_ids[0]}')

    best_checkpoint_dir = best_trial.checkpoint.value
    model_state, optimizer_state = torch.load(os.path.join(best_checkpoint_dir,"checkpoint"))
    net.load_state_dict(model_state,strict=False)



    testloader = torch.utils.data.DataLoader(partition['test'],
                                            batch_size=args.test_batch_size,
                                            shuffle=False,
                                            num_workers=32)



    net.eval()

    correct = 0
    total = 0

    for i, data in enumerate(testloader,0):
        image, label = data
        image = image.to(f'cuda:7')
        label = label.to(f'cuda:7')
        #image = image.cuda()
        #label = label.cuda()
        output = net(image)

        _, predicted = torch.max(output.data,1)
        total += label.size(0)
        correct += (predicted == label).sum().item()

    test_acc = 100 * correct / total

    return test_acc, best_checkpoint_dir
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
def experiment(partition,config,args): #in_channels,out_dim
    scheduler = ASHAScheduler(metric='loss',
                             mode='min',
                             max_t=args.epoch,
                             grace_period=1,
                             reduction_factor=2)


    reporter = CLIReporter(parameter_columns=['layer1','layer2','lr'],
                           metric_columns=['loss','accuracy','training_iteration'])


    result = tune.run(partial(train_validate,partition=partition,args=args),
                      resources_per_trial={'cpu':32,'gpu':8},
                      config=config,
                      progress_reporter=reporter)

    best_trial = result.get_best_trial('loss','min','last')
    
    
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(best_trial.last_result['loss']))
    print("Best trial final valiation accuracy {}".format(best_trial.last_result['accuracy']))

    test_acc, best_checkpoint_dir = test(best_trial, partition, args)


    return vars(args), result, test_acc, best_checkpoint_dir
## ==================================== ##





## ========= Run Experiment and saving result ========= ##
# define result-saving function
def save_exp_result(setting,result,best_model_test_acc, best_checkpoint_dir):
    exp_name = setting['exp_name']
    setting['test_acc'] = best_model_test_acc
    setting['best_checkpoint_dir'] = best_checkpoint_dir


    best_trial = result.get_best_trial("loss","min","last")

    result_summary = {}
    result_summary['best_trial_last_valloss'] = best_trial.last_result['loss']
    result_summary['best_trial_last_valacc'] = best_trial.last_result['accuracy']
    result_summary['best_trial_last_epoch'] = best_trial.last_result['training_iteration']

    setting.update(best_trial.config)
    setting.update(result_summary)


    hash_key = hashlib.sha1(str(setting).encode()).hexdigest()[:6]
    filename = '/scratch/3DCNN/VGGNet_results/{}-{}_lrTuning.json'.format(exp_name, hash_key)


    with open(filename, 'w') as f:
        json.dump(setting, f)

# seed number
seed = 1234
np.random.seed(seed)
torch.manual_seed(seed)

# Run Experiment and save result
name_var = 'model_code'
#list_var = ['VGG11']
list_var = ['VGG11']

checkpoint_dir = '/home/connectome/dhkdgmlghks/docker/share/VGGNet_results/'

#config = {
#    "layer1":4096,
#    "layer2":19
#}

config = {
    "layer1":4096,
    "layer2":25,
    "lr":tune.grid_search([1e-6,0.0000015,0.000002,0.0000025])
}


for var in list_var:
    setattr(args,name_var,var)
    setting, result, best_model_test_acc, best_checkpoint_dir  = experiment(partition, config,deepcopy(args))
    save_exp_result(setting,result,best_model_test_acc, best_checkpoint_dir)
    torch.cuda.empty_cache()
## ==================================================== ##
