import nibabel as nib

def plot(input) :
    if type(input) == np.ndarray:
        image = input
    else:
        image = get_image(input)

    fig = plt.figure()
    ax = plt.imshow(image, cmap=plt.cm.gray)
    ax = plt.Axes(fig,[0,0,1,1])
    plt.axis('off')
    plt.show()

from glob import glob
import logging
import os
import sys

import random
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm

import pandas as pd
import re

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import monai
from monai.data import CSVSaver, ImageDataset, DistributedWeightedRandomSampler
from monai.transforms import AddChannel, Compose, RandRotate90, Resize, ScaleIntensity, Flip, ToTensor
from monai.utils import set_determinism
from monai.apps import CrossValidation

from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix

#pip install torchvision

from typing import Callable

#import pandas as pd
#import torch
import torch.utils.data
import torchvision

import argparse

import csv

## ========= Argument Setting ========= ##
parser = argparse.ArgumentParser()

parser.add_argument('-o', '--optim', type=int, required=False, default=0, help='optimizer option')
parser.add_argument('--debug',  default=False, action='store_true')
parser.add_argument('-e', '--epoch', type=int, required=False, default=100, help='total # of epoch')
parser.add_argument('-p', '--path', type=str, required=False, default='./docker/share/B_DenseNet/images', help='relative path to preprocessed dataset')
parser.add_argument('-i', '--id', type=int, required=False, default=0, help='id')
parser.add_argument('--lr', '-lr', type=float, default=1e-5, required = False, help = 'learning rate')
parser.add_argument('--l2', '-l2', type=float, default=0.00001, required = False, help = 'weight decay')


args = parser.parse_args()
## ==================================== ##


class IDS(torch.utils.data.sampler.Sampler):
    #!pip install https://github.com/ufoym/imbalanced-dataset-sampler/archive/master.zip
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices: a list of indices
        num_samples: number of samples to draw
        callback_get_label: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(self, dataset, indices: list = None, num_samples: int = None, callback_get_label: Callable = None):
        # if indices is not provided, all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) if indices is None else indices

        # define custom callback
        self.callback_get_label = callback_get_label

        # if num_samples is not provided, draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) if num_samples is None else num_samples

        # distribution of classes in the dataset
        df = pd.DataFrame()
        df["label"] = self._get_labels(dataset)
        df.index = self.indices
        df = df.sort_index()

        label_to_count = df["label"].value_counts()

        weights = 1.0 / label_to_count[df["label"]]

        self.weights = torch.DoubleTensor(weights.to_list())

    def _get_labels(self, dataset):
        if self.callback_get_label:
            return self.callback_get_label(dataset)
        elif isinstance(dataset, torchvision.datasets.MNIST):
            return dataset.train_labels.tolist()
        elif isinstance(dataset, torchvision.datasets.ImageFolder):
            return dataset.imgs[:][1]
        elif isinstance(dataset, torchvision.datasets.DatasetFolder):
            return dataset.samples[:][1]
        elif isinstance(dataset, torch.utils.data.Subset):
            return dataset.dataset.imgs[:][1]
        elif isinstance(dataset, monai.data.image_dataset.ImageDataset):
            return dataset.labels
        elif isinstance(dataset, torch.utils.data.Dataset):
            return dataset.get_labels()

        else:
            raise NotImplementedError

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples

if args.debug:
    set_determinism(seed=0)


# ## ========= Data loading ========= ##
# path = args.path + '/*/*'
# images = glob(path) ## 210617 masked images
# random.shuffle(images)
# labels = np.array([1 if "female" in f else 0 for f in images], dtype=np.int64)


## ======== NEW DATA LOADER ======== ##

### get subject ID and target variables
target = 'sex'

subject_dir = '/home/connectome/bettybetty3k/docker/share/data/ABCD'
os.chdir(subject_dir)
subject_data = pd.read_csv('ABCD_phenotype_total.csv')
subject_data = subject_data.loc[:,['subjectkey',target]]
subject_data = subject_data.sort_values(by='subjectkey')
subject_data = subject_data.dropna(axis = 0)
subject_data = subject_data.reset_index(drop=True) # removing subject have NA values in sex
print("Loading subject list is completed")

data_dir = '/home/connectome/bettybetty3k/docker/share/preprocessed_masked'
os.chdir(data_dir) # if I do not change directory here, image data is not loaded
# get subject ID and target variables as sorted list

image_files = glob('*.npy')
image_files = sorted(image_files)

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

random.shuffle(imageFiles_labels)

images = []
labels = []
for imageFile_label in imageFiles_labels:
    image, label = imageFile_label
    images.append(image)
    labels.append(label)

if args.debug:
    print(np.unique(labels, return_counts= True))

np.load(images[0]).shape

weight = 1/np.unique(labels, return_counts= True)[1]*100
weight = [weight[0],weight[1]]
if args.debug:
    print(weight)

# array size (99, 117, 95)
size = 96
resize = (size,size,size)
train_transforms = Compose([ScaleIntensity(), AddChannel(), #Resize(resize), #RandRotate90(), Flip(),
                            ToTensor()])
val_transforms = Compose([ScaleIntensity(), AddChannel(), #Resize(resize),
                          ToTensor()])
test_transforms = Compose([ScaleIntensity(), AddChannel(), #Resize(resize),
                           ToTensor()])

length = len(images)
print("dataset size: ", length)
images_train = images[:int(length*0.8)]
images_val = images[int(length*0.8):int(length*0.9)]
images_test = images[int(length*0.9):]

labels_train = labels[:int(length*0.8)]
labels_val = labels[int(length*0.8):int(length*0.9)]
labels_test = labels[int(length*0.9):]

df1 = pd.DataFrame(images_train, columns=["path"])
df1["data"] = "train"

df2 = pd.DataFrame(images_val, columns=["path"])
df2["data"] = "val"

df3 = pd.DataFrame(images_test, columns=["path"])
df3["data"] = "test"

pd.concat([df1,df2,df3]).to_csv("result/dataset_abcd.csv", index=False)

BATCH_SIZE=16

# create a data loader with monai
train_ds = ImageDataset(image_files=images_train, labels=labels_train, transform=train_transforms)
#train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available())
train_loader = DataLoader(train_ds, sampler = IDS(train_ds), batch_size=BATCH_SIZE, num_workers=2, pin_memory=torch.cuda.is_available())
val_ds = ImageDataset(image_files=images_val, labels=labels_val, transform=val_transforms)
val_loader = DataLoader(val_ds, batch_size=1, num_workers=2, pin_memory=torch.cuda.is_available())
test_ds = ImageDataset(image_files=images_test, labels=labels_test, transform=test_transforms)
test_loader = DataLoader(test_ds, batch_size=1, num_workers=2, pin_memory=torch.cuda.is_available())
## ==================================== ##

## ===== Model, Optimizer, Scheduler ===== ##
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create DenseNet121, CrossEntropyLoss and Adam optimizer
model = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=1, out_channels=2).to(device)

if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
  model = nn.DataParallel(model)
# if you want to load previous training result, 
# model.load_state_dict(torch.load("model/ABCD_3DCNN-2.pth"))

loss_function = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor(weight).cuda())

if args.optim == 0:
    # Optimizer 1: Simple Adam
    optimizer = torch.optim.Adam(model.parameters(), args.lr=1e-5, args.l2=1e-5)
elif args.optim == 1:
    # Optimizer 2: SGD with three step learning rate
    optimizer = torch.optim.SGD(model.parameters(), args.lr=0.001, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,60], gamma=0.1)
else:
    # Optimizer 3: SGD with scheduler
    optimizer = torch.optim.SGD(model.parameters(), args.lr=0.001, args.l2=1e-4, momentum=0.9, nesterov=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=4)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
## ==================================== ##


## ============ Training ============ ##
val_interval = 2
best_metric_acc = -1
best_metric_loss = 1000000
best_metric_epoch = -1
best_metric_epoch1 = -1
best_metric_epoch2 = -1
epoch_loss_values = list()
epoch_acc_values = list()
metric_values = list()
metric_values_loss = list()
# true_positive = list()
# true_negative = list()
# false_positive = list()
# false_negative = list()
writer = SummaryWriter(log_dir='./temp')

pth_path = '/home/connectome/bettybetty3k/'
if args.id == 0:
    pth_path = pth_path + 'model'
else:
    pth_path = pth_path + 'weight' + str(args.id)

print('pth_path: ', pth_path)

epochs = args.epoch
for epoch in range(epochs):
    print("-" * 10)
    print(f"epoch {epoch + 1}/{epochs}")
    current_lr = get_lr(optimizer)
    model.train()
    epoch_loss = 0
    epoch_val_loss = 0
    step = 0
    epoch_acc = 0
    for batch_data in train_loader:
        step += 1
        epoch_len = len(train_ds) // train_loader.batch_size
        if step > epoch_len:
            break
        inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
        if args.debug:
            print(labels)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
        predicted = outputs.argmax(dim=1)
        if args.debug:
            print(predicted)
        train_correct = torch.eq(predicted, labels)
        num_corr = train_correct.sum().item()
        train_acc = num_corr / len(labels)
        epoch_acc += train_acc
        print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}, train_acc: {train_acc:.4f}")
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    epoch_acc /= step
    epoch_acc_values.append(epoch_acc)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
    if args.optim == 1:
        scheduler.step()
    elif args.optim > 1:
        scheduler.step(epoch_loss)

    if (epoch + 1) % val_interval == 0:
        model.eval()

        with torch.no_grad():
            num_correct = 0.0
            metric_count = 0
            female = 0
            # total_labels = list()
            # total_predict = list()
            for val_data in val_loader:
                val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
                val_outputs = model(val_images)
                val_loss = loss_function(val_outputs, val_labels)
                epoch_val_loss += val_loss.item()
                val_predict = val_outputs.argmax(dim=1)
                value = torch.eq(val_predict, val_labels)
                female_count = torch.eq(val_predict, 1)
                metric_count += len(value)
                num_correct += value.sum().item()
                female += female_count.sum().item()
            #     total_labels.append(val_labels)
            #     total_predict.append(val_predict)
            # if args.debug:
            #     print(total_labels)
            #     print(total_predict)
            # c_mat = confusion_matrix(np.array(total_labels), np.array(total_predict))
            # if args.debug:
            #     print(c_mat)
            # true_positive.append(c_mat[1,1])
            # true_negative.append(c_mat[1,0])
            # false_positive.append(c_mat[0,1])
            # false_negative.append(c_mat[0,0])
            # if args.debug:
            #     print(true_positive)
            #     print(true_negative)
            #     print(false_positive)
            #     print(false_negative)
            metric = num_correct / metric_count
            print(f"male: {metric_count-female}/{metric_count}, female: {female}/{metric_count}")
            metric_values.append(metric)
            metric_values_loss.append(epoch_val_loss)
            if epoch_val_loss < best_metric_loss:
                best_metric_loss = epoch_val_loss
                best_metric_epoch1 = epoch + 1
                torch.save(model.state_dict(), pth_path + "/ABCD_3DCNN-1.pth")
                print("saved new best metric model")
            if metric > best_metric_acc:
                best_metric_acc = metric
                best_metric_epoch2 = epoch + 1
                torch.save(model.state_dict(), pth_path + "/ABCD_3DCNN-2.pth")
                print("saved new best metric model")
            print(
                "current epoch: {} current loss: {:.4f} best loss: {:.4f} at epoch {}".format(
                    epoch + 1, epoch_val_loss, best_metric_loss, best_metric_epoch1
                )
            )
            print(
                "current epoch: {} current accuracy: {:.4f} best accuracy: {:.4f} at epoch {}".format(
                    epoch + 1, metric, best_metric_acc, best_metric_epoch2
                )
            )
            writer.add_scalar("val_accuracy", metric, epoch + 1)
        if current_lr != get_lr(optimizer) and args.optim > 0:
            print("update lr: loading best model weights")
            model.load_state_dict(torch.load(pth_path + "/ABCD_3DCNN-1.pth"))

print(f"train completed, best_metric: {best_metric_loss:.4f} at epoch: {best_metric_epoch1}")
print(f"train completed, best_metric: {best_metric_acc:.4f} at epoch: {best_metric_epoch2}")
writer.close()

with open('result/loss_and_acc.csv', 'w', newline='') as f:
    write = csv.writer(f)
    # write.writerow(epoch_loss_values)
    # write.writerow(epoch_acc_values)
    write.writerow(metric_values)
    write.writerow(metric_values_loss)
    # write.writerow(true_positive)
    # write.writerow(true_negative)
    # write.writerow(false_positive)
    # write.writerow(false_negative)
## ==================================== ##


## ============ Graphs ============ ##
plt.figure("train", (12, 6))
plt.subplot(1, 2, 1)
plt.title("Epoch Average Loss")
x = [i + 1 for i in range(len(epoch_loss_values))]
y = epoch_loss_values
plt.xlabel("epoch")
plt.plot(x, y)
plt.subplot(1, 2, 2)
plt.title("Val ACC")
x = [val_interval * (i + 1) for i in range(len(metric_values))]
y = metric_values
plt.xlabel("epoch")
plt.plot(x, y)
plt.show()
## ==================================== ##


## ============ Display Info ============ ##
model = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=1, out_channels=2).to(device)
if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
  model = nn.DataParallel(model)
model = model.to(device)
model.load_state_dict(torch.load(pth_path + "/ABCD_3DCNN-1.pth"))
print("Validation result - weight with the least loss")
model.eval()
y_pred = torch.tensor([], dtype=torch.float32, device=device)
y = torch.tensor([], dtype=torch.long, device=device)
y_pred_auc = []
for val_data in val_loader:
    val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
    outputs = model(val_images)
    y_pred = torch.cat([y_pred, outputs.argmax(dim=1)], dim=0)
    y_pred_auc.extend([outputs[0].tolist()])
    y = torch.cat([y, val_labels], dim=0)
print(confusion_matrix(y.cpu().numpy(), y_pred.cpu().numpy()))
print(classification_report(
    y.cpu().numpy(),
    y_pred.cpu().numpy(),
    #target_names=["non-lesion", "lesion"]
)
)
cm = confusion_matrix(
    y.cpu().numpy(),
    y_pred.cpu().numpy(),
    normalize="true",
)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    #display_labels=["non-lesion", "lesion"],
)
disp.plot(ax=plt.subplots(1, 1, facecolor="white")[1])

model = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=1, out_channels=2).to(device)
if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
  model = nn.DataParallel(model)
model = model.to(device)
model.load_state_dict(torch.load(pth_path + "/ABCD_3DCNN-2.pth"))
print("Validation result - weight with the best accuracy")

model.eval()
y_pred = torch.tensor([], dtype=torch.float32, device=device)
y = torch.tensor([], dtype=torch.long, device=device)
y_pred_auc = []
for val_data in val_loader:
    val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
    outputs = model(val_images)
    y_pred = torch.cat([y_pred, outputs.argmax(dim=1)], dim=0)
    y_pred_auc.extend([outputs[0].tolist()])
    y = torch.cat([y, val_labels], dim=0)
print(confusion_matrix(y.cpu().numpy(), y_pred.cpu().numpy()))
print(classification_report(
    y.cpu().numpy(),
    y_pred.cpu().numpy(),
    #target_names=["non-lesion", "lesion"]
)
)
cm = confusion_matrix(
    y.cpu().numpy(),
    y_pred.cpu().numpy(),
    normalize="true",
)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    #display_labels=["non-lesion", "lesion"],
)
disp.plot(ax=plt.subplots(1, 1, facecolor="white")[1])

print("Test result - weight with the best accuracy")

model.eval()
y_pred = torch.tensor([], dtype=torch.float32, device=device)
y = torch.tensor([], dtype=torch.long, device=device)
y_pred_auc = []
for val_data in test_loader:
    val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
    outputs = model(val_images)
    y_pred = torch.cat([y_pred, outputs.argmax(dim=1)], dim=0)
    y_pred_auc.extend([outputs[0].tolist()])
    y = torch.cat([y, val_labels], dim=0)
print(confusion_matrix(y.cpu().numpy(), y_pred.cpu().numpy()))
print(classification_report(
    y.cpu().numpy(),
    y_pred.cpu().numpy(),
    #target_names=["non-lesion", "lesion"]
)
)
cm = confusion_matrix(
    y.cpu().numpy(),
    y_pred.cpu().numpy(),
    normalize="true",
)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    #display_labels=["non-lesion", "lesion"],
)
disp.plot(ax=plt.subplots(1, 1, facecolor="white")[1])

###### grad cam
resize = (99,117,95)

win_size = resize # (96,96,96)

# cam = monai.visualize.CAM(nn_module=model_3d, target_layers="class_layers.relu", fc_layers="class_layers.out")
cam = monai.visualize.GradCAM(nn_module=model.module, target_layers="class_layers.relu")
# cam = monai.visualize.GradCAMpp(nn_module=model_3d, target_layers="class_layers.relu")
print("original feature shape",
      cam.feature_map_size([1, 1] + list(win_size), device),
)
print("upsampled feature shape", [1, 1] + list(win_size))

size_stride = 1 # 4
size_mask = 3 # 12

occ_sens = monai.visualize.OcclusionSensitivity(nn_module=model, mask_size = size_mask, n_batch=1, stride=size_stride)

# For occlusion sensitivity, inference must be run many times. Hence, we can use a
# bounding box to limit it to a 2D plane of interest (z=the_slice) where each of
# the arguments are the min and max for each of the dimensions (in this case CHWD).
the_slice = train_ds[0][0].shape[-1] // 2
occ_sens_b_box = [-1, -1, -1, -1, -1, -1, the_slice, the_slice]

#plt.imshow(occ_result.cpu().numpy()[0,0,:,:,0], cmap="jet")

train_transforms.set_random_state(42)
n_examples = 2
subplot_shape = [3, n_examples]
fig, axes = plt.subplots(*subplot_shape, figsize=(25, 15), facecolor="white")
items = np.random.choice(len(train_ds), size=len(train_ds))

example = 0
for item in items:

    data = train_ds[item]  # this fetches training data with random augmentations
    image, label = data[0].to(device).unsqueeze(0), data[1]
    y_pred = model(image)
    pred_label = y_pred.argmax(1).item()
    # Only display tumours images
    if label != 1 or label != pred_label:
        continue

    img = image.detach().cpu().numpy()[..., the_slice]

    name = "actual: "
    name += "ASD" if label == 1 else "non-ASD"
    name += "\npred: "
    name += "ASD" if pred_label == 1 else "non-ASD"
    name += f"\nASD: {y_pred[0,1]:.3}"
    name += f"\nnon-ASD: {y_pred[0,0]:.3}"

    # run CAM
    cam_result = cam(x=image, class_idx=None)
    cam_result = cam_result[..., the_slice]

    # run occlusion
    occ_result, _ = occ_sens(x=image, b_box=occ_sens_b_box)
    occ_result = occ_result[..., pred_label]

    for row, (im, title) in enumerate(
        zip(
            [img, cam_result, occ_result],
            [name, "CAM", "Occ. sens."],
        )
    ):
        cmap = "gray" if row == 0 else "jet"
        ax = axes[row, example]
        if isinstance(im, torch.Tensor):
            im = im.cpu().detach()
        im_show = ax.imshow(im[0][0], cmap=cmap)

        ax.set_title(title, fontsize=25)
        ax.axis("off")
        fig.colorbar(im_show, ax=ax)

    example += 1
    if example == n_examples:
        break

plt.imshow(occ_result.cpu().numpy()[0][0], cmap=cmap)

np.save("abcd_example1.npy",occ_result.cpu().numpy())
np.save("abcd_example2.npy",cam_result.cpu().numpy())

