import os
from os import listdir
from os.path import isfile, join
import glob


from utils.utils import case_control_count
from dataloaders.preprocessing import preprocessing_cat, preprocessing_num
from dataloaders.data_augmentation import _set_augmentation 
from dataloaders.view_generator import ContrastiveLearningViewGenerator

import random
from tqdm.auto import tqdm

import pandas as pd
import numpy as np

import monai
from monai.transforms import AddChannel, Compose, RandRotate90, Resize, ScaleIntensity, Flip, ToTensor
from monai.data import ImageDataset

def loading_images(image_dir, args):
    os.chdir(image_dir)
    image_files = glob.glob('*.nii.gz')
    image_files = sorted(image_files)
    image_files = image_files[:100]
    print("Loading image file names as list is completed")
    return image_files

def loading_phenotype(phenotype_dir, args):
    targets = args.cat_target + args.num_target
    col_list = targets + ['eid']

    ### get subject ID and target variables
    subject_data = pd.read_csv(phenotype_dir)
    subject_data = subject_data.loc[:,col_list]
    subject_data = subject_data.sort_values(by='eid')
    subject_data = subject_data.dropna(axis = 0)
    subject_data = subject_data.reset_index(drop=True) # removing subject have NA values in sex

    ### preprocessing categorical variables and numerical variables
    subject_data = preprocessing_cat(subject_data, args)
    subject_data = preprocessing_num(subject_data, args)
    
    return subject_data, targets


## combine categorical + numeric
def combining_image_target(subject_data, image_files, target_list):
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

    col_list = target_list + ['image_files']
    
    for i in tqdm(range(len(subject_data))):
        imageFile_label = {}
        for j, col in enumerate(col_list):
            imageFile_label[col] = subject_data[col][i]
        imageFiles_labels.append(imageFile_label)
        
    return imageFiles_labels



def partition_dataset_simCLR(imageFiles,args):
    """train_set for training simCLR
        finetuning_set for finetuning simCLR for prediction task 
        tests_set for evaluating simCLR for prediction task"""
    #random.shuffle(imageFiles_labels)

    transformation_list = _set_augmentation(args)
    train_transform = Compose(transformation_list)

    val_transform = Compose([ScaleIntensity(),
                               AddChannel(),
                               Resize(tuple(args.resize)),
                              ToTensor()])


    # number of total / train,val, test
    num_total = len(imageFiles)
    num_train = int(num_total*(1 - args.finetuning_size))

    num_finetuning_all = num_total - num_train
    num_finetuning = int(num_finetuning_all*(1 - args.val_size - args.test_size))
    num_val = int((num_finetuning_all - num_finetuning)*args.val_size)
    

    # image information of SSL training
    images_train = imageFiles[:num_train]

    # imageinformation of fine tuning
    images_finetuning = imageFiles[num_train:num_train+num_finetuning]

    # image information of valid
    images_val = imageFiles[num_train+num_finetuning:num_train+num_finetuning+num_val]

    # image information of test
    images_test = imageFiles[num_train+num_finetuning+num_val:]

    train_set = ImageDataset(image_files=images_train,transform=ContrastiveLearningViewGenerator(train_transform)) # return of ContrastiveLearningViewGenerator is [image1, image2]
    finetuning_set = ImageDataset(image_files=images_finetuning,transform=val_transform)
    val_set = ImageDataset(image_files=images_val,transform=val_transform)
    test_set = ImageDataset(image_files=images_test,transform=val_transform)

    partition = {}
    partition['train'] = train_set
    partition['finetuning'] = finetuning_set
    partition['val'] = val_set
    partition['test'] = test_set

    return partition
## ====================================== ##



def partition_dataset_finetuning(imageFiles_labels,args):
    """train_set for training simCLR
        finetuning_set for finetuning simCLR for prediction task 
        tests_set for evaluating simCLR for prediction task"""
    #random.shuffle(imageFiles_labels)

    images = []
    labels = []
    targets = args.cat_target + args.num_target

    for imageFile_label in imageFiles_labels:
        image = imageFile_label['image_files']
        label = {}

        for label_name in targets[:len(targets)]:
            label[label_name]=imageFile_label[label_name]

        images.append(image)
        labels.append(label)


    resize = tuple(args.resize)
    augmentation_list = _set_augmentation(args)
    transform_list = augmentation_list + [ScaleIntensity(), AddChannel(), Resize(resize), ToTensor()]
    train_transform = Compose(transform_list)

    val_transform = Compose([ScaleIntensity(),
                               AddChannel(),
                               Resize(resize),
                              ToTensor()])


    # number of total / train,val, test
    num_total = len(images)
    num_train = int(num_total*(1 - args.finetuning_size))

    num_finetuning_all = num_total - num_train
    num_finetuning = int(num_finetuning_all*(1 - args.val_size - args.test_size))
    num_val = int((num_finetuning_all - num_finetuning)*args.val_size)
    

    # image and label information of SSL training
    images_train = images[:num_train]
    labels_train = labels[:num_train]

    # image and label information of fine tuning
    images_finetuning = images[num_train:num_train+num_finetuning]
    labels_finetuning = labels[num_train:num_train+num_finetuning]

    # image and label information of valid
    images_val = images[num_train+num_finetuning:num_train+num_finetuning+num_val]
    labels_val = labels[num_train+num_finetuning:num_train+num_finetuning+num_val]

    # image and label information of test
    images_test = images[num_train+num_finetuning+num_val:]
    labels_test = labels[num_train+num_finetuning+num_val:]

    train_set = ImageDataset(image_files=images_train,labels=labels_train,transform=ContrastiveLearningViewGenerator(train_transform)) # return of ContrastiveLearningViewGenerator is [image1, image2]
    finetuning_set = ImageDataset(image_files=images_finetuning,labels=labels_finetuning,transform=val_transform)
    val_set = ImageDataset(image_files=images_val,labels=labels_val,transform=val_transform)
    test_set = ImageDataset(image_files=images_test,labels=labels_test,transform=val_transform)

    partition = {}
    partition['train'] = train_set
    partition['finetuning'] = finetuning_set
    partition['val'] = val_set
    partition['test'] = test_set

    #case_control_count(labels_train, 'train', args)
    #case_control_count(labels_val, 'validation', args)
    #case_control_count(labels_test, 'test', args)

    return partition
## ====================================== ##