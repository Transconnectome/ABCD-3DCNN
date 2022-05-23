import os
from os import listdir
from os.path import isfile, join
import glob


from utils.utils import case_control_count
from dataloaders.preprocessing import preprocessing_cat, preprocessing_num
from dataloaders.view_generator import ContrastiveLearningViewGenerator

import random
from tqdm.auto import tqdm

import pandas as pd
import numpy as np

import monai
from monai.transforms import AddChannel, Compose, RandRotate90, Resize, NormalizeIntensity, Flip, ToTensor, RandSpatialCrop
from monai.data import ImageDataset

def loading_images(image_dir, args):
    os.chdir(image_dir)
    image_files = glob.glob('*.nii.gz')
    image_files = sorted(image_files)
    #image_files = image_files[:1000]
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
    subject_data = subject_data.sort_values(by='eid')

    col_list = target_list + ['image_files']
    
    for i in tqdm(range(len(subject_data))):
        imageFile_label = {}
        for j, col in enumerate(col_list):
            imageFile_label[col] = subject_data[col][i]
        imageFiles_labels.append(imageFile_label)
        
    return imageFiles_labels



def partition_dataset_simCLR(imageFiles,args):
    """
    train_set for training simCLR
    finetuning_set for finetuning simCLR for prediction task 
    tests_set for evaluating simCLR for prediction task
    """

    """
    Imagedataset 클래스 선언시 들어가는 transform에 crop and resize를 넣으면, 같은 이미지에서 나온 위치가 다른 view image (roi box)가 생성됨. 
    반면에 data loader iterator로 불러낸 이후에 crop and resize를 넣으면, 같은 이미지에서 나온 view image들은 모두 동일한 위치 (roi box)임 
    """
    
    """
    Flow of data augmentation is as follows. 
    intensity_crop_resize (together image set1 and image set2) at CPU -> cuda -> transform (sepreately applied to image set1 and image set2) at GPU. 
    By applying scale intensity, crop, and resize, it can resolve CPU -> GPU bottleneck problem which occur when too many augmentation techniques are sequentially operated at CPU.
    That's because we can apply augmentation technique to all of samples in mini-batches simulateously by tensor operation at GPU.
    However, GPUs have limitations in their RAM memory. Thus, too large matrix couldn't be attached. 
    So, in this code, crop and resizing operations are done at CPU, afterward other augmentations are applied at GPU.
    
    This strategy dramatically reduce training time by resolving the CPU ->GPU bottleneck problem

    Of note, crop and resizing operations are done before data loader iterator stack mini-batches because each image size could be different. 
    """
    
    train_transform = Compose([AddChannel(),
                               RandSpatialCrop(roi_size= [78, 93, 78],max_roi_size=[156, 186, 156], random_center=True, random_size=True),
                               Resize(tuple(args.resize)),
                               ToTensor()]) #augmentation is done after data loader iteration. Refer to def simCLR_train


    val_transform = Compose([AddChannel(),
                             Resize(tuple(args.resize)),
                             NormalizeIntensity(),
                             ToTensor()])


    # number of total / train,val, test
    num_total = len(imageFiles)
    num_train = int(num_total*(1 - args.val_size - args.test_size))
    num_val = int(num_total*args.val_size)
    num_test = int(num_total*args.test_size)
    

    # image for SSL training and linear classifier training (linear classifier is trained during linear evaluation protocol) 
    images_train = imageFiles[:num_train]

    # image for validation set during fine tuning (exactly saying linear classifier training during linear evaluation protocol)
    images_val = imageFiles[num_train:num_train+num_val]

    # image for test set during fine tuning (exactly saying linear classifier training during linear evaluation protocol)
    images_test = imageFiles[num_train+num_val:]

    print("Training Sample: {}".format(len(images_train)))

    train_set = ImageDataset(image_files=images_train,transform=ContrastiveLearningViewGenerator(train_transform)) # return of ContrastiveLearningViewGenerator is [image1, image2]
    val_set = ImageDataset(image_files=images_val,transform=val_transform)
    test_set = ImageDataset(image_files=images_test,transform=val_transform)

    partition = {}
    partition['train'] = train_set
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


    val_transform = Compose([AddChannel(),
                             Resize(tuple(args.resize)),
                             NormalizeIntensity(),
                             ToTensor()])


    # number of total / train,val, test
    num_total = len(images)
    num_train = int(num_total*(1 - args.val_size - args.test_size))
    num_val = int(num_total*args.val_size)
    num_test = int(num_total*args.test_size)
    

    # image and label for SSL training and linear classifier training (linear classifier is trained during linear evaluation protocol) 
    images_train = images[:num_train]
    labels_train = labels[:num_train]

    # image for validation set during fine tuning (exactly saying linear classifier training during linear evaluation protocol)
    images_val = images[num_train:num_train+num_val]
    labels_val = labels[num_train:num_train+num_val]

    # image for test set during fine tuning (exactly saying linear classifier training during linear evaluation protocol)
    images_test = images[num_train+num_val:]
    labels_test = labels[num_train+num_val:]

    print("Training Sample: {}. Validation Sample: {}. Test Sample: {}".format(len(images_train), len(images_val), len(images_test)))

    train_set = ImageDataset(image_files=images_train,labels=labels_train,transform=val_transform) # return of ContrastiveLearningViewGenerator is [image1, image2]
    val_set = ImageDataset(image_files=images_val,labels=labels_val,transform=val_transform)
    test_set = ImageDataset(image_files=images_test,labels=labels_test,transform=val_transform)

    partition = {}
    partition['train'] = train_set
    partition['val'] = val_set
    partition['test'] = test_set

    #case_control_count(labels_train, 'train', args)
    #case_control_count(labels_val, 'validation', args)
    #case_control_count(labels_test, 'test', args)

    return partition
## ====================================== ##