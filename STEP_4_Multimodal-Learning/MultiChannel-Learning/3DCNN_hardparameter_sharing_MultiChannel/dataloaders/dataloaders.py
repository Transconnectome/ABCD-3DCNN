from ast import Add
import os
from os import listdir
from os.path import isfile, join
import glob


from utils.utils import case_control_count
from dataloaders.preprocessing import preprocessing_cat, preprocessing_num
from dataloaders.concatenate_channels import concatenate_channels

import random
from tqdm.auto import tqdm

import pandas as pd
import numpy as np
import torch

import monai
from monai.transforms import AddChannel, Compose, RandRotate90, Resize, ScaleIntensity, Flip, ToTensor
from monai.data import ImageDataset, Dataset, NibabelReader


def loading_images(image_dir, args):
    image_files = glob.glob(os.path.join(image_dir ,'*.nii.gz'))
    image_files = sorted(image_files)
    #image_files = image_files[:1000]
    print("Loading image file names as list is completed")
    return image_files

def loading_phenotype(phenotype_dir, args):
    targets = args.cat_target + args.num_target
    col_list = targets + ['subjectkey']

    ### get subject ID and target variables
    subject_data = pd.read_csv(phenotype_dir)
    subject_data = subject_data.loc[:,col_list]
    subject_data = subject_data.sort_values(by='subjectkey')
    subject_data = subject_data.dropna(axis = 0)
    subject_data = subject_data.reset_index(drop=True) # removing subject have NA values in sex

    ### preprocessing categorical variables and numerical variables
    subject_data = preprocessing_cat(subject_data, args)
    subject_data = preprocessing_num(subject_data, args)
    
    return subject_data, targets

def extract_subjectkey_images(subject_data, image_files, img_dir, keys):
    dir_len = len(img_dir)

    subj= []
    if type(subject_data['subjectkey'][0]) == np.str_ or type(subject_data['subjectkey'][0]) == str:
        for i in range(len(image_files)):
            subj.append(str(image_files[i][dir_len:-7]))
    elif type(subject_data['subjectkey'][0]) == np.int_ or type(subject_data['subjectkey'][0]) == int:
        for i in range(len(image_files)):
            subj.append(int(image_files[i][dir_len:-7]))

    image_list = pd.DataFrame({'subjectkey':subj, keys: image_files})

    
    return image_list


## combine categorical + numeric
def combining_image_target(subject_data, image_files, img_dir, target_list):
    imageFiles_labels = []

    FA_image_list = extract_subjectkey_images(subject_data, image_files['FA'], img_dir['FA'], 'FA')
    MD_image_list = extract_subjectkey_images(subject_data, image_files['MD'], img_dir['MD'], 'MD')
    RD_image_list = extract_subjectkey_images(subject_data, image_files['RD'], img_dir['RD'], 'RD')


    subject_data = pd.merge(subject_data, FA_image_list, how='inner', on='subjectkey')
    subject_data = pd.merge(subject_data, MD_image_list, how='inner', on='subjectkey')
    subject_data = pd.merge(subject_data, RD_image_list, how='inner', on='subjectkey')
    subject_data = subject_data.sort_values(by='subjectkey')

    col_list = target_list + ['FA', 'MD', 'RD']
    #col_list = target_list + ['FA',  'RD']
    
    for i in tqdm(range(len(subject_data))):
        imageFile_label = {}
        for j, col in enumerate(col_list):
            imageFile_label[col] = subject_data[col][i]
        imageFiles_labels.append(imageFile_label)
        

    return imageFiles_labels


def partition_dataset(imageFiles_labels,args):
    #random.shuffle(imageFiles_labels)

    data = []
    targets = args.cat_target + args.num_target

    for imageFile_label in imageFiles_labels:
        # restoring images
        image_label = {}

        for channel in ['FA', 'MD', 'RD']:
            image_label[channel] = imageFile_label[channel]

        # restoring labels
        label = {}
        for label_name in targets[:len(targets)]:
            label[label_name]=imageFile_label[label_name]
        image_label['label'] = label

        data.append(image_label)


    # number of total / train,val, test
    num_total = len(data)
    num_train = int(num_total*(1 - args.val_size - args.test_size))
    num_val = int(num_total*args.val_size)
    num_test = int(num_total*args.test_size)

    # image and label information of train
    train = data[:num_train]

    # image and label information of valid
    val = data[num_train:num_train+num_val]

    # image and label information of test
    test = data[num_train+num_val:]


    train_set = Dataset(data)
    val_set = Dataset(data)
    test_set = Dataset(data)

    partition = {}
    partition['train'] = train_set
    partition['val'] = val_set
    partition['test'] = test_set

    #case_control_count(labels_train, 'train', args)
    #case_control_count(labels_val, 'validation', args)
    #case_control_count(labels_test, 'test', args)

    return partition

## ====================================== ##
## reference 1: https://sanghyu.tistory.com/90
## reference 2: https://intrepidgeeks.com/tutorial/add-parameter-to-torch-collate-fn
## how to make custom dataloader. Especially, how to make dataloaders for map-style dataset (dictionary type dataset) 

class _custom_collate_fn(object):
    def __init__(self, resize):
        self.transform = Compose([ScaleIntensity(),
                                          Resize(tuple(resize)),
                                          AddChannel(),
                                          ToTensor()])
        self.ImageReader = NibabelReader()

    def __call__(self, batch):
        images = []
        labels = []

        for i in range(len(batch)):
            img_FA = self.ImageReader.read(batch[i]['FA'])
            img_FA, _ = self.ImageReader.get_data(img_FA)
            img_FA = self.transform(img_FA)

            img_MD = self.ImageReader.read(batch[i]['MD'])
            img_MD, _ = self.ImageReader.get_data(img_MD)
            img_MD = self.transform(img_MD)

            img_RD = self.ImageReader.read(batch[i]['RD'])
            img_RD, _ = self.ImageReader.get_data(img_RD)
            img_RD = self.transform(img_RD)

            img = torch.cat([img_FA, img_MD, img_RD], dim=0)
            img = img.unsqueeze(0)

            images.append(img)
            labels.append(batch[i]['label'])
        
        return torch.cat(images,dim=0), labels