import os
import glob

from dataloaders.preprocessing import preprocessing_cat, preprocessing_num

import random
from tqdm.auto import tqdm

import pandas as pd
import numpy as np

from monai.transforms import AddChannel, Compose, RandRotate90, Resize, ScaleIntensity, Flip, ToTensor, RandAffine
from monai.data import ImageDataset


def case_control_count(labels, dataset_type, args):
    if args.cat_target:
        for cat_target in args.cat_target:
            target_labels = []

            for label in labels:
                target_labels.append(label[cat_target])
            
            n_control = target_labels.count(0)
            n_case = target_labels.count(1) + target_labels.count(2) # revising - count also 2 for UKB data
            print('In {} dataset, {} contains {} CASE and {} CONTROL'.format(dataset_type, cat_target,n_case, n_control))
            

## ========= Define directories of data ========= ##
# revising
ABCD_data = {'fmriprep':'/scratch/connectome/3DCNN/data/1.ABCD/1.sMRI_fmriprep/preprocessed_masked',
            'freesurfer':'/scratch/connectome/3DCNN/data/1.ABCD/2.sMRI_freesurfer',
            'FA_unwarpped_nii':'/scratch/connectome/3DCNN/data/1.ABCD/3.1.FA_unwarpped_nii',
            'FA_warpped_nii':'/scratch/connectome/3DCNN/data/1.ABCD/3.2.FA_warpped_nii',
            'MD_unwarpped_nii':'/scratch/connectome/3DCNN/data/1.ABCD/3.3.MD_unwarpped_nii',
            'MD_unwarpped_npy':'/scratch/connectome/3DCNN/data/1.ABCD/3.3.MD_unwarpped_npy',
            'MD_warpped_nii':'/scratch/connectome/3DCNN/data/1.ABCD/3.4.MD_warpped_nii',
            'RD_warpped':'/scratch/connectome/3DCNN/data/1.ABCD/3.5.RD_warpped'}
ABCD_phenotype_dir = '/scratch/connectome/3DCNN/data/1.ABCD/4.demo_qc/ABCD_phenotype_total.csv'

UKB_data_dir = '/scratch/connectome/3DCNN/data/2.UKB/1.sMRI_fs_cropped/'
UKB_phenotype_dir = '/scratch/connectome/3DCNN/data/2.UKB/2.demo_qc/UKB_phenotype.csv'


## ========= Define helper functions ========= ##

def loading_images(image_dir, args):
    os.chdir(image_dir)
    image_files = pd.Series(glob.glob('*.npy')) # revising
    image_files = pd.concat([image_files, pd.Series(glob.glob('*.nii.gz'))])
    image_files.sort_values(inplace=True)
    subjects = image_files.map(lambda x: x.split('.')[0]) # revising
    #image_files = image_files[:100]
    return image_files


def filter_phenotype(subject_data, filters):
    for fil in filters:
        fil_name, fil_option = fil.split(':')
        fil_option = np.float64(fil_option)
        subject_data = subject_data[subject_data[fil_name] == fil_option]
        
    return subject_data


def loading_phenotype(phenotype_dir, target_list, args):
    col_list = target_list + [subjectkey]

    ### get subject ID and target variables
    subject_data = pd.read_csv(phenotype_dir)
    subject_data = filter_phenotype(subject_data, args.filter)
    subject_data = subject_data.loc[:,col_list]
    subject_data = subject_data.sort_values(by=subjectkey)
    subject_data = subject_data.dropna(axis = 0)
    subject_data = subject_data.reset_index(drop=True) # removing subject have NA values in sex
    if args.transfer == 'MAE' and args.dataset == 'ABCD':
        return subject_data
    elif args.scratch == 'MAE':
        return subject_data

    ### preprocessing categorical variables and numerical variables
    subject_data = preprocessing_cat(subject_data, args)
    subject_data = preprocessing_num(subject_data, args)
    
    return subject_data


# combine categorical + numeric
def combining_image_target(subject_data, image_files, target_list): # revising
    if 'str' in str(type(subject_data[subjectkey][0])): 
        image_subjectkeys = image_files.map(lambda x: str(x.split('.')[0]))
    elif 'int' in str(type(subject_data[subjectkey][0])):
        image_subjectkeys = image_files.map(lambda x: int(x.split('.')[0]))

    image_list = pd.DataFrame({subjectkey:image_subjectkeys, 'image_files':image_files})
    subject_data = pd.merge(subject_data, image_list, how='inner', on=subjectkey)  

    col_list = target_list + ['image_files']
    
    imageFiles_labels = []
    for i in tqdm(range(len(subject_data))):
        imageFile_label = {}
        for j, col in enumerate(col_list):
            imageFile_label[col] = subject_data[col][i]
        imageFiles_labels.append(imageFile_label) 

    return imageFiles_labels


# defining train,val, test set splitting function
def partition_dataset(imageFiles_labels, target_list, args):
    # make list of images & lables
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
    
    # define transform function
    resize = tuple(args.resize)
    
    default_transforms = [ScaleIntensity(),
                          AddChannel(),
                          Resize(resize),
                          ToTensor()]    
    aug_transforms = []
    if 'shift' in args.augmentation:
        aug_transforms.append(RandAffine(prob=0.1,translate_range=(0,2),padding_mode='zeros'))
    elif 'flip' in args.augmentation:
        aug_transforms.append(RandFlip(prob=0.1, spatial_axis=0))

    train_transform = Compose(default_transforms+aug_transforms)
    val_transform = Compose(default_transforms)
    test_transform = Compose(default_transforms)

    # number of total / train,val, test
    num_total = len(images)
    num_val = int(num_total*args.val_size)
    num_test = int(num_total*args.test_size)
    num_train = int(num_total*(1 - args.val_size - args.test_size))
    print(f"Total subjects={num_total}, train={num_train}, val={num_val}, test={num_test}")

    # image and label information of train, val, test
    images_train, images_val, images_test = np.split(images, [num_train, num_train+num_val]) # revising
    labels_train, labels_val, labels_test = np.split(labels, [num_train, num_train+num_val])

    train_set = ImageDataset(image_files=images_train,labels=labels_train,transform=train_transform)
    val_set = ImageDataset(image_files=images_val,labels=labels_val,transform=val_transform)
    test_set = ImageDataset(image_files=images_test,labels=labels_test,transform=test_transform)

    partition = {}
    partition['train'] = train_set
    partition['val'] = val_set
    partition['test'] = test_set

    case_control_count(labels_train, 'train', args)
    case_control_count(labels_val, 'validation', args)
    case_control_count(labels_test, 'test', args)

    return partition
## ====================================== ##


## ========= Main function that makes partition of dataset  ========= ##
def make_dataset(args): # revising
    image_dir = ABCD_data[args.data] if args.dataset == 'ABCD' else UKB_data_dir
    phenotype_dir = ABCD_phenotype_dir if args.dataset == 'ABCD' else UKB_phenotype_dir
    
    global subjectkey
    subjectkey = 'subjectkey' if args.dataset == 'ABCD' else 'eid'
    
    target_list = args.cat_target + args.num_target

    image_files = loading_images(image_dir, args)
    subject_data= loading_phenotype(phenotype_dir, target_list, args)
    os.chdir(image_dir)

    # data preprocesing categorical variable and numerical variables 
    imageFiles_labels = combining_image_target(subject_data, image_files, target_list)

    # partitioning dataset and preprocessing (change the range of categorical variables and standardize numerical variables )
    partition = partition_dataset(imageFiles_labels, target_list, args)
    print("*** Making a dataset is completed *** \n")
    
    return partition, subject_data

