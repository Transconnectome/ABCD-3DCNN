import os
import re
import glob
import random

import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from monai.transforms import (AddChannel, Compose, CenterSpatialCrop, Flip, RandAffine,
                              RandFlip, RandRotate90, Resize, ScaleIntensity, ToTensor)
from monai.data import ImageDataset, NibabelReader

from dataloaders.custom_transform import MaskTissue
from dataloaders.custom_dataset import MultiModalImageDataset
from dataloaders.preprocessing import preprocessing_cat, preprocessing_num

ABCD_data_dir = {
    'fmriprep':'/scratch/connectome/3DCNN/data/1.ABCD/1.sMRI_fmriprep/preprocessed_masked/',
    'freesurfer':'/scratch/connectome/3DCNN/data/1.ABCD/2.sMRI_freesurfer/',
    'freesurfer_256':'/scratch/connectome/3DCNN/data/1.ABCD/2.2.sMRI_freesurfer_256/',
    'FA_unwarpped_nii':'/scratch/connectome/3DCNN/data/1.ABCD/3.1.FA_unwarpped_nii/',
    'FA_warpped_nii':'/scratch/connectome/3DCNN/data/1.ABCD/3.2.FA_warpped_nii/',
    'MD_unwarpped_nii':'/scratch/connectome/3DCNN/data/1.ABCD/3.3.MD_unwarpped_nii/',
    'MD_warpped_nii':'/scratch/connectome/3DCNN/data/1.ABCD/3.4.MD_warpped_nii/',
    'RD_unwarpped_nii':'/scratch/connectome/3DCNN/data/1.ABCD/3.5.RD_unwarpped_nii/',
    'RD_warpped_nii':'/scratch/connectome/3DCNN/data/1.ABCD/3.6.RD_warpped_nii/',
    '5tt_warped_nii':'/scratch/connectome/3DCNN/data/1.ABCD/3.7.5tt_warped_nii/',
    'freesurfer_crop_resize128':'/home/connectome/jubin/3DCNN/ABCD/freesurfer_crop_resize128/',
    'FA_crop_resize128':'/home/connectome/jubin/3DCNN/ABCD/FA_crop_resize128/'
}

ABCD_phenotype_dir = {
    'total':'/scratch/connectome/3DCNN/data/1.ABCD/4.demo_qc/ABCD_phenotype_total.csv',
    'ADHD_case':'/scratch/connectome/3DCNN/data/1.ABCD/4.demo_qc/ABCD_ADHD.csv',
    'suicide_case':'/scratch/connectome/3DCNN/data/1.ABCD/4.demo_qc/ABCD_suicide_case.csv',
    'suicide_control':'/scratch/connectome/3DCNN/data/1.ABCD/4.demo_qc/ABCD_suicide_control.csv'
}

UKB_data_dir = '/scratch/connectome/3DCNN/data/2.UKB/1.sMRI_fs_cropped/'
UKB_phenotype_dir = '/scratch/connectome/3DCNN/data/2.UKB/2.demo_qc/UKB_phenotype.csv'


def case_control_count(labels, dataset_type, args):
    if args.cat_target:
        for cat_target in args.cat_target:
            curr_cnt = labels[cat_target].value_counts()
            print(f'In {dataset_type},\t"{cat_target}" contains {curr_cnt[1]} CASE and {curr_cnt[0]} CONTROL')

            
def loading_images(image_dir, args):
    image_files = pd.DataFrame()
    data_types = args.data_type if (args.tissue == None) else args.data_type + ['5tt_warped_nii']
    for brain_modality in data_types:
        curr_dir = image_dir[brain_modality]
        curr_files = pd.DataFrame({brain_modality:glob.glob(curr_dir+'*[yz]')}) # to get .npy(sMRI) & .nii.gz(dMRI) files
        curr_files[subjectkey] = curr_files[brain_modality].map(lambda x: x.split("/")[-1].split('.')[0].split('sub-')[-1])
        if args.dataset == 'UKB':
            curr_files[subjectkey] = curr_files[subjectkey].map(int)
        curr_files.sort_values(by=subjectkey, inplace=True)
        
        if len(image_files) == 0:
            image_files = curr_files
        else:
            image_files = pd.merge(image_files, curr_files, how='inner', on=subjectkey)
            
    if args.debug:
        image_files = image_files[:100]
        
    return image_files


def get_available_subjects(subject_data, args):
    case  = pd.read_csv(ABCD_phenotype_dir['ADHD_case'])[subjectkey]
    control = pd.read_csv(ABCD_phenotype_dir['suicide_control'])[subjectkey]
    filtered_subjectkey = pd.concat([case,control]).reset_index(drop=True)
    subject_data = subject_data[subject_data[subjectkey].isin(filtered_subjectkey)]
    
    return subject_data


def filter_phenotype(subject_data, filters):
    for fil in filters:
        fil_name, fil_option = fil.split(':')
        fil_option = np.float64(fil_option)
        subject_data = subject_data[subject_data[fil_name] == fil_option]
        
    return subject_data


def loading_phenotype(phenotype_dir, target_list, args):
    col_list = target_list + [subjectkey]

    ## get subject ID and target variables
    subject_data = pd.read_csv(phenotype_dir)
    subject_data = subject_data.loc[:,col_list]
    if 'Attention.Deficit.Hyperactivity.Disorder.x' in target_list:
        subject_data = get_available_subjects(subject_data, args)
    subject_data = filter_phenotype(subject_data, args.filter)
    subject_data = subject_data.sort_values(by=subjectkey)
    subject_data = subject_data.dropna(axis = 0)
    subject_data = subject_data.reset_index(drop=True)

    ### preprocessing categorical variables and numerical variables
    subject_data = preprocessing_cat(subject_data, args)
    if args.num_normalize == True:
        subject_data = preprocessing_num(subject_data, args)
    
    return subject_data


def make_balanced_testset(il, num_test, args):
    n_case = num_test//2
    n_control = num_test - n_case
    t_case, rest_case = np.split(il[il[args.cat_target[0]]==0], (n_case,))
    t_control, rest_control = np.split(il[il[args.cat_target[0]]==1],(n_control,))
    
    test = pd.concat((t_case, t_control))
    rest = pd.concat((rest_case, rest_control))
    
    test = test.sort_values(by=subjectkey)
    rest = rest.sort_values(by=subjectkey)
    
    imageFiles_labels = pd.concat((rest,test)).reset_index(drop=True)
    
    return imageFiles_labels


# defining train,val, test set splitting function
def partition_dataset(imageFiles_labels, target_list, args):
    ## Random shuffle according to args.seed -> Disable.
#     imageFiles_labels = imageFiles_labels.sample(frac=1).reset_index(drop=True)
    
    ## Dataset split    
    num_total = len(imageFiles_labels)
    num_train = int(num_total*(1 - args.val_size - args.test_size))
    num_val = int(num_total*args.val_size) if args.cv == None else int((num_total-num_test)/5)
    num_test = int(num_total*args.test_size)
    
#     num_test = int(num_total*args.test_size)
#     num_val = int(num_total*args.val_size) if args.cv == None else int((num_total-num_test)/5)
#     num_train = num_total - (num_val+num_test)
    
    if len(args.cat_target) > 0:
        imageFiles_labels = make_balanced_testset(imageFiles_labels, num_test, args)
    images = imageFiles_labels[args.data_type]
    labels = imageFiles_labels[target_list].to_dict('records')
    
    ## split dataset by 5-fold cv or given split size
    if args.cv == None:
        images_train, images_val, images_test = np.split(images, [num_train, num_train+num_val]) # revising
        labels_train, labels_val, labels_test = np.split(labels, [num_train, num_train+num_val])
    else:
        split_points = [num_val, 2*num_val, 3*num_val, 4*num_val, num_total-num_test]
        images_total, labels_total = np.split(images, split_points), np.split(labels, split_points)
        images_test, labels_test = images_total.pop(), labels_total.pop()
        images_val, labels_val = images_total.pop(args.cv-1), labels_total.pop(args.cv-1)
        images_train, labels_train = np.concatenate(images_total), np.concatenate(labels_total)
        num_train, num_val = images_train.shape[0], images_val.shape[0]
        
    print(f"Total subjects={num_total}, train={num_train}, val={num_val}, test={num_test}")

    ## Define transform function
    resize = tuple(args.resize)
    default_transforms = [ScaleIntensity(), AddChannel(), Resize(resize), ToTensor()] 
    
    if 'resize128' in args.data_type:
        default_transforms.pop(2)
    if 'crop' in args.transform:
        default_transforms.insert(2, CenterSpatialCrop(192))
    if args.tissue:
        dMRI_transform = [MaskTissue(imageFiles_labels['5tt_warped_nii'], args.tissue)]
        dMRI_transform += default_transforms
        
    aug_transforms = []
    if 'shift' in args.augmentation:
        aug_transforms.append(RandAffine(prob=0.1,translate_range=(0,2),padding_mode='zeros'))
    elif 'flip' in args.augmentation:
        aug_transforms.append(RandFlip(prob=0.1, spatial_axis=0))
    
    train_transforms, val_transforms, test_transforms = [], [], []
    for brain_modality in args.data_type:
        curr_transform = dMRI_transform if args.tissue else default_transforms
        train_transforms.append(Compose(curr_transform + aug_transforms))
        val_transforms.append(Compose(curr_transform))
        test_transforms.append(Compose(curr_transform))
    
    ## make splitted dataset
    train_set = MultiModalImageDataset(image_files=images_train, labels=labels_train, transform=train_transforms)
    val_set = MultiModalImageDataset(image_files=images_val, labels=labels_val, transform=val_transforms)
    test_set = MultiModalImageDataset(image_files=images_test, labels=labels_test, transform=test_transforms)

    partition = {}
    partition['train'] = train_set
    partition['val'] = val_set
    partition['test'] = test_set

#     case_control_count(labels_train, 'train', args)
#     case_control_count(labels_val, 'validation', args)
#     case_control_count(labels_test, 'test', args)

    return partition


def make_dataset(args):
    global subjectkey
    subjectkey = 'subjectkey' if args.dataset == 'ABCD' else 'eid'
    image_dir = ABCD_data_dir if args.dataset == 'ABCD' else UKB_data_dir
    phenotype_dir = ABCD_phenotype_dir['total'] if args.dataset == 'ABCD' else UKB_phenotype_dir
    target_list = args.cat_target + args.num_target
    
    image_files = loading_images(image_dir, args)
    subject_data = loading_phenotype(phenotype_dir, target_list, args)

    # combining image files & labels
    imageFiles_labels = pd.merge(subject_data, image_files, how='inner', on=subjectkey)

    # partitioning dataset and preprocessing (change the range of categorical variables and standardize numerical variables)
    partition = partition_dataset(imageFiles_labels, target_list, args)
    print("*** Making a dataset is completed *** \n")
    
    return partition, subject_data
