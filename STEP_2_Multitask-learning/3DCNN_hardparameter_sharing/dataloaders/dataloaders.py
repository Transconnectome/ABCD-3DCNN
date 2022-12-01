import os
from os import listdir
from os.path import isfile, join
import glob


from utils.utils import case_control_count
from dataloaders.preprocessing import preprocessing_cat, preprocessing_num

import random
from tqdm.auto import tqdm

import pandas as pd
import numpy as np

import monai
from monai.transforms import AddChannel, Compose, RandRotate90, Resize, ScaleIntensity, Flip, ToTensor
from monai.data import ImageDataset

def check_study_sample(study_sample):
    if study_sample == 'UKB':
        #image_dir = '/scratch/connectome/3DCNN/data/2.UKB/1.sMRI_fs_cropped'
        #phenotype_dir = '/scratch/connectome/3DCNN/data/2.UKB/2.demo_qc/UKB_phenotype.csv'
        image_dir = '/home/ubuntu/dhkdgmlghks/2.UKB/1.sMRI_fs_cropped'
        phenotype_dir = '/home/ubuntu/dhkdgmlghks/2.UKB/2.demo_qc/UKB_phenotype.csv'
    elif study_sample == 'ABCD':
        image_dir = '/scratch/connectome/3DCNN/data/1.ABCD/2.sMRI_freesurfer'
        phenotype_dir = '/scratch/connectome/3DCNN/data/1.ABCD/4.demo_qc/ABCD_phenotype_total.csv'  
        #image_dir = '/home/ubuntu/dhkdgmlghks/1.ABCD/2.sMRI_freesurfer'
        #phenotype_dir = '/home/ubuntu/dhkdgmlghks/1.ABCD/4.demo_qc/ABCD_phenotype_total.csv'
    elif study_sample == 'ABCD_MNI':
        #image_dir = '/scratch/connectome/3DCNN/data/1.ABCD/2.sMRI_fmriprep'
        #phenotype_dir = '/scratch/connectome/3DCNN/data/1.ABCD/4.demo_qc/ABCD_phenotype_total.csv'  
        image_dir = '/home/ubuntu/dhkdgmlghks/1.ABCD/1.sMRI_freesurfer'
        phenotype_dir = '/home/ubuntu/dhkdgmlghks/1.ABCD/4.demo_qc/ABCD_phenotype_total.csv'
    elif study_sample == 'ABCD_ADHD':
        #image_dir = '/scratch/connectome/3DCNN/data/1.ABCD/2.sMRI_freesurfer'
        #phenotype_dir = '/scratch/connectome/3DCNN/data/1.ABCD/4.demo_qc/ABCD_ADHD.csv'   
        image_dir = '/home/ubuntu/dhkdgmlghks/1.ABCD/2.sMRI_freesurfer'       
        #phenotype_dir = '/home/ubuntu/dhkdgmlghks/1.ABCD/4.demo_qc/ABCD_ADHD_TotalSymp_and_KSADS.csv'
        phenotype_dir = '/home/ubuntu/dhkdgmlghks/1.ABCD/4.demo_qc/ABCD_ADHD_TotalSymp_and_KSADS_and_CBCL.csv'
    return image_dir, phenotype_dir 


def loading_images(image_dir, args, study_sample='UKB'):
    if study_sample.find('UKB') != -1:
        image_files = glob.glob(os.path.join(image_dir,'*.nii.gz'))
    elif study_sample.find('ABCD') != -1:
        image_files = glob.glob(os.path.join(image_dir,'*.npy'))
    image_files = sorted(image_files)
   
    #image_files = image_files[:100]
    print("Loading image file names as list is completed")
    return image_files


def loading_phenotype(phenotype_dir, args, study_sample='UKB'):
    if study_sample.find('UKB') != -1:
        subject_id_col = 'eid'
    elif study_sample.find('ABCD') != -1:
        subject_id_col = 'subjectkey'

    targets = args.cat_target + args.num_target
    col_list = targets + [subject_id_col]

    ### get subject ID and target variables
    subject_data = pd.read_csv(phenotype_dir)
    subject_data = subject_data.loc[:,col_list]
    subject_data = subject_data.sort_values(by=subject_id_col)
    subject_data = subject_data.dropna(axis = 0)
    subject_data = subject_data.reset_index(drop=True) # removing subject have NA values in sex
    

    ### preprocessing categorical variables and numerical variables
    if args.cat_target:
        subject_data = preprocessing_cat(subject_data, args)
        num_classes = int(subject_data[args.cat_target].nunique().values)
    if args.num_target:
        #subject_data = preprocessing_num(subject_data, args)
        num_classes = 1 
    
    return subject_data, targets, num_classes


## combine categorical + numeric
def combining_image_target(subject_data, image_files, target_list, study_sample='UKB'):
    if study_sample.find('UKB') !=- 1:
        subject_id_col = 'eid'
        suffix_len = -12
    elif study_sample.find('ABCD') != -1:
        subject_id_col = 'subjectkey'
        suffix_len = -4
    imageFiles_labels = []
    
    subj = []
    if type(subject_data[subject_id_col][0]) == np.str_ or type(subject_data[subject_id_col][0]) == str:
        for i in range(len(image_files)):
            subject_id = os.path.split(image_files[i])[-1]
            subj.append(str(subject_id[:suffix_len]))
    elif type(subject_data[subject_id_col][0]) == np.int_ or type(subject_data[subject_id_col][0]) == int:
        for i in range(len(image_files)):
            subject_id = os.path.split(image_files[i])[-1]
            subj.append(int(subject_id[:suffix_len]))

    image_list = pd.DataFrame({subject_id_col:subj, 'image_files': image_files})
    subject_data = pd.merge(subject_data, image_list, how='inner', on=subject_id_col)
    subject_data = subject_data.sort_values(by=subject_id_col)

    col_list = target_list + ['image_files']
    
    for i in tqdm(range(len(subject_data))):
        imageFile_label = {}
        for j, col in enumerate(col_list):
            imageFile_label[col] = subject_data[col][i]
        imageFiles_labels.append(imageFile_label)
        
    return imageFiles_labels


# defining train,val, test set splitting function
def partition_dataset(imageFiles_labels,args):
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

    case_control_count(labels_train, 'train', args)
    case_control_count(labels_val, 'validation', args)
    case_control_count(labels_test, 'test', args)

    return partition
## ====================================== ##
