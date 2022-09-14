import os
from os import listdir
from os.path import isfile, join
import glob


from util.utils import case_control_count
from dataloaders.preprocessing import preprocessing_cat, preprocessing_num

import random
from tqdm.auto import tqdm

import pandas as pd
import numpy as np

import monai
from monai.transforms import AddChannel, Compose, RandRotate90, Resize, NormalizeIntensity, Flip, ToTensor, RandSpatialCrop, ScaleIntensity, RandAxisFlip
from monai.data import ImageDataset

def check_study_sample(study_sample):
    if study_sample == 'UKB':
        image_dir = '/scratch/connectome/3DCNN/data/2.UKB/1.sMRI_fs_cropped'
        phenotype_dir = '/scratch/connectome/3DCNN/data/2.UKB/2.demo_qc/UKB_phenotype.csv'
    elif study_sample == 'ABCD':
        image_dir = '/scratch/connectome/3DCNN/data/1.ABCD/2.sMRI_freesurfer'
        phenotype_dir = '/scratch/connectome/3DCNN/data/1.ABCD/4.demo_qc/ABCD_phenotype_total.csv'  
    return image_dir, phenotype_dir 



def loading_images(image_dir, args, study_sample='UKB'):
    if study_sample == 'UKB':
        image_files = glob.glob(os.path.join(image_dir,'*.nii.gz'))
    elif study_sample == 'ABCD':
        image_files = glob.glob(os.path.join(image_dir,'*.npy'))
    image_files = sorted(image_files)
    #image_files = image_files[:1000]
    print("Loading image file names as list is completed")
    return image_files

def loading_phenotype(phenotype_dir, args, study_sample='UKB'):
    if study_sample == 'UKB':
        subject_id_col = 'eid'
    elif study_sample == 'ABCD':
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
    if study_sample == 'UKB':
        subject_id_col = 'eid'
        suffix_len = -12
    elif study_sample == 'ABCD':
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

    assert len(target_list) == 1  
    
    for i in tqdm(range(len(subject_data))):
        imageFile_label = (subject_data['image_files'][i], subject_data[target_list[0]][i])
        imageFiles_labels.append(imageFile_label)
        
    return imageFiles_labels



def partition_dataset_pretrain(imageFiles,args):

    train_transform = Compose([AddChannel(),
                               Resize(tuple(args.img_size)),
                               RandAxisFlip(prob=0.5),
                               NormalizeIntensity(),
                               ToTensor()])

    val_transform = Compose([AddChannel(),
                             Resize(tuple(args.img_size)),
                             NormalizeIntensity(),
                             ToTensor()])


    # number of total / train,val, test
    num_total = len(imageFiles)
    num_train = int(num_total*(1 - args.val_size - args.test_size))
    num_val = int(num_total*args.val_size)
    num_test = int(num_total*args.test_size)
    

    # image for MAE training and linear classifier training (linear classifier is trained during linear evaluation protocol) 
    images_train = imageFiles[:num_train]

    # image for validation set during fine tuning (exactly saying linear classifier training during linear evaluation protocol)
    images_val = imageFiles[num_train:num_train+num_val]

    # image for test set during fine tuning (exactly saying linear classifier training during linear evaluation protocol)
    #images_test = imageFiles[num_train+num_val:]

    print("Training Sample: {}".format(len(images_train)))

    train_set = ImageDataset(image_files=images_train,transform=train_transform) 
    val_set = ImageDataset(image_files=images_val,transform=val_transform)
    #test_set = ImageDataset(image_files=images_test,transform=val_transform)

    partition = {}
    partition['train'] = train_set
    partition['val'] = val_set
    #partition['test'] = test_set

    return partition
## ====================================== ##



def partition_dataset_finetuning(imageFiles_labels, args):
    """train_set for training simCLR
        finetuning_set for finetuning simCLR for prediction task 
        tests_set for evaluating simCLR for prediction task"""
    #random.shuffle(imageFiles_labels)

    images = []
    labels = []

    for imageFile_label in imageFiles_labels:
        image, label = imageFile_label
        images.append(image)
        labels.append(label)

    train_transform = Compose([AddChannel(),
                               Resize(tuple(args.img_size)),
                               RandAxisFlip(prob=0.5),
                               NormalizeIntensity(),
                               ToTensor()])

    val_transform = Compose([AddChannel(),
                             Resize(tuple(args.img_size)),
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

    train_set = ImageDataset(image_files=images_train,labels=labels_train,transform=train_transform)
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