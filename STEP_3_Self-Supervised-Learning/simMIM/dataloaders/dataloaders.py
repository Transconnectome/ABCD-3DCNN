import os
from os import listdir
from os.path import isfile, join
import glob
from typing import List

from matplotlib import transforms


from util.utils import case_control_count
from dataloaders.preprocessing import preprocessing_cat, preprocessing_num

import random
from tqdm.auto import tqdm

import pandas as pd
import numpy as np

import monai
from monai.transforms import AddChannel, Compose, RandRotate90, Resize, NormalizeIntensity, Flip, ToTensor, RandSpatialCrop, ScaleIntensity, RandAxisFlip, RandCoarseDropout
from dataloaders.dataset import Image_Dataset
from dataloaders.preprocessing import MaskGenerator
#from monai.data import ImageDataset

def check_study_sample(study_sample):
    if study_sample == 'UKB':
        image_dir = '/scratch/connectome/3DCNN/data/2.UKB/1.sMRI_fs_cropped'
        phenotype_dir = '/scratch/connectome/3DCNN/data/2.UKB/2.demo_qc/UKB_phenotype.csv'
        #image_dir = '/home/ubuntu/dhkdgmlghks/2.UKB/1.sMRI_fs_cropped'
        #phenotype_dir = '/home/ubuntu/dhkdgmlghks/2.UKB/2.demo_qc/UKB_phenotype.csv'
    elif study_sample == 'ABCD':
        image_dir = '/scratch/connectome/3DCNN/data/1.ABCD/2.sMRI_freesurfer'
        phenotype_dir = '/scratch/connectome/3DCNN/data/1.ABCD/4.demo_qc/ABCD_phenotype_total.csv'  
        #image_dir = '/home/ubuntu/dhkdgmlghks/1.ABCD/2.sMRI_freesurfer'
        #phenotype_dir = '/home/ubuntu/dhkdgmlghks/1.ABCD/4.demo_qc/ABCD_phenotype_total.csv'
    elif study_sample == 'ABCD_MNI':
        image_dir = '/scratch/connectome/3DCNN/data/1.ABCD/1.1.sMRI_MNI_warped'
        phenotype_dir = '/scratch/connectome/3DCNN/data/1.ABCD/4.demo_qc/BMI_prediction/ABCD_phenotype_total.csv'  
        #image_dir = '/home/ubuntu/dhkdgmlghks/1.ABCD/1.1.sMRI_warped'
        #phenotype_dir = '/home/ubuntu/dhkdgmlghks/1.ABCD/4.demo_qc/BMI_prediction/ABCD_phenotype_total.csv'
    elif study_sample == 'ABCD_ADHD':
        #image_dir = '/scratch/connectome/3DCNN/data/1.ABCD/2.sMRI_freesurfer'
        #phenotype_dir = '/scratch/connectome/3DCNN/data/1.ABCD/4.demo_qc/ABCD_ADHD.csv'   
        image_dir = '/home/ubuntu/dhkdgmlghks/1.ABCD/2.sMRI_freesurfer'       
        #phenotype_dir = '/home/ubuntu/dhkdgmlghks/1.ABCD/4.demo_qc/ABCD_ADHD_TotalSymp_and_KSADS.csv'
        phenotype_dir = '/home/ubuntu/dhkdgmlghks/1.ABCD/4.demo_qc/ABCD_ADHD_TotalSymp_and_KSADS_and_CBCL.csv'

    return image_dir, phenotype_dir 

def check_synthetic_sample(include_synthetic=True):
    if include_synthetic: 
        #image_dir = '/home/ubuntu/3.LDM/1.sMRI'
        image_dir = '/scratch/connectome/3DCNN/data/tmp/LDM_100k_preproc'
    else: 
        image_dir = None 
    return image_dir 


def loading_images(image_dir, args, study_sample='UKB', include_synthetic=False):
    if study_sample.find('UKB') != -1:
        image_files = glob.glob(os.path.join(image_dir,'*.nii.gz'))
    elif study_sample.find('ABCD') != -1:
        image_files = glob.glob(os.path.join(image_dir,'*.nii.gz'))
    image_files = sorted(image_files)
   
    #image_files = image_files[:100]
    print("Loading image file names as list is completed")

    synthetic_image_dir = check_synthetic_sample(include_synthetic)
    if synthetic_image_dir is not None: 
        synthetic_image_files = glob.glob(os.path.join(synthetic_image_dir,'*.nii.gz'))
        synthetic_image_files = sorted(synthetic_image_files)
        print("Loading synthetic image file names as list is completed")
    else: 
        synthetic_image_files = None 
    return image_files, synthetic_image_files

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
        suffix_len = -7
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



def partition_dataset_pretrain(imageFiles_list: list=None, synthetic_imageFiles=None, args=None):

    images_train, images_val = [], []
    for imageFiles in imageFiles_list: 
        # number of total / train,val, test
        if args.train_size + args.val_size + args.test_size != 1: 
            print('PLZ CHECK WHETHER YOU WANT TO USE ONLY THE PORTION OF DATA')
        num_total = len(imageFiles)
        num_train = int(num_total*args.train_size)
        num_val = int(num_total*args.val_size)
        num_test = int(num_total*args.test_size)
        

        # image for MAE training and linear classifier training (linear classifier is trained during linear evaluation protocol) 
        images_train_tmp = imageFiles[:num_train]

        # image for validation set during fine tuning (exactly saying linear classifier training during linear evaluation protocol)
        images_val_tmp = imageFiles[num_train:num_train+num_val]

        # image for test set during fine tuning (exactly saying linear classifier training during linear evaluation protocol)
        #images_test = imageFiles[num_train+num_val:]

        images_train, images_val = images_train + images_train_tmp, images_val + images_val_tmp

    if synthetic_imageFiles is not None: 
        images_train = images_train + synthetic_imageFiles
        print("{} Synthetic imaages are included in the training data".format(len(synthetic_imageFiles)))

    print("Training Sample: {}".format(len(images_train)))


    train_transform = Compose([ScaleIntensity(),
                               AddChannel(),
                               Resize(tuple(args.img_size)),
                               RandRotate90(prob=0.5),
                               RandAxisFlip(prob=0.5)
                               ])

    val_transform = Compose([ScaleIntensity(),
                             AddChannel(),
                             Resize(tuple(args.img_size))
                             ])

    train_set = Image_Dataset(image_files=images_train,transform=MaskGenerator(train_transform, input_size=args.img_size[0], mask_ratio=args.mask_ratio, mask_patch_size=args.mask_patch_size)) 
    val_set = Image_Dataset(image_files=images_val,transform=MaskGenerator(val_transform, input_size=args.img_size[0], mask_ratio=args.mask_ratio, mask_patch_size=args.mask_patch_size))
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
    if args.shuffle_data: 
        random.shuffle(imageFiles_labels)

    # number of total / train,val, test
    num_total = len(imageFiles_labels)
    #if args.train_size + args.val_size + args.test_size != 1: 
    #    print('PLZ CHECK WHETHER YOU WANT TO USE ONLY THE PORTION OF DATA')
    num_train = int(num_total*args.train_size)
    num_val = int(num_total*args.val_size)
    num_test = int(num_total*args.test_size)


    images = []
    labels = []

    for imageFile_label in imageFiles_labels:
        image, label = imageFile_label
        images.append(image)
        labels.append(label)
    
    if args.dataset_split == 'test':
        train_set_tmp, val_set_tmp, test_set_tmp = balancing_testset(imageFiles_labels, num_train, num_val ,num_test)
        images_train, labels_train = train_set_tmp
        images_val, labels_val = val_set_tmp 
        images_test, labels_test = test_set_tmp
    elif args.dataset_split == 'train_test': 
        train_set_tmp, val_set_tmp, test_set_tmp = balancing_trainANDtestset(imageFiles_labels, num_train, num_val ,num_test, args)
        images_train, labels_train = train_set_tmp
        images_val, labels_val = val_set_tmp 
        images_test, labels_test = test_set_tmp        
    elif args.dataset_split == 'all': 
        train_set_tmp, val_set_tmp, test_set_tmp = balancing_ALLset(imageFiles_labels, num_train, num_val ,num_test, args)
        images_train, labels_train = train_set_tmp
        images_val, labels_val = val_set_tmp 
        images_test, labels_test = test_set_tmp    
    else:
        # image and label for SSL training and linear classifier training (linear classifier is trained during linear evaluation protocol) 
        images_train = images[:num_train]
        labels_train = labels[:num_train]

        # image for validation set during fine tuning (exactly saying linear classifier training during linear evaluation protocol)
        images_val = images[num_train:num_train+num_val]
        labels_val = labels[num_train:num_train+num_val]

        # image for test set during fine tuning (exactly saying linear classifier training during linear evaluation protocol)
        images_test = images[num_train+num_val:num_train+num_val+num_test]
        labels_test = labels[num_train+num_val:num_train+num_val+num_test]

    print("Training Sample: {}. Validation Sample: {}. Test Sample: {}".format(len(images_train), len(images_val), len(images_test)))

    cutout_size = (32, 32, 32)
    num_holes = 1 
    train_transform = Compose([ScaleIntensity(),
                               AddChannel(),
                               Resize(tuple(args.img_size)),
                               RandCoarseDropout(holes=num_holes,spatial_size=cutout_size, prob=0.5),
                               RandRotate90(prob=0.5),
                               RandAxisFlip(prob=0.5),
                               ToTensor()
                               ])

    val_transform = Compose([ScaleIntensity(),
                             AddChannel(),
                             Resize(tuple(args.img_size)),
                             ToTensor()
                             ])
    
    train_set = Image_Dataset(image_files=images_train,labels=labels_train,transform=train_transform)
    val_set = Image_Dataset(image_files=images_val,labels=labels_val,transform=val_transform)
    test_set = Image_Dataset(image_files=images_test,labels=labels_test,transform=val_transform)

    partition = {}
    partition['train'] = train_set
    partition['val'] = val_set
    partition['test'] = test_set

    #case_control_count(labels_train, 'train', args)
    #case_control_count(labels_val, 'validation', args)
    #case_control_count(labels_test, 'test', args)

    return partition
## ====================================== ##





def balancing_testset(imageFiles_labels: List[tuple], num_train, num_val ,num_test:int) -> tuple:
    num_case = num_test // 2 
    num_control = num_test = num_case 

    # dataset list for train and validation 
    images = [] 
    labels = [] 

    images_case = []
    labels_case = [] 
    images_control = []
    labels_control = [] 
    
    count_case = 0 
    count_control = 0
    for imageFile_label in imageFiles_labels: 
        image, label = imageFile_label

        if label == 0: 
            if count_control < num_control: 
                images_control.append(image)
                labels_control.append(label)
                count_control += 1 
            else: 
                images.append(image)
                labels.append(label)

        elif label == 1: 
            if count_case < num_case: 
                images_case.append(image)
                labels_case.append(label)   
                count_case += 1 
            else: 
                images.append(image)
                labels.append(label)             


    images_test = images_control + images_case 
    labels_test = labels_control + labels_case 

    images_train = images[:num_train]
    labels_train = labels[:num_train]

    images_val = images[num_train:]
    labels_val = labels[num_train:]


    print(f'TRAIN (CONTROL/CASE): {labels_train.count(0)}/{labels_train.count(1)}.')
    print(f'VALIDATION (CONTROL/CASE): {labels_val.count(0)}/{labels_val.count(1)}.')
    print(f'TEST (CONTROL/CASE): {labels_test.count(0)}/{labels_test.count(1)}.')
    return (images_train, labels_train), (images_val, labels_val), (images_test, labels_test) 


def balancing_trainANDtestset(imageFiles_labels: List[tuple], num_train, num_val ,num_test:int, args) -> tuple:
    total_num_control = 0 
    total_num_case = 0 
    for imageFile_label in imageFiles_labels:
        image, label = imageFile_label
        if label == 0: 
            total_num_control += 1 
        elif label == 1: 
            total_num_case += 1 
    

    num_case_train = int(total_num_case*args.train_size)
    num_control_train = num_case_train 
    num_case_test = int(total_num_case*args.test_size)
    num_control_test = num_case_test
    

    # dataset list for train and validation 
    images = [] 
    labels = [] 

    images_case_train = []
    labels_case_train = [] 
    images_control_train = []
    labels_control_train = [] 

    images_case_test = []
    labels_case_test = [] 
    images_control_test = []
    labels_control_test = [] 

    count_case_train = 0 
    count_control_train = 0    
    count_case_test = 0 
    count_control_test = 0

    for imageFile_label in imageFiles_labels: 
        image, label = imageFile_label

        if label == 0: 
            if count_control_test < num_control_test: 
                images_control_test.append(image)
                labels_control_test.append(label)
                count_control_test += 1 
            else:
                if count_control_train < num_control_train:
                    images_control_train.append(image)
                    labels_control_train.append(label)     
                    count_control_train += 1               
                else:
                    images.append(image)
                    labels.append(label)

        elif label == 1: 
            if count_case_test < num_case_test: 
                images_case_test.append(image)
                labels_case_test.append(label)   
                count_case_test += 1 
            else: 
                if count_case_train < num_case_train:
                    images_case_train.append(image)
                    labels_case_train.append(label)
                    count_case_train += 1 
                else:
                    images.append(image)
                    labels.append(label)             


    images_test = images_control_test + images_case_test 
    labels_test = labels_control_test + labels_case_test 

    images_train = images_control_train + images_case_train
    labels_train = labels_control_train + labels_case_train

    images_val = images
    labels_val = labels


    print(f'TRAIN (CONTROL/CASE): {labels_train.count(0)}/{labels_train.count(1)}.')
    print(f'VALIDATION (CONTROL/CASE): {labels_val.count(0)}/{labels_val.count(1)}.')
    print(f'TEST (CONTROL/CASE): {labels_test.count(0)}/{labels_test.count(1)}.')
    return (images_train, labels_train), (images_val, labels_val), (images_test, labels_test) 


def balancing_ALLset(imageFiles_labels: List[tuple], num_train, num_val ,num_test:int, args) -> tuple:
    total_num_control = 0 
    total_num_case = 0 
    for imageFile_label in imageFiles_labels:
        image, label = imageFile_label
        if label == 0: 
            total_num_control += 1 
        elif label == 1: 
            total_num_case += 1 
    

    num_case_train = int(total_num_case*args.train_size)
    num_control_train = num_case_train 
    num_case_test = int(total_num_case*args.test_size)
    num_control_test = num_case_test
    num_case_val = total_num_case - num_case_train - num_case_test
    num_control_val = num_case_val
    
    

    # dataset list for train and validation 
    images_case_train = []
    labels_case_train = [] 
    images_control_train = []
    labels_control_train = [] 

    images_case_test = []
    labels_case_test = [] 
    images_control_test = []
    labels_control_test = [] 

    images_case_val = []
    labels_case_val = [] 
    images_control_val = []
    labels_control_val = [] 

    count_case_train = 0 
    count_control_train = 0    
    count_case_test = 0 
    count_control_test = 0
    count_case_val = 0 
    count_control_val = 0 

    for imageFile_label in imageFiles_labels: 
        image, label = imageFile_label

        if label == 0: 
            if count_control_test < num_control_test: 
                images_control_test.append(image)
                labels_control_test.append(label)
                count_control_test += 1 
            else:
                if count_control_train < num_control_train:
                    images_control_train.append(image)
                    labels_control_train.append(label)     
                    count_control_train += 1               
                else:
                    if count_control_val < num_control_val:
                        images_control_val.append(image)
                        labels_control_val.append(label)
                        count_control_val += 1 

        elif label == 1: 
            if count_case_test < num_case_test: 
                images_case_test.append(image)
                labels_case_test.append(label)   
                count_case_test += 1 
            else: 
                if count_case_train < num_case_train:
                    images_case_train.append(image)
                    labels_case_train.append(label)
                    count_case_train += 1 
                else:
                    if count_case_val < num_case_val: 
                        images_case_val.append(image)
                        labels_case_val.append(label)  
                        count_case_val += 1            


    images_test = images_control_test + images_case_test 
    labels_test = labels_control_test + labels_case_test 

    images_train = images_control_train + images_case_train
    labels_train = labels_control_train + labels_case_train

    images_val = images_control_val + images_case_val
    labels_val = labels_control_val + labels_case_val


    print(f'TRAIN (CONTROL/CASE): {labels_train.count(0)}/{labels_train.count(1)}.')
    print(f'VALIDATION (CONTROL/CASE): {labels_val.count(0)}/{labels_val.count(1)}.')
    print(f'TEST (CONTROL/CASE): {labels_test.count(0)}/{labels_test.count(1)}.')
    return (images_train, labels_train), (images_val, labels_val), (images_test, labels_test) 