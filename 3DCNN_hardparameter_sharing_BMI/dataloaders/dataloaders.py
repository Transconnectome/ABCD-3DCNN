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
from monai.transforms import AddChannel, Compose, RandRotate90, Resize, ScaleIntensity, Flip, ToTensor, RandAxisFlip
from monai.data import ImageDataset

def check_study_sample(study_sample):
    if study_sample == 'UKB':
        image_dir = '/scratch/connectome/3DCNN/data/2.UKB/1.sMRI_fs_cropped'
        phenotype_dir = '/scratch/connectome/3DCNN/data/2.UKB/2.demo_qc/UKB_phenotype.csv'
        #image_dir = '/home/ubuntu/dhkdgmlghks/2.UKB/1.sMRI_fs_cropped'
        #phenotype_dir = '/home/ubuntu/dhkdgmlghks/2.UKB/2.demo_qc/UKB_phenotype.csv'
    elif study_sample == 'ABCD':
        image_dir = '/scratch/connectome/3DCNN/data/1.ABCD/2.sMRI_freesurfer'
        phenotype_dir = '/scratch/connectome/3DCNN/data/1.ABCD/4.demo_qc/BMI_prediction/ABCD_phenotype_total.csv'  
        #image_dir = '/home/ubuntu/dhkdgmlghks/1.ABCD/2.sMRI_freesurfer'
        #phenotype_dir = '/home/ubuntu/dhkdgmlghks/1.ABCD/4.demo_qc/BMI_prediction/ABCD_phenotype_total.csv'
    elif study_sample == 'ABCD_MNI':
        image_dir = '/scratch/connectome/3DCNN/data/1.ABCD/1.1.sMRI_MNI_warped'
        phenotype_dir = '/scratch/connectome/3DCNN/data/1.ABCD/4.demo_qc/BMI_prediction/ABCD_phenotype_total.csv'  
        #image_dir = '/home/ubuntu/dhkdgmlghks/1.ABCD/1.1.sMRI_warped'
        #phenotype_dir = '/home/ubuntu/dhkdgmlghks/1.ABCD/4.demo_qc/BMI_prediction/ABCD_phenotype_total.csv'
        phenotype_dir = '/home/ubuntu/dhkdgmlghks/1.ABCD/4.demo_qc/BMI_prediction/ABCD_ADHD_TotalSymp_and_KSADS_and_CBCL.csv'
    elif study_sample == 'ABCD_1y_after':
        image_dir = '/scratch/connectome/3DCNN/data/1.ABCD/1.1.sMRI_MNI_warped'
        phenotype_dir = '/scratch/connectome/3DCNN/data/1.ABCD/4.demo_qc/BMI_prediction/ABCD_phenotype_total_1years_revised.csv' 
        #image_dir = '/home/ubuntu/dhkdgmlghks/1.ABCD/1.1.sMRI_warped'
        #image_dir = '/home/ubuntu/dhkdgmlghks/1.ABCD/2.sMRI_freesurfer'  
        #phenotype_dir = '/home/ubuntu/dhkdgmlghks/1.ABCD/4.demo_qc/BMI_prediction/ABCD_phenotype_total_1years_revised.csv' 
    elif study_sample == 'ABCD_2y_after':
        image_dir = '/scratch/connectome/3DCNN/data/1.ABCD/1.1.sMRI_MNI_warped'
        phenotype_dir = '/scratch/connectome/3DCNN/data/1.ABCD/4.demo_qc/BMI_prediction/ABCD_phenotype_total_2years_revised.csv' 
        #image_dir = '/home/ubuntu/dhkdgmlghks/1.ABCD/1.1.sMRI_warped'
        #image_dir = '/home/ubuntu/dhkdgmlghks/1.ABCD/2.sMRI_freesurfer'  
        #phenotype_dir = '/home/ubuntu/dhkdgmlghks/1.ABCD/4.demo_qc/BMI_prediction/ABCD_phenotype_total_2years_revised.csv'     
    elif study_sample == 'ABCD_male':
        image_dir = '/scratch/connectome/3DCNN/data/1.ABCD/1.1.sMRI_MNI_warped'
        phenotype_dir = '/scratch/connectome/3DCNN/data/1.ABCD/4.demo_qc/BMI_prediction/ABCD_phenotype_total_male.csv' 
        #image_dir = '/home/ubuntu/dhkdgmlghks/1.ABCD/2.sMRI_freesurfer'  
        #image_dir = '/home/ubuntu/dhkdgmlghks/1.ABCD/1.1.sMRI_warped'
        #phenotype_dir = '/home/ubuntu/dhkdgmlghks/1.ABCD/4.demo_qc/BMI_prediction/ABCD_phenotype_total_male.csv'  
    elif study_sample == 'ABCD_female':
        image_dir = '/scratch/connectome/3DCNN/data/1.ABCD/1.1.sMRI_MNI_warped'
        phenotype_dir = '/scratch/connectome/3DCNN/data/1.ABCD/4.demo_qc/BMI_prediction/ABCD_phenotype_total_female.csv' 
        #image_dir = '/home/ubuntu/dhkdgmlghks/1.ABCD/2.sMRI_freesurfer'  
        #image_dir = '/home/ubuntu/dhkdgmlghks/1.ABCD/1.1.sMRI_warped'
        #phenotype_dir = '/home/ubuntu/dhkdgmlghks/1.ABCD/4.demo_qc/BMI_prediction/ABCD_phenotype_total_female.csv'
    elif study_sample == 'ABCD_gps':
        image_dir = '/scratch/connectome/3DCNN/data/1.ABCD/1.1.sMRI_MNI_warped'
        phenotype_dir = '/scratch/connectome/3DCNN/data/1.ABCD/4.demo_qc/BMI_prediction/ABCD_phenotype_total_gps.csv' 
        #image_dir = '/home/ubuntu/dhkdgmlghks/1.ABCD/2.sMRI_freesurfer'  
        #image_dir = '/home/ubuntu/dhkdgmlghks/1.ABCD/1.1.sMRI_warped'
        #phenotype_dir = '/home/ubuntu/dhkdgmlghks/1.ABCD/4.demo_qc/BMI_prediction/ABCD_phenotype_total_gps.csv'      
    elif study_sample == 'ABCD_2y_after_BMIgain':
        image_dir = '/scratch/connectome/3DCNN/data/1.ABCD/1.1.sMRI_MNI_warped'
        phenotype_dir = '/scratch/connectome/3DCNN/data/1.ABCD/4.demo_qc/BMI_prediction/ABCD_phenotype_total_2years_revised_BMIgain.csv' 
        #image_dir = '/home/ubuntu/dhkdgmlghks/1.ABCD/1.1.sMRI_warped'
        #image_dir = '/home/ubuntu/dhkdgmlghks/1.ABCD/2.sMRI_freesurfer'  
        #phenotype_dir = '/home/ubuntu/dhkdgmlghks/1.ABCD/4.demo_qc/BMI_prediction/ABCD_phenotype_total_2years_revised_BMIgain.csv'   
    elif study_sample == 'ABCD_2y_after_BMIloss':
        image_dir = '/scratch/connectome/3DCNN/data/1.ABCD/1.1.sMRI_MNI_warped'
        phenotype_dir = '/scratch/connectome/3DCNN/data/1.ABCD/4.demo_qc/BMI_prediction/ABCD_phenotype_total_2years_revised_BMIloss.csv' 
        #image_dir = '/home/ubuntu/dhkdgmlghks/1.ABCD/1.1.sMRI_warped'
        #image_dir = '/home/ubuntu/dhkdgmlghks/1.ABCD/2.sMRI_freesurfer'  
        #phenotype_dir = '/home/ubuntu/dhkdgmlghks/1.ABCD/4.demo_qc/BMI_prediction/ABCD_phenotype_total_2years_revised_BMIloss.csv'
    elif study_sample == 'ABCD_1y_after_BMIgain':
        image_dir = '/scratch/connectome/3DCNN/data/1.ABCD/1.1.sMRI_MNI_warped'
        phenotype_dir = '/scratch/connectome/3DCNN/data/1.ABCD/4.demo_qc/BMI_prediction/ABCD_phenotype_total_1years_revised_BMIgain.csv' 
        #image_dir = '/home/ubuntu/dhkdgmlghks/1.ABCD/1.1.sMRI_warped'
        #image_dir = '/home/ubuntu/dhkdgmlghks/1.ABCD/1.1.sMRI_warped'  
        #phenotype_dir = '/home/ubuntu/dhkdgmlghks/1.ABCD/4.demo_qc/BMI_prediction/ABCD_phenotype_total_1years_revised_BMIgain.csv'   
    elif study_sample == 'ABCD_1y_after_BMIloss':
        image_dir = '/scratch/connectome/3DCNN/data/1.ABCD/1.1.sMRI_MNI_warped'
        phenotype_dir = '/scratch/connectome/3DCNN/data/1.ABCD/4.demo_qc/BMI_prediction/ABCD_phenotype_total_1years_revised_BMIloss.csv' 
        #image_dir = '/home/ubuntu/dhkdgmlghks/1.ABCD/1.1.sMRI_warped'
        #image_dir = '/home/ubuntu/dhkdgmlghks/1.ABCD/2.sMRI_freesurfer'  
        #phenotype_dir = '/home/ubuntu/dhkdgmlghks/1.ABCD/4.demo_qc/BMI_prediction/ABCD_phenotype_total_1years_revised_BMIloss.csv' 
    elif study_sample == 'ABCD_1y_after_become_overweight':
        image_dir = '/scratch/connectome/3DCNN/data/1.ABCD/1.1.sMRI_MNI_warped'
        phenotype_dir = '/scratch/connectome/3DCNN/data/1.ABCD/4.demo_qc/BMI_prediction/ABCD_phenotype_total_1years_become_overweight_partitioned.csv'   
    elif study_sample == 'ABCD_1y_after_become_overweight_pretrain':
        image_dir = '/scratch/connectome/3DCNN/data/1.ABCD/1.1.sMRI_MNI_warped'
        phenotype_dir = '/scratch/connectome/3DCNN/data/1.ABCD/4.demo_qc/BMI_prediction/ABCD_phenotype_total_for_pretraining_1y_after_become_overweight.csv'   
    elif study_sample == 'ABCD_1y_after_become_normal':
        image_dir = '/scratch/connectome/3DCNN/data/1.ABCD/1.1.sMRI_MNI_warped'
        phenotype_dir = '/scratch/connectome/3DCNN/data/1.ABCD/4.demo_qc/BMI_prediction/ABCD_phenotype_total_1years_become_normal_partitioned.csv'   
    elif study_sample == 'ABCD_1y_after_become_normal_pretrain':
        image_dir = '/scratch/connectome/3DCNN/data/1.ABCD/1.1.sMRI_MNI_warped'
        phenotype_dir = '/scratch/connectome/3DCNN/data/1.ABCD/4.demo_qc/BMI_prediction/ABCD_phenotype_total_for_pretraining_1y_after_become_overweight.csv'  
    elif study_sample == 'ABCD_2y_after_become_overweight':
        image_dir = '/scratch/connectome/3DCNN/data/1.ABCD/1.1.sMRI_MNI_warped'
        phenotype_dir = '/scratch/connectome/3DCNN/data/1.ABCD/4.demo_qc/BMI_prediction/ABCD_phenotype_total_2years_become_overweight_partitioned.csv'   
    elif study_sample == 'ABCD_2y_after_become_overweight_pretrain':
        image_dir = '/scratch/connectome/3DCNN/data/1.ABCD/1.1.sMRI_MNI_warped'
        phenotype_dir = '/scratch/connectome/3DCNN/data/1.ABCD/4.demo_qc/BMI_prediction/ABCD_phenotype_total_for_pretraining_2y_after_become_overweight.csv'             
    elif study_sample == 'ABCD_2y_after_become_normal':
        image_dir = '/scratch/connectome/3DCNN/data/1.ABCD/1.1.sMRI_MNI_warped'
        phenotype_dir = '/scratch/connectome/3DCNN/data/1.ABCD/4.demo_qc/BMI_prediction/ABCD_phenotype_total_2years_become_normal_partitioned.csv'   
    elif study_sample == 'ABCD_2y_after_become_normal_pretrain':
        image_dir = '/scratch/connectome/3DCNN/data/1.ABCD/1.1.sMRI_MNI_warped'
        phenotype_dir = '/scratch/connectome/3DCNN/data/1.ABCD/4.demo_qc/BMI_prediction/ABCD_phenotype_total_for_pretraining_2y_after_become_normal.csv' 
    return image_dir, phenotype_dir 


def loading_images(image_dir, args, study_sample='UKB'):
    if study_sample.find('UKB') != -1:
        image_files = glob.glob(os.path.join(image_dir,'*.nii.gz'))
    elif study_sample.find('ABCD') != -1:
        image_files = glob.glob(os.path.join(image_dir,'*.nii.gz'))
    image_files = sorted(image_files)
   
    #image_files = image_files[:1000]
    print("Loading image file names as list is completed")
    return image_files


def loading_phenotype(phenotype_dir, args, study_sample='UKB', undersampling_dataset_target=None, partitioned_dataset_number=None):
    if study_sample.find('UKB') != -1:
        subject_id_col = 'eid'
    elif study_sample.find('ABCD') != -1:
        subject_id_col = 'subjectkey'

    targets = args.cat_target + args.num_target
    if undersampling_dataset_target:
        if not undersampling_dataset_target in targets: 
            col_list = targets + [subject_id_col] + [undersampling_dataset_target]
        else: 
            col_list = targets + [subject_id_col]
    elif partitioned_dataset_number is not None: 
        assert undersampling_dataset_target is None 
        col_list = targets + [subject_id_col] + ["partition%s" % partitioned_dataset_number]
    else: 
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
        #num_classes = int(subject_data[args.cat_target].nunique().values)
    #if args.num_target:
        #subject_data = preprocessing_num(subject_data, args)
        #num_classes = 1 
    
    return subject_data, targets


## combine categorical + numeric
def combining_image_target(subject_data, image_files, target_list, undersampling_dataset_target=None,partitioned_dataset_number=None ,study_sample='UKB'):
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

    if undersampling_dataset_target:
        if not undersampling_dataset_target in target_list:
            col_list = target_list + [undersampling_dataset_target] + ['image_files']
        else:
            col_list = target_list + ['image_files']
    elif partitioned_dataset_number is not None:
        assert undersampling_dataset_target is None 
        col_list = target_list + ["partition%s" % partitioned_dataset_number] + ['image_files']

    else: 
        col_list = target_list + ['image_files']
    
    for i in tqdm(range(len(subject_data))):
        imageFile_label = {}
        for j, col in enumerate(col_list):
            imageFile_label[col] = subject_data[col][i]
        imageFiles_labels.append(imageFile_label)
        
    return imageFiles_labels



# defining train,val, test set splitting function
def partition_dataset(imageFiles_labels, targets, args):
    #random.shuffle(imageFiles_labels)

    images = []
    labels = []

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
                               RandRotate90(prob=0.3),
                               RandAxisFlip(prob=0.3),
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

    print("Training Samples: {}. Valdiation Samples: {}. Test Samples: {}". format(len(labels_train), len(labels_val), len(labels_test)))

    case_control_count(labels_train, 'train', args)
    case_control_count(labels_val, 'validation', args)
    case_control_count(labels_test, 'test', args)

    return partition
## ====================================== ##


def matching_partition_dataset(imageFiles_labels, reference_dataset, targets, args):
    #random.shuffle(imageFiles_labels)

    images_train = []
    labels_train = []
    images_val = []
    labels_val = []
    images_test = []
    labels_test = []
    

    for imageFile_label in imageFiles_labels:
        image = imageFile_label['image_files']
        label = {}

        for label_name in targets[:len(targets)]:
            label[label_name]=imageFile_label[label_name]

        if image in reference_dataset['train'].image_files: 
            images_train.append(image)
            labels_train.append(label)
        elif image in reference_dataset['val'].image_files: 
            images_val.append(image)
            labels_val.append(label)
        elif image in reference_dataset['test'].image_files: 
            images_test.append(image)
            labels_test.append(label)
    


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
    """
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
    """
    train_set = ImageDataset(image_files=images_train,labels=labels_train,transform=train_transform)
    val_set = ImageDataset(image_files=images_val,labels=labels_val,transform=val_transform)
    test_set = ImageDataset(image_files=images_test,labels=labels_test,transform=test_transform)

    partition = {}
    partition['train'] = train_set
    partition['val'] = val_set
    partition['test'] = test_set

    print("The number of Training samples: {}. The number of Validation samples: {}. The number of Test samples: {}".format(len(labels_train), len(labels_val), len(labels_test)))

    return partition
## ====================================== ##


def undersampling_ALLset(imageFiles_labels, targets, undersampling_dataset_target, args):
    #random.shuffle(imageFiles_labels)

    total_num_control = 0 
    total_num_case = 0 
    for imageFile_label in imageFiles_labels:
        if imageFile_label[undersampling_dataset_target] == 0: 
            total_num_control += 1 
        elif imageFile_label[undersampling_dataset_target] == 1: 
            total_num_case += 1 

    num_case_train = int(total_num_case*args.train_size)
    num_control_train = num_case_train 
    num_case_test = int(total_num_case*args.test_size)
    num_control_test = num_case_test
    num_case_val = total_num_case - num_case_train - num_case_test
    num_control_val = num_case_val
    

    # dataset list for train, validation, and test 
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
        image = imageFile_label['image_files']
        label = {}
        for label_name in targets[:len(targets)]:
            label[label_name]=imageFile_label[label_name]

        if imageFile_label[undersampling_dataset_target] == 0: 
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

        elif imageFile_label[undersampling_dataset_target] == 1: 
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

    images_test = images_control_test + images_case_test 
    labels_test = labels_control_test + labels_case_test 

    images_train = images_control_train + images_case_train
    labels_train = labels_control_train + labels_case_train

    images_val = images_control_val + images_case_val
    labels_val = labels_control_val + labels_case_val

    train_set = ImageDataset(image_files=images_train,labels=labels_train,transform=train_transform)
    val_set = ImageDataset(image_files=images_val,labels=labels_val,transform=val_transform)
    test_set = ImageDataset(image_files=images_test,labels=labels_test,transform=test_transform)

    partition = {}
    partition['train'] = train_set
    partition['val'] = val_set
    partition['test'] = test_set

    print("Training Samples: {}. Valdiation Samples: {}. Test Samples: {}". format(len(labels_train), len(labels_val), len(labels_test)))

    case_control_count(labels_train, 'train', args)
    case_control_count(labels_val, 'validation', args)
    case_control_count(labels_test, 'test', args)

    return partition


def matching_undersampling_ALLset(imageFiles_labels, reference_dataset,  undersampling_dataset_target, args):
    #random.shuffle(imageFiles_labels)
    targets = args.cat_target + args.num_target

    total_num_control = 0 
    total_num_case = 0 
    for imageFile_label in imageFiles_labels:
        if imageFile_label[undersampling_dataset_target] == 0: 
            total_num_control += 1 
        elif imageFile_label[undersampling_dataset_target] == 1: 
            total_num_case += 1 

    num_case_train = int(total_num_case*args.train_size)
    num_control_train = num_case_train 
    num_case_test = int(total_num_case*args.test_size)
    num_control_test = num_case_test
    num_case_val = total_num_case - num_case_train - num_case_test
    num_control_val = num_case_val

    # dataset list for train, validation, and test 
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
        image = imageFile_label['image_files']
        label = {}
        for label_name in targets[:len(targets)]:
            label[label_name]=imageFile_label[label_name]

        if imageFile_label[undersampling_dataset_target] == 0: 
            if count_control_test < num_control_test:
                if image in reference_dataset['test'].image_files:  
                    images_control_test.append(image)
                    labels_control_test.append(label)
                count_control_test += 1 
            else:
                if count_control_train < num_control_train:
                    if image in reference_dataset['train'].image_files: 
                        images_control_train.append(image)
                        labels_control_train.append(label)     
                    count_control_train += 1               
                else:
                    if count_control_val < num_control_val:
                        if image in reference_dataset['val'].image_files: 
                            images_control_val.append(image)
                            labels_control_val.append(label)
                        count_control_val += 1 

        elif imageFile_label[undersampling_dataset_target] == 1: 
            if count_case_test < num_case_test:
                if image in reference_dataset['test'].image_files:  
                    images_case_test.append(image)
                    labels_case_test.append(label)   
                count_case_test += 1 
            else: 
                if count_case_train < num_case_train:
                    if image in reference_dataset['train'].image_files: 
                        images_case_train.append(image)
                        labels_case_train.append(label)
                    count_case_train += 1 
                else:
                    if count_case_val < num_case_val: 
                        if image in reference_dataset['val'].image_files: 
                            images_case_val.append(image)
                            labels_case_val.append(label)  
                        count_case_val += 1 
    


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
    """
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
    """

    images_test = images_control_test + images_case_test 
    labels_test = labels_control_test + labels_case_test 

    images_train = images_control_train + images_case_train
    labels_train = labels_control_train + labels_case_train

    images_val = images_control_val + images_case_val
    labels_val = labels_control_val + labels_case_val


    train_set = ImageDataset(image_files=images_train,labels=labels_train,transform=train_transform)
    val_set = ImageDataset(image_files=images_val,labels=labels_val,transform=val_transform)
    test_set = ImageDataset(image_files=images_test,labels=labels_test,transform=test_transform)

    partition = {}
    partition['train'] = train_set
    partition['val'] = val_set
    partition['test'] = test_set

    print("The number of Training samples: {}. The number of Validation samples: {}. The number of Test samples: {}".format(len(labels_train), len(labels_val), len(labels_test)))

    return partition
## ====================================== ##


# defining train,val, test set splitting function
def partition_dataset_predefined(imageFiles_labels, targets, partitioned_dataset_number, args):
    #random.shuffle(imageFiles_labels)

    images_train = []
    labels_train = []
    images_val = []
    labels_val = []
    images_test = []
    labels_test = []

    for imageFile_label in imageFiles_labels:
        image = imageFile_label['image_files']
        label = {}
        for label_name in targets[:len(targets)]:
            label[label_name]=imageFile_label[label_name]

        if imageFile_label['partition%s' % partitioned_dataset_number] =='train': 
            images_train.append(image)
            labels_train.append(label)
        elif imageFile_label['partition%s' % partitioned_dataset_number] =='val': 
            images_val.append(image)
            labels_val.append(label)    
        elif imageFile_label['partition%s' % partitioned_dataset_number] =='test': 
            images_test.append(image)
            labels_test.append(label)

    resize = tuple(args.resize)
    train_transform = Compose([ScaleIntensity(),
                               AddChannel(),
                               Resize(resize),
                               RandRotate90(prob=0.3),
                               RandAxisFlip(prob=0.3),
                              ToTensor()])

    val_transform = Compose([ScaleIntensity(),
                               AddChannel(),
                               Resize(resize),
                              ToTensor()])

    test_transform = Compose([ScaleIntensity(),
                               AddChannel(),
                               Resize(resize),
                              ToTensor()])


    train_set = ImageDataset(image_files=images_train,labels=labels_train,transform=train_transform)
    val_set = ImageDataset(image_files=images_val,labels=labels_val,transform=val_transform)
    test_set = ImageDataset(image_files=images_test,labels=labels_test,transform=test_transform)

    partition = {}
    partition['train'] = train_set
    partition['val'] = val_set
    partition['test'] = test_set

    print("Training Samples: {}. Valdiation Samples: {}. Test Samples: {}". format(len(labels_train), len(labels_val), len(labels_test)))

    case_control_count(labels_train, 'train', args)
    case_control_count(labels_val, 'validation', args)
    case_control_count(labels_test, 'test', args)

    return partition