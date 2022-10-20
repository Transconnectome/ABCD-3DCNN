import os
from os import listdir
from os.path import isfile, join
import glob
from tkinter import NONE

from matplotlib import transforms


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
        #image_dir = '/home/ubuntu/dhkdgmlghks/2.UKB/1.sMRI_fs_cropped'
        #phenotype_dir = '/home/ubuntu/dhkdgmlghks/2.UKB/2.demo_qc/UKB_phenotype.csv'
    elif study_sample == 'ABCD':
        image_dir = '/scratch/connectome/3DCNN/data/1.ABCD/2.sMRI_freesurfer'
        phenotype_dir = '/scratch/connectome/3DCNN/data/1.ABCD/4.demo_qc/ABCD_phenotype_total.csv'  
        #image_dir = '/home/ubuntu/dhkdgmlghks/1.ABCD/2.sMRI_freesurfer'
        #phenotype_dir = '/home/ubuntu/dhkdgmlghks/1.ABCD/4.demo_qc/ABCD_phenotype_total.csv'
    elif study_sample == 'ABCD_ADHD':
        image_dir = '/scratch/connectome/3DCNN/data/1.ABCD/2.sMRI_freesurfer'
        phenotype_dir = '/scratch/connectome/3DCNN/data/1.ABCD/4.demo_qc/ABCD_ADHD.csv'          
    return image_dir, phenotype_dir 

def check_synthetic_sample(include_synthetic=True):
    if include_synthetic: 
        image_dir = '/home/ubuntu/3.LDM/1.sMRI'
    else: 
        image_dir = None 
    return image_dir 


def loading_images(image_dir, args, study_sample='UKB', include_synthetic=False):
    if study_sample == 'UKB':
        image_files = glob.glob(os.path.join(image_dir,'*.nii.gz'))
    elif study_sample == 'ABCD':
        image_files = glob.glob(os.path.join(image_dir,'*.npy'))
    image_files = sorted(image_files)
   
    #image_files = image_files[:1000]
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



def partition_dataset_pretrain(imageFiles_list: list=None, synthetic_imageFiles=None, args=None):

    train_transform = Compose([AddChannel(),
                               Resize(tuple(args.img_size)),
                               RandAxisFlip(prob=0.5),
                               ScaleIntensity(),
                               ToTensor()])

    val_transform = Compose([AddChannel(),
                             Resize(tuple(args.img_size)),
                             ScaleIntensity(),
                             ToTensor()])

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
        image_train = image_train + synthetic_imageFiles
        print("{} Synthetic imaages are included in the training data".format(len(synthetic_imageFiles)))

    print("Training Sample: {}".format(len(images_train)))

    train_set = ImageDataset(image_files=images_train,transform=MaskGenerator(train_transform, input_size=args.img_size[0], mask_ratio=args.mask_ratio, mask_patch_size=args.mask_patch_size)) 
    val_set = ImageDataset(image_files=images_val,transform=MaskGenerator(val_transform, input_size=args.img_size[0], mask_ratio=args.mask_ratio, mask_patch_size=args.mask_patch_size))
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
                               ScaleIntensity(),
                               ToTensor()])

    val_transform = Compose([AddChannel(),
                             Resize(tuple(args.img_size)),
                             ScaleIntensity(),
                             ToTensor()])


    # number of total / train,val, test
    num_total = len(images)
    if args.train_size + args.val_size + args.test_size != 1: 
        print('PLZ CHECK WHETHER YOU WANT TO USE ONLY THE PORTION OF DATA')
    num_train = int(num_total*args.train_size)
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


class MaskGenerator:
    def __init__(self, transform, input_size=192, mask_patch_size=16, model_patch_size=4, mask_ratio=0.6):
        self.transform = transform
        self.input_size = input_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio

        if isinstance(mask_patch_size, tuple):
            assert mask_patch_size[0] == mask_patch_size[1] == mask_patch_size[2]
            self.mask_patch_size = mask_patch_size[0]
        elif isinstance(mask_patch_size, int): 
            self.mask_patch_size = mask_patch_size

        assert self.input_size % self.mask_patch_size == 0
        assert self.mask_patch_size % self.model_patch_size == 0
        
        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size
        
        self.token_count = self.rand_size ** 3
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))
    
    def update_config(self, model_patch_size):
        if isinstance(model_patch_size, tuple):
            assert model_patch_size[0] == model_patch_size[1] == model_patch_size[2]
            model_patch_size = model_patch_size[0]
        self.model_patch_size = model_patch_size
        self.scale = self.mask_patch_size // model_patch_size
        
    def __call__(self, img):
        mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1
        
        mask = mask.reshape((self.rand_size, self.rand_size, self.rand_size))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1).repeat(self.scale, axis=2)
        
        return (self.transform(img), mask)