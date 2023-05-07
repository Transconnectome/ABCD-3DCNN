import os
import re
import glob
import random
from collections import defaultdict

import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from sklearn.preprocessing import OneHotEncoder 
from skmultilearn.model_selection import IterativeStratification

import torch
from monai.transforms import (AddChannel, Compose, CenterSpatialCrop, Flip, RandAffine,
                              RandFlip, RandRotate90, Resize, ScaleIntensity, ToTensor)
from monai.data import ImageDataset, NibabelReader

from dataloaders.custom_transform import MaskTissue
from dataloaders.custom_dataset import MultiModalImageDataset
from dataloaders.preprocessing import preprocessing_cat, preprocessing_num

root_dir="/scratch/x2519a05/"
if 'x2519a05' in os.getcwd():
    root_dir = "/scratch/x2519a05/"
else:
    root_dir = '/scratch/connectome/3DCNN/'

ABCD_data_dir = {
    'fmriprep': f'{root_dir}/data/1.ABCD/1.sMRI_fmriprep/preprocessed_masked/',
    'freesurfer': f'{root_dir}/data/1.ABCD/2.sMRI_freesurfer/',
    'freesurfer_256': f'{root_dir}/data/1.ABCD/2.2.sMRI_freesurfer_256/',
    'T1_MNI_resize128': f'{root_dir}/data/1.ABCD/T1_MNI_resize128/',
    'FA_MNI_resize128': f'{root_dir}/data/1.ABCD/FA_MNI_resize128/',
    'FA_wm_MNI_resize128': f'{root_dir}/data/1.ABCD/FA_wm_MNI_resize128/',
    'FA_unwarpped_nii': f'{root_dir}/data/1.ABCD/3.1.FA_unwarpped_nii/',
    'FA_warpped_nii': f'{root_dir}/data/1.ABCD/3.2.FA_warpped_nii/',
    'FA_crop_resize128': f'{root_dir}/data/1.ABCD/FA_crop_resize128/',
    'MD_unwarpped_nii': f'{root_dir}/data/1.ABCD/3.3.MD_unwarpped_nii/',
    'MD_warpped_nii': f'{root_dir}/data/1.ABCD/3.4.MD_warpped_nii/',
    'RD_unwarpped_nii': f'{root_dir}/data/1.ABCD/3.5.RD_unwarpped_nii/',
    'RD_warpped_nii': f'{root_dir}/data/1.ABCD/3.6.RD_warpped_nii/',
    '5tt_warped_nii': f'{root_dir}/data/1.ABCD/3.7.5tt_warped_nii/',
    'T1_fastsurfer_resize128': f'{root_dir}/data/1.ABCD/T1_fastsurfer_resize128/', # added T1 fastsurfer 128
    'FA_hippo': f'{root_dir}/data/1.ABCD/FA_hippo/',
    'MD_hippo': f'{root_dir}/data/1.ABCD/MD_hippo/',    
}
if os.uname()[1] in ['node1', 'node3']:
    ABCD_data_dir['FA_crop_resize128'] = '/home/connectome/jubin/3DCNN/ABCD/FA_crop_resize128/'
    ABCD_data_dir['T1_fastsurfer_resize128'] = '/scratch/connectome/jubin/data/1.ABCD/T1_fastsurfer_resize128/'

ABCD_phenotype_dir = {
    'total': f'{root_dir}/data/1.ABCD/4.demo_qc/ABCD_phenotype_total_balanced_multitarget.csv',
    'ADHD_case': f'{root_dir}/data/1.ABCD/4.demo_qc/ABCD_ADHD.csv',
    'ADHD_KSADS': f"{root_dir}/data/1.ABCD/4.demo_qc/ABCD_ADHD_TotalSymp_and_KSADS.csv",
    'ADHD_KSADS_CBCL': f"{root_dir}/data/1.ABCD/4.demo_qc/ABCD_ADHD_TotalSymp_and_KSADS_and_CBCL.csv",
    'BMI': f'{root_dir}/data/1.ABCD/4.demo_qc/BMI_prediction/ABCD_phenotype_total.csv', 
    'suicide_case': f'{root_dir}/data/1.ABCD/4.demo_qc/ABCD_suicide_case.csv',
    'suicide_control': f'{root_dir}/data/1.ABCD/4.demo_qc/ABCD_suicide_control.csv'
}

UKB_data_dir = f'{root_dir}/data/2.UKB/1.sMRI_fs_cropped/'
UKB_phenotype_dir = {'total': f'{root_dir}/data/2.UKB/2.demo_qc/UKB_phenotype.csv'}

CHA_data_dir = {'freesurfer_256': f'{root_dir}/data/CHA_bigdata/sMRI_brain/',
                'T1_resize128': f'{root_dir}/data/CHA_bigdata/sMRI_resize128/'}
CHA_phenotype_dir = {'total': f'{root_dir}/data/CHA_bigdata/metadata/sMRI_BSID_imputed.csv',
                     'ASDvsGDD': f'{root_dir}/data/CHA_bigdata/metadata/CHA_psm_split.csv',
                     'psm2': f'{root_dir}/data/CHA_bigdata/metadata/CHA_psm2.csv',}

data_dict = {'ABCD': ABCD_data_dir,
             'UKB': UKB_data_dir,
             'CHA': CHA_data_dir}
phenotype_dict = {'ABCD': ABCD_phenotype_dir,
                  'UKB': UKB_phenotype_dir,
                  'CHA': CHA_phenotype_dir}


def case_control_count(labels, dataset_type, args):
    if args.cat_target:
        df_labels = pd.DataFrame.from_records(labels)
        for cat_target in args.cat_target:
            curr_cnt = df_labels[cat_target].value_counts()
            print(f'In {dataset_type},\t"{cat_target}" contains {curr_cnt[1]} CASE and {curr_cnt[0]} CONTROL')

            
def loading_images(image_dir, args):
    image_files = pd.DataFrame()
    data_types = args.data_type if (args.tissue == None) else args.data_type + ['5tt_warped_nii']
    for brain_modality in data_types:
        curr_dir = image_dir[brain_modality]
        curr_files = pd.DataFrame({brain_modality:glob.glob(curr_dir+'*[yz]')}) # to get .npy(sMRI) & .nii.gz(dMRI) files
        curr_files[subjectkey] = curr_files[brain_modality].map(lambda x: x.split("/")[-1].split('.')[0])
        if args.dataset == 'UKB':
            curr_files[subjectkey] = curr_files[subjectkey].map(lambda x: int(x.split('sub-')[-1]))
        curr_files.sort_values(by=subjectkey, inplace=True)
        
        if len(image_files) == 0:
            image_files = curr_files
        else:
            image_files = pd.merge(image_files, curr_files, how='inner', on=subjectkey)
            
    if args.debug:
        image_files = image_files[:160]
        
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


def split_cv(subject_data, args):    
    assert len(args.cv.split('_')) == 3, "args.cv should NumFold_ValFold_TestFold form"
    subject_data['split']=None
    N, val, test = [ int(x) for x in args.cv.split('_') ]
    subject_data.loc[~subject_data['fold'].isin([test,val]),'split'] = 'train'
    subject_data.loc[subject_data['fold']==val,'split'] = 'val'
    subject_data.loc[subject_data['fold']==test,'split'] = 'test'
    
    return subject_data


def loading_phenotype(phenotype_dir, target_list, args):
    col_list = target_list + [subjectkey]
    if 'multitarget' in args.balanced_split:
        col_list.append('split')

    ## get subject ID and target variables
    subject_data = pd.read_csv(phenotype_dir)
    if args.cv != None:
        subject_data = split_cv(subject_data, args)
    raw_csv = subject_data.copy()
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
    
    return subject_data, raw_csv


def slice_index(array, n_chunks ):
    partitioned_list = np.array_split(np.sort(array), n_chunks)
    return [i[-1] for i in partitioned_list]
    

def multilabel_matrix_maker(df, binary_cols=None, multiclass_cols=None, continuous_cols=None, n_chunks=3) :
    """
    returns matrix that will be used for multilabel, taking into account columns that are either multiclass or continuous
    * df : the dataframe to be split
    * binary_cols : LIST of cols (str)just cols that will be used (binarized)
    * multiclass_cols : LIST of the cols (str) that are multi class
    * continuous_cols : LIST of the cols (str) that will be split (continouous)
    * n_chunks : if using continouous cols are used, how many split?
    
    outputs matrix that has binarized binarized for all columns (only needs to be used during iskf to get the indices)
    """
    df = df.copy() #copy becasue we don't want to modify the original df
    if binary_cols == multiclass_cols == continuous_cols == None : #i.e. if all are None
        raise ValueError("at least one of the cols have to be put.. currently all cols are selected as None")
    if type(binary_cols)!= list or type(multiclass_cols)!= list or type(continuous_cols)!= list: 
        raise ValueError("the cols have to be lists!")
    #checking if NaN exist => raise error (sanity check)\
    if df[binary_cols+multiclass_cols+continuous_cols].isnull().values.any():
        raise ValueError("Your provided df had some NaN in columns that you are wanting to do iskf on")    
 
    #now adding binarized columns for each column types and aggregating them into total_cols
    total_cols = []
    if binary_cols : 
        for col in binary_cols :
            df[col] = pd.factorize(df[col], sort = True)[0] 
            total_cols.append(df[col].values) #or single []?  ([[]] : df 로 만드는 것, [] : series로 만듬) 
            
    if multiclass_cols :
        for col in multiclass_cols : 
            df_col = df[[col]] #[[]] not [] because of dims 
            ohe = OneHotEncoder()
            ohe.fit(df_col)
            binarized_col = ohe.transform(df_col).todense() 
            total_cols.append(binarized_col)
            
    if continuous_cols: 
        for col in continuous_cols:
            df[col] = df[col].astype('float') #change to float when doing 
            array = df[col].values
            boundaries = slice_index(array, n_chunks)  
            i_below = -np.infty
            for i in boundaries:
                extracted_df = (df[col]>i_below) & (df[col]<=i) 
                i_below = i #update i_below
                total_cols.append(extracted_df.values.astype(float))     
    
    #adding all together,
    final_arr = np.column_stack(total_cols)
    
    return final_arr


def get_info_fold(df, train_idx, valid_idx, test_idx, target_col):
    """
    * kf_split : the `kf.split(XX)`된것
    * df : the dataframe with the metadata I will use
    * target_col : the columns in the df that I'll take statistics of 
    """
    train_dict = defaultdict(list)
    valid_dict = defaultdict(list)
    test_dict = defaultdict(list)

    label_train = df.iloc[train_idx]
    label_valid = df.iloc[valid_idx]
    label_test = df.iloc[test_idx]
    
    for col in target_col:
        if df[col].nunique()<=10: # case: categorical variable
            keys=list(map(lambda x: f'{col}[{x}]', label_train[col].unique()))
            train_counts=label_train[col].value_counts()
            valid_counts=label_valid[col].value_counts()
            test_counts=label_test[col].value_counts()
            for i, key in enumerate(keys):
                train_dict[key].append(train_counts[train_counts.index[i]])
                valid_dict[key].append(valid_counts[valid_counts.index[i]])
                test_dict[key].append(test_counts[test_counts.index[i]])
        else: # case: continuous variable
            train_dict[f'{col}-mean/std'].append(f'{label_train[col].mean():.2f} / {label_train[col].std():.2f}')
            valid_dict[f'{col}-mean/std'].append(f'{label_valid[col].mean():.2f} / {label_valid[col].std():.2f}')
            test_dict[f'{col}-mean/std'].append(f'{label_test[col].mean():.2f} / {label_test[col].std():.2f}')
                    
    print("=== Fold-wise categorical values information of training set ===")
    print(pd.DataFrame(train_dict))
    print("=== Fold-wise categorical values information of validation set ===")
    print(pd.DataFrame(valid_dict))
    print("=== Fold-wise categorical values information of test set ===")
    print(pd.DataFrame(test_dict))    


def iterative_stratification(imageFiles_labels, raw_merged, num_total, args):
    if args.val_size % args.test_size != 0:
        print("Validation size & Test size aren't matched to make folds. change val size to test size")
        args.val_size = args.test_size

    binary_target = list(set(['sex']+args.cat_target))
    continuous_target = list(set(['age']+args.num_target))
    floatized_arr = multilabel_matrix_maker(raw_merged, n_chunks=10,
                                            binary_cols=binary_target,
                                            multiclass_cols=[],
                                            continuous_cols=continuous_target)
    skf_target = [floatized_arr, floatized_arr]        
    args.num_folds = int(1/args.test_size)
    kf = IterativeStratification(n_splits=args.num_folds, order=10, random_state = np.random.seed(0))
    folds = [*kf.split(*skf_target)]
    indices = [ x[1] for x in folds ] 

    # split dataset
    fold_test, fold_val = int(args.num_folds*args.test_size), int(args.num_folds*args.val_size)
    split_indices = indices[:fold_test], indices[fold_test:fold_test+fold_val], indices[fold_test+fold_val:]
    test_idx, val_idx, train_idx = [ np.concatenate(idx) for idx in split_indices ]
    get_info_fold(raw_merged, train_idx, val_idx, test_idx, binary_target+continuous_target)

    num_train, num_val =len(train_idx), len(val_idx)
    new_idx = np.concatenate([train_idx, val_idx, test_idx])
    imageFiles_labels = imageFiles_labels.iloc[new_idx]
        
    return num_train, num_val, imageFiles_labels


def make_balanced_testset(il, raw_merged, num_total, args):
    if 'iter_strat' in args.balanced_split:
        num_train, num_val, imageFiles_labels = iterative_stratification(il, raw_merged, num_total, args)
        
    elif 'multitarget' in args.balanced_split:
        num_train, num_val = il['split'].value_counts()[['train','val']]
        splits = [il[il['split']=='train'], il[il['split']=='val'], il[il['split']=='test']]
        imageFiles_labels = pd.concat(splits)
        
    elif 'BMI' in args.balanced_split:        
        predefined_set = pd.read_csv(f'{root_dir}/data/1.ABCD/4.demo_qc/'+args.balanced_split)
        num_test = round(len(predefined_set)*args.test_size)
        num_train = round(len(predefined_set)*(1-args.val_size-args.test_size))
        num_val = num_total - num_train - num_test
        test_idx = il[il.subjectkey.isin(predefined_set['subjectkey'][-num_test:])].index
        train_idx = il[il.subjectkey.isin(predefined_set['subjectkey'][:num_train])].index
        val_idx = list(set(il.index) - set(train_idx) - set(test_idx))
        imageFiles_labels = pd.concat([il.iloc[train_idx],il.iloc[val_idx],il.iloc[test_idx]])
        
    elif len(args.cat_target) > 0:
        num_test = round(num_total*args.test_size)
        num_val = round(num_total*args.val_size)
        num_train = num_total - num_val - num_test
        n_case = num_test//2
        n_control = num_test - n_case
        t_case, rest_case = np.split(il[il[args.cat_target[0]]==0], (n_case,))
        t_control, rest_control = np.split(il[il[args.cat_target[0]]==1],(n_control,))
        test = pd.concat((t_case, t_control))
        rest = pd.concat((rest_case, rest_control))
        
        test = test.sort_values(by=subjectkey)
        rest = rest.sort_values(by=subjectkey)

        imageFiles_labels = pd.concat((rest,test)).reset_index(drop=True)
#     elif len(args.num_target) > 0:    
                        
    return num_train, num_val, imageFiles_labels


# defining train,val, test set splitting function
def partition_dataset(imageFiles_labels, raw_merged, target_list, args):
    ## Random shuffle according to args.seed -> Disable.
#     imageFiles_labels = imageFiles_labels.sample(frac=1).reset_index(drop=True)
    if args.N != None:
        imageFiles_labels = imageFiles_labels.sample(n=args.N).reset_index(drop=True)
        raw_merged = pd.merge(raw_merged, imageFiles_labels, on=subjectkey, how='right',suffixes=('','x'))
    ## Dataset split    
    num_total = len(imageFiles_labels) 
    num_train = int(num_total*(1 - args.val_size - args.test_size))
    num_val = int(num_total*args.val_size) #if args.cv == None else int((num_total-num_test)/5)
    num_test = int(num_total*args.test_size)
        
    if args.balanced_split:
        num_train, num_val, imageFiles_labels = make_balanced_testset(imageFiles_labels, raw_merged, num_total, args)
    images = imageFiles_labels[args.data_type]
    labels = imageFiles_labels[target_list].to_dict('records')
    
    ## split dataset by 5-fold cv or given split size
    # if args.cv == None:
    images_train, images_val, images_test = np.split(images, [num_train, num_train+num_val]) # revising
    labels_train, labels_val, labels_test = np.split(labels, [num_train, num_train+num_val])
    # else:
    #     split_points = [num_val, 2*num_val, 3*num_val, 4*num_val, num_total-num_test]
    #     images_total, labels_total = np.split(images, split_points), np.split(labels, split_points)
    #     images_test, labels_test = images_total.pop(), labels_total.pop()
    #     images_val, labels_val = images_total.pop(args.cv-1), labels_total.pop(args.cv-1)
    #     images_train, labels_train = np.concatenate(images_total), np.concatenate(labels_total)
    #     num_train, num_val = images_train.shape[0], images_val.shape[0]

    if target_list == []:
        label, labels_train, labels_val, labels_test = None, None, None, None
        
    print(f"Total subjects={len(images)}, train={len(images_train)}, val={len(images_val)}, test={len(images_test)}")

    ## Define transform function
    resize = tuple(args.resize)
    default_transforms = [ScaleIntensity(), AddChannel(), Resize(resize), ToTensor()] 
    
    if 'resize128' in args.data_type[0]:
        default_transforms.pop(2)
        args.resize = (128,128,128)
    if 'crop' in args.transform:
        default_transforms.insert(2, CenterSpatialCrop(192))
    if args.tissue:
        dMRI_transform = [MaskTissue(imageFiles_labels['5tt_warped_nii'], args.tissue)]
        dMRI_transform += default_transforms
        
    aug_transforms = []
    if 'affine' in args.augmentation:
        aug_transforms.append(RandAffine(prob=0.2, padding_mode='zeros',
                                         translate_range=(int(resize[0]*0.1),)*3,
                                         rotate_range=(np.pi/36,)*3,
                                         spatial_size=args.resize, cache_grid=True))
    elif 'flip' in args.augmentation:
        aug_transforms.append(RandFlip(prob=0.2, spatial_axis=0))
    
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

    partition = {'train': train_set,
                 'val': val_set,
                 'test': test_set}

    case_control_count(labels_train, 'train', args)
    case_control_count(labels_val, 'validation', args)
    case_control_count(labels_test, 'test', args)

    return partition


def setting_dataset(args):
    subjectkey = 'eid' if args.dataset == 'UKB' else 'subjectkey'
    image_dir = data_dict[args.dataset]
    phenotype_dir = phenotype_dict[args.dataset]
    phenotype_csv = phenotype_dir[args.phenotype]

    return subjectkey, image_dir, phenotype_csv


def make_dataset(args):
    global subjectkey
    subjectkey, image_dir, phenotype_dir = setting_dataset(args)
    target_list = args.cat_target + args.num_target
    
    image_files = loading_images(image_dir, args)
    subject_data, raw_csv = loading_phenotype(phenotype_dir, target_list, args)

    # combining image files & labels
    raw_merged = pd.merge(raw_csv, image_files, how='inner', on=subjectkey)
    if 'multitarget' in args.balanced_split:
        imageFiles_labels = (pd.merge(subject_data, image_files, how='left', on=subjectkey)).dropna()
    else:
        imageFiles_labels = pd.merge(subject_data, image_files, how='inner', on=subjectkey)

    # partitioning dataset and preprocessing (change the range of categorical variables and standardize numerical variables)
    partition = partition_dataset(imageFiles_labels, raw_merged, target_list, args)
    print("*** Making a dataset is completed *** \n")
    
    return partition, subject_data


def make_dataloaders(partition, args):
    def seed_worker(worker_id):
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
    g = torch.Generator()
    g.manual_seed(args.seed)
    
    trainloader = torch.utils.data.DataLoader(partition['train'],
                                              batch_size=args.train_batch_size,
                                              shuffle=True,
                                              persistent_workers=True,
                                              num_workers=args.num_workers,
                                              worker_init_fn=seed_worker,
                                              generator=g)
    
    valloader = torch.utils.data.DataLoader(partition['val'],
                                            batch_size=args.val_batch_size,
                                            shuffle=True,
                                            persistent_workers=True,
                                            num_workers=args.num_workers,
                                            worker_init_fn=seed_worker,
                                            generator=g)   
    
    return trainloader, valloader
