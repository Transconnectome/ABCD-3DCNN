import os
import random
import json
import argparse 
from collections import defaultdict, OrderedDict

import numpy as np
import pandas as pd
import torch

import models.simple3d as simple3d 
import models.vgg3d as vgg3d 
import models.resnet3d as resnet3d 
import models.densenet3d as densenet3d 
import models.sfcn as sfcn

def argument_setting():
    parser = argparse.ArgumentParser()

    # Options for model setting
    parser.add_argument("--model", type=str, required=True, help='Select model. e.g. densenet3D121, sfcn.',
                        choices=['simple3D', 'sfcn', 'vgg3D11', 'vgg3D13', 'vgg3D16', 'vgg3D19',
                                 'resnet3D50', 'resnet3D101', 'resnet3D152',
                                 'densenet3D121', 'densenet3D169', 'densenet201', 'densenet264',
                                 'densenet3D121attn', 'densenet3D121MM'])
    parser.add_argument("--in_channels", default=1, type=int, help='')

    # Options for dataset and data type, split ratio, CV, resize, augmentation
    parser.add_argument("--dataset", type=str, choices=['UKB','ABCD','CHA'], required=True, help='Select dataset') 
    parser.add_argument("--data_type", nargs='+', type=str, help='Select data type(sMRI, dMRI)',
                        choices=['fmriprep', 'freesurfer', 'freesurfer_256', 'freesurfer_crop_resize128',
                                 'T1_MNI_resize128', 'T1_resize128', 'T1_fastsurfer_resize128', # added T1 fastsurfer 128
                                 'FA_crop_resize128','FA_MNI_resize128', 'FA_wm_MNI_resize128',
                                 'FA_unwarpped_nii', 'FA_warpped_nii', 'FA_hippo',
                                 'MD_unwarpped_nii', 'MD_warpped_nii', 'MD_hippo',
                                 'RD_unwarpped_nii', 'RD_warpped_nii',
                                 'T1_MNI_resize_areamode', 'FA_MNI_resize_areamode'])
    parser.add_argument("--phenotype", default='total', type=str, help='')
    parser.add_argument("--balanced_split", default='', type=str, help='')
    parser.add_argument("--N", default=None, type=int, help='')
    parser.add_argument("--tissue", default=None, type=str, help='Select tissue mask(Cortical grey matter, \
                        Sub-cortical grey matter, White matter, CSF, Pathological tissue)',
                        choices=['cgm', 'scgm', 'wm', 'csf', 'pt'])
    parser.add_argument("--metric", default='', type=str, help='')
    parser.add_argument("--val_size", default=0.1, type=float, help='')
    parser.add_argument("--test_size", default=0.1, type=float, help='')
    parser.add_argument("--cv", default=None, type=int, choices=[1,2,3,4,5], help="option for 5-fold CV. 1~5.")
    parser.add_argument("--resize", nargs="*", default=(96, 96, 96), type=int, help='')
    parser.add_argument("--transform", nargs="*", default=[], type=str, choices=['crop'],
                        help="option for additional transform - [crop] are available")
    parser.add_argument("--augmentation", nargs="*", default=[], type=str, choices=['shift','flip'],
                        help="Data augmentation - [shift, flip] are available")

    # Hyperparameters for model training
    parser.add_argument("--lr", default=0.01, type=float, help='')
    parser.add_argument("--lr_adjust", default=0.01, type=float, help='')
    parser.add_argument("--epoch", type=int, required=True, help='')
    parser.add_argument("--epoch_FC", type=int, default=0, help='Option for training only FC layer')
    parser.add_argument("--optim", default='Adam', type=str, choices=['Adam','SGD','RAdam','AdamW'], help='')
    parser.add_argument("--weight_decay", default=0.01, type=float, help='')
    parser.add_argument("--scheduler", default='', type=str, help='') 
    parser.add_argument("--early_stopping", default=None, type=int, help='')
    parser.add_argument("--num_workers", default=3, type=int, help='')
    parser.add_argument('--accumulation_steps', default=None, type=int, required=False)
    parser.add_argument("--train_batch_size", default=16, type=int, help='')
    parser.add_argument("--val_batch_size", default=16, type=int, help='')
    parser.add_argument("--test_batch_size", default=1, type=int, help='')

    # Options for experiment setting
    parser.add_argument("--cat_target", nargs='+', default=[], type=str, help='')
    parser.add_argument("--num_target", nargs='+', default=[], type=str, help='')
    parser.add_argument("--num_normalize", type=str, default=True, help='')
    parser.add_argument("--confusion_matrix",  nargs='*', default=[], type=str, help='')
    parser.add_argument("--filter", nargs="*", default=[], type=str,
                        help='options for filter data by phenotype. usage: --filter abcd_site:10 sex:1')
    parser.add_argument("--mode", default='pretraining', type=str,  choices=['pretraining','finetuning','transfer'],
                        help='Option for learning from scratch')
    parser.add_argument("--load", default='', type=str, help='Load model weight that mathces {your_exp_dir}/result/*{load}*')
    parser.add_argument("--unfrozen_layers", default='all', type=str, help='Select the number of layers that would be unfrozen')
    parser.add_argument("--init_unfrozen", default='', type=str, help='Initializes unfrozen layers')
    
    # Options for environment setting
    parser.add_argument("--exp_name", type=str, required=True, help='')
    parser.add_argument("--wandb", type=str, default=None, help='')    
    parser.add_argument("--gpus", nargs='+', type=int, help='')
    parser.add_argument("--sbatch", type=str, choices=['True', 'False'])
    parser.add_argument("--debug", default='', type=str, help='')
        
    args = parser.parse_args()
    if ((args.cat_target + args.num_target) == []) and 'MM' not in args.model:
        raise ValueError('--num-target or --cat-target should be specified')
        
    print(f"*** Categorical target labels are {args.cat_target} and Numerical target labels are {args.num_target} *** \n")

    return args


def seed_all(SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    
# Return a specific model based on argument setting
def select_model(subject_data, args):
    # Simple CNN
    if args.model == 'simple3D':
        assert args.resize == [96, 96, 96]
        net = simple3d.simple3D(subject_data, args)
    # VGGNet
    elif args.model == 'vgg3D11':
        assert args.resize == [96, 96, 96]
        net = vgg3d.vgg3D11(subject_data, args)
    elif args.model == 'vgg3D13':
        assert args.resize == [96, 96, 96]
        net = vgg3d.vgg3D13(subject_data, args)
    elif args.model == 'vgg3D16':
        assert args.resize == [96, 96, 96]
        net = vgg3d.vgg3D16(subject_data, args)
    elif args.model == 'vgg3D19':
        assert args.resize == [96, 96, 96]
        net = vgg3d.vgg3D19(subject_data, args)
    # ResNet
    elif args.model == 'resnet3D50':
        net = resnet3d.resnet3D50(subject_data, args)
    elif args.model == 'resnet3D101':
        net = resnet3d.resnet3D101(subject_data, args)
    elif args.model == 'resnet3D152':
        net = resnet3d.resnet3D152(subject_data, args)
    # DenseNet
    elif 'densenet3D121' in args.model:
        net = densenet3d.densenet3D121(subject_data, args)
    elif args.model == 'densenet3D161':
        net = densenet3d.densenet3D161(subject_data, args) 
    elif args.model == 'densenet3D169':
        net = densenet3d.densenet3D169(subject_data, args) 
    elif args.model == 'densenet3D201':
        net = densenet3d.densenet3D201(subject_data, args)
    # SFCN
    elif args.model.lower() == 'sfcn':
        net = sfcn.SFCN(subject_data, args)
        
    return net

# Experiment 
def CLIreporter(train_loss, train_acc, val_loss, val_acc):
    '''command line interface reporter per every epoch during experiments'''
    print("="*80)
    visual_report = defaultdict(list)
    for label_name in train_loss:
        loss_value = f'{train_loss[label_name]:2.4f} / {val_loss[label_name]:2.4f}'
        if 'contrastive_loss' not in label_name:
            acc_value = f'{train_acc[label_name]:2.4f} / {val_acc[label_name]:2.4f}' 
        else:
            acc_value = None
        visual_report['Loss (train/val)'].append(loss_value)
        visual_report['R2 or ACC (train/val)'].append(acc_value)
    print(pd.DataFrame(visual_report, index=train_loss.keys()))
    
    return None


# define checkpoint-saving function
def checkpoint_save(net, epoch, args):
    """checkpoint is saved only if validation performance for all target tasks are improved """
    if args.debug=='1':
        return None
    if os.path.isdir(os.path.join(args.save_dir, 'model')) == False:
        makedir(os.path.join(args.save_dir, 'model'))
    
    checkpoint_dir = os.path.join(args.save_dir, f'model/{args.model}_{args.exp_name}.pth')
    if epoch == args.epoch:
        checkpoint_dir = checkpoint_dir[:-4]+"_last.pth"
    torch.save(net.module.state_dict(), checkpoint_dir)

    return checkpoint_dir


def checkpoint_load(net, checkpoint_dir, args, test=False): #230313change
    if hasattr(net, 'module'):
        net = net.module
    
    model_state = torch.load(checkpoint_dir, map_location = 'cpu')
    if 'MM' in args.model and args.mode != 'pretraining' and test == False:
        try:
            net.load_state_dict(model_state, strict=True)
        except:
            extractor_state = OrderedDict()
            for k, v in model_state.items():
                if 'feature_extractors' not in k:
                    break
                new_k = '.'.join(k.split('.')[1:])
                extractor_state[new_k] = model_state[k]
            net.feature_extractors.load_state_dict(extractor_state)
    else:
        net.load_state_dict(model_state, strict=True)
        
    print('The best checkpoint is loaded')

    return net
    
    
# define result-saving function
def save_exp_result(setting, result):
    if setting['debug']=='1':
        return None
    makedir(setting['save_dir'])
    exp_name = setting['exp_name']
    del setting['epoch']
    del setting['test_batch_size']

    filename = setting['save_dir'] + f'/{exp_name}.json'
    result.update(setting)

    with open(filename, 'w') as f:
        json.dump(result, f, indent=4)


def makedir(path):
    if not os.path.isdir(path):
        os.makedirs(path)
