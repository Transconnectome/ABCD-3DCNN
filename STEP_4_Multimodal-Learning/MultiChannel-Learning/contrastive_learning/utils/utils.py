import os
import json
import argparse 
from collections import defaultdict

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
                                 'densenet3D121', 'densenet3D169', 'densenet201', 'densenet264'])
    parser.add_argument("--in_channels", default=1, type=int, help='')
    
    # Options for dataset and data type, split ratio, CV, resize, augmentation
    parser.add_argument("--dataset", type=str, choices=['UKB','ABCD'], required=True, help='Selelct dataset')
    parser.add_argument("--data_type", nargs='+', type=str, help='Select data type(sMRI, dMRI)',
                        choices=['fmriprep', 'freesurfer', 'FA_unwarpped_nii', 'FA_warpped_nii',
                                 'MD_unwarpped_nii', 'MD_warpped_nii', 'RD_unwarpped_nii', 'RD_warpped_nii'])
    parser.add_argument("--val_size", default=0.1, type=float, help='')
    parser.add_argument("--test_size", default=0.1, type=float, help='')
    parser.add_argument("--cv", default=None, type=int, choices=[1,2,3,4,5], help="option for 5-fold CV. 1~5.")
    parser.add_argument("--resize", nargs="*", default=(96, 96, 96), type=int, help='')
    parser.add_argument("--augmentation", nargs="*", default=[], type=str, choices=['shift','flip'],
                        help="Data augmentation - [shift, flip] are available")
    
    # Hyperparameters for model training
    parser.add_argument("--lr", default=0.01, type=float, help='')
    parser.add_argument("--lr_adjust", default=0.01, type=float, help='')
    parser.add_argument("--epoch", type=int, required=True, help='')
    parser.add_argument("--epoch_FC", type=int, default=0, help='Option for training only FC layer')
    parser.add_argument("--optim", default='Adam', type=str, choices=['Adam','SGD','RAdam','AdamW'], help='')
    parser.add_argument("--weight_decay", default=0.001, type=float, help='')
    parser.add_argument("--scheduler", default='', type=str, help='') 
    parser.add_argument("--early_stopping", default=None, type=int, help='')
    parser.add_argument("--train_batch_size", default=16, type=int, help='')
    parser.add_argument("--val_batch_size", default=16, type=int, help='')
    parser.add_argument("--test_batch_size", default=1, type=int, help='')
   
    # Options for experiment setting
    parser.add_argument("--exp_name", type=str, required=True, help='')
    parser.add_argument("--gpus", nargs='+', type=int, help='')
    parser.add_argument("--sbatch", type=str, choices=['True', 'False'])
    parser.add_argument("--cat_target", nargs='+', default=[], type=str, help='')
    parser.add_argument("--num_target", nargs='+', default=[], type=str, help='')
    parser.add_argument("--confusion_matrix",  nargs='*', default=[], type=str, help='')
    parser.add_argument("--filter", nargs="*", default=[], type=str,
                        help='options for filter data by phenotype. usage: --filter abcd_site:10 sex:1')
    parser.add_argument("--load", default='', type=str, help='Load model weight that mathces {your_exp_dir}/result/*{load}*')
    parser.add_argument("--scratch", default='', type=str, help='Option for learning from scratch')
    parser.add_argument("--transfer", default='', type=str, choices=['sex','age','simclr','MAE'],
                        help='Choose pretrained model according to your option')
    parser.add_argument("--unfrozen_layer", default='0', type=str, help='Select the number of layers that would be unfrozen')
    parser.add_argument("--init_unfrozen", default='', type=str, help='Initializes unfrozen layers')
    parser.add_argument("--debug", default='', type=str, help='')
        
    args = parser.parse_args()
    if args.cat_target == args.num_target:
        raise ValueError('--num-target or --cat-target should be specified')
        
    print(f"*** Categorical target labels are {args.cat_target} and Numerical target labels are {args.num_target} *** \n")

    return args

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
    elif args.model == 'densenet3D121':
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
    visual_report = defaultdict(list)
    for label_name in train_loss:
        loss_value = f'{train_loss[label_name]:2.4f} / {val_loss[label_name]:2.4f}'
        if 'contrastive_loss' not in label_name:
            acc_value = f'{train_acc[label_name]:2.4f} / {val_acc[label_name]:2.4f}' 
        else:
            acc_value = None
            
        visual_report['Loss (train/val)'].append(loss_value)
        visual_report['R2 or ACC (train/val)'].append(acc_value)
    print("="*80)
    print(pd.DataFrame(visual_report, index=train_loss.keys()))


# define checkpoint-saving function
def checkpoint_save(net, epoch, current_result, previous_result, args):
    """checkpoint is saved only if validation performance for all target tasks are improved """
    if os.path.isdir(os.path.join(args.save_dir, 'model')) == False:
        makedir(os.path.join(args.save_dir, 'model'))
    
    checkpoint_dir = os.path.join(args.save_dir, f'model/{args.model}_{args.exp_name}.pth')
    best_checkpoint_votes = 0

    if args.cat_target:
        for cat_target in args.cat_target:
            if current_result[cat_target] >= max(previous_result[cat_target]):
                best_checkpoint_votes += 1
    if args.num_target:
        for num_target in args.num_target:
            if current_result[num_target] >= max(previous_result[num_target]):
                best_checkpoint_votes += 1
        
    if best_checkpoint_votes == len(args.cat_target + args.num_target):
        torch.save(net.module.state_dict(), checkpoint_dir)
        print("Best iteration until now is %d" % (epoch + 1))

    return checkpoint_dir

sex_model_dir = '/scratch/connectome/dhkdgmlghks/3DCNN_test/3DCNN_hardparameter_sharing/result/model/UKB_sex_densenet3D121_6cbde7.pth'
age_model_dir = '/scratch/connectome/dhkdgmlghks/UKB_sex_densenet3D121_6cbde7.pth'
MAE_model_dir = '/scratch/connectome/dhkdgmlghks/3DCNN_test/3DCNN_hardparameter_sharing/result/model/densenet3D121_UKB_age_d167e4.pth'
simclr_dir = '/scratch/connectome/jubin/Simclr_Contrastive_MRI_epoch_99.pth'

def checkpoint_load(net, checkpoint_dir):
    if hasattr(net, 'module'):
        net = net.module
        
    if checkpoint_dir == 'sex':
        checkpoint_dir = sex_model_dir
    elif checkpoint_dir == 'age':
        checkpoint_dir = age_model_dir
    elif checkpoint_dir == 'simclr':
        checkpoint_dir = simclr_dir
    elif checkpoint_dir == 'MAE':
        checkpoint_dir = MAE_model_dir
    
    model_state = torch.load(checkpoint_dir, map_location = 'cpu')
    net.load_state_dict(model_state, strict=False)
    print('The best checkpoint is loaded')

    return net
    
    
# define result-saving function
def save_exp_result(setting, result):
    makedir(setting['save_dir'])
    exp_name = setting['exp_name']
    del setting['epoch']
    del setting['test_batch_size']

    filename = setting['save_dir'] + f'/{exp_name}.json'
    result.update(setting)

    with open(filename, 'w') as f:
        json.dump(result, f)


def makedir(path):
    if not os.path.isdir(path):
        os.makedirs(path)




