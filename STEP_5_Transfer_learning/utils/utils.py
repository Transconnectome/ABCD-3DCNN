from genericpath import isdir
import os
import pandas as pd
import hashlib
import json
import argparse 
import torch
from copy import deepcopy

import models.simple3d as simple3d #model script
import models.vgg3d as vgg3d #model script
import models.resnet3d as resnet3d #model script
import models.densenet3d as densenet3d #model script

def argument_setting():
    parser = argparse.ArgumentParser()

    #parser.add_argument("--GPU_NUM",default=1,type=int,required=True,help='')
    parser.add_argument("--model",required=True,type=str,help='',choices=['simple3D','vgg3D11','vgg3D13','vgg3D16','vgg3D19','resnet3D50','resnet3D101','resnet3D152', 'densenet3D121', 'densenet3D169','densenet201','densenet264'])
    parser.add_argument("--dataset",required=True, type=str, choices=['UKB','ABCD'],help='') # revising
    parser.add_argument("--data", type=str, help='select data type') # revising
    parser.add_argument("--val_size",default=0.1,type=float,required=False,help='')
    parser.add_argument("--test_size",default=0.1,type=float,required=False,help='')
    parser.add_argument("--resize",default=(96, 96, 96),type=int,nargs="*",required=False,help='')
    parser.add_argument("--train_batch_size",default=16,type=int,required=False,help='')
    parser.add_argument("--val_batch_size",default=16,type=int,required=False,help='')
    parser.add_argument("--test_batch_size",default=1,type=int,required=False,help='')
    parser.add_argument("--in_channels",default=1,type=int,required=False,help='')
    parser.add_argument("--optim",type=str,required=True,help='', choices=['Adam','SGD','RAdam','AdamW'])
    parser.add_argument("--scheduler",type=str,default='',help='') # revising
    parser.add_argument("--lr", default=0.01,type=float,required=False,help='')
    parser.add_argument("--lr_adjust", default=0.01, type=float, required=False,help='')   
    parser.add_argument("--weight_decay",default=0.001,type=float,required=False,help='')
    parser.add_argument("--epoch",type=int,required=True,help='')
    parser.add_argument("--epoch_FC",type=int,required=False,default=0,help='')
    parser.add_argument("--exp_name",type=str,required=True,help='')
    parser.add_argument("--cat_target", type=str, nargs='*', required=False, help='')
    parser.add_argument("--num_target", type=str,nargs='*', required=False, help='')
    parser.add_argument("--confusion_matrix", type=str, nargs='*',required=False, help='')
    parser.add_argument("--gpus", type=int,nargs='*', required=False, help='')
    parser.add_argument("--sbatch", type=str, required=False, choices=['True', 'False'])
    parser.add_argument("--transfer", type=str, required=False, default="", choices=['sex','age','simclr','MAE'])
    parser.add_argument("--unfrozen_layer", type=str, required=False, default='0') 
    parser.add_argument("--load", type=str, required=False, default="")
    parser.add_argument("--init_unfrozen", type=str, required=False, default="",help='init unfrozen layers')
    
    args = parser.parse_args()
    print("*** Categorical target labels are {} and Numerical target labels are {} *** \n".format(
        args.cat_target, args.num_target)
         )

    if not args.cat_target:
        args.cat_target = []
    elif not args.num_target:
        args.num_target = []
    elif not args.cat_target and args.num_target:
        raise ValueError('YOU SHOULD SELECT THE TARGET!')

    return args

# Return a specific model based on argument setting
def select_model(subject_data, args): # revising
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
        
    return net

# Experiment 
def CLIreporter(targets, train_loss, train_acc, val_loss, val_acc):
    '''command line interface reporter per every epoch during experiments'''
    var_column = []
    visual_report = {}
    visual_report['Loss (train/val)'] = []
    visual_report['R2 or ACC (train/val)'] = []
    for label_name in targets:
        var_column.append(label_name)

        loss_value = '{:2.4f} / {:2.4f}'.format(train_loss[label_name],val_loss[label_name])
        acc_value = '{:2.4f} / {:2.4f}'.format(train_acc[label_name],val_acc[label_name])
        visual_report['Loss (train/val)'].append(loss_value)
        visual_report['R2 or ACC (train/val)'].append(acc_value)

    print(pd.DataFrame(visual_report, index=var_column))


# define checkpoint-saving function
"""checkpoint is saved only when validation performance for all target tasks are improved """
def checkpoint_save(net, save_dir, epoch, current_result, previous_result,  args):
    if os.path.isdir(os.path.join(save_dir,'model')) == False:
        makedir(os.path.join(save_dir,'model'))
    
    checkpoint_dir = os.path.join(save_dir, 'model/{}_{}.pth'.format(args.model, args.exp_name))
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
def save_exp_result(save_dir, setting, result):
    makedir(save_dir)
    exp_name = setting['exp_name']
    del setting['epoch']
    del setting['test_batch_size']

    filename = save_dir + '/{}.json'.format(exp_name)
    result.update(setting)

    with open(filename, 'w') as f:
        json.dump(result, f)


def makedir(path):
    if not os.path.isdir(path):
        os.makedirs(path)




