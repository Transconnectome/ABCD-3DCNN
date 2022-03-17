import os
import argparse
import yaml
import json
import random

import torch
from torchvision import utils as vu
import torch.nn as nn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def makedir(folder):
    if not os.path.isdir(folder):
        os.makedirs(folder)


class Initializer:
    def __init__(self):
        pass

    @staticmethod
    def initialize(model, initialization, **kwargs):

        def weights_init(m):
            if isinstance(m, nn.Conv2d):
                initialization(m.weight.data, **kwargs)
                try:
                    initialization(m.bias.data)
                except:
                    pass

            elif isinstance(m, nn.Conv3d):
                initialization(m.weight.data, **kwargs)
                try:
                    initialization(m.bias.data)
                except:
                    pass

            elif isinstance(m, nn.Linear):
                initialization(m.weight.data, **kwargs)
                try:
                    initialization(m.bias.data)
                except:
                    pass

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.fill_(0)

            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1.0)
                m.bias.data.fill_(0)

        model.apply(weights_init)


def print_separator(text, total_len=50):
    print('#' * total_len)
    left_width = (total_len - len(text))//2
    right_width = total_len - len(text) - left_width
    print("#" * left_width + text + "#" * right_width)
    print('#' * total_len)


def print_yaml(opt):
    lines = []
    if isinstance(opt, dict):
        for key in opt.keys():
            tmp_lines = print_yaml(opt[key])
            tmp_lines = ["%s.%s" % (key, line) for line in tmp_lines]
            lines += tmp_lines
    else:
        lines = [": " + str(opt)]
    return lines


def create_path(opt):
    for k, v in opt['paths'].items():
        makedir(os.path.join(v, opt['exp_name']))


def read_yaml():
    # read in yaml
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str, help="Path for the config file")
    parser.add_argument("--exp_ids", type=int, nargs='+', default=[0])
    parser.add_argument("--gpus", type=int, nargs='+', default=[0])
    parser.add_argument("--exp_name", required=True, type=str, default='ABCD_multitask')

    parser.add_argument("--cat_target", type=str, nargs='*', required=False, help='')
    parser.add_argument("--num_target", type=str,nargs='*', required=False, help='')
    
    parser.add_argument("--model",required=False,type=str,help='',choices=['ResNet3D50','ResNet3D101','ResNet3D152'])
    parser.add_argument("--warmup_size",default=0.3,type=float,required=False,help='')
    parser.add_argument("--train_batch_size",default=16,type=int,required=False,help='')
    parser.add_argument("--val_batch_size",default=16,type=int,required=False,help='')
    parser.add_argument("--test_batch_size",default=16,type=int,required=False,help='')

    parser.add_argument("--resize",default=(96,96,96),required=False,help='')
    
    parser.add_argument("--lr", default=1e-3,type=float,required=False,help='')
    parser.add_argument("--backbone_lr", default=1e-6,type=float,required=False,help='')
    parser.add_argument("--policy_lr", default=1e-4,type=float,required=False,help='')
    parser.add_argument("--weight_decay",default=0.001,type=float,required=False,help='')
    parser.add_argument("--epoch",type=int,required=False,help='')

    args = parser.parse_args()

    # setting options
    with open(args.config) as f:
        opt = yaml.load(f, Loader=yaml.FullLoader)

    opt['exp_name'] = args.exp_name
    
    opt['task']['cat_target'] = args.cat_target
    opt['task']['num_target'] = args.num_target
    if not opt['task']['cat_target']:
        opt['task']['cat_target'] = []
    if not opt['task']['num_target'] :
        opt['task']['num_target'] = []
    opt['task']['targets'] = opt['task']['cat_target'] + opt['task']['num_target']


    opt['data_split']['warmup_size'] = args.warmup_size
    opt['data_split']['train_batch_size'] = args.train_batch_size
    opt['data_split']['val_batch_size'] = args.val_batch_size
    
    opt['train']['lr'] = args.lr
    opt['train']['policy_lr'] = args.policy_lr
    opt['train']['backbone_lr'] = args.backbone_lr
    opt['train']['weight_decay'] = args.weight_decay

    opt['data_augmentation']['resize'] = args.resize

    opt['lambdas'] = [1] * len(opt['task']['targets']) # This lambda value is the weight of loss of tasks. Now set as 1, so all loss of tasks are treated samely


    return opt, args.gpus, args.exp_ids


def fix_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_config():
    print_separator('READ YAML')
    opt, gpu_ids = read_yaml()
    fix_random_seed(opt["seed"])
    create_path(opt)
    # print yaml on the screen
    lines = print_yaml(opt)
    for line in lines: print(line)
    print('-----------------------------------------------------')
    # print to file
    with open(os.path.join(opt['paths']['log_dir'], opt['exp_name'], 'opt.txt'), 'w+') as f:
        f.writelines(lines)
    return opt, gpu_ids


def summarizing_results(opt ,results_iter, results_final, test = False):
    # summarizing 
    if test:
        if opt['task']['cat_target']:
            for cat_target in opt['task']['cat_target']:
                test_loss = np.mean(results_iter[cat_target]['test']['loss'])
                test_acc = np.mean(results_iter[cat_target]['test']['ACC or R2'])
                results_iter[cat_target]['test']['loss'] = test_loss
                results_iter[cat_target]['test']['ACC or R2'] = test_acc
                results_final[cat_target]['test']['loss'].append(test_loss)
                results_final[cat_target]['test']['ACC or R2'].append(test_acc) 

        if opt['task']['num_target']:
            for num_target in opt['task']['num_target']:
                test_loss = np.mean(results_iter[num_target]['test']['loss'])
                test_acc = np.mean(results_iter[num_target]['test']['ACC or R2'])
                results_iter[num_target]['test']['loss'] = test_loss
                results_iter[num_target]['test']['ACC or R2'] = test_acc
                results_final[num_target]['test']['loss'].append(test_loss)
                results_final[num_target]['test']['ACC or R2'].append(test_acc) 

    else:
        if opt['task']['cat_target']:
            for cat_target in opt['task']['cat_target']:
                train_loss = np.mean(results_iter[cat_target]['train']['loss'])
                train_acc = np.mean(results_iter[cat_target]['train']['ACC or R2']) 
                val_loss = np.mean(results_iter[cat_target]['val']['loss'])     
                val_acc = np.mean(results_iter[cat_target]['val']['ACC or R2'])
                results_iter[cat_target]['train']['loss'] = train_loss
                results_iter[cat_target]['train']['ACC or R2'] = train_acc
                results_iter[cat_target]['val']['loss'] =  val_loss
                results_iter[cat_target]['val']['ACC or R2'] = val_acc 
                results_final[cat_target]['train']['loss'].append(train_loss)
                results_final[cat_target]['train']['ACC or R2'].append(train_acc) 
                results_final[cat_target]['val']['loss'].append(val_loss)
                results_final[cat_target]['val']['ACC or R2'].append(val_acc)
            
        if opt['task']['num_target']:
            for num_target in opt['task']['num_target']:
                train_loss = np.mean(results_iter[num_target]['train']['loss'])
                train_acc = np.mean(results_iter[num_target]['train']['ACC or R2']) 
                val_loss = np.mean(results_iter[num_target]['val']['loss'])     
                val_acc = np.mean(results_iter[num_target]['val']['ACC or R2'])
                results_iter[num_target]['train']['loss'] = train_loss
                results_iter[num_target]['train']['ACC or R2'] = train_acc
                results_iter[num_target]['val']['loss'] =  val_loss
                results_iter[num_target]['val']['ACC or R2'] = val_acc 
                results_final[num_target]['train']['loss'].append(train_loss)
                results_final[num_target]['train']['ACC or R2'].append(train_acc) 
                results_final[num_target]['val']['loss'].append(val_loss)
                results_final[num_target]['val']['ACC or R2'].append(val_acc)

    return results_iter, results_final


def CLIreporter(results, opt, test= False):
    '''command line interface reporter per every epoch during experiments'''
    if test:
        var_column = []
        visual_report = {}
        visual_report['Loss'] = []
        visual_report['R2 or ACC'] = []

        if opt['task']['cat_target']:
            for cat_target in opt['task']['cat_target']:
                var_column.append(cat_target)
                loss_value = '{:2.2f}'.format(results[cat_target]['test']['loss'])
                acc_value = '{:2.2f}'.format(results[cat_target]['test']['ACC or R2'])
                visual_report['Loss'].append(loss_value)
                visual_report['R2 or ACC'].append(acc_value)
        
        if opt['task']['num_target']:
            for num_target in opt['task']['num_target']:
                var_column.append(num_target)
                loss_value = '{:2.2f}'.format(results[num_target]['test']['loss'])
                acc_value = '{:2.2f}'.format(results[num_target]['test']['ACC or R2'])                
                visual_report['Loss'].append(loss_value)
                visual_report['R2 or ACC'].append(acc_value)

    else:
        var_column = []
        visual_report = {}
        visual_report['Loss (train/val)'] = []
        visual_report['R2 or ACC (train/val)'] = []

        if opt['task']['cat_target']:
            for cat_target in opt['task']['cat_target']:
                var_column.append(cat_target)
                loss_value = '{:2.2f} / {:2.2f}'.format(results[cat_target]['train']['loss'],results[cat_target]['val']['loss'])
                acc_value = '{:2.2f} / {:2.2f}'.format(results[cat_target]['train']['ACC or R2'],results[cat_target]['val']['ACC or R2'])
                visual_report['Loss (train/val)'].append(loss_value)
                visual_report['R2 or ACC (train/val)'].append(acc_value)
        
        if opt['task']['num_target']:
            for num_target in opt['task']['num_target']:
                var_column.append(num_target)
                loss_value = '{:2.2f} / {:2.2f}'.format(results[num_target]['train']['loss'],results[num_target]['val']['loss'])           
                acc_value = '{:2.2f} / {:2.2f}'.format(results[num_target]['train']['ACC or R2'],results[num_target]['val']['ACC or R2'])
                visual_report['Loss (train/val)'].append(loss_value)
                visual_report['R2 or ACC (train/val)'].append(acc_value)

    print(pd.DataFrame(visual_report, index=var_column))


def making_results_template(opt, mode=None):
    results_final = {}
    if mode == 'test':
        if opt['task']['cat_target']:
            for cat_target in opt['task']['cat_target']:
                results_final[cat_target] = {'test':{'loss':[], 'ACC or R2':[]}}
        if opt['task']['num_target']:
            for num_target in opt['task']['num_target']:
                results_final[num_target] = {'test':{'loss':[], 'ACC or R2':[]}}      
    else:
        if opt['task']['cat_target']:
            for cat_target in opt['task']['cat_target']:
                results_final[cat_target] = {'train':{'loss':[], 'ACC or R2':[]}, 'val':{'loss':[], 'ACC or R2':[]}}
        if opt['task']['num_target']:
            for num_target in opt['task']['num_target']:
                results_final[num_target] = {'train':{'loss':[], 'ACC or R2':[]}, 'val':{'loss':[], 'ACC or R2':[]}}
    
    return results_final


def save_exp_results(results_final, opt, mode=None):
    exp_name = opt['exp_name']
    file_name = '{}_results_{}.json'.format(exp_name, mode)
    file_name = os.path.join(opt['paths']['result_dir'], exp_name, file_name)
    opt.update(results_final)

    with open(file_name, 'w') as f:
        json.dump(opt, f)


def ImageList_loading(file_name):
    image_files = []
    with open(file_name, 'r') as f:
        while True:
            subject = f.readline()
            if not subject: break
            image_files.append(str(subject.strip('\n')))
    
    return image_files