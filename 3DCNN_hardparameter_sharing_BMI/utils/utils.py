
from genericpath import isdir
from importlib.util import module_for_loader
import os
import random
import pandas as pd
import numpy as np 
import hashlib
import json
import argparse 
import torch
from copy import deepcopy
from typing import Dict


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def case_control_count(labels, dataset_type, args):
    if args.cat_target:
        for cat_target in args.cat_target:
            target_labels = []

            for label in labels:
                target_labels.append(label[cat_target])
            
            n_control = target_labels.count(0)
            n_case = target_labels.count(1)
            print('In {} dataset, {} contains {} CASE and {} CONTROL'.format(dataset_type, cat_target,n_case, n_control))


# Experiment 
def CLIreporter(targets, train_loss, train_acc, val_loss, val_acc):
    '''command line interface reporter per every epoch during experiments'''
    """
    var_column = []
    visual_report = {}
    visual_report['Loss (train/val)'] = []
    visual_report['R2 or ACC (train/val)'] = []
    for label_name in targets:
        var_column.append(label_name)

        loss_value = '{:2.2f} / {:2.2f}'.format(train_loss[label_name],val_loss[label_name])
        acc_value = '{:2.2f} / {:2.2f}'.format(train_acc[label_name],val_acc[label_name])
        visual_report['Loss (train/val)'].append(loss_value)
        visual_report['R2 or ACC (train/val)'].append(acc_value)

    print(pd.DataFrame(visual_report, index=var_column))
    """
    for target in targets: 
        train_acc[target]['train_loss'] = train_loss[target]
        val_acc[target]['val_loss'] = val_loss[target]
    print(" Train {} \n Valid {}".format(pd.DataFrame(train_acc), pd.DataFrame(val_acc)))


# define checkpoint-saving function
"""checkpoint is saved only when validation performance for all target tasks are improved """
def checkpoint_save(net, save_dir, epoch, current_result, previous_result,  args):
    if os.path.isdir(os.path.join(save_dir,'model')) == False:
        makedir(os.path.join(save_dir,'model'))
    
    checkpoint_dir = os.path.join(save_dir, 'model/{}_{}.pth'.format(args.model, args.exp_name))
    best_checkpoint_votes = 0
    
    if args.cat_target:
        for cat_target in args.cat_target:
            if current_result[cat_target]['ACC'] >= max(previous_result[cat_target][:-1]):
                best_checkpoint_votes += 1
    if args.num_target:
        for num_target in args.num_target:
            if current_result[num_target]['r_square'] >= max(previous_result[num_target][:-1]):
                best_checkpoint_votes += 1
        
    if best_checkpoint_votes == len(args.cat_target + args.num_target):
        torch.save(net.module.state_dict(), checkpoint_dir)
        print("Best iteration until now is %d" % (epoch + 1))
    
    return checkpoint_dir


def checkpoint_load(net, checkpoint_dir, layers='all'):
    if hasattr(net, 'module'):
        net = net.module

    model_state = torch.load(checkpoint_dir, map_location = 'cpu')
    
    if layers == 'all':
        net.load_state_dict(model_state)
        print('All layers of the best checkpoint are loaded')
    elif layers == 'conv': 
        for checkpoint_key in model_state.keys():
            if checkpoint_key.find('features') != -1:
                for net_key, net_values in net.named_parameters():
                    if checkpoint_key == net_key:
                        setattr(net_values, 'data', model_state[checkpoint_key])
        print('Convolution layers of the best checkpoint are loaded')

    return net
            

# define result-saving function
def save_exp_result(save_dir, setting, result):
    makedir(save_dir)
    exp_name = setting['exp_name']


    filename = save_dir + '/{}.json'.format(exp_name)
    result.update(setting)

    with open(filename, 'w') as f:
        json.dump(result, f)


def makedir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def combine_pred_subjid(outputs: dict, subject_ids: list) -> dict:
    def getting_subjid(subject_ids):
        for i, subj in enumerate(subject_ids):
            _, subj = os.path.split(subj)
            if subj.find('.nii.gz') != -1: 
                subj = subj.replace('.nii.gz','')
            elif subj.find('.npy') != -1: 
                subj = subj.replace('.npy','')
            subject_ids[i] = subj
        return subject_ids

    subject_ids = getting_subjid(subject_ids=subject_ids)    
    for i, pred_score in enumerate(outputs): 
        pred_subjid = (subject_ids[i], pred_score)
        outputs[i] = tuple(pred_subjid)
    return outputs


def combine_emb_subjid(embs: torch.Tensor, subject_ids: list) -> Dict[str, np.array]:
    def getting_subjid(subject_ids):
        for i, subj in enumerate(subject_ids):
            _, subj = os.path.split(subj)
            if subj.find('.nii.gz') != -1: 
                subj = subj.replace('.nii.gz','')
            elif subj.find('.npy') != -1: 
                subj = subj.replace('.npy','')
            subject_ids[i] = subj
        return subject_ids

    subject_ids = getting_subjid(subject_ids=subject_ids)
    embs_subjid = {}
    for i in range(embs.shape[0]):
        embs_subjid[subject_ids[i]] = embs[i].detach().cpu().numpy()

    return embs_subjid

# define result-saving function
def save_emb_result(save_dir, exp_name, embeddings):
    makedir(save_dir)
    filename = save_dir + '/{}.npy'.format(exp_name)
    np.save(filename, embeddings)


def freeze_conv(net): 
    for k, v in net.module.named_parameters():
        if k.find('features') != -1:
            if k.find('norm') != -1:    # When Fine-tuning, all layers should be frozen but the BathcNormalization layers need to be trainable
                v.requires_grad = True
            else: 
                v.requires_grad = False
    return net 

def unfreeze_conv(net): 
    """
    - Ref 1: https://forums.fast.ai/t/why-are-batchnorm-layers-set-to-trainable-in-a-frozen-model/46560
    - Ref 2: https://luvbb.tistory.com/50
    The BathcNormalization layers need to be kept frozen (more details). If they are also turned to trainable, the first epoch after unfreezing will significantly reduce accuracy.
    Each block needs to be all turned on or off. This is because the architecture includes a shortcut from the first layer to the last layer for each block. Not respecting blocks also significantly harms the final performance.
    """
    for k, v in net.module.named_parameters():
        if k.find('features') != -1:
            v.requires_grad = True
    return net 