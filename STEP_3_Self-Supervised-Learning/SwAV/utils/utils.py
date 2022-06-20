
import models.resnet3d as resnet3d #model script
import models.densenet3d as densenet3d #model script

from genericpath import isdir
import os
import pandas as pd
import numpy as np
import hashlib
import json
import argparse 
import torch
import copy
from copy import deepcopy


def argument_setting_SwAV():
    parser = argparse.ArgumentParser()

    #########################
    #### data parameters ####
    #########################
    parser.add_argument("--val_size",default=0.1,type=float,required=False,help='')
    parser.add_argument("--test_size",default=0.1,type=float,required=False,help='')
    parser.add_argument("--resize",default=[96, 96, 96],type=int,nargs="*",required=False,help='')
    parser.add_argument("--augmentation", type=str, nargs='*', required=True, choices=['RandRotate','RandRotate90','RandFlip','RandAdjustContrast','RandGaussianSmooth', 'RandGibbsNoise','RandCoarseDropout'])

    #########################
    ### batch size params ###
    #########################
    parser.add_argument("--train_batch_size",default=16,type=int,required=False,help='')

    #########################
    ## swav specific params #
    #########################
    parser.add_argument("--model",required=True,type=str,help='',choices=['simple3D','vgg3D11','vgg3D13','vgg3D16','vgg3D19','resnet3D50','resnet3D101','resnet3D152', 'densenet3D121', 'densenet3D169','densenet201','densenet264'])
    parser.add_argument("--nmb_standard_views", default=2, type=int, help = 'number of standard resolution views (= n_views in simCLR). This is for multi crop strategy')
    parser.add_argument("--nmb_low_views", default=4, type=int, help = 'number of low resolution views. This is for multi crop strategy')
    parser.add_argument("--nmb_prototypes", default=3000, type=int, help="number of prototypes")
    parser.add_argument("--queue_length", type=int, default=0, help="length of the queue (0 for no queue). In other words, the number of stored features in the dictionary from the previous batches.")
    parser.add_argument("--epoch_queue_starts", type=int, default=15, help="from this epoch, we start using a queue")
    parser.add_argument("--feat_dim", default=128, type=int, help="feature dimension")
    parser.add_argument("--sinkhorn_iterations", default=3, type=int, help="number of iterations in Sinkhorn-Knopp algorithm")
    parser.add_argument("--temperature", default=0.1, type=float, help="temperature parameter in training loss")
    parser.add_argument("--epsilon", default=0.05, type=float, help="regularization parameter for Sinkhorn-Knopp algorithm")

    ##########################
    #### optim parameters ####
    ##########################
    parser.add_argument("--optim",type=str,required=True,help='', choices=['Adam','SGD', 'LARS', 'LAMB'])
    parser.add_argument("--lr", default=0.01,type=float,required=False,help='')
    parser.add_argument("--weight_decay",default=0.001,type=float,required=False,help='')
    parser.add_argument("--epoch",type=int,required=True,help='')
    parser.add_argument("--freeze_prototypes_niters", default=313, type=int, help="freeze the prototypes during this many iterations from the start")
    
    ##########################
    #### other parameters ####
    ##########################
    parser.add_argument("--in_channels",default=1,type=int,required=False,help='')
    parser.add_argument("--exp_name",type=str,required=True,help='')
    parser.add_argument("--checkpoint_dir", type=str, default=None,required=False)
    parser.add_argument("--resume", type=str, default='True', required=True)
    
    #########################
    #### dist parameters ####
    #########################
    parser.add_argument("--sbatch", type=str, required=False, choices=['True', 'False'])
    parser.add_argument("--world_size", default=-1, type=int, help="""number of processes: it is set automatically and should not be passed as argument""")
    parser.add_argument("--rank", default=0, type=int, help="""rank of this process:it is set automatically and should not be passed as argument""")
    parser.add_argument("--local_rank", default=0, type=int,help="this argument is not used and should be ignored")



    ####global args
    args = parser.parse_args()

    return args


def argument_setting_finetuning():
    parser = argparse.ArgumentParser()

    #########################
    #### data parameters ####
    #########################
    parser.add_argument("--finetuning_size",default=0.1,type=float,required=False,help='')
    parser.add_argument("--val_size",default=0.1,type=float,required=False,help='')
    parser.add_argument("--test_size",default=0.1,type=float,required=False,help='')
    parser.add_argument("--train_batch_size",default=16,type=int,required=False,help='')
    parser.add_argument("--val_batch_size",default=16,type=int,required=False,help='')
    parser.add_argument("--test_batch_size",default=1,type=int,required=False,help='')
    parser.add_argument("--resize",default=[96, 96, 96],type=int,nargs="*",required=False,help='')
    parser.add_argument("--cat_target", type=str, nargs='*', required=False, help='')
    parser.add_argument("--num_target", type=str,nargs='*', required=False, help='')

    ##########################
    #### model parameters ####
    ##########################
    parser.add_argument("--model",required=True,type=str,help='',choices=['simple3D','vgg3D11','vgg3D13','vgg3D16','vgg3D19','resnet3D50','resnet3D101','resnet3D152', 'densenet3D121', 'densenet3D169','densenet201','densenet264'])
    parser.add_argument("--in_channels",default=1,type=int,required=False,help='')

    ##########################
    #### optim parameters ####
    ##########################
    parser.add_argument("--optim",type=str,required=True,help='', choices=['Adam','SGD'])
    parser.add_argument("--lr", default=0.01,type=float,required=False,help='')
    parser.add_argument("--weight_decay",default=0.001,type=float,required=False,help='')
    parser.add_argument("--epoch",type=int,required=True,help='')

    ##########################
    #### other parameters ####
    ##########################
    parser.add_argument("--pretrained_model_dir", type=str, required=False)
    parser.add_argument("--checkpoint_dir", type=str, default=None, required=False)
    parser.add_argument("--resume", type=str, default='True', required=True)
    parser.add_argument("--exp_name",type=str,required=True,help='')
    parser.add_argument("--confusion_matrix", type=str, nargs='*',required=False, help='')
    parser.add_argument("--sbatch", type=str, required=False, choices=['True', 'False'])

    args = parser.parse_args()
    print("Categorical target labels are {} and Numerical target labels are {}".format(args.cat_target, args.num_target))

    if not args.cat_target:
        args.cat_target = []
    elif not args.num_target:
        args.num_target = []
    elif not args.cat_target and args.num_target:
        raise ValueError('YOU SHOULD SELECT THE TARGET!')

    return args


def set_random_seed(seed):
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
def set_backbone(args):
    if args.model == 'resnet3D50':
        net = resnet3d.resnet3D50()
        setattr(net, 'args', args)
        num_features = getattr(net,'num_features')
    elif args.model == 'resnet3D101':
        net = resnet3d.resnet3D101()
        setattr(net, 'args', args)
        num_features = getattr(net,'num_features')
    elif args.model == 'resnet3D152':
        net = resnet3d.resnet3D152()
        setattr(net, 'args', args)
        num_features = getattr(net,'num_features')
    # DenseNet
    elif args.model == 'densenet3D121':
        net = densenet3d.densenet3D121()
        setattr(net, 'args', args)
        num_features = getattr(net,'num_features')
    elif args.model == 'densenet3D161':
        net = densenet3d.densenet3D161() 
        setattr(net, 'args', args)
        num_features = getattr(net,'num_features')
    elif args.model == 'densenet3D169':
        net = densenet3d.densenet3D169()
        setattr(net, 'args', args) 
        num_features = getattr(net,'num_features')
    elif args.model == 'densenet3D201':
        net = densenet3d.densenet3D201()
        setattr(net, 'args', args)
        num_features = getattr(net,'num_features')

    return net, num_features


def device_as(t1, t2):
    """Moves tensor1 (t1) to the device of tensor2 (t2)"""
    return t1.to(t2.device)


def get_queue_path(save_dir, args): 
    if os.path.isdir(os.path.join(save_dir)) == False:
        makedir(os.path.join(save_dir,"queue"))
    return os.path.join(save_dir, "queue", "queue" + str(args.rank) + ".path") 
        


def CLIreporter(targets, train_loss, train_acc, val_loss, val_acc):
    '''command line interface reporter per every epoch during experiments'''
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


# define checkpoint-saving function
"""checkpoint is saved only when validation performance for all target tasks are improved """
def checkpoint_save(net, optimizer, save_dir, epoch, scheduler, args, current_result=None, previous_result=None, mode=None):
    # if not resume, making checkpoint file. And if resume, overwriting on existing files  
    if args.resume == 'False':
        if os.path.isdir(os.path.join(save_dir,'model')) == False:
            makedir(os.path.join(save_dir,'model'))
        checkpoint_dir = os.path.join(save_dir, 'model/{}_{}.pth'.format(args.model, args.exp_name))
    
    else:
        checkpoint_dir = copy.copy(args.checkpoint_dir)
        
    
    # saving model
    if mode == 'simCLR':
        torch.save({'backbone':net.module.backbone.state_dict(), 
                    'projection_head': net.module.projection_head.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr': optimizer.param_groups[0]['lr'],
                    'scheduler': scheduler.state_dict(),
                    'epoch':epoch}, checkpoint_dir)
        print("Checkpoint is saved")
    elif mode == 'prediction':
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
            torch.save({'backbone':net.module.backbone.state_dict(),
                        'FClayers':net.module.FClayers.state_dict(),
                        'optimizer':optimizer.state_dict(),
                        'lr': optimizer.param_groups[0]['lr'],
                        'epoch': epoch}, checkpoint_dir)
            print("Best iteration until now is %d" % (epoch + 1))

    return checkpoint_dir



def checkpoint_load(net, checkpoint_dir, optimizer, scheduler, args,  mode='simCLR'):
    if mode == 'simCLR':
        model_state = torch.load(checkpoint_dir, map_location = 'cpu')
        net.backbone.load_state_dict(model_state['backbone'])
        net.projection_head.load_state_dict(model_state['projection_head'])
        optimizer.load_state_dict(model_state['optimizer'])
        scheduler.load_state_dict(model_state['scheduler'])
        print('The last checkpoint is loaded')
        #return net, optimizer, model_state['epoch']
    
    elif mode == 'finetuing':
        model_state = torch.load(checkpoint_dir, map_location='cpu')
        # loading network
        net.backbone.load_state_dict(model_state['backbone'])
        FClayers = model_state['FClayers']
        if args.version == 'simCLR_v1':
            net.FClayers.FClayer.load_state_dict(FClayers['FClayer'])
        elif args.version == 'simCLR_v2':
            net.FClayers.head1.load_state_dict(FClayers['head1'])
            net.FClayers.FClayer.load_state_dict(FClayers['FClayer'])
        #loading optimizers
        optimizer.load_state_dict(model_state['optimizer'])
        scheduler.load_state_dict(model_state['scheduler'])
        print('The last checkpoint is loaded')
        #return net, optimizer, model_state['epoch']
    
    elif mode == 'eval':
        if hasattr(net, 'module'):
            net = net.module
        model_state = torch.load(checkpoint_dir, map_location='cpu')
        net.backbone.load_state_dict(model_state['backbone'])
        net.FClayers.load_state_dict(model_state['FClayers'])
        optimizer.load_state_dict(model_state['optimizer'])
        scheduler.load_state_dict(model_state['scheduler'])
        print('The best checkpoint is loaded')
    

    return net, optimizer, scheduler, model_state['epoch'] + 1, model_state['lr'] 
 


def load_pretrained_model(net, checkpoint_dir):
    model_state = torch.load(checkpoint_dir, map_location = 'cpu')
    net.backbone.load_state_dict(model_state['backbone'])
    print('The pre-trained model is loaded')

    return net    



def freezing_layers(module):
    for param in module.parameters():
        param.requires_grad = False
    return module



# define result-saving function
def save_exp_result(save_dir, setting, result, resume='False'):
    if os.path.isdir(save_dir) == False:
        makedir(save_dir)

    filename = save_dir + '/{}.json'.format(setting['exp_name'])
    result.update(setting)

    with open(filename, 'w') as f:
        json.dump(result, f)


def makedir(path):
    if not os.path.isdir(path):
        os.makedirs(path)




