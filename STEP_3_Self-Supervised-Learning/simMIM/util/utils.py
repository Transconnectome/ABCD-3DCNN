import os
import pandas as pd
import numpy as np
import hashlib
import json
 
import torch
import copy
from copy import deepcopy




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


def device_as(t1, t2):
    """Moves tensor1 (t1) to the device of tensor2 (t2)"""
    return t1.to(t2.device)

        

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
def checkpoint_save(net, optimizer, save_dir, epoch, scheduler, scaler, args, performance_result=None, mode=None):
    # if not resume, making checkpoint file. And if resume, overwriting on existing files  
    if args.resume == False:
        if os.path.isdir(os.path.join(save_dir,'model')) == False:
            makedir(os.path.join(save_dir,'model'))
        checkpoint_dir = os.path.join(save_dir, 'model/{}_{}.pth'.format(args.model, args.exp_name))
    
    else:
        checkpoint_dir = copy.copy(args.checkpoint_dir)
    

    if mode == 'pretrain':
        torch.save({'net':net.module.state_dict(), 
                    'optimizer': optimizer.state_dict(),
                    'lr': optimizer.param_groups[0]['lr'],
                    'scheduler': scheduler.state_dict(),
                    'amp_state': scaler.state_dict(),
                    'epoch':epoch}, checkpoint_dir)

        print("Checkpoint is saved")
    elif mode == 'finetune':
        torch.save({'net':net.module.state_dict(), 
                    'optimizer': optimizer.state_dict(),
                    'lr': optimizer.param_groups[0]['lr'],
                    'scheduler': scheduler.state_dict(),
                    'amp_state': scaler.state_dict(),
                    'performance': performance_result,
                    'epoch':epoch}, checkpoint_dir) 
        print("Checkpoint is saved")       

            
    """
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
    """
    return checkpoint_dir



def checkpoint_load(net, checkpoint_dir, optimizer, scheduler, scaler, mode='pretrain'):
    if mode == 'pretrain':
        model_state = torch.load(checkpoint_dir, map_location = 'cpu')
        net.load_state_dict(model_state['net'])
        optimizer.load_state_dict(model_state['optimizer'])
        scheduler.load_state_dict(model_state['scheduler'])
        scaler.load_state_dict(model_state['amp_state'])
        print('The last checkpoint is loaded')
        #return net, optimizer, model_state['epoch']
        return net, optimizer, scheduler, model_state['epoch'] + 1, model_state['lr'], scaler  

    elif mode == 'finetuning': 
        model_state = torch.load(checkpoint_dir, map_location = 'cpu')
        net = load_pretrained_model(net, model_state)
        return net 

    """
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
    """

    
 

def saving_outputs(net, pred, mask, target, save_dir):
    img_with_mask = mask.unsqueeze(-1) * target 
    np.save(os.path.join(save_dir,'img_with_mask.npy'), net.module.unpatchify_3D(img_with_mask)[:10].detach().cpu().numpy())
    np.save(os.path.join(save_dir,'target.npy'),net.module.unpatchify_3D(target)[:10].detach().cpu().numpy())
    np.save(os.path.join(save_dir,'pred.npy'),net.module.unpatchify_3D(pred).detach()[:10].cpu().numpy())
    print('==== DONE SAVING EXAMPLE IMAGES ====')


def load_attention_blocks(net, model_state):
    for i, block in enumerate(net.blocks): 
        # initial norm layer
        setattr(net, 'blocks[%s].norm1.weight.data' % str(i), model_state['net']['blocks.%s.norm1.weight' % str(i)])
        setattr(net, 'blocks[%s].norm1.bias.data' % str(i), model_state['net']['blocks.%s.norm1.bias' % str(i)])
        # attention qkv parameters 
        setattr(net, 'blocks[%s].attn.qkv.weight.data' % str(i), model_state['net']['blocks.%s.attn.qkv.weight' % str(i)])
        setattr(net, 'blocks[%s].attn.qkv.bias.data' % str(i), model_state['net']['blocks.%s.attn.qkv.bias' % str(i)])
        # attention projection layer
        setattr(net, 'blocks[%s].attn.proj.weight.data' % str(i), model_state['net']['blocks.%s.attn.proj.weight' % str(i)])
        setattr(net, 'blocks[%s].attn.proj.bias.data' % str(i), model_state['net']['blocks.%s.attn.proj.bias' % str(i)])
        # last norm layer
        setattr(net, 'blocks[%s].norm2.weight.data' % str(i), model_state['net']['blocks.%s.norm2.weight' % str(i)])
        setattr(net, 'blocks[%s].norm2.bias.data' % str(i), model_state['net']['blocks.%s.norm2.bias' % str(i)])
        # fc layer 
        setattr(net, 'blocks[%s].mlp.fc1.weight.data' % str(i), model_state['net']['blocks.%s.mlp.fc1.weight' % str(i)])
        setattr(net, 'blocks[%s].mlp.fc1.bias.data' % str(i), model_state['net']['blocks.%s.mlp.fc1.bias' % str(i)])
        setattr(net, 'blocks[%s].mlp.fc2.weight.data' % str(i), model_state['net']['blocks.%s.mlp.fc2.weight' % str(i)])
        setattr(net, 'blocks[%s].mlp.fc2.bias.data' % str(i), model_state['net']['blocks.%s.mlp.fc2.bias' % str(i)])
        return net


def load_pretrained_model(net, model_state):
    # load positional embedding
    net.pos_embed.data = model_state['net']['pos_embed']
    # load patch embedding
    setattr(net, 'patch_embed.proj.weight.data', model_state['net']['patch_embed.proj.weight'])
    setattr(net, 'patch_embed.proj.bias.data', model_state['net']['patch_embed.proj.bias'])
    # load attention blocks 
    net = load_attention_blocks(net, model_state)
    # load last norm layer 
    setattr(net, 'norm.weight.data', model_state['net']['norm.weight'])
    setattr(net, 'norm.bias.data', model_state['net']['norm.bias'])
    print('The pre-trained model is loaded')
    return net    



def _n2p(w, t=True):
    if w.ndim == 4 and w.shape[0] == w.shape[1] == w.shape[2] == 1:
        w = w.flatten()
    if t:
        if w.ndim == 4:
             w = w.transpose([3, 2, 0, 1])
        elif w.ndim == 3:
            w = w.transpose([2, 0, 1])
        elif w.ndim == 2:
            w = w.transpose([1, 0])
    return torch.from_numpy(w)


def load_imagenet_pretrained_attention_blocks(net, w):
    for i, block in enumerate(net.blocks):
        # initial norm layer
        setattr(net, 'blocks[%s].norm1.weight.data' % str(i), _n2p(w['Transformer/encoderblock_%s/LayerNorm_0/scale' % str(i)]))
        setattr(net, 'blocks[%s].norm1.bias.data' % str(i),  _n2p(w['Transformer/encoderblock_%s/LayerNorm_0/bias' % str(i)]))
        # attention qkv parameters 
        #setattr(net, 'blocks[%s].attn.qkv.weight.data' % str(i), _n2p(w['Transformer/encoderblock_%s/LayerNorm_0/scale' % str(i)]))
        setattr(net, 'blocks[%s].attn.qkv.weight.data' % str(i), torch.cat([_n2p(w['Transformer/encoderblock_%s/MultiHeadDotProductAttention_1/%s/kernel' % (str(i), n)], t=False).flatten(1).T for n in ('query', 'key', 'value')]))
        setattr(net, 'blocks[%s].attn.qkv.bias.data' % str(i), torch.cat([_n2p(w['Transformer/encoderblock_%s/MultiHeadDotProductAttention_1/%s/bias' % (str(i), n)], t=False).flatten(1).T for n in ('query', 'key', 'value')]))
        # attention projection layer
        setattr(net, 'blocks[%s].attn.proj.weight.data' % str(i), _n2p(w['Transformer/encoderblock_%s/MultiHeadDotProductAttention_1/out/kernel' % str(i)]))
        setattr(net, 'blocks[%s].attn.proj.bias.data' % str(i), _n2p(w['Transformer/encoderblock_%s/MultiHeadDotProductAttention_1/out/bias' % str(i)]))
        # last norm layer
        setattr(net, 'blocks[%s].norm2.weight.data' % str(i), _n2p(w['Transformer/encoderblock_%s/LayerNorm_2/scale' % str(i)]))
        setattr(net, 'blocks[%s].norm2.bias.data' % str(i), _n2p(w['Transformer/encoderblock_%s/LayerNorm_2/scale' % str(i)]))
        # fc layer 
        setattr(net, 'blocks[%s].mlp.fc1.weight.data' % str(i), _n2p(w['Transformer/encoderblock_%s/MlpBlock_3/Dense_0/kernel' % str(i)]))
        setattr(net, 'blocks[%s].mlp.fc1.bias.data' % str(i), _n2p(w['Transformer/encoderblock_%s/MlpBlock_3/Dense_0/bias' % str(i)]))
        setattr(net, 'blocks[%s].mlp.fc2.weight.data' % str(i), _n2p(w['Transformer/encoderblock_%s/MlpBlock_3/Dense_1/kernel' % str(i)]))
        setattr(net, 'blocks[%s].mlp.fc2.bias.data' % str(i), _n2p(w['Transformer/encoderblock_%s/MlpBlock_3/Dense_1/bias' % str(i)]))
    return net
             

def load_imagenet_pretrained_weight(net, args):
    """ 
    reference: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    Load weights from .npz checkpoints for official Google Brain Flax implementation
    """
    if args.model.find('vit_base_patch16_3D') != -1:
        # load numpy file of imagenet pretrained model
        w = np.load('/scratch/connectome/dhkdgmlghks/3DCNN_test/MAE_DDP/pretrained_model/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_384.npz')
        # load weight on attention blocks 
        net = load_imagenet_pretrained_attention_blocks(net, w)
        # load weight on last norm layer
        setattr(net, 'norm.weight.data', _n2p(w['Transformer/encoder_norm/scale']))
        setattr(net, 'norm.bias.data',_n2p(w['Transformer/encoder_norm/bias']))    

    elif args.model.find('vit_large_patch16_3D') != -1:
        # load numpy file of imagenet pretrained model
        w = np.load('/scratch/connectome/dhkdgmlghks/3DCNN_test/MAE_DDP/pretrained_model/L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_384.npz')
        # load weight on attention blocks 
        net = load_imagenet_pretrained_attention_blocks(net, w)
        # load weight on last norm layer
        setattr(net, 'norm.weight.data', _n2p(w['Transformer/encoder_norm/scale']))
        setattr(net, 'norm.bias.data',_n2p(w['Transformer/encoder_norm/bias'])) 
    
    elif args.model.find('vit_huge_patch14_3D') != -1:
        # load numpy file of imagenet pretrained model
        w = np.load('/scratch/connectome/dhkdgmlghks/3DCNN_test/MAE_DDP/pretrained_model/ViT-H_14.npz')
        # load weight on attention blocks 
        net = load_imagenet_pretrained_attention_blocks(net, w)
        # load weight on last norm layer
        setattr(net, 'norm.weight.data', _n2p(w['Transformer/encoder_norm/scale']))
        setattr(net, 'norm.bias.data',_n2p(w['Transformer/encoder_norm/bias']))
    print('The ImageNet21K pre-trained model is loaded')     
    del w 

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







