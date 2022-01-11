import sys

from matplotlib.pyplot import get
sys.path.insert(0, '..')
import glob

import os
import time

import pandas as pd

from torch.utils.data import DataLoader

from dataloaders.data_preprocessing import *
from dataloaders.data_loading import *

from envs.blockdrop_env import BlockDropEnv

import torch
from utils.util import makedir, print_separator, read_yaml, create_path, print_yaml, fix_random_seed, CLIreporter
import numpy as np
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

"""
def eval(environ, dataloader, tasks, policy=False, num_train_layers=None, hard_sampling=False, num_seg_cls=-1,
         eval_iter=10):
    batch_size = []
    records = {}
    val_metrics = {}
    if 'seg' in tasks:
        assert (num_seg_cls != -1)
        records['seg'] = {'mIoUs': [], 'pixelAccs': [],  'errs': [], 'conf_mat': np.zeros((num_seg_cls, num_seg_cls)),
                          'labels': np.arange(num_seg_cls)}
    if 'sn' in tasks:
        records['sn'] = {'cos_similaritys': []}
    if 'depth' in tasks:
        records['depth'] = {'abs_errs': [], 'rel_errs': [], 'sq_rel_errs': [], 'ratios': [], 'rms': [], 'rms_log': []}
    if 'keypoint' in tasks:
        records['keypoint'] = {'errs': []}
    if 'edge' in tasks:
        records['edge'] = {'errs': []}

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            #if eval_iter != -1:
            #    if batch_idx > eval_iter:
            #        break

            environ.set_inputs(batch)
            # seg_pred, seg_gt, pixelAcc, cos_similarity = environ.val(policy, num_train_layers, hard_sampling)
            metrics = environ.val2(policy, num_train_layers, hard_sampling)

            # environ.networks['mtl-net'].task1_logits
            # mIoUs.append(mIoU)
            if 'seg' in tasks:
                new_mat = confusion_matrix(metrics['seg']['gt'], metrics['seg']['pred'], records['seg']['labels'])
                assert (records['seg']['conf_mat'].shape == new_mat.shape)
                records['seg']['conf_mat'] += new_mat
                records['seg']['pixelAccs'].append(metrics['seg']['pixelAcc'])
                records['seg']['errs'].append(metrics['seg']['err'])
            if 'sn' in tasks:
                records['sn']['cos_similaritys'].append(metrics['sn']['cos_similarity'])
            if 'depth' in tasks:
                records['depth']['abs_errs'].append(metrics['depth']['abs_err'])
                records['depth']['rel_errs'].append(metrics['depth']['rel_err'])
                records['depth']['sq_rel_errs'].append(metrics['depth']['sq_rel_err'])
                records['depth']['ratios'].append(metrics['depth']['ratio'])
                records['depth']['rms'].append(metrics['depth']['rms'])
                records['depth']['rms_log'].append(metrics['depth']['rms_log'])
            if 'keypoint' in tasks:
                records['keypoint']['errs'].append(metrics['keypoint']['err'])
            if 'edge' in tasks:
                records['edge']['errs'].append(metrics['edge']['err'])
            batch_size.append(len(batch['img']))

    # overall_mIoU = (np.array(mIoUs) * np.array(batch_size)).sum() / sum(batch_size)
    if 'seg' in tasks:
        val_metrics['seg'] = {}
        jaccard_perclass = []
        for i in range(num_seg_cls):
            if not records['seg']['conf_mat'][i, i] == 0:
                jaccard_perclass.append(records['seg']['conf_mat'][i, i] / (np.sum(records['seg']['conf_mat'][i, :]) +
                                                                            np.sum(records['seg']['conf_mat'][:, i]) -
                                                                            records['seg']['conf_mat'][i, i]))

        val_metrics['seg']['mIoU'] = np.sum(jaccard_perclass) / len(jaccard_perclass)

        val_metrics['seg']['Pixel Acc'] = (np.array(records['seg']['pixelAccs']) * np.array(batch_size)).sum() / sum(
            batch_size)

        val_metrics['seg']['err'] = (np.array(records['seg']['errs']) * np.array(batch_size)).sum() / sum(batch_size)

    if 'sn' in tasks:
        val_metrics['sn'] = {}
        overall_cos = np.clip(np.concatenate(records['sn']['cos_similaritys']), -1, 1)

        angles = np.arccos(overall_cos) / np.pi * 180.0
        val_metrics['sn']['cosine_similarity'] = overall_cos.mean()
        val_metrics['sn']['Angle Mean'] = np.mean(angles)
        val_metrics['sn']['Angle Median'] = np.median(angles)
        val_metrics['sn']['Angle RMSE'] = np.sqrt(np.mean(angles ** 2))
        val_metrics['sn']['Angle 11.25'] = np.mean(np.less_equal(angles, 11.25)) * 100
        val_metrics['sn']['Angle 22.5'] = np.mean(np.less_equal(angles, 22.5)) * 100
        val_metrics['sn']['Angle 30'] = np.mean(np.less_equal(angles, 30.0)) * 100
        val_metrics['sn']['Angle 45'] = np.mean(np.less_equal(angles, 45.0)) * 100

    if 'depth' in tasks:
        val_metrics['depth'] = {}
        records['depth']['abs_errs'] = np.stack(records['depth']['abs_errs'], axis=0)
        records['depth']['rel_errs'] = np.stack(records['depth']['rel_errs'], axis=0)
        records['depth']['sq_rel_errs'] = np.stack(records['depth']['sq_rel_errs'], axis=0)
        records['depth']['ratios'] = np.concatenate(records['depth']['ratios'], axis=0)
        records['depth']['rms'] = np.concatenate(records['depth']['rms'], axis=0)
        records['depth']['rms_log'] = np.concatenate(records['depth']['rms_log'], axis=0)
        records['depth']['rms_log'] = records['depth']['rms_log'][~np.isnan(records['depth']['rms_log'])]
        val_metrics['depth']['abs_err'] = (records['depth']['abs_errs'] * np.array(batch_size)).sum() / sum(batch_size)
        val_metrics['depth']['rel_err'] = (records['depth']['rel_errs'] * np.array(batch_size)).sum() / sum(batch_size)
        val_metrics['depth']['sq_rel_err'] = (records['depth']['sq_rel_errs'] * np.array(batch_size)).sum() / sum(
            batch_size)
        val_metrics['depth']['sigma_1.25'] = np.mean(np.less_equal(records['depth']['ratios'], 1.25)) * 100
        val_metrics['depth']['sigma_1.25^2'] = np.mean(np.less_equal(records['depth']['ratios'], 1.25 ** 2)) * 100
        val_metrics['depth']['sigma_1.25^3'] = np.mean(np.less_equal(records['depth']['ratios'], 1.25 ** 3)) * 100
        val_metrics['depth']['rms'] = (np.sum(records['depth']['rms']) / len(records['depth']['rms'])) ** 0.5
        # val_metrics['depth']['rms_log'] = (np.sum(records['depth']['rms_log']) / len(records['depth']['rms_log'])) ** 0.5

    if 'keypoint' in tasks:
        val_metrics['keypoint'] = {}
        val_metrics['keypoint']['err'] = (np.array(records['keypoint']['errs']) * np.array(batch_size)).sum() / sum(
            batch_size)

    if 'edge' in tasks:
        val_metrics['edge'] = {}
        val_metrics['edge']['err'] = (np.array(records['edge']['errs']) * np.array(batch_size)).sum() / sum(
            batch_size)

    return val_metrics
"""

def train_and_eval_iter(environ, trainloader, valloader, current_iter, results_iter, optimizing_opt, opt):
    environ.train()
    start_time = time.time()

    # training model
    for batch_idx, batch in enumerate(trainloader, 0):
        environ.set_inputs(batch)
        environ.optimize(lambdas=optimizing_opt['lambdas'], is_policy=optimizing_opt['is_policy'], flag=optimizing_opt['flag'], num_train_layers=optimizing_opt['num_train_layers'], hard_sampling=optimizing_opt['hard_sampling'])
    
        # summarizing results from mini-batches 
        results = getattr(environ,'results')
        if opt['task']['cat_target']:
            for cat_target in opt['task']['cat_target']:
                results_iter[cat_target]['train']['loss'].append(results[cat_target]['loss'].item())       # if item is extracted by item() in the class of environ, loss could be not backpropagated. Thus it is extracted by item() in here.  
                results_iter[cat_target]['train']['ACC or MSE'].append(results[cat_target]['ACC or MSE'])
        if opt['task']['num_target']:
            for num_target in opt['task']['num_target']:
                results_iter[num_target]['train']['loss'].append(results[num_target]['loss'].item())       # if item is extracted by item() in the class of environ, loss could be not backpropagated. Thus it is extracted by item() in here.  
                results_iter[num_target]['train']['ACC or MSE'].append(results[num_target]['ACC or MSE'])
                        

    # validation
    environ.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(valloader):
            #if eval_iter != -1:
            #    if batch_idx > eval_iter:
            #        break

            environ.set_inputs(batch)
            environ.val(policy=False, num_train_layers=None, hard_sampling=False)

            # summarizing results from mini-batches
            results = getattr(environ, 'results')
            if opt['task']['cat_target']:
                for cat_target in opt['task']['cat_target']:
                    results_iter[cat_target]['val']['loss'].append(results[cat_target]['loss'].item())       # if item is extracted by item() in the class of environ, loss could be not backpropagated. Thus it is extracted by item() in here.  
                    results_iter[cat_target]['val']['ACC or MSE'].append(results[cat_target]['ACC or MSE'])
            if opt['task']['num_target']:
                for num_target in opt['task']['num_target']:
                    results_iter[num_target]['val']['loss'].append(results[num_target]['loss'].item())       # if item is extracted by item() in the class of environ, loss could be not backpropagated. Thus it is extracted by item() in here.  
                    results_iter[num_target]['val']['ACC or MSE'].append(results[num_target]['ACC or MSE'])          
    end_time = time.time()

                
    # save the model
    environ.save('latest', current_iter)
    
    return results_iter , end_time-start_time
    

def train():

    """
    한가지 고려사항: 원래 Adashare 코드에서는 train 과정에서 train data를 trainset, trainset1, trainset2 와 같이 3등분을 한다. 
    trainset은 warmup을 할 때에, trainset1은 learning phase에서, trainset2는 policy network를 업데이트할 때에. 
    그런데 이렇게 굳이 3등분 해서, 서로 독립적으로 network 학습에 사용하는 이유가 무엇인가? trainset 하나로 이 모든 걸 진행하면 안되나? 
    각 과정에서 독립적으로 처리해야하는 이유가 있는 것인가?
    => 원래 curriculum learning에서는 쉬운 샘플로 학습을 시키고 그 다음에 점차 어려운 샘플로 학습시켜 나간다는 점에서, 각 샘플을 독립적으로 다루는 것이 이해는 감. 
    그런데 우리의 실험 세팅에서도 굳이 나눌 필요가 있을까? 어차피 여기에서 curriculum learning을 사용하는 것은 policy of network를 추정하기 위한 parameter 학습 -> network weight 학습이지 않은가?
    
    Ver1. 일단 trainset을 총 3개로 나누어서 구현"""
    # ********************************************************************
    # ****************** create folders and print options ****************
    # ********************************************************************
    # read the yaml
    print_separator('READ YAML')
    opt, gpu_ids, _ = read_yaml()
    fix_random_seed(opt["seed"][0])
    create_path(opt)
    # print yaml on the screen
    lines = print_yaml(opt)
    for line in lines: print(line)
    # print to file
    with open(os.path.join(opt['paths']['log_dir'], opt['exp_name'], 'opt.txt'), 'w+') as f:
        f.writelines(lines)

    # ********************************************************************
    # ******************** Prepare the dataloaders ***********************
    # ********************************************************************
    
    # load the dataloader
    print_separator('DATA PREPROCSESSING AND CREATE DATALOADER')
    
    ## ========= get image, subject ID and target variables ========= ##
    if opt['dataload']['dataset'] == 'ABCD':
        os.chdir(opt['dataload']['img_dataroot_train'])
        image_files_train = glob.glob('*.npy')
        image_files_train = sorted(image_files_train)
        image_files_train = image_files_train[:30]

        os.chdir(opt['dataload']['img_dataroot_val'])
        image_files_val = glob.glob('*.npy')
        image_files_val = sorted(image_files_val)
        image_files_val = image_files_val[:30]

    if not opt['task']['cat_target'] and opt['task']['num_target']:
        raise ValueError('YOU SHOULD SELECT THE TARGET!')

    targets = opt['task']['targets']
    col_list = targets + ['subjectkey']

    ### get subject ID and target variables
    subject_data = pd.read_csv(opt['dataload']['label_dataroot'])
    subject_data = subject_data.loc[:,col_list]
    subject_data = subject_data.sort_values(by='subjectkey')
    subject_data = subject_data.dropna(axis = 0)
    subject_data = subject_data.reset_index(drop=True) # removing subject have NA values
  

    ## ========= Data Loading ========= ##
    if opt['task']['cat_target']:
        preprocessing_cat(subject_data, opt)

    if opt['task']['num_target']:
        preprocessing_num(subject_data, opt)

    imageFiles_labels_train = combining_image_target(image_files_train, subject_data, opt)
    imageFiles_labels_val = combining_image_target(image_files_val, subject_data, opt)

    partition = partitioning_loading(imageFiles_labels_train, opt)
    partition['val'] = loading(imageFiles_labels_val, opt)

    # count the class of labels (= output dimension of neural networks) 
    opt['task']['tasks_num_class'] = []
    for cat_label in opt['task']['cat_target']:
        opt['task']['tasks_num_class'].append(len(subject_data[cat_label].value_counts()))
    for num_label in opt['task']['cat_target']:
        opt['task']['tasks_num_class'].append(int(1))    

    print("Loading image file names as list is completed")
    
    print('size of training_warmup set: ', len(partition['train_warmup']))
    print('size of training set1: ', len(partition['train1']))
    print('size of training set2: ', len(partition['train2']))
    print('size of validation set: ', len(partition['val']))

    ## ====================================== ##  

    warminguploader = DataLoader(partition['train_warmup'],batch_size=opt['data_split']['train_batch_size'], drop_last=True, num_workers=24, shuffle=True)
    trainloader1 = DataLoader(partition['train1'], batch_size=opt['data_split']['train_batch_size'], drop_last=True, num_workers=24, shuffle=True)
    trainloader2 = DataLoader(partition['train2'], batch_size=opt['data_split']['train_batch_size'], drop_last=True, num_workers=24, shuffle=True)
    valloader = DataLoader(partition['val'], batch_size=opt['data_split']['val_batch_size'], drop_last=True, num_workers=24, shuffle=True)
    
    """
    이 네트워크의 경우 전체 모델의 alpha (policy)를 업데이트 하는 구간과 weight (neural net weights)를 업데이트 하는 구간이 나누어져 있다. 
    그래서 한 epoch (iteration) 동안은 alpha를 업데이트 했다가, 다른 한 epoch 동안에는 weight를 업데이트 하는 식으로 왔다갔다 모델을 학습시킨다. 
    이떄 전환 시점을 알려주는 option이 weight_iter_alternate와 alpha_iter_alternate이다. """
    opt['train']['weight_iter_alternate'] = opt['train'].get('weight_iter_alternate', len(trainloader1))  
    opt['train']['alpha_iter_alternate'] = opt['train'].get('alpha_iter_alternate', len(valloader))

    # ********************************************************************
    # ********************Create the environment *************************
    # ********************************************************************

    # create the model and the pretrain model
    print_separator('CREATE THE ENVIRONMENT')
    environ = BlockDropEnv(opt['paths']['log_dir'], opt['paths']['checkpoint_dir'], opt['exp_name'],
                           opt['task']['tasks_num_class'], opt['init_neg_logits'], gpu_ids[0],
                           opt['train']['init_temp'], opt['train']['decay_temp'], is_train=True, opt=opt)

    current_iter = 0
    current_iter_w, current_iter_a = 0, 0
    if opt['train']['resume']:
        current_iter = environ.load(opt['train']['which_iter'])
        environ.networks['mtl-net'].reset_logits()

    environ.define_optimizer(False)
    environ.define_scheduler(False)
    if torch.cuda.is_available():
        environ.cuda(gpu_ids)

    # ********************************************************************
    # ***************************  Training  *****************************
    # ********************************************************************
    flag = 'update_w'
    environ.fix_alpha()
    environ.free_w(opt['fix_BN'])
    
    best_value, best_iter = 0, 0
    best_metrics = None
    p_epoch = 0
    flag_warmup = True
    if opt['backbone'] == 'ResNet50':
        num_blocks = 18
    elif opt['backbone'] == 'ResNet101':
        num_blocks = 33
    elif opt['backbone'] == 'ResNet152':
        num_blocks = 50
    else:
        raise ValueError('Backbone %s is invalid' % opt['backbone'])

    with tqdm(total = opt['train']['total_iters']) as progress_bar:
        while current_iter < opt['train']['total_iters']:
            current_iter += 1
            if current_iter == 1:
                print_separator('START WARMING UP PHASE...')        

            # making template for reporting results
            results_iter = {}
            if opt['task']['cat_target']:
                for cat_target in opt['task']['cat_target']:
                    results_iter[cat_target] = {'train':{'loss':[], 'ACC or MSE':[]}, 'val':{'loss':[], 'ACC or MSE':[]}}
            if opt['task']['num_target']:
                for num_target in opt['task']['num_target']:
                    results_iter[num_target] = {'train':{'loss':[], 'ACC or MSE':[]}, 'val':{'loss':[], 'ACC or MSE':[]}}

            # warm up update weigths of networks
            if current_iter < opt['train']['warm_up_iters']:
                # options used for optimizing neural networks
                optimizing_opt = {}
                optimizing_opt['lambdas'] = opt['lambdas']
                optimizing_opt['is_policy'] = False
                optimizing_opt['flag'] = 'update_w'
                optimizing_opt['num_train_layers'] = None
                optimizing_opt['hard_sampling'] = False

                results_iter, time= train_and_eval_iter(environ=environ, trainloader=warminguploader, valloader=valloader,current_iter=current_iter, results_iter=results_iter, optimizing_opt=optimizing_opt, opt=opt)

                # summarizing and report results per epoch 
                CLIreporter(results_iter, opt)
                print('Epoch {}. Took {:2.2f} sec'.format(current_iter, time))

            elif current_iter == opt['train']['warm_up_iters']:
                print_separator('END WARMING UP PHASE...') 
                environ.save('warmup', current_iter)
                environ.fix_alpha()
                
                
            # After the warming up phase, learning phase is starting 
            else:
                print_separator('START LEARNING PHASE...') 
                
                if flag_warmup:
                    environ.define_optimizer(policy_learning=True)
                    environ.define_scheduler(True)
                    flag_warmup = False

                # Update the network weights
                if flag == 'update_w':
                    current_iter_w += 1

                    if opt['is_curriculum']:
                        num_train_layers = p_epoch // opt['curriculum_speed'] + 1
                    else:
                        num_train_layers = None

                    # options used for optimizing neural networks
                    optimizing_opt = {}
                    optimizing_opt['lambdas'] = opt['lambdas']
                    optimizing_opt['is_policy'] = opt['policy']
                    optimizing_opt['flag'] = flag
                    optimizing_opt['num_train_layers'] = num_train_layers
                    optimizing_opt['hard_sampling'] = opt['train']['hard_sampling']

                    results_iter, time = train_and_eval_iter(environ=environ, trainloader=trainloader1, valloader=valloader, current_iter=current_iter, results_iter=results_iter, optimizing_opt=optimizing_opt, opt=opt)
                    
                    # summarizing and report results per epoch 
                    CLIreporter(results_iter, opt)
                    print('Epoch {}. Took {:2.2f} sec'.format(current_iter, time))

                    # change the flag from 'update_weight' to 'update_alpha'
                    flag = 'update_alpha'
                    environ.fix_w()
                    environ.free_alpha()
                    print("## ====================== CHANGING THE LEARNING STATUR FROM **update_weigth** to **update_alpha** ====================== ##"  )

                # update the policy network
                elif flag == 'update_alpha':
                    current_iter_a += 1

                    if opt['is_curriculum']:
                        num_train_layers = p_epoch // opt['curriculum_speed'] + 1
                    else:
                        num_train_layers = None

                    # options used for optimizing neural networks
                    optimizing_opt = {}
                    optimizing_opt['lambdas'] = opt['lambdas']
                    optimizing_opt['is_policy'] = opt['policy']
                    optimizing_opt['flag'] = flag
                    optimizing_opt['num_train_layers'] = num_train_layers
                    optimizing_opt['hard_sampling'] = opt['train']['hard_sampling']

                    results_iter, time = train_and_eval_iter(environ=environ, trainloader=trainloader2, valloader=valloader, current_iter=current_iter, results_iter=results_iter, optimizing_opt=optimizing_opt, opt=opt)
                    
                    # print the distribution
                    dists = environ.get_policy_prob()
                    print(np.concatenate(dists, axis=-1))
                    p_epoch += 1

                    # summarizing and report results per epoch 
                    CLIreporter(results_iter, opt)
                    print('Epoch {}. Took {:2.2f} sec'.format(current_iter, time))

                    # change the flag from 'update_alpha' to 'update_weight'
                    flag = 'update_weight'
                    environ.fix_alpha()
                    environ.decay_temperature()
                    print("## ====================== CHANGING THE LEARNING STATUR FROM **update_alpha** to **update_weight** ====================== ##"  )

                else:
                    raise ValueError('flag %s is not recognized' % flag)
            progress_bar.update(1)

if __name__ == "__main__":
    train()
