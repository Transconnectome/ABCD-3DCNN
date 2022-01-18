import sys
sys.path.insert(0, '..')
import glob

import os
import time
import numpy as np

from torch.utils.data import DataLoader

from dataloaders.data_preprocessing import *
from dataloaders.data_loading import *

from envs.blockdrop_env import BlockDropEnv
import torch
from utils.util import print_separator, read_yaml, create_path, print_yaml,  fix_random_seed, CLIreporter, summarizing_results
from copy import deepcopy
from sklearn.metrics import confusion_matrix
from tqdm import tqdm


def train_and_eval_iter_fix_policy(environ, trainloader, valloader, current_iter, results_iter, opt):
    environ.train()
    start_time = time.time()

    # training model
    for batch_idx, batch in enumerate(trainloader, 0):
        environ.set_inputs(batch)
        environ.optimize_fix_policy(lambdas=opt['lambdas'])
    
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
            environ.set_inputs(batch)
            environ.val_fix_policy()

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

def _train(exp_id, opt, gpu_ids):

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

    tasks = opt['task']['targets']
    col_list = tasks + ['subjectkey']

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

    partition = {}
    partition['train'] = loading(imageFiles_labels_train, opt)
    partition['val'] = loading(imageFiles_labels_val, opt)

    # count the class of labels (= output dimension of neural networks) 
    opt['task']['tasks_num_class'] = []
    for cat_label in opt['task']['cat_target']:
        opt['task']['tasks_num_class'].append(len(subject_data[cat_label].value_counts()))
    for num_label in opt['task']['cat_target']:
        opt['task']['tasks_num_class'].append(int(1))    

    print("Loading image file names as list is completed")
    print('size of training set: ', len(partition['train']))
    print('size of validation set: ', len(partition['val']))

    trainloader = DataLoader(partition['train'], batch_size=opt['data_split']['train_batch_size'], drop_last=True, num_workers=24, shuffle=True)
    valloader = DataLoader(partition['val'], batch_size=opt['data_split']['val_batch_size'], drop_last=True, num_workers=24, shuffle=True)


    # ********************************************************************
    # ********************Create the environment *************************
    # ********************************************************************

    # create the model and the pretrain model
    print_separator('CREATE THE ENVIRONMENT')
    environ = BlockDropEnv(opt['paths']['log_dir'], opt['paths']['checkpoint_dir'], opt['exp_name'],
                           opt['task']['tasks_num_class'], opt['init_neg_logits'],
                           gpu_ids[0], opt['train']['init_temp'], opt['train']['decay_temp'],
                           is_train=True, opt=opt)

    current_iter = 0
    policy_label = 'Iter%s_rs%04d' % (opt['train']['policy_iter'], opt['seed'][exp_id])
    if opt['train']['retrain_resume']:
        current_iter = environ.load(opt['train']['which_iter'])
        if opt['policy_model'] == 'task-specific':
            environ.load_policy(policy_label)
    else:
        if opt['policy_model'] == 'task-specific':
            init_state = deepcopy(environ.get_current_state(0))      # getting the number of the last iterations from "train.py"
            if environ.check_exist_policy(policy_label):
                environ.load_policy(policy_label)
            else:
                environ.load(opt['train']['policy_iter'])
                dists = environ.get_policy_prob()
                overall_dist = np.concatenate(dists, axis=-1)
                print(overall_dist)
                environ.sample_policy(opt['train']['hard_sampling'])
                environ.save_policy(policy_label)

            if opt['retrain_from_pl']:
                environ.load(opt['train']['policy_iter'])
            else:
                environ.load_snapshot(init_state)

    if opt['policy_model'] == 'task-specific':
        policys = environ.get_current_policy()
        overall_policy = np.concatenate(policys, axis=-1)
        print(overall_policy)

    environ.define_optimizer(False)
    environ.define_scheduler(False)
    if torch.cuda.is_available():
        environ.cuda(gpu_ids)

    # ********************************************************************
    # ***************************  Training  *****************************
    # ********************************************************************
    environ.fix_alpha()
    environ.free_w(fix_BN=opt['fix_BN'])


    # setting the values for comparing results per epoch and the bset results
    best_value = {}
    best_iter = 0.0
    if opt['task']['cat_target']:
        for cat_target in opt['task']['cat_target']:
            best_value[cat_target] = 0.0       # best_value[cat_target] is compared to the validation Accuracy  
    if opt['task']['num_target']:
        for num_target in opt['task']['num_target']:
            best_value[num_target] = 100000.0       # best_value[num_target] is compared to the validation MSE Loss. This value should be set according to its cases


    opt['train']['retrain_total_iters'] = opt['train'].get('retrain_total_iters', opt['train']['total_iters'])


    with tqdm(total = opt['train']['total_iters']) as progress_bar:
        while current_iter < opt['train']['retrain_total_iters']:
            environ.train()
            current_iter += 1

            # making template for reporting results
            results_iter = {}
            if opt['task']['cat_target']:
                for cat_target in opt['task']['cat_target']:
                    results_iter[cat_target] = {'train':{'loss':[], 'ACC or MSE':[]}, 'val':{'loss':[], 'ACC or MSE':[]}}
            if opt['task']['num_target']:
                for num_target in opt['task']['num_target']:
                    results_iter[num_target] = {'train':{'loss':[], 'ACC or MSE':[]}, 'val':{'loss':[], 'ACC or MSE':[]}}
            
            # Training 
            results_iter, time= train_and_eval_iter_fix_policy(environ=environ, trainloader=trainloader, valloader=valloader, current_iter=current_iter, results_iter=results_iter, opt=opt)
            results_iter = summarizing_results(results_iter, opt)
            CLIreporter(results_iter, opt)
                        

            # comparing the current results to best results. If current results are bettter than the best results, then saving the best model 
            best_checkpoints_vote = 0
            if opt['task']['cat_target']:
                for cat_target in opt['task']['cat_target']:
                    pass
                #    print(results_iter[cat_target]['val']['ACC or MSE'])
                    if results_iter[cat_target]['val']['ACC or MSE'] >= best_value[cat_target]:
                        best_checkpoints_vote += 1
                        best_value[cat_target] = results_iter[cat_target]['val']['ACC or MSE']
            if opt['task']['num_target']:
                for num_target in opt['task']['num_target']:
                    pass
                #    print(results_iter[num_target]['val']['ACC or MSE'])
                    if results_iter[num_target]['val']['ACC or MSE'] <= best_value[num_target]:
                        best_checkpoints_vote += 1      
                        best_value[num_target] = results_iter[num_target]['val']['ACC or MSE']
                    

            if best_checkpoints_vote == len(opt['task']['targets']):
                best_iter = current_iter
                environ.save('retrain%03d_policyIter%s_best' % (exp_id, opt['train']['policy_iter']), current_iter)
                print("Best iteration until now is %d" % best_iter)


            # summarizing and report results per epoch 
            print('Epoch {}. Took {:2.2f} sec'.format(current_iter, time))
            progress_bar.update(1)



def train():
    # ********************************************************************
    # ****************** create folders and print options ****************
    # ********************************************************************
    print_separator('READ YAML')
    opt, gpu_ids, exp_ids = read_yaml()
    # fix_random_seed(opt["seed"])
    create_path(opt)
    # print yaml on the screen
    lines = print_yaml(opt)
    for line in lines: print(line)
    # print to file
    with open(os.path.join(opt['paths']['log_dir'], opt['exp_name'], 'opt.txt'), 'w+') as f:
        f.writelines(lines)

    best_results = {}
    for exp_id in exp_ids:
        fix_random_seed(opt["seed"][exp_id])
        # fix_random_seed(48)
        _train(exp_id, opt, gpu_ids)


if __name__ == "__main__":
    train()
