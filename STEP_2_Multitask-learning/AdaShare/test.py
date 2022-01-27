import sys
sys.path.insert(0, '..')
import glob

import os
import numpy as np

from torch.utils.data import DataLoader

from dataloaders.data_preprocessing import *
from dataloaders.data_loading import *

from envs.blockdrop_env import BlockDropEnv
import torch
from utils.util import print_separator, read_yaml, create_path, print_yaml, fix_random_seed, CLIreporter, summarizing_results, making_results_template, save_exp_results, ImageList_loading
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

def eval_iter_fix_policy(environ, testloader, results_iter, opt):
    environ.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(testloader):
            environ.set_inputs(batch)
            environ.val_fix_policy()

            # summarizing results from mini-batches
            results = getattr(environ, 'results')
            if opt['task']['cat_target']:
                for cat_target in opt['task']['cat_target']:
                    results_iter[cat_target]['test']['loss'].append(results[cat_target]['loss'])       # if item is extracted by item() in the class of environ, loss could be not backpropagated. Thus it is extracted by item() in here.  
                    results_iter[cat_target]['test']['ACC or MSE'].append(results[cat_target]['ACC or MSE'])
            if opt['task']['num_target']:
                for num_target in opt['task']['num_target']:
                    results_iter[num_target]['test']['loss'].append(results[num_target]['loss'])       # if item is extracted by item() in the class of environ, loss could be not backpropagated. Thus it is extracted by item() in here.  
                    results_iter[num_target]['test']['ACC or MSE'].append(results[num_target]['ACC or MSE'])          

    
    return results_iter


def test():
    # # ********************************************************************
    # # ****************** create folders and print options ****************
    # # ********************************************************************
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


    # ********************************************************************
    # ******************** Prepare the dataloaders ***********************
    # ********************************************************************
    # load the dataloader
    print_separator('DATA PREPROCSESSING AND CREATE DATALOADER')
    os.chdir(opt['dataload']['img_dataroot']) # important line
    
    ## ========= get image, subject ID and target variables ========= ##
    if opt['dataload']['dataset'] == 'ABCD':
        image_files_test = ImageList_loading(os.path.join(opt['dataload']['img_dataroot'],'image_test_SubjectList.txt'))
        image_files_test = sorted(image_files_test)
        #image_files_test = image_files_test[:30]


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

    imageFiles_labels_test = combining_image_target(image_files_test, subject_data, opt)

    partition = {}
    partition['test'] = loading(imageFiles_labels_test, opt)

    # count the class of labels (= output dimension of neural networks) 
    opt['task']['tasks_num_class'] = []
    if opt['task']['cat_target']:
        for cat_label in opt['task']['cat_target']:
            opt['task']['tasks_num_class'].append(len(subject_data[cat_label].value_counts()))
    if opt['task']['num_target']:
        for num_label in opt['task']['num_target']:
            opt['task']['tasks_num_class'].append(int(1))    

    print("Loading image file names as list is completed")
    print('size of test set: ', len(partition['test']))

    testloader = DataLoader(partition['test'], batch_size=opt['data_split']['train_batch_size'], drop_last=True, num_workers=24, shuffle=True)


    # ********************************************************************
    # ********************Create the environment *************************
    # ********************************************************************
    # create the model and the pretrain model
    print_separator('CREATE THE ENVIRONMENT')
    environ = BlockDropEnv(opt['paths']['log_dir'], opt['paths']['checkpoint_dir'], opt['exp_name'],
                           opt['task']['tasks_num_class'], device=gpu_ids[0], is_train=False, opt=opt)

    current_iter = environ.load('retrain%03d_policyIter%s_best' % (exp_ids[0], opt['train']['policy_iter']))

    print('Evaluating the snapshot saved at %d iter' % current_iter)

    policy_label = 'Iter%s_rs%04d' % (opt['train']['policy_iter'], opt['seed'][exp_ids[0]])

    if environ.check_exist_policy(policy_label):
        environ.load_policy(policy_label)

    policys = environ.get_current_policy()
    overall_policy = np.concatenate(policys, axis=-1)
    print(overall_policy)

    if torch.cuda.is_available():
        environ.cuda(gpu_ids)

    # creating final results template 
    results_final = making_results_template(opt, mode='test')


    # ********************************************************************
    # ***************************  Test  *********************************
    # ********************************************************************
    # making template for reporting results
    results_iter = {}
    if opt['task']['cat_target']:
        for cat_target in opt['task']['cat_target']:
            results_iter[cat_target] = {'test':{'loss':[], 'ACC or MSE':[]}}
    if opt['task']['num_target']:
        for num_target in opt['task']['num_target']:
            results_iter[num_target] = {'test':{'loss':[], 'ACC or MSE':[]}}
    
    # Test/ Inference 
    results_iter = eval_iter_fix_policy(environ, testloader, results_iter=results_iter, opt=opt)
    results_iter, results_final = summarizing_results(opt, results_iter, results_final, test=True)
    CLIreporter(results_iter, opt, test=True)


    # *********************************************************************
    # ************************  Saving Results  ***************************
    # *********************************************************************
    # saving results as json file
    save_exp_results(results_final, opt, mode='test')


if __name__ == "__main__":
    test()