## ======= load module ======= ##
import os
import glob
import time
import datetime
import hashlib
from copy import deepcopy
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from tqdm.auto import tqdm

from utils.optimizer import CosineAnnealingWarmupRestarts
from utils.utils import argument_setting, seed_all, select_model, CLIreporter, save_exp_result, checkpoint_save, checkpoint_load
from dataloaders.dataloaders import make_dataset, make_dataloaders
from envs.experiments import train, validate, test
from envs.transfer import setting_transfer

import warnings
warnings.filterwarnings("ignore")

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system') # to prevent "Too many open files" error.

## ========= Helper Functions =============== ##
def setup_results(args):
    result = defaultdict(list)
    result['train_losses'] = defaultdict(list)
    result['train_accs'] = defaultdict(list)
    result['val_losses'] = defaultdict(list)
    result['val_accs'] = defaultdict(list)
    
    return result

    
def set_optimizer(args, net):
    if args.scheduler.lower() == 'cos':
        init_lr = 1e-9
    else:
        init_lr = args.lr
        
    if args.optim == 'SGD':
        optimizer = optim.SGD(params = filter(lambda p: p.requires_grad, net.parameters()),
                              lr=init_lr, momentum=0.9)
    elif args.optim == 'Adam':
        optimizer = optim.Adam(params = filter(lambda p: p.requires_grad, net.parameters()),
                               lr=init_lr, weight_decay=args.weight_decay)
    elif args.optim =='RAdam':
        optimizer = optim.RAdam(params = filter(lambda p: p.requires_grad, net.parameters()),
                                lr=init_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay)
    elif args.optim == 'AdamW':
        optimizer = optim.AdamW(params = filter(lambda p: p.requires_grad, net.parameters()),
                                lr=init_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay)
    else:
        raise ValueError('Invalid optimizer choice')
        
    return optimizer
    
    
def set_lr_scheduler(args, optimizer, len_dataloader):
    if args.scheduler == '':
        scheduler = None
    elif args.scheduler == 'on':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,'max', patience=10, factor=0.2, min_lr=1e-7)
    elif args.scheduler.lower() == 'cos':
        scheduler = CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=40, cycle_mult=1,
                                                  max_lr=args.lr, min_lr=1e-9, warmup_steps=10, gamma=0.75)
#        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2, eta_min=0)
    elif 'step' in args.scheduler:
        step_size = 80 if len(args.scheduler.split('_')) != 2 else int(args.scheduler.split('_')[1])        
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size, gamma=0.1)
    elif args.scheduler.lower() == 'onecycle':
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=args.epoch)
    else:
        raise Exception("Invalid scheduler option")
        
    return scheduler
    
    
def add_epoch_result(result, train_result, val_result): #230313change
    loss_acc_sum = defaultdict(int)
    for target_name in train_result['loss']:
        result['train_losses'][target_name].append(train_result['loss'][target_name])
        result['val_losses'][target_name].append(val_result['loss'][target_name])
        loss_acc_sum['train_loss'] += train_result['loss'][target_name]
        loss_acc_sum['val_loss'] += val_result['loss'][target_name]
        acc_done=0
        for acc in train_result:
            if 'MM' in args.model and 'contrastive_loss' in target_name:
                loss_acc_sum[f'val_metric'] += -val_result[acc][target_name]
            if ('contrastive_loss' not in target_name) and acc != 'loss':
                loss_acc_sum[f'train_{acc}'] += train_result[acc][target_name]
                loss_acc_sum[f'val_{acc}'] += val_result[acc][target_name]
                if acc_done == 0 and 'loss' not in acc:
                    loss_acc_sum[f'val_metric'] += val_result[acc][target_name]
                    result[f'train_accs'][target_name].append(train_result[acc][target_name])
                    result[f'val_accs'][target_name].append(val_result[acc][target_name])
                    acc_done=1
    
    return loss_acc_sum

    
def run_experiment(args, net, partition, result, mode):
    epoch_exp = args.epoch if mode == 'ALL' else args.epoch_FC
    if args.mode != 'pretraining':
        unfrozen_layers = args.unfrozen_layers if mode == 'ALL' else '0' #230313change
        setting_transfer(args, net.module, unfrozen_layers)
          
    scaler = torch.cuda.amp.GradScaler()
    optimizer = set_optimizer(args, net)
    scheduler = set_lr_scheduler(args, optimizer, len(partition['train']))
    
    trainloader, valloader = make_dataloaders(partition, args)

    best_loss_acc = {'train_loss': float('inf'), 'val_loss': float('inf'), 'val_metric': -float('inf')}
    checkpoint_dir = None
    patience = 0

    for epoch in tqdm(range(epoch_exp)):
        ts = time.time()
        curr_lr = optimizer.param_groups[0]['lr']
        net, train_result = train(net, trainloader, optimizer, scaler, args)
        val_result = validate(net, valloader, scheduler, args)

        ## sorting the results
        loss_acc_sum = add_epoch_result(result, train_result, val_result)
        if args.wandb:
            wandb.log(data=dict(loss_acc_sum, **{'learning_rate':curr_lr}), step=epoch+1)
        
        ## Check if best epoch, save the checkpoint and results, visualize the result.
        is_best = is_best_metric = (loss_acc_sum['val_metric'] > best_loss_acc['val_metric'])
        is_best_loss = (loss_acc_sum['val_loss'] < best_loss_acc['val_loss'])
        if is_best_metric:
            result['best_epoch'] = epoch
            best_loss_acc.update(loss_acc_sum)
            if args.wandb:
                wandb.summary.update({'best_'+k:v for k, v in best_loss_acc.items()})
            checkpoint_dir = checkpoint_save(net, epoch, args)
        patience = (patience+1) * (not is_best)
        save_exp_result(vars(args).copy(), result) 
        metric = list(filter(lambda x: 'loss' not in x, train_result.keys()))[0]
        CLIreporter(train_result['loss'], train_result[metric],
                    val_result['loss'], val_result[metric])
        
        te = time.time()
        print(f"Epoch {epoch+1}. Current learning rate {curr_lr:.4e}. Took {te-ts:2.2f} sec. {is_best*'Best epoch'}")

        ## Early-Stopping
        if args.early_stopping != None and patience >= args.early_stopping and epoch >= 50:
            print(f"*** Validation Loss patience reached {args.early_stopping} epochs. Early Stopping Experiment ***")
            break
                
    if args.debug:
        return result, None
    
    checkpoint_save(net, args.epoch, args)

    opt = '' if mode == 'ALL' else '_FC'
    result[f'best_val_loss{opt}'] = best_loss_acc['val_loss']
    result[f'best_train_loss{opt}'] = best_loss_acc['train_loss']
        
    return result, checkpoint_dir


## ========= Experiment =============== ##
def experiment(partition, subject_data, args):
    # selecting a model
    net = select_model(subject_data, args)
    
    # loading pretrained model if transfer option is given
    if args.mode == 'pretraining':
        print("*** Model setting for learning from scratch ***")
    else:
        print("*** Model setting for transfer learning & fine tuning *** \n")
        model_dir = glob.glob(f'result/model/*{args.load}*')[0]
        print(f"Loaded {model_dir[:-4]}")
        net = checkpoint_load(net, model_dir, args) #0313change
    
    # setting a DataParallel and model on GPU
    if args.sbatch == "True":
        devices = list(range(torch.cuda.device_count()))
    elif args.gpus:
        devices = args.gpus
    else:
        raise ValueError("GPU DEVICE IDS SHOULD BE ASSIGNED")
    net = nn.DataParallel(net, device_ids = devices)        
    net.to(f'cuda:{net.device_ids[0]}')
    
    wandb.watch(net)
    
    # setting for results' DataFrame
    result = setup_results(args)
    
    # training a model
    print("*** Start training a model *** \n")
    if args.epoch_FC != 0:
        print("*** Transfer Learning - Training FC layers *** \n")
        result, _ = run_experiment(args, net, partition, result, 'FC')
                
        print(f"Adjust learning rate for Training unfrozen layers from {args.lr} to {args.lr*args.lr_adjust}")
        args.lr *= args.lr_adjust
        result['lr_adjusted'] = args.lr
            
    print("*** Training unfrozen layers *** \n")
    result, checkpoint_dir = run_experiment(args, net, partition, result, 'ALL')
                    
    # testing a model
    if args.debug:
        return vars(args), result
    
    print("\n*** Start testing a model *** \n")
    net.to('cpu')
    torch.cuda.empty_cache()

    net = checkpoint_load(net, checkpoint_dir, args, test=True) #230313change
    if args.sbatch == 'True':
        net.cuda()
    else:
        net.to(f'cuda:{args.gpus[0]}')
    test_acc, confusion_matrices = test(net, partition, args)
    result['test_acc'] = test_acc
    print(f"===== Test result for {args.exp_name} =====") 
    print(test_acc)
    if args.wandb:
            wandb.log(data=test_acc, step=0)

    if confusion_matrices != None:
        print("===== Confusion Matrices =====")
        print(confusion_matrices,'\n')
        result['confusion_matrices'] = confusion_matrices
        
    return vars(args), result
## ==================================== ##


if __name__ == "__main__":
    ## ========= Setting ========= ##
    args = argument_setting()
    args.save_dir = os.getcwd() + '/result'
    
    # Seed number
    args.seed = 1234
    seed_all(args.seed)
    
    # Set exp_name
    time_hash = datetime.datetime.now().time()
    hash_key = hashlib.sha1(str(time_hash).encode()).hexdigest()[:6]
    args.exp_name = args.exp_name + f'_{hash_key}'

    ## ========= Run Experiment and saving result ========= ##    
    # Initialize wandb
    if args.wandb:
        wandb.init(project=f'{args.dataset}_{str(args.cat_target+args.num_target)}',
                   group=f'modality-{str(args.data_type)}_split-{args.balanced_split}',
                   name=args.exp_name, config=args)
    
    # Run Experiment
    print(f"*** Experiment {args.exp_name} Start ***")
    partition, subject_data = make_dataset(args)
    setting, result = experiment(partition, subject_data, deepcopy(args))
    print("===== Experiment Setting Report =====")
    print(args)
    
    # Save result
    if args.debug:
        quit()
    
    save_exp_result(setting, result)
    print("*** Experiment Done ***\n")
