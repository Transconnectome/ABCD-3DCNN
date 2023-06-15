## ======= load module ======= ##
import os
import glob
import time
import datetime
import hashlib
from copy import deepcopy
from collections import defaultdict

import optuna
from optuna.trial import TrialState
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm

from utils.optimizer import CosineAnnealingWarmupRestarts
from utils.utils import argument_setting, seed_all, select_model, CLIreporter, save_exp_result, checkpoint_save, checkpoint_load
from dataloaders.dataloaders import make_dataset, make_dataloaders
from envs.experiments import train, validate, test
from envs.transfer import setting_transfer

import warnings
warnings.filterwarnings("ignore")

## ========= Helper Functions =============== ##
def setup_results(args):
    train_losses = defaultdict(list)
    train_accs = defaultdict(list)
    val_losses = defaultdict(list)
    val_accs = defaultdict(list)

    result = {}
    result['train_losses'] = train_losses
    result['train_accs'] = train_accs
    result['val_losses'] = val_losses
    result['val_accs'] = val_accs
    
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
    elif 'on' in args.scheduler.lower() or args.scheduler == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,'max', patience=10, factor=0.1, min_lr=1e-8)
    elif args.scheduler.lower() == 'cos' or args.scheduler == 'ReduceLROnCosineAnnealingWarmupRestartsPlateau':
        scheduler = CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=15, cycle_mult=2,
                                                  max_lr=args.lr, min_lr=1e-9, warmup_steps=10, gamma=0.75)
#        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2, eta_min=0)
    elif args.scheduler == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = args.epoch)
    elif 'step' in args.scheduler or args.scheduler == 'StepLR':
        step_size = 40 if len(args.scheduler.split('_')) != 2 else int(args.scheduler.split('_')[1])        
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size, gamma=0.1)
    elif args.scheduler == 'OneCycleLR':
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=args.epoch)
    else:
        raise Exception("Invalid scheduler option")
        
    return scheduler
    
    
def add_epoch_result(result, train_loss, train_acc, val_loss, val_acc): #230313change
    loss_acc_sum = {'train_loss':0, 'val_loss':0, 'val_acc':0}
    for target_name in train_loss:
        result['train_losses'][target_name].append(train_loss[target_name])
        result['val_losses'][target_name].append(val_loss[target_name])
        loss_acc_sum['train_loss'] += train_loss[target_name]
        loss_acc_sum['val_loss'] += val_loss[target_name]
        if 'contrastive_loss' not in target_name:
            result['train_accs'][target_name].append(train_acc[target_name])
            result['val_accs'][target_name].append(val_acc[target_name])
            loss_acc_sum['val_acc'] += val_acc[target_name]
            
    return loss_acc_sum

    
def run_experiment(args, net, partition, result, mode, trial=None):
    if trial:
        # args.optim = trial.suggest_categorical("optimizer", ["AdamW", "SGD"])
        args.weight_decay = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        args.lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        # args.scheduler = trial.suggest_categorical("scheduler", ["ReduceLROnPlateau", "ReduceLROnCosineAnnealingWarmupRestartsPlateau"])
        
    epoch_exp = args.epoch if mode == 'ALL' else args.epoch_FC
    if args.mode != 'pretraining':
        unfrozen_layers = args.unfrozen_layers if mode == 'ALL' else '0' #230313change
        setting_transfer(args, net.module, unfrozen_layers)
          
    scaler = torch.cuda.amp.GradScaler()
    optimizer = set_optimizer(args, net)
    scheduler = set_lr_scheduler(args, optimizer, len(partition['train']))
    
    trainloader, valloader = make_dataloaders(partition, args)

    val_metric = 'loss' if 'MM' in args.model else 'acc'
    best_loss_acc = {'train_loss': float('inf'), 'val_loss': float('inf'), 'val_acc': -float('inf')}
    patience = 0

    for epoch in tqdm(range(epoch_exp)):
        curr_lr = optimizer.param_groups[0]['lr']
        ts = time.time()
        net, train_loss, train_acc = train(net, trainloader, optimizer, scaler, args)
        val_loss, val_acc = validate(net, valloader, scheduler, args)
        te = time.time()

        ## sorting the results
        loss_acc_sum = add_epoch_result(result, train_loss, train_acc, val_loss, val_acc) #230313change
        if args.wandb:
            wandb.log(data=(loss_acc_sum | {'learning_rate':curr_lr}), step=epoch+1)
        if val_metric == 'loss':
            is_best = (loss_acc_sum['val_loss'] < best_loss_acc['val_loss'])
        else:
            is_best = (loss_acc_sum['val_acc'] > best_loss_acc['val_acc']) 
        
        ## Check if best epoch, save the checkpoint and results, visualize the result.
        if is_best:
            result['best_epoch'] = epoch
            best_loss_acc.update(loss_acc_sum)
            if args.wandb:
                wandb.summary.update({'best_'+k:v for k, v in best_loss_acc.items()})
            checkpoint_dir = checkpoint_save(net, epoch, args)
        patience = (patience+1) * (not is_best)
        save_exp_result(vars(args).copy(), result) 
        CLIreporter(train_loss, train_acc, val_loss, val_acc)
        print(f"Epoch {epoch+1}. Current learning rate {curr_lr:.4e}. Took {te-ts:2.2f} sec. {is_best*'Best epoch'}")
        
        ## Optuna result report & prune        
        trial.report(loss_acc_sum['val_loss'], epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

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
def experiment(partition, subject_data, args, trial=None): # trial for optuna
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
    if '2.0' in torch.__version__:
        net = torch.compile(net)
    wandb.watch(net)
    
    # setting for results' DataFrame
    result = setup_results(args)
    
    # training a model
    print("*** Start training a model *** \n")
    if args.epoch_FC != 0:
        print("*** Transfer Learning - Training FC layers *** \n")
        result, _ = run_experiment(args, net, partition, result, 'FC', trial)
                
        print(f"Adjust learning rate for Training unfrozen layers from {args.lr} to {args.lr*args.lr_adjust}")
        args.lr *= args.lr_adjust
        result['lr_adjusted'] = args.lr
            
    print("*** Training unfrozen layers *** \n")
    result, checkpoint_dir = run_experiment(args, net, partition, result, 'ALL', trial)
                    
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

    if confusion_matrices != None:
        print("===== Confusion Matrices =====")
        print(confusion_matrices,'\n')
        result['confusion_matrices'] = confusion_matrices
        
    return vars(args), result

def objective(trial):
    ## ========= Args Setting ========= ##
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
    # Run Experiment
    if args.wandb:
        wandb.init(project=f'{args.dataset}_{str(args.cat_target+args.num_target)}',
                   group=f'modality-{str(args.data_type)}_split-{args.balanced_split}_{trial.number}',
                   name=args.exp_name, config=args)
        
    print(f"*** Experiment {args.exp_name} Start ***")
    partition, subject_data = make_dataset(args)
    setting, result = experiment(partition, subject_data, deepcopy(args), trial) # change for optuna
    print("===== Experiment Setting Report =====")
    print(args)
    
    save_exp_result(setting, result)
    if 'test_acc' not in result:
        pths_need_pruned = glob.glob(f'{args.save_dir}/model/{args.exp_name}*pth')
        try:
            list(map(lambda x: os.remove(x), pths_need_pruned))
        except:
            print('removing pruned pths failed')
    print("*** Experiment Done ***\n")
    
    return result['best_val_loss'] # for minimizing direction of optuna.create_study

## ==================================== ##


if __name__ == "__main__":
    ## ========= Optuna Setting ========= ##
    study = optuna.create_study(direction="minimize") # minimize loss
    study.optimize(objective, n_trials=300)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
