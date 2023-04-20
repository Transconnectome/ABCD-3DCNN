import time 
from tqdm import tqdm
import os 

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, utils
import torch.optim as optim
from utils.optimizer import SGDW

from envs.loss_functions import calculating_loss, calculating_eval_metrics, MixUp_loss, CutMix_loss, C_MixUp_loss
from utils.utils import combine_pred_subjid, combine_emb_subjid

from utils.utils import CLIreporter, checkpoint_save, checkpoint_load, MOPED_network
from utils.lr_scheduler import *
from utils.early_stopping import * 


### ========= Train,Validate, and Test ========= ###
'''The process of calcuating loss and accuracy metrics is as follows.
   1) sequentially calculate loss and accuracy metrics of target labels with for loop.
   2) store the result information with dictionary type.
   3) return the dictionary, which form as {'cat_target':value, 'num_target:value}
   This process is intended to easily deal with loss values from each target labels.'''


'''All of the loss from predictions are summated and this loss value is used for backpropagation.'''
# define training step
def train(net,partition,optimizer, global_steps, args):
    '''GradScaler is for calculating gradient with float 16 type'''
    scaler = torch.cuda.amp.GradScaler()

    trainloader = torch.utils.data.DataLoader(partition['train'],
                                             batch_size=args.batch_size,
                                             shuffle=True,
                                             num_workers=12)

    net.train()

    if args.mixup != 0.0:
        loss_fn = MixUp_loss(args=args, device=f'cuda:{net.device_ids[0]}', beta=args.beta) # 
    elif args.cutmix != 0.0 : 
        loss_fn = CutMix_loss(args=args, device=f'cuda:{net.device_ids[0]}', beta=args.beta)
    elif args.c_mixup != 0.0: 
        loss_fn = C_MixUp_loss(args=args, device=f'cuda:{net.device_ids[0]}', beta=args.beta, cutmix=True)
    elif args.manifold_mixup != 0.0:
        loss_fn = MixUp_loss(args=args, device=f'cuda:{net.device_ids[0]}', beta=args.beta) # 
    else:
        loss_fn = calculating_loss(device=f'cuda:{net.device_ids[0]}', args=args)
    eval_metrices_log = calculating_eval_metrics(args=args)


    optimizer.zero_grad()
    for i, data in enumerate(trainloader,0):
        image, targets = data
        image = image.to(f'cuda:{net.device_ids[0]}')

        if args.mixup != 0.0:
            r = np.random.rand(1) 
            if r < args.mixup: 
                image, label_a, label_b, lam = loss_fn.mixup_data(image, targets) 
                with torch.cuda.amp.autocast():
                    output = net(image)
                loss, train_loss = loss_fn(output, label_a, label_b, lam)
            else: 
                with torch.cuda.amp.autocast():
                    output = net(image)
                loss, train_loss = loss_fn(output, targets)
        elif args.cutmix != 0.0: 
            r = np.random.rand(1) 
            if r < args.cutmix: 
                image, label_a, label_b, lam = loss_fn.generate_mixed_sample(image, targets) 
                with torch.cuda.amp.autocast():
                    output = net(image)
                    loss, train_loss = loss_fn(output, label_a, label_b, lam)
            else: 
                with torch.cuda.amp.autocast():
                    output = net(image)
                    loss, train_loss = loss_fn(output, targets)
        elif args.c_mixup!= 0.0: 
            r = np.random.rand(1) 
            if r < args.c_mixup: 
                image, label_a, label_b, lam = loss_fn.mixup_data(image, targets) 
                with torch.cuda.amp.autocast():
                    output = net(image)
                loss, train_loss = loss_fn(output, label_a, label_b, lam)
            else: 
                with torch.cuda.amp.autocast():
                    output = net(image)
                loss, train_loss = loss_fn(output, targets)            
        elif args.manifold_mixup != 0.0:
            """
            ref: https://github.com/vikasverma1077/manifold_mixup/tree/870ef77caaa5092144d82c56f26b07b29eefabec
            """
            r = np.random.rand(1) 
            if r < args.manifold_mixup: 
                y = targets[loss_fn.target].to(f'cuda:{net.device_ids[0]}')
                with torch.cuda.amp.autocast():
                    lam = loss_fn.get_lambda(tensor=True).to(f'cuda:{net.device_ids[0]}')  
                    output, label_a, label_b = net(image, y=y,target=loss_fn.target, manifold_mixup=True, lam=lam.tile((len(net.device_ids),))) # because of Data Parallel, replicate lambda. the size of lambda = (# of device,)
                    loss, train_loss = loss_fn(output, label_a, label_b, lam)    
            else: 
                with torch.cuda.amp.autocast():
                    output = net(image)
                    loss, train_loss = loss_fn(output, targets)  
            
        else: 
            
            # feed forward network with floating point 16
            with torch.cuda.amp.autocast():
                output = net(image)
            loss, train_loss = loss_fn(targets, output)
        eval_metrices_log.store(output, targets)
        if args.accumulation_steps:
            loss = loss / args.accumulation_steps
            # pytorch 2.0
            scaler.scale(loss).sum().backward()
            #scaler.scale(loss).backward()
            if  (i + 1) % args.accumulation_steps == 0:
                # gradient clipping 
                if args.gradient_clipping == True:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5, error_if_nonfinite=False)   
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                global_steps += 1 
            


    # calculating total loss and acc of separate mini-batch
    if args.cat_target:
        for cat_target in args.cat_target:
            train_loss[cat_target] = np.mean(train_loss[cat_target])

    if args.num_target:
        for num_target in args.num_target:
            train_loss[num_target] = np.mean(train_loss[num_target])


    return net, train_loss, eval_metrices_log.get_result(),  global_steps


# define validation step
def validate(net,partition,scheduler,args):
    valloader = torch.utils.data.DataLoader(partition['val'],
                                           batch_size=args.batch_size,
                                           shuffle=True,
                                           num_workers=12)

    net.eval()

    loss_fn = calculating_loss(device=f'cuda:{net.device_ids[0]}', args=args)
    eval_metrices_log = calculating_eval_metrics(args=args)

    with torch.no_grad():
        for i, data in enumerate(valloader,0):
            image, targets = data
            image = image.to(f'cuda:{net.device_ids[0]}')

            with torch.cuda.amp.autocast():
                output = net(image)
            loss, val_loss = loss_fn(targets, output)
            eval_metrices_log.store(output, targets)

    if args.cat_target:
        for cat_target in args.cat_target:
            val_loss[cat_target] = np.mean(val_loss[cat_target])

    if args.num_target:
        for num_target in args.num_target:
            val_loss[num_target] = np.mean(val_loss[num_target])


    # learning rate scheduler
    #scheduler.step(sum(val_acc.values())) #if you want to use ReduceLROnPlateau lr scheduler, activate this line and deactivate the below line 
    scheduler.step()

    return val_loss, eval_metrices_log.get_result()



# define test step
def test(net,partition,args):
    # flag for data shuffle 
    data_shuffle = False 

    assert data_shuffle == False 
    testloader = torch.utils.data.DataLoader(partition['test'],
                                            batch_size=args.batch_size,
                                            shuffle=data_shuffle,
                                            num_workers=12)

    net.eval()

    device = 'cuda:0'

    loss_fn = calculating_loss(device=device, args=args)
    eval_metrices_log = calculating_eval_metrics(args=args)

    with torch.no_grad():
        for i, data in enumerate(tqdm(testloader),0):
            image, targets = data
            image = image.to(device)

            with torch.cuda.amp.autocast():
                output = net(image)
            loss, test_loss = loss_fn(targets, output)
            eval_metrices_log.store(output, targets)
    

    if args.get_predicted_score:
        if args.cat_target:
            for cat_target in args.cat_target: 
                eval_metrices_log.pred_score[cat_target] = eval_metrices_log.pred[cat_target].squeeze(-1).tolist()
                eval_metrices_log.pred_score[cat_target] = combine_pred_subjid(eval_metrices_log.pred_score[cat_target], partition['test'].image_files)
                eval_metrices_log.pred_score["predicted_%s" % cat_target] = eval_metrices_log.pred_score.pop(cat_target)
        if args.num_target:
            for num_target in args.num_target: 
                eval_metrices_log.pred_score[num_target] = eval_metrices_log.pred[num_target].squeeze(-1).tolist()
                eval_metrices_log.pred_score[num_target] = combine_pred_subjid(eval_metrices_log.pred_score[num_target], partition['test'].image_files)
                eval_metrices_log.pred_score["predicted_%s" % num_target] = eval_metrices_log.pred_score.pop(num_target)
        return eval_metrices_log.get_result(), None, eval_metrices_log.pred_score
    else: 
        return eval_metrices_log.get_result(), None, None
## ============================================ ##



## ========= Experiment =============== ##
def experiment(partition, subject_data, save_dir, args): #in_channels,out_dim
    targets = args.cat_target + args.num_target

    # DenseNet
    if args.model == 'densenet3D121':
        import models.densenet3d as densenet3d #model script
        from envs.experiments import train, validate, test 
        net = densenet3d.densenet3D121(subject_data, args)
    elif args.model == 'densenet3D169':
        import models.densenet3d as densenet3d #model script
        from envs.experiments import train, validate, test 
        net = densenet3d.densenet3D169(subject_data, args) 
    elif args.model == 'densenet3D201':
        import models.densenet3d as densenet3d #model script
        from envs.experiments import train, validate, test 
        net = densenet3d.densenet3D201(subject_data, args)
    # DenseNet with CBAM module
    if args.model == 'densenet3D121_cbam':
        import models.densenet3d_cbam as densenet3d_cbam #model script
        from envs.experiments import train, validate, test 
        net = densenet3d_cbam.densenet3D121_cbam(subject_data, args)
    elif args.model == 'densenet3D169_cbam':
        import models.densenet3d_cbam as densenet3d_cbam #model script
        from envs.experiments import train, validate, test 
        net = densenet3d_cbam.densenet3D169_cbam(subject_data, args) 
    elif args.model == 'densenet3D201_cbam':
        import models.densenet3d_cbam as densenet3d_cbam #model script
        from envs.experiments import train, validate, test 
        net = densenet3d_cbam.densenet3D201_cbam(subject_data, args)
    # Bayesian (variational) DenseNet
    elif args.model == 'variational_densenet3D121':
        import models.variational_densenet3d as variational_densenet3d
        from envs.bayesian_experiments import train, validate, test 
        if args.moped: 
            import models.densenet3d as densenet3d
            bayes_net = variational_densenet3d.densenet3D121(subject_data, args)
            det_net = densenet3d.densenet3D121(subject_data, args)
        else: 
            net = variational_densenet3d.densenet3D121(subject_data, args)
    elif args.model == 'variational_densenet3D169':
        import models.variational_densenet3d as variational_densenet3d
        from envs.bayesian_experiments import train, validate, test 
        if args.moped: 
            import models.densenet3d as densenet3d
            bayes_net = variational_densenet3d.densenet3D169(subject_data, args)
            det_net = densenet3d.densenet3D169(subject_data, args)
        else: 
            net = variational_densenet3d.densenet3D169(subject_data, args)
    elif args.model == 'variational_densenet3D201':
        import models.variational_densenet3d as variational_densenet3d
        from envs.bayesian_experiments import train, validate, test 
        if args.moped: 
            import models.densenet3d as densenet3d
            bayes_net = variational_densenet3d.densenet3D201(subject_data, args)
            det_net = densenet3d.densenet3D201(subject_data, args)
        else: 
            net = variational_densenet3d.densenet3D201(subject_data, args)
    # Bayesian (flipout) DenseNet
    elif args.model == 'flipout_densenet3D121':
        import models.flipout_densenet3d as flipout_densenet3d
        from envs.bayesian_experiments import train, validate, test 
        if args.moped: 
            import models.densenet3d as densenet3d
            bayes_net = flipout_densenet3d.densenet3D121(subject_data, args)
            det_net = densenet3d.densenet3D121(subject_data, args)
        else: 
            net = flipout_densenet3d.densenet3D121(subject_data, args)
    elif args.model == 'flipout_densenet3D169':
        import models.flipout_densenet3d as flipout_densenet3d
        from envs.bayesian_experiments import train, validate, test 
        if args.moped: 
            import models.densenet3d as densenet3d
            bayes_net = flipout_densenet3d.densenet3D169(subject_data, args)
            det_net = densenet3d.densenet3D169(subject_data, args)
        else: 
            net = flipout_densenet3d.densenet3D169(subject_data, args)
    elif args.model == 'flipout_densenet3D201':
        import models.flipout_densenet3d as flipout_densenet3d
        from envs.bayesian_experiments import train, validate, test 
        if args.moped: 
            import models.densenet3d as densenet3d
            bayes_net = flipout_densenet3d.densenet3D201(subject_data, args)
            det_net = densenet3d.densenet3D201(subject_data, args)
        else: 
            net = flipout_densenet3d.densenet3D201(subject_data, args)
    # EfficientNet V1 
    elif args.model.find('efficientnet3D') != -1: 
        import models.efficientnet3d as efficientnet3d
        from envs.experiments import train, validate, test 
        net = efficientnet3d.efficientnet3D(subject_data,args)

    # load checkpoint
    if args.moped: 
        assert args.checkpoint_dir is not None 
        det_net = checkpoint_load(det_net, args.checkpoint_dir, layers='conv') 
        net = MOPED_network(bayes_net=bayes_net, det_net=det_net)   # weights of FC layer does not used for prior of Bayesian DNN if checkpoint load only convolution layers
        del det_net
        print("Prior of Bayesian DNN are set with parameters from Deterministic DNN")
    else: 
        if args.checkpoint_dir is not None: 
            net = checkpoint_load(net, args.checkpoint_dir, layers='conv')


    if args.optim == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)
    elif args.optim == 'SGDW':
        optimizer = SGDW(net.parameters(), lr=0, momentum=0.9)
    elif args.optim == 'Adam':
        optimizer = optim.Adam(net.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    elif args.optim == 'AdamW': 
        optimizer = optim.AdamW(net.parameters(), lr=0, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    else:
        raise ValueError('In-valid optimizer choice')

    # learning rate schedluer
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,'max', patience=20) #if you want to use this scheduler, you should activate the line 134 of envs/experiments.py
    scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=30, T_mult=2, eta_max=args.lr, T_up=5, gamma=0.5)
    # apply early stopping 
    early_stopping = EarlyStopping(patience=30)

    # setting DataParallel
    devices = []
    for d in range(torch.cuda.device_count()):
        devices.append(d)
    net = nn.DataParallel(net, device_ids = devices)
    # pytorch 2.0 
    net = torch.compile(net)
    
    # attach network and optimizer to cuda device
    net.cuda()


    """
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(f'cuda:{net.device_ids[0]}')
    """
    
    # setting for results' data frame
    train_losses = {}
    train_accs = {}
    val_losses = {}
    val_accs = {}

    for target_name in targets:
        train_losses[target_name] = []
        train_accs[target_name] = [-10000.0]
        #train_accs[target_name] = [10000.0]
        val_losses[target_name] = []
        val_accs[target_name] = [-10000.0]
        #val_accs[target_name] = [10000.0]
        
    global_steps = 0
    for epoch in tqdm(range(args.epoch)):
        ts = time.time()
        net, train_loss, train_acc, global_steps = train(net, partition,optimizer, global_steps, args)
        torch.cuda.empty_cache()
        val_loss, val_acc = validate(net,partition,scheduler,args)
        te = time.time()

         # sorting the results
        if args.cat_target: 
            for cat_target in args.cat_target: 
                train_losses[cat_target].append(train_loss[cat_target])
                train_accs[cat_target].append(train_acc[cat_target]['ACC'])
                val_losses[cat_target].append(val_loss[cat_target])
                val_accs[cat_target].append(val_acc[cat_target]['ACC'])
                early_stopping(val_acc[cat_target]['ACC'])
        if args.num_target: 
            for num_target in args.num_target: 
                train_losses[num_target].append(train_loss[num_target])
                train_accs[num_target].append(train_acc[num_target]['r_square'])
                #train_accs[num_target].append(train_acc[num_target]['abs_loss'])
                val_losses[num_target].append(val_loss[num_target])
                val_accs[num_target].append(val_acc[num_target]['r_square'])
                #val_accs[num_target].append(val_acc[num_target]['abs_loss'])
                early_stopping(val_acc[num_target]['r_square'])
                #early_stopping(val_acc[num_target]['abs_loss'])            

        # visualize the result
        CLIreporter(targets, train_loss, train_acc, val_loss, val_acc)
        print('Epoch {}. Current learning rate {}. Took {:2.2f} sec'.format(epoch+1,optimizer.param_groups[0]['lr'],te-ts))

        # saving the checkpoint
        #if train_acc[targets[0]] > 0.9:
        checkpoint_dir = checkpoint_save(net, save_dir, epoch, val_acc, val_accs, args)

        # early stopping 
        #if early_stopping.early_stop: 
        #    break

    # test
    net.to('cpu')
    torch.cuda.empty_cache()

    net = checkpoint_load(net, checkpoint_dir)

    # setting DataParallel
    devices = []
    for d in range(torch.cuda.device_count()):
        devices.append(d)
    net = nn.DataParallel(net, device_ids = devices)
    net.cuda()

    if args.model.find('flipout_') != -1: 
        test_acc, confusion_matrices, predicted_score = test(net, partition, args, num_monete_carlo=50)
    else: 
        test_acc, confusion_matrices, predicted_score = test(net, partition, args)

    # summarize results
    result = {}
    result['train_losses'] = train_losses
    result['train_accs'] = train_accs
    result['val_losses'] = val_losses
    result['val_accs'] = val_accs

    result['train_acc'] = train_acc
    result['val_acc'] = val_acc
    result['test_acc'] = test_acc 
    if args.get_predicted_score: 
        result['predicted_score'] = predicted_score  
    
    if confusion_matrices != None:
        result['confusion_matrices'] = confusion_matrices

    return vars(args), result
## ==================================== ##


def extract_embedding(net, partition_dataset, args):
    # flag for data shuffle 
    data_shuffle = False 

    assert data_shuffle == False 
    dataloader = torch.utils.data.DataLoader(partition_dataset,
                                            batch_size=args.batch_size,
                                            shuffle=data_shuffle,
                                            num_workers=24)

    net.eval()
    if hasattr(net, 'module'):
        device = net.device_ids[0]
    else: 
        if args.sbatch =='True':
            device = 'cuda:0'
        else:
            device = f'cuda:{args.gpus[0]}'


    embeddings = torch.tensor([])
    with torch.no_grad():
        for i, data in enumerate(tqdm(dataloader),0):
            image, _ = data
            image = image.to(device)
            conv_emb = net._hook_embeddings(image).detach().cpu()
            embeddings = torch.cat([embeddings, conv_emb]) 

    embeddings = combine_emb_subjid(embeddings, partition_dataset.image_files)

    return embeddings

