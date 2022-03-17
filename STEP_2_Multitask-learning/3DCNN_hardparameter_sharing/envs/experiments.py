import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

from envs.loss_functions import calculating_loss, calculating_acc

### ========= Train,Validate, and Test ========= ###
'''The process of calcuating loss and accuracy metrics is as follows.
   1) sequentially calculate loss and accuracy metrics of target labels with for loop.
   2) store the result information with dictionary type.
   3) return the dictionary, which form as {'cat_target':value, 'num_target:value}
   This process is intended to easily deal with loss values from each target labels.'''


'''All of the loss from predictions are summated and this loss value is used for backpropagation.'''

# define training step
def train(net,partition,optimizer,args):
    '''GradScaler is for calculating gradient with float 16 type'''
    scaler = torch.cuda.amp.GradScaler()

    trainloader = torch.utils.data.DataLoader(partition['train'],
                                             batch_size=args.train_batch_size,
                                             shuffle=True,
                                             num_workers=24)

    net.train()

    correct = {}
    total = {}


    train_loss = {}
    train_acc = {}  # collecting r-square of contiunuous varibale directly
    

    if args.cat_target:
        for cat_target in args.cat_target:
            correct[cat_target] = 0
            total[cat_target] = 0
            train_acc[cat_target] = []
            train_loss[cat_target] = []

    if args.num_target:
        for num_target in args.num_target:
            train_acc[num_target] = []
            train_loss[num_target] = []


    for i, data in enumerate(trainloader,0):
        optimizer.zero_grad()

        image, targets = data
        image = image.to(f'cuda:{net.device_ids[0]}')
        output = net(image)

        loss, train_loss = calculating_loss(targets, output, train_loss,net, args)
        train_acc = calculating_acc(targets, output, correct, total, train_acc, net, args)

        scaler.scale(loss).backward()# multi-head model sum all the loss from predicting each target variable and back propagation
        scaler.step(optimizer)
        scaler.update()

    # calculating total loss and acc of separate mini-batch
    if args.cat_target:
        for cat_target in args.cat_target:
            train_acc[cat_target] = np.mean(train_acc[cat_target])
            train_loss[cat_target] = np.mean(train_loss[cat_target])

    if args.num_target:
        for num_target in args.num_target:
            train_acc[num_target] = np.mean(train_acc[num_target])
            train_loss[num_target] = np.mean(train_loss[num_target])


    return net, train_loss, train_acc


# define validation step
def validate(net,partition,scheduler,args):
    valloader = torch.utils.data.DataLoader(partition['val'],
                                           batch_size=args.val_batch_size,
                                           shuffle=True,
                                           num_workers=24)

    net.eval()

    correct = {}
    total = {}


    val_loss = {}
    val_acc = {}  # collecting r-square of contiunuous varibale directly


    if args.cat_target:
        for cat_target in args.cat_target:
            correct[cat_target] = 0
            total[cat_target] = 0
            val_acc[cat_target] = []
            val_loss[cat_target] = []

    if args.num_target:
        for num_target in args.num_target:
            val_acc[num_target] = []
            val_loss[num_target] = []


    with torch.no_grad():
        for i, data in enumerate(valloader,0):
            image, targets = data
            image = image.to(f'cuda:{net.device_ids[0]}')
            output = net(image)

            loss, val_loss = calculating_loss(targets, output, val_loss,net, args)
            val_acc = calculating_acc(targets, output, correct, total, val_acc, net, args)

    if args.cat_target:
        for cat_target in args.cat_target:
            val_acc[cat_target] = np.mean(val_acc[cat_target])
            val_loss[cat_target] = np.mean(val_loss[cat_target])

    if args.num_target:
        for num_target in args.num_target:
            val_acc[num_target] = np.mean(val_acc[num_target])
            val_loss[num_target] = np.mean(val_loss[num_target])


    # learning rate scheduler
    scheduler.step(sum(val_acc.values()))


    return val_loss, val_acc



# define test step
def test(net,partition,args):
    testloader = torch.utils.data.DataLoader(partition['test'],
                                            batch_size=args.test_batch_size,
                                            shuffle=False,
                                            num_workers=24)

    net.eval()

    correct = {}
    total = {}

    test_acc = {}


    if args.cat_target:
        for cat_target in args.cat_target:
            correct[cat_target] = 0
            total[cat_target] = 0
            test_acc[cat_target] = []

    if args.num_target:
        for num_target in args.num_target:
            test_acc[num_target] = []


    for i, data in enumerate(testloader,0):
            image, targets = data
            image = image.to(f'cuda:{net.device_ids[0]}')
            output = net(image)

            test_acc = calculating_acc(targets, output, correct, total, test_acc, net, args)

    if args.cat_target:
        for cat_target in args.cat_target:
            test_acc[cat_target] = np.mean(test_acc[cat_target])

    if args.num_target:
        for num_target in args.num_target:
            test_acc[num_target] = np.mean(test_acc[num_target])


    return test_acc
## ============================================ ##
