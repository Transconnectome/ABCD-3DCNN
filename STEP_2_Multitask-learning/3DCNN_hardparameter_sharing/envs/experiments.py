import torch
from torch.utils.data import Dataset, DataLoader

from envs.loss_functions import calculating_loss_acc, calculating_acc

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
        for cat_label in args.cat_target:
            correct[cat_label] = 0
            total[cat_label] = 0
            train_loss[cat_label] = 0.0

    if args.num_target:
        for num_label in args.num_target:
            train_acc[num_label] = 0.0
            train_loss[num_label] = 0.0


    for i, data in enumerate(trainloader,0):
        optimizer.zero_grad()

        image, targets = data
        image = image.to(f'cuda:{net.device_ids[0]}')
        output = net(image)
        loss, correct, total, train_loss,train_acc = calculating_loss_acc(targets,output, args.cat_target, args.num_target, correct, total, train_loss, train_acc,net)

        scaler.scale(loss).backward()# multi-head model sum all the loss from predicting each target variable and back propagation
        scaler.step(optimizer)
        scaler.update()

    # calculating total loss and acc of separate mini-batch
    if args.cat_target:
        for cat_label in args.cat_target:
            train_acc[cat_label] = 100 * correct[cat_label] / total[cat_label]
            train_loss[cat_label] = train_loss[cat_label] / len(trainloader)

    if args.num_target:
        for num_label in args.num_target:
            train_acc[num_label] = train_acc[num_label] / len(trainloader)
            train_loss[num_label] = train_loss[num_label] / len(trainloader)


    return net, train_loss, train_acc


# define validation step
def validate(net,partition,scheduler,args):
    valloader = torch.utils.data.DataLoader(partition['val'],
                                           batch_size=args.val_batch_size,
                                           shuffle=False,
                                           num_workers=24)

    net.eval()

    correct = {}
    total = {}


    val_loss = {}
    val_acc = {}  # collecting r-square of contiunuous varibale directly


    if args.cat_target:
        for cat_label in args.cat_target:
            correct[cat_label] = 0
            total[cat_label] = 0
            val_loss[cat_label] = 0.0

    if args.num_target:
        for num_label in args.num_target:
            val_acc[num_label] = 0.0
            val_loss[num_label] = 0.0


    with torch.no_grad():
        for i, data in enumerate(valloader,0):
            image, targets = data
            image = image.to(f'cuda:{net.device_ids[0]}')
            output = net(image)

            loss, correct, total, val_loss,val_acc = calculating_loss_acc(targets,output, args.cat_target, args.num_target, correct, total, val_loss, val_acc,net)


    if args.cat_target:
        for cat_label in args.cat_target:
            val_acc[cat_label] = 100 * correct[cat_label] / total[cat_label]
            val_loss[cat_label] = val_loss[cat_label] / len(valloader)

    if args.num_target:
        for num_label in args.num_target:
            val_acc[num_label] = val_acc[num_label] / len(valloader)
            val_loss[num_label] = val_loss[num_label] / len(valloader)


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
        for cat_label in args.cat_target:
            correct[cat_label] = 0
            total[cat_label] = 0

    if args.num_target:
        for num_label in args.num_target:
            test_acc[num_label] = 0.0


    for i, data in enumerate(testloader,0):
            image, targets = data
            image = image.to(f'cuda:{net.device_ids[0]}')
            output = net(image)

            correct, total, test_acc = calculating_acc(targets,output, args.cat_target, args.num_target, correct, total, test_acc,net)


    if args.cat_target:
        for cat_label in args.cat_target:
            test_acc[cat_label] = 100 * correct[cat_label] / total[cat_label]

    if args.num_target:
        for num_label in args.num_target:
            test_acc[num_label] = test_acc[num_label] / len(testloader)


    return test_acc
## ============================================ ##
