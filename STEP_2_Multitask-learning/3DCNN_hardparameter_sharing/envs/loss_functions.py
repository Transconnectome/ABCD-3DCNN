import torch
import torch.nn as nn

def calculating_loss_acc(targets, output, cat_target, num_target, correct, total, loss_dict, acc_dict,net):
    '''define calculating loss and accuracy function used during training and validation step'''
    loss = 0.0

    for cat_label in cat_target:
        label = targets[cat_label]
        label = label.to(f'cuda:{net.device_ids[0]}')
        tmp_output = output[cat_label]

        criterion = nn.CrossEntropyLoss()

        tmp_loss = criterion(tmp_output.float(),label.long())
        loss += tmp_loss                         # loss is for aggregating all the losses from predicting each target variable

        # restoring train loss and accuracy for each task (target variable)
        loss_dict[cat_label] += tmp_loss.item()     # train_loss is for restoring loss from predicting each target variable
        _, predicted = torch.max(tmp_output.data,1)
        correct[cat_label] += (predicted == label).sum().item()
        total[cat_label] += label.size(0)
        #print(label.size(0))

    for num_label in num_target:
        y_true = targets[num_label]
        y_true = y_true.to(f'cuda:{net.device_ids[0]}')
        tmp_output = output[num_label]

        criterion =nn.MSELoss()

        tmp_loss = criterion(tmp_output.float(),y_true.float().unsqueeze(1))
        loss += tmp_loss
                #print(tmp_loss)

        # restoring train loss and accuracy for each task (target variable)
        loss_dict[num_label] += tmp_loss.item()     # train_loss is for restoring loss from predicting each target variable
        acc_dict[num_label] += tmp_loss.item() # RMSE for evaluating continuous variable

    return loss, correct, total, loss_dict,acc_dict


def calculating_acc(targets, output, cat_target, num_target, correct, total, acc_dict, net):
    '''define calculating accuracy function used during test step'''

    for cat_label in cat_target:
        label = targets[cat_label]
        label = label.to(f'cuda:{net.device_ids[0]}')
        tmp_output = output[cat_label]


        _, predicted = torch.max(tmp_output.data,1)
        correct[cat_label] += (predicted == label).sum().item()
        total[cat_label] += label.size(0)


    for num_label in num_target:
        y_true = targets[num_label]
        y_true = y_true.to(f'cuda:{net.device_ids[0]}')
        tmp_output = output[num_label]

        criterion =nn.MSELoss()

        tmp_loss = criterion(tmp_output.float(),y_true.float().unsqueeze(1))


        # restoring train loss and accuracy for each task (target variable)
        acc_dict[num_label] += tmp_loss.item() # RMSE for evaluating continuous variable
    return correct, total, acc_dict
