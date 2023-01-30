import torch
import torch.nn as nn

def calculating_loss_acc(targets, output, loss_dict, correct, total, acc_dict, net, args):
    '''define calculating loss and accuracy function used during training and validation step'''
    cat_weight = 1-(len(args.num_target)/(len(args.cat_target)+len(args.num_target)))
    num_weight = 1-(len(args.cat_target)/(len(args.cat_target)+len(args.num_target)))
    loss = 0.0

    if args.cat_target: 
        for cat_target in args.cat_target:
            label = targets[cat_target]
            label = label.to(f'cuda:{net.device_ids[0]}')
            tmp_output = output[cat_target]
            
            # Loss
            criterion = nn.CrossEntropyLoss()
            tmp_loss = criterion(tmp_output.float(),label.long())
            loss += tmp_loss * cat_weight                            # loss is for aggregating all the losses from predicting each target variable

            # restoring train loss and accuracy for each task (target variable)
            loss_dict[cat_target].append(tmp_loss.item())     # train_loss is for restoring loss from predicting each target variable
            # Acc
            _, predicted = torch.max(tmp_output.data,1)
            correct[cat_target] += (predicted == label).sum().item()
            total[cat_target] += label.size(0)
            acc_dict[cat_target].append(100 * correct[cat_target] / total[cat_target]) 
            

    if args.num_target:
        for num_target in args.num_target:
            y_true = targets[num_target]
            y_true = y_true.to(f'cuda:{net.device_ids[0]}')
            tmp_output = output[num_target]
            if args.transfer != '' and args.transfer != 'MAE':
                criterion = nn.MSELoss()
            elif args.transfer == 'MAE' or args.scratch == 'MAE':
                criterion = nn.L1Loss()
            else:
                assert args.transfer == 'MAE', print("Invalid age loss option")
            tmp_loss = criterion(tmp_output.float(),y_true.float().unsqueeze(1))
            loss += tmp_loss * num_weight
            
            # Loss
            # restoring train loss and r-square for each task (target variable)
            loss_dict[num_target].append(tmp_loss.item())     # train_loss is for restoring loss from predicting each target variable
            # Acc
            if args.transfer == 'MAE' or args.scratch =='MAE':
                criterion = nn.MSELoss() 
                tmp_loss = criterion(tmp_output.float(),y_true.float().unsqueeze(1))
                
            y_var = torch.var(y_true)
            r_square = 1 - (tmp_loss / y_var)
            acc_dict[num_target].append(r_square.item()) # RMSE for evaluating continuous variable

    return loss, loss_dict, acc_dict





