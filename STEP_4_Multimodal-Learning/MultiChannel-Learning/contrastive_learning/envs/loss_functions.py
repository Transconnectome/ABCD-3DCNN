import torch
import torch.nn as nn

def calculating_loss_acc(targets, output, loss_dict, correct, total, acc_dict, net, args):
    '''define calculating loss and accuracy function used during training and validation step'''
    cat_weight = 1 - (len(args.num_target)/(len(args.cat_target)+len(args.num_target)))
    num_weight = 1 - cat_weight
    loss = 0.0

    if args.cat_target: 
        for cat_target in args.cat_target:
            label = targets[cat_target].to(f'cuda:{net.device_ids[0]}')
            tmp_output = output[cat_target]
            
            # Loss
            criterion = nn.CrossEntropyLoss()
            tmp_loss = criterion(tmp_output.float(), label.long())
            loss += tmp_loss * cat_weight  # loss is for aggregating all the losses from predicting each target variable
            loss_dict[cat_target].append(tmp_loss.item()) 
            
            # Acc
            _, predicted = torch.max(tmp_output.data, 1)
            correct[cat_target] += (predicted == label).sum().item()
            total[cat_target] += label.size(0)
            acc_dict[cat_target].append(100 * correct[cat_target] / total[cat_target]) 
            
    if args.num_target:
        for num_target in args.num_target:
            y_true = targets[num_target].to(f'cuda:{net.device_ids[0]}')
            tmp_output = output[num_target]
            
            #Loss
            criterion = nn.MSELoss() if 'MAE' not in [args.transfer, args.scratch] else nn.L1Loss()
            tmp_loss = criterion(tmp_output.float(), y_true.float().unsqueeze(1))
            loss += tmp_loss * num_weight
            loss_dict[num_target].append(tmp_loss.item())   
            
            # Acc
            if 'MAE' in [args.transfer, args.scratch]:
                criterion = nn.MSELoss() 
                tmp_loss = criterion(tmp_output.float(), y_true.float().unsqueeze(1))
                
            y_var = torch.var(y_true, unbiased=False)
            r_square = 1 - (tmp_loss / y_var)
            acc_dict[num_target].append(r_square.item())
            
    return loss, loss_dict, acc_dict
