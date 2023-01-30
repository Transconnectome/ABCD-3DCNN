import torch
import torch.nn as nn

def contrastive_loss(output, metric='cos'):
    # << should be implemented later >> case where len(args.data_type) >= 3 
    embedding_1, embedding_2 = output['embeddings']
    embedding_2_rolled = embedding_2.roll(1,0)
    
    if metric == 'cos':
        label_positive = torch.ones(embedding_1.shape[0], device='cuda:0')
        label_negative = -torch.ones(embedding_1.shape[0], device='cuda:0')

        criterion_ssim = nn.CosineEmbeddingLoss(margin=0.0, reduction='mean')
        loss_positive = criterion_ssim(embedding_1, embedding_2, label_positive)
        loss_negative = criterion_ssim(embedding_1, embedding_2_rolled, label_negative)
    
    elif metric.upper() == 'L2':
        criterion_ssim = nn.MSELoss()
        loss_positive = criterion_ssim(embedding_1, embedding_2)
        loss_negative = torch.zeros(loss_positive.shape) # setting this to 0 raises error when call loss_negative.item() later
    
    output.pop('embeddings')
    
    return loss_positive, loss_negative
    

def calc_acc(tmp_output, label, args, tmp_loss=None):
    _, predicted = torch.max(tmp_output.data, 1)
    correct = (predicted == label).sum().item()
    total = label.size(0)
    
    return (100 * correct / total)


def calc_R2(tmp_output, y_true, args, tmp_loss=None):
    if ('MAE' in [args.transfer, args.scratch]) or tmp_loss == None:
        criterion = nn.MSELoss()
        tmp_loss = criterion(tmp_output.float(), y_true.float().unsqueeze(1))

    y_var = torch.var(y_true, unbiased=False)
    r_square = 1 - (tmp_loss / y_var)
                    
    return r_square.item()


def calc_MAE_MSE_R2(tmp_output, y_true, args, tmp_loss=None):
    pred, true = tmp_output.float(), y_true.float().unsqueeze(1)
    abs_loss = torch.nn.functional.l1_loss(pred, true)
    mse_loss = torch.nn.functional.mse_loss(pred, true)
    y_var = torch.var(true, unbiased=False)
    r_square = 1 - (mse_loss / y_var)
    result = {
        'abs_loss': abs_loss.item(),
        'mse_loss': mse_loss.item(),
        'r_square': r_square.item()
    }
    
    return result


def calculating_loss_acc(targets, output, loss_dict, acc_dict, net, args):
    '''define calculating loss and accuracy function used during training and validation step'''
    # << should be implemented later >> how to set ssim_weight?
    cat_weight = (len(args.cat_target)/(len(args.cat_target)+len(args.num_target)))
    num_weight = 1 - cat_weight
    loss = 0.0
    
    # calculate constrastive_loss
    if (len(args.data_type) > 1 and args.in_channels == 1):
        loss_positive, loss_negative = contrastive_loss(output, args.metric)
        loss += (loss_positive + loss_negative)/2
        loss_dict['contrastive_loss_positive'].append(loss_positive.item())
        loss_dict['contrastive_loss_negative'].append(loss_negative.item())
        
    # calculate target_losses & accuracies
    for curr_target in targets:
        tmp_output = output[curr_target]
        label = targets[curr_target].to('cuda:0')
#         label = label.repeat(2)
        tmp_label = label.long() if curr_target in args.cat_target else label.float().unsqueeze(1)
        weight = cat_weight if curr_target in args.cat_target else num_weight
        
        if curr_target in args.cat_target:
            criterion = nn.CrossEntropyLoss()
        elif curr_target == 'age' and 'MAE' in [args.transfer, args.scratch]:
            criterion = nn.L1Loss()
        else:
            criterion = nn.MSELoss()
        
        # Loss
        tmp_loss = criterion(tmp_output.float(), tmp_label)
        loss += tmp_loss * weight
        loss_dict[curr_target].append(tmp_loss.item())
        
        # Acc
        acc_func = calc_acc if curr_target in args.cat_target else calc_R2
        acc = acc_func(tmp_output, label, args, tmp_loss)
        acc_dict[curr_target].append(acc) 
            
    return loss
