import torch
import torch.nn as nn

def constrastive_loss(output):
    # << should be implemented later >> case where len(args.data_type) >= 3 
    embedding_1, embedding_2 = output['embeddings']
    embedding_2_rolled = embedding_2.roll(1,0)
    
    label_positive = torch.ones(embedding_1.shape[0]).to('cuda:0')
    label_negative = -label_positive

    criterion_ssim = nn.CosineEmbeddingLoss(margin=0.0, reduction='mean').to('cuda:0')
    loss_positive = criterion_ssim(embedding_1, embedding_2, label_positive)
    loss_negative = criterion_ssim(embedding_1, embedding_2_rolled, label_negative)
    
    output.pop('embeddings')

    return loss_positive, loss_negative


def calc_acc(tmp_output, label, args, tmp_loss=None):
    _, predicted = torch.max(tmp_output.data, 1)
    correct = (predicted == label).sum().item()
    total = label.size(0)
    
    return (100 * correct / total)


def calc_R2(tmp_output, y_true, args, tmp_loss=None):
    if ('MAE' in [args.transfer, args.scratch]) or tmp_loss == None:
        criterion = nn.MSELoss().to('cuda:0')
        tmp_loss = criterion(tmp_output.float(), y_true.float().unsqueeze(1))

    y_var = torch.var(y_true, unbiased=False)
    r_square = 1 - (tmp_loss / y_var)
                    
    return r_square.item()


def calculating_loss_acc(targets, output, loss_dict, acc_dict, net, args):
    '''define calculating loss and accuracy function used during training and validation step'''
    # << should be implemented later >> how to set ssim_weight?
    cat_weight = (len(args.cat_target)/(len(args.cat_target)+len(args.num_target)))
    num_weight = 1 - cat_weight
    loss = 0.0
    
    # calculate constrastive_loss
    if len(args.data_type) > 1:
        loss_positive, loss_negative = constrastive_loss(output)
        loss += (loss_positive + loss_negative)/2
        loss_dict['contrastive_loss_positive'].append(loss_positive.item())
        loss_dict['contrastive_loss_negative'].append(loss_negative.item())
        
    # calculate target_losses & accuracies
    for curr_target in output:
        tmp_output = output[curr_target]
        label = targets[curr_target].to('cuda:0')
#         label = label.repeat(2)
        tmp_label = label.long() if curr_target in args.cat_target else label.float().unsqueeze(1)
        weight = cat_weight if curr_target in args.cat_target else num_weight
        
        if curr_target in args.cat_target:
            criterion = nn.CrossEntropyLoss().to('cuda:0')
        elif curr_target == 'age' and 'MAE' in [args.transfer, args.scratch]:
            criterion = nn.L1Loss().to('cuda:0')
        else:
            criterion = nn.MSELoss().to('cuda:0')
        
        # Loss
        tmp_loss = criterion(tmp_output.float(), tmp_label)
        loss += tmp_loss * weight
        loss_dict[curr_target].append(tmp_loss.item())
        
        # Acc
        acc_func = calc_acc if curr_target in args.cat_target else calc_R2
        acc = acc_func(tmp_output, label, args, tmp_loss)
        acc_dict[curr_target].append(acc) 
            
    return loss