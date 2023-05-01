import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

LARGE_NUM = 1e9

class NTXentLoss(torch.nn.Module):
    def __init__(self, device, temperature, alpha_weight):
        """Compute loss for model.
        temperature: a `floating` number for temperature scaling.
        weights: a weighting number or vector.
        """
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.alpha_weight = alpha_weight
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def softXEnt(self, target, logits):
        """
        From the pytorch discussion Forum:
        https://discuss.pytorch.org/t/soft-cross-entropy-loss-tf-has-it-does-pytorch-have-it/69501
        """
        logprobs = torch.nn.functional.log_softmax(logits, dim=1)
        loss = -(target * logprobs).sum() / logits.shape[0]
        return loss

    def forward(self, zis, zjs, norm=True):
        temperature = self.temperature
        alpha = self.alpha_weight

        # Get (normalized) hidden1 and hidden2.
        if norm:
            zis = F.normalize(zis, p=2, dim=1)
            zjs = F.normalize(zjs, p=2, dim=1)

        hidden1, hidden2 = zis, zjs
        batch_size = hidden1.shape[0]

        hidden1_large = hidden1
        hidden2_large = hidden2
        labels = F.one_hot(
            torch.arange(start=0, end=batch_size, dtype=torch.int64),
            num_classes=batch_size,
        ).float()
        labels = labels.to(self.device)

        # Different from Image-Image contrastive learning
        # In the case of Image-Gen contrastive learning we do not compute the intra-modal similarity
#        masks = F.one_hot(
#            torch.arange(start=0, end=batch_size, dtype=torch.int64),
#            num_classes=batch_size,
#        ).to(device)
#        logits_aa = torch.matmul(hidden1, torch.transpose(hidden1_large,0, 1)) / temperature
#        logits_aa = logits_aa - masks * LARGE_NUM
#        logits_bb = torch.matmul(hidden2,  torch.transpose(hidden2_large,0, 1)) / temperature
#        logits_bb = logits_bb - masks * LARGE_NUM

        logits_ab = (
            torch.matmul(hidden1, torch.transpose(hidden2_large, 0, 1)) / temperature
        )
        logits_ba = (
            torch.matmul(hidden2, torch.transpose(hidden1_large, 0, 1)) / temperature
        )

        loss_a = self.softXEnt(labels, logits_ab)
        loss_b = self.softXEnt(labels, logits_ba)

        return alpha * loss_a + (1 - alpha) * loss_b


def contrastive_loss(output, result, metric='cos'):
    # << should be implemented later >> case where len(args.data_type) >= 3 
    loss_name = 'contrastive_loss_'+metric
    embedding_1, embedding_2 = output  
    if metric == 'NTXent':
        criterion_ssim = NTXentLoss(device='cuda', temperature=0.5, alpha_weight=0.5)
        loss = criterion_ssim(embedding_1, embedding_2, norm=False) # vectors are already normalized in densenet3DMM
        result['loss'][loss_name].append(loss.item())
    
    else:
        embedding_2_rolled = embedding_2.roll(1,0)        
        if metric == 'cos':
            criterion_ssim = nn.CosineEmbeddingLoss(margin=0.0, reduction='mean')
            label_positive = torch.ones(embedding_1.shape[0], device='cuda:0')
            label_negative = -torch.ones(embedding_1.shape[0], device='cuda:0')
            loss_positive = criterion_ssim(embedding_1, embedding_2, label_positive)
            loss_negative = criterion_ssim(embedding_1, embedding_2_rolled, label_negative)
            result['loss'][f'{loss_name}_positive'].append(loss_positive.item())
            result['loss'][f'{loss_name}_negative'].append(loss_negative.item())
            loss = (loss_positive + loss_negative)/2
        elif metric.upper() == 'L2':
            criterion_ssim = nn.MSELoss()
            loss = criterion_ssim(embedding_1, embedding_2)
            result['loss'][loss_name].append(loss.item())            
        
    return loss
    
    
def calc_acc(tmp_output, label, args, tmp_loss=None):
    _, predicted = torch.max(tmp_output.data, 1)
    correct = (predicted == label).sum().item()
    total = label.size(0)
    acc = (100 * correct / total)

    return {'acc': acc}


def calc_acc_auroc(tmp_output, label, args, tmp_loss=None):
    _, predicted = torch.max(tmp_output.data, 1)
    correct = (predicted == label).sum().item()
    total = label.size(0)   
    acc = (100 * correct / total)
    auroc = roc_auc_score(label.detach().cpu(), tmp_output.data[:, 1].detach().cpu())
    result = {'acc': acc, 'auroc': auroc.item()}
    
    return result


def calc_R2(tmp_output, y_true, args, tmp_loss=None): #230313change
    if ('MAE' in args.exp_name or tmp_loss == None):
        criterion = nn.MSELoss()
        tmp_loss = criterion(tmp_output.float(), y_true.float().unsqueeze(1))

    y_var = torch.var(y_true, unbiased=False)
    r_square = 1 - (tmp_loss / y_var)
                    
    return {'r_square': r_square.item()}


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


def calculating_loss_acc(targets, output, result, net, args):
    '''define calculating loss and accuracy function used during training and validation step'''
    # << should be implemented later >> how to set ssim_weight?
    if targets != []:
        cat_weight = (len(args.cat_target)/(len(args.cat_target)+len(args.num_target)))
        num_weight = 1 - cat_weight
    loss = 0.0
    
    # calculate constrastive_loss
    if args.metric and (len(args.data_type) > 1 and args.in_channels == 1):
        loss = contrastive_loss(output, result, args.metric)
        
    # calculate target_losses & accuracies
    for curr_target in targets:
        tmp_output = output[curr_target]
        label = targets[curr_target].to('cuda:0')
        tmp_label = label.long() if curr_target in args.cat_target else label.float().unsqueeze(1)
        weight = cat_weight if curr_target in args.cat_target else num_weight
        
        if curr_target in args.cat_target:
            criterion = nn.CrossEntropyLoss()
        elif 'MAE' in args.exp_name: #230313change
            criterion = nn.L1Loss()
        else:
            criterion = nn.MSELoss()
        
        # Loss
        tmp_loss = criterion(tmp_output.float(), tmp_label)
        loss += tmp_loss * weight
        result['loss'][curr_target].append(tmp_loss.item())
        
        # Acc
        acc_func = calc_acc_auroc if curr_target in args.cat_target else calc_MAE_MSE_R2
        acc = acc_func(tmp_output, label, args, tmp_loss)
        for k, v in acc.items():
            result[k][curr_target].append(v) 
            
    return loss
