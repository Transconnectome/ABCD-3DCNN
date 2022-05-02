import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import device_as

def calculating_loss(targets, output, loss_dict,net, args):
    '''define calculating loss and accuracy function used during training and validation step'''
    cat_weight = 1-(len(args.num_target)/(len(args.cat_target)+len(args.num_target)))
    num_weight = 1-(len(args.cat_target)/(len(args.cat_target)+len(args.num_target)))
    loss = 0.0

    if args.cat_target: 
        for cat_target in args.cat_target:
            label = targets[cat_target]
            label = label.to(f'cuda:{net.device_ids[0]}')
            tmp_output = output[cat_target]

            criterion = nn.CrossEntropyLoss()
            tmp_loss = criterion(tmp_output.float(),label.long())
            loss += tmp_loss * cat_weight                            # loss is for aggregating all the losses from predicting each target variable

            # restoring train loss and accuracy for each task (target variable)
            loss_dict[cat_target].append(tmp_loss.item())     # train_loss is for restoring loss from predicting each target variable

    if args.num_target:
        for num_target in args.num_target:
            y_true = targets[num_target]
            y_true = y_true.to(f'cuda:{net.device_ids[0]}')
            tmp_output = output[num_target]

            criterion =nn.MSELoss()
            tmp_loss = criterion(tmp_output.float(),y_true.float().unsqueeze(1))
            loss += tmp_loss * num_weight
    
            # restoring train loss and r-square for each task (target variable)
            loss_dict[num_target].append(tmp_loss.item())     # train_loss is for restoring loss from predicting each target variable


    return loss, loss_dict

def calculating_acc(targets, output, correct, total, acc_dict, net, args):

    if args.cat_target:
        for cat_target in args.cat_target:
            label = targets[cat_target]
            label = label.to(f'cuda:{net.device_ids[0]}')
            tmp_output = output[cat_target]

            _, predicted = torch.max(tmp_output.data,1)
            correct[cat_target] += (predicted == label).sum().item()
            total[cat_target] += label.size(0)
            acc_dict[cat_target].append(100 * correct[cat_target] / total[cat_target]) 

    if args.num_target:
        for num_target in args.num_target:
            y_true = targets[num_target]
            y_true = y_true.to(f'cuda:{net.device_ids[0]}')
            tmp_output = output[num_target]

            criterion =nn.MSELoss()
            tmp_loss = criterion(tmp_output.float(),y_true.float().unsqueeze(1))

            # restoring train loss and accuracy for each task (target variable)
            y_var = torch.var(y_true)
            r_square = 1 - (tmp_loss / y_var)
            acc_dict[num_target].append(r_square.item()) # RMSE for evaluating continuous variable

    return acc_dict

class ContrastiveLoss(nn.Module):
    """Vanilla Contrastive loss, also called InfoNceLoss as in SimCLR paper"""

    def __init__(self, batch_size, temperature = 0.5):
        super().__init__()
        self.batch_size = batch_size 
        self.temperature = temperature
        self.mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float() # '~' means 'not'
    
    def calc_similarity_batch(self, a, b):
        representations = torch.cat([a,b], dim=0)
        return F.cosine_similarity(representations.unsqueeze(1),representations.unsqueeze(0), dim=2)

    def forward(self, proj_1, proj_2):
        """
        proj_1 and proj_2 are batched embeddings [batch, embedding_dim]
        where corresponding indices are pairs
        z_i, z_j in the SimCLR paper
        """

        batch_size = proj_1.shape[0]
        z_i = F.normalize(proj_1, p = 2, dim = 1)
        z_j = F.normalize(proj_2, p=2, dim=1)

        similarity_matrix = self.calc_similarity_batch(z_i, z_j)

        sim_ij = torch.diag(similarity_matrix, batch_size)
        sim_ji = torch.diag(similarity_matrix, -batch_size)

        positives = torch.cat([sim_ij, sim_ji], dim=0)

        nominator = torch.exp(positives/self.temperature)
        denominator = device_as(self.mask, similarity_matrix) * torch.exp(similarity_matrix / self.temperature)

        all_losses = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(all_losses) / (2 * self.batch_size)

        return loss 







