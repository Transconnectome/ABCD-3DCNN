import torch
import torch.nn as nn
import torch.functional as F
import numpy as np 
import monai
from sklearn.metrics import roc_auc_score, f1_score, r2_score
from sklearn.neighbors import KernelDensity


def get_class_weight(label):
    unique_values, counts = torch.unique(label ,return_counts=True)
    weight = [1 - (x / torch.sum(counts)) for x in counts ]
    return torch.tensor(weight, device=label.device)


class calculating_loss(torch.nn.Module): 
    def __init__(self, device, args):
        super().__init__()
        self.targets = args.cat_target + args.num_target
        self.cat_target = args.cat_target 
        self.num_target = args.num_target 
        self.loss_dict = {} 
        self.device = device 
        self.criterion_cat = None 
        self.criterion_num = None
        if args.cat_target: 
            for cat_target in args.cat_target: 
                self.loss_dict[cat_target] = []
        if args.num_target: 
            for num_target in args.num_target: 
                self.loss_dict[num_target] = [] 

    def forward(self, targets, output,): 
        loss = 0.0
        if self.cat_target: 
            for cat_target in self.cat_target:
                label = targets[cat_target]
                label = label.to(self.device)
                tmp_output = output[cat_target]

                criterion = nn.CrossEntropyLoss(label_smoothing=0.15)
                tmp_loss = criterion(tmp_output.float(),label.long())
                loss += tmp_loss * (1/len(self.targets))                           # loss is for aggregating all the losses from predicting each target variable

                # restoring train loss and accuracy for each task (target variable)
                self.loss_dict[cat_target].append(tmp_loss.item())     # train_loss is for restoring loss from predicting each target variable            

        if self.num_target:
            for num_target in self.num_target:
                y_true = targets[num_target]
                y_true = y_true.to(self.device)
                tmp_output = output[num_target]

                criterion = nn.MSELoss()
                #criterion = nn.L1Loss()
                tmp_loss = criterion(tmp_output.float(),y_true.float().unsqueeze(1))
                loss += tmp_loss * (1/len(self.targets)) 
        
                # restoring train loss and r-square for each task (target variable)
                self.loss_dict[num_target].append(tmp_loss.item())     # train_loss is for restoring loss from predicting each target variable

        return loss, self.loss_dict


class calculating_eval_metrics(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.targets = args.cat_target + args.num_target
        self.cat_target = args.cat_target 
        self.num_target = args.num_target 
        self.pred = {}
        self.true = {} 
        self.pred_score = {}

        if args.cat_target:
            for cat_target in args.cat_target:
                self.pred[cat_target] = torch.tensor([])
                self.true[cat_target] = torch.tensor([])
                self.pred_score[cat_target] = torch.tensor([])

        if args.num_target:
            for num_target in args.num_target:
                self.pred[num_target] = torch.tensor([])
                self.true[num_target] = torch.tensor([])
                self.pred_score[num_target] = torch.tensor([])
                

    def store(self, pred: dict, true: dict, pred_score=None): 
        for target in self.targets: 
            self.pred[target] = torch.cat([self.pred[target], pred[target].detach().cpu()])
            self.true[target] = torch.cat([self.true[target], true[target].detach().cpu()])
            if pred_score is not None: 
                    self.pred_score[target] = torch.cat([self.pred_score[target], pred_score[target].detach().cpu()])
    
    def get_result(self): 
        result = {} 
        if self.cat_target: 
            for cat_target in self.cat_target: 
                result[cat_target] = {}
                # calculating ACC
                _, predicted = torch.max(self.pred[cat_target],1)
                correct = (predicted == self.true[cat_target]).sum().item()
                total = self.true[cat_target].shape[0]
                result[cat_target]['ACC'] = 100 * correct / total
                # calculating AUROC if the task is binary classification 
                if len(torch.unique(self.true[cat_target])) == 2:
                    result[cat_target]['F1_score'] = f1_score(self.true[cat_target], predicted, pos_label=1) 
                    result[cat_target]['AUROC'] = roc_auc_score(self.true[cat_target], self.pred[cat_target][:, 1]) 

        if self.num_target: 
            for num_target in self.num_target: 
                result[num_target] = {} 
                # calculating Absolute loss 
                result[num_target]['abs_loss'] = torch.nn.functional.l1_loss(self.pred[num_target], self.true[num_target]).item()
                # calculating Mean Squared loss 
                mse_loss = torch.nn.functional.mse_loss(self.pred[num_target], self.true[num_target])
                result[num_target]['mse_loss'] = mse_loss.item()
                # calculating R square
                #r_square = 1 - (mse_loss / torch.var(self.true[num_target]))
                r_square = r2_score(self.true[num_target], self.pred[num_target])
                result[num_target]['r_square'] = r_square.item()
        return result

