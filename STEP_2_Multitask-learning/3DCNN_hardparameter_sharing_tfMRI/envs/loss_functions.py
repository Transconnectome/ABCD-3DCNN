import torch
import torch.nn as nn
import torch.functional as F
import numpy as np 
import monai
from sklearn.metrics import roc_auc_score, f1_score, r2_score
from sklearn.neighbors import KernelDensity

from typing import List, Dict

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

    def forward(self, targets, output, kl=None): 
        loss = 0.0
        if self.cat_target: 
            for cat_target in self.cat_target:
                label = targets[cat_target]
                label = label.to(self.device)
                tmp_output = output[cat_target]

                criterion = nn.CrossEntropyLoss(label_smoothing=0.15)
                tmp_loss = criterion(tmp_output.float(),label.long())
                if kl is not None: 
                    tmp_loss += kl 
                #criterion = monai.losses.FocalLoss(gamma=2.0, to_onehot_y=True, weight=[0.7, 0.3])
                #tmp_loss = criterion(tmp_output.float(),label.long().unsqueeze(-1))
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
                if kl is not None:
                    tmp_loss += kl
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


class MixUp_loss(torch.nn.Module):
    def __init__(self, args, device, beta=1.0): 
        """
        If beta == 1.0, beta distribtuion is same as uniform distribution. 
        """
        super().__init__()
        # only one categorical target could be used to mixup
        assert (len(args.num_target) == 1) or (len(args.cat_target) == 1) and (len(args.cat_target + args.num_target) == 1)
        self.beta = beta 
        self.device = device
        self.cat_target = args.cat_target
        self.num_target = args.num_target

        if self.cat_target: 
            self.target = args.cat_target[0] 
            #self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.15)
            self.criterion = torch.nn.CrossEntropyLoss()
        elif self.num_target: 
            self.target = args.num_target[0]
            self.criterion = torch.nn.MSELoss()
        self.loss_dict = {} 
        self.loss_dict[self.target] = []

    def get_lambda(self, tensor=False):
        if tensor: 
            return torch.tensor(np.random.beta(self.beta, self.beta)).unsqueeze(0)
        else: 
            return np.random.beta(self.beta, self.beta)
        
             
    def mixup_data(self, x, y:dict, index: torch.Tensor=None):
        y: torch.Tensor = y[self.target].to(self.device)
        y_a , y_b= {}, {} 
        lam = self.get_lambda()
        batch_size = x.shape[0]
        if index is None: 
            index = torch.randperm(batch_size)
        mixed_x = lam * x + (1 - lam) * x[index]
        y_a[self.target], y_b[self.target] = y, y[index]
        return mixed_x, y_a, y_b, lam 

    def forward(self, pred_y, y_a: dict, y_b:dict=None, lam=None, kl=None): 
        if y_b is not None and lam is not None: 
            if self.cat_target: 
                # target is categorical
                y_a, y_b = y_a[self.target].long(), y_b[self.target].long()
            elif self.num_target:
                # target is numerical 
                y_a, y_b = y_a[self.target].float(), y_b[self.target].float()

            pred_y = pred_y[self.target]
            loss = lam * self.criterion(pred_y, y_a) + (1 - lam) * self.criterion(pred_y, y_b) 
            if kl is not None: 
                loss += kl 
        else: 
            if self.cat_target: 
                y = y_a[self.target].to(self.device).long()
            elif self.num_target: 
                y = y_a[self.target].to(self.device).float()
            pred_y = pred_y[self.target]
            loss = self.criterion(pred_y, y)
            if kl is not None: 
                loss += kl 
        self.loss_dict[self.target].append(loss.item())
        return loss, self.loss_dict



class CutMix_loss(torch.nn.Module): 
    """
    ref: https://github.com/clovaai/CutMix-PyTorch/blob/master/train.py

    If beta == 1.0, beta distribtuion is same as uniform distribution. 
    """
    def __init__(self, args, beta=1.0, device=None):
        super().__init__()
        assert (len(args.num_target) == 1) or (len(args.cat_target) == 1) and (len(args.cat_target + args.num_target) == 1)
        assert beta > 0 
        self.beta = beta
        self.device = device 
        self.cat_target = args.cat_target
        self.num_target = args.num_target

        if self.cat_target: 
            self.target = args.cat_target[0] 
            #self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.15)
            self.criterion = torch.nn.CrossEntropyLoss()
        elif self.num_target: 
            self.target = args.num_target[0]
            self.criterion = torch.nn.MSELoss()
        self.loss_dict = {} 
        self.loss_dict[self.target] = []

    def rand_bbox(self, x, lam): 
        _, _, H, W, D = x.shape
        cut_w = np.int(W * lam)
        cut_h = np.int(H * lam)
        cut_d = np.int(D * lam)
        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        cz = np.random.randint(D)
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbz1 = np.clip(cz - cut_d // 2, 0, D)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        bbz2 = np.clip(cz + cut_d // 2, 0, D)

        return bbx1, bby1, bbz1, bbx2, bby2, bbz2


    def generate_mixed_sample(self, x:torch.Tensor, y:dict): 
        # generate mixed sample
        y: torch.Tensor = y[self.target].to(self.device)
        y_a , y_b= {}, {} 
        lam = np.random.beta(self.beta, self.beta)
        index = torch.randperm(x.shape[0])
        y_a[self.target], y_b[self.target] = y, y[index]
        bbx1, bby1, bbz1, bbx2, bby2, bbz2 = self.rand_bbox(x, lam)
        x[:, :, bbx1:bbx2, bby1:bby2, bbz1:bbz2] = x[index, :, bbx1:bbx2, bby1:bby2, bbz1:bbz2]
        # adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1)  *  (bbz2 - bbz1)/ (x.shape[-1] * x.shape[-2] * x.shape[-3]))

        return x, y_a, y_b, lam 


    def forward(self, pred_y, y_a: dict, y_b:dict=None, lam=None, kl=None): 
        if y_b is not None and lam is not None: 
            if self.cat_target: 
                # target is categorical
                y_a, y_b = y_a[self.target].long(), y_b[self.target].long()
            elif self.num_target:
                # target is numerical 
                y_a, y_b = y_a[self.target].float(), y_b[self.target].float()
            pred_y = pred_y[self.target]
            loss = lam * self.criterion(pred_y, y_a) + (1 - lam) * self.criterion(pred_y, y_b) 
            if kl is not None: 
                loss += kl 
        else: 
            if self.cat_target: 
                y = y_a[self.target].to(self.device).long()
            elif self.num_target: 
                y = y_a[self.target].to(self.device).float()
            pred_y = pred_y[self.target]
            loss = self.criterion(pred_y, y)
            if kl is not None: 
                loss += kl 
        self.loss_dict[self.target].append(loss.item())
        return loss, self.loss_dict
        

class C_MixUp_loss(torch.nn.Module):
    """
    ref: https://github.com/huaxiuyao/C-Mixup

    If beta == 1.0, beta distribtuion is same as uniform distribution. 
    In this code, only the "kde" mixup type is implemented. (kde = kernel density estimator)

    - logic of C_MixUp: 
    1. compare the distance of each sample in a mini-batch to another samples in a mini-batch 
    2. mixup with the nearest samples  
    """
    def __init__(self, args, device, beta=1.0, kernel_bandwidth=0.2, cutmix=False): 
        super().__init__()
        # only one continuous target could be used to mixup
        assert not args.cat_target and len(args.num_target) == 1 
        self.beta = beta
        self.kernel_bandwith = kernel_bandwidth
        self.device = device 
        self.num_target = args.num_target[0] 
        self.loss_dict = {} 
        self.loss_dict[self.num_target] = []
        self.cutmix = cutmix
        self.criterion = nn.MSELoss()


    def rand_bbox(self, x, lam): 
        _, _, H, W, D = x.shape
        cut_w = np.int(W * lam)
        cut_h = np.int(H * lam)
        cut_d = np.int(D * lam)
        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        cz = np.random.randint(D)
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbz1 = np.clip(cz - cut_d // 2, 0, D)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        bbz2 = np.clip(cz + cut_d // 2, 0, D)

        return bbx1, bby1, bbz1, bbx2, bby2, bbz2


    # for each batch calculate get_mixup_sample_rate
    def get_mixup_sample_rate(self, train_y:torch.tensor):  
        """
        Compare the distance between each sample with another samples with gaussian kernel estimation.
        Based on the distance, assigning higher probability for more reasonable mixing pairs. 
        
        ##NOTE##
        This is modified version of C-Mixup-batch. (ref: page 17 of original paper)
        In the original paper, sampling two mini-batches, and calculating pairwise distance and performing mixing between two different batches. 
        However, in this code, sampling only one batch, and calculating pairwise distance within a mini-batch.
        Then, set the probability of sampling (batch_i, batch_i) as 0 (in other wirds, set the probability of same images used for mixing), so that only the different image could be mixed up.
        """
        N = train_y.shape[0]
        mix_idx = []
        train_y = train_y.unsqueeze(-1)
        for i in range(N): 
            # get kde (kernel density estimation) sample rate
            kd = KernelDensity(kernel='gaussian', bandwidth=self.kernel_bandwith).fit(train_y[i].unsqueeze(-1))  # should be 2D
            each_rate = np.exp(kd.score_samples(train_y))
            each_rate[i] = 0    # ##NOTE##: additional line for setting the probability of same images used for mixing up as 0 
            each_rate /= np.sum(each_rate)  # norm
            mix_idx.append(each_rate)        
        return np.array(mix_idx)


    def mixup_data(self, x, y:dict, mixup_idx_sample_rate=None):
        """
        On each mini-batch, cacluating the distance between each mixing pair and assigning higher probability for more reasonable mixing pairs. 
        Sample mixing pairs based on the probability. In other words, the more two samples are similar, the more frequently make pairs.
        """
        
        batch_size = x.shape[0]
        mixup_idx_sample_rate = self.get_mixup_sample_rate(train_y=y[self.num_target])
        y: torch.Tensor = y[self.num_target].to(self.device)
        y_a , y_b= {}, {} 
        lam = np.random.beta(self.beta, self.beta)
        index = np.array([np.random.choice(np.arange(batch_size), p=mixup_idx_sample_rate[sel_index]) for sel_index in np.arange(batch_size)])
        # shuffling and mixup X and, shuffling y
        if self.cutmix:
            bbx1, bby1, bbz1, bbx2, bby2, bbz2 = self.rand_bbox(x, lam)
            y_a[self.num_target], y_b[self.num_target] = y, y[index]
            x[:, :, bbx1:bbx2, bby1:bby2, bbz1:bbz2] = x[index, :, bbx1:bbx2, bby1:bby2, bbz1:bbz2]
            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1)  *  (bbz2 - bbz1)/ (x.shape[-1] * x.shape[-2] * x.shape[-3]))
        else: 
            x = lam * x + (1 - lam) * x[index]
            y_a[self.num_target], y_b[self.num_target] = y, y[index]
        return x, y_a, y_b, lam
        """
        ##TODO##
        ## original version 
        batch_size = x.shape[0]
        lam = np.random.beta(self.beta, self.beta)
        mixup_idx_sample_rate = self.get_mixup_sample_rate(train_y=y[self.num_target])
        y: torch.Tensor = y[self.num_target].to(self.device)
        y_a, y_b= {}, {}
        x_a, y_a[self.num_target] = x[:batch_size // 2], y[:batch_size // 2]

        index = np.array([np.random.choice(np.arange(batch_size), p=mixup_idx_sample_rate[sel_index]) for sel_index in np.arange(batch_size // 2)])
        y_b[self.num_target] = y[index]

        if self.cutmix:
            bbx1, bby1, bbz1, bbx2, bby2, bbz2 = self.rand_bbox(x[index], lam)
            x[:, :, bbx1:bbx2, bby1:bby2, bbz1:bbz2] = x_a[index, :, bbx1:bbx2, bby1:bby2, bbz1:bbz2]
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) * (bbz2 - bbz1)/ (x.shape[-1] * x.shape[-2] * x.shape[-3]))
            # original batch = B; final batch = B // 2
            return x_a, y_a, y_b, lam
        else:
            x = lam * x_a + (1 - lam) * x_b
            return x, y_a, y_b, lam
        """
       
        
        """
        index1 = torch.randperm(batch_size)
        index2 = np.array([np.random.choice(np.arange(batch_size), p=mixup_idx_sample_rate[sel_index]) for sel_index in index1])
        # shuffling and mixup X and, shuffling y             
        mixed_x = lam * x[index1] + (1 - lam) * x[index2]
        y_a[self.num_target], y_b[self.num_target] = y[index1], y[index2]
        """
         


    def forward(self, pred_y, y_a: dict, y_b:dict=None, lam=None, kl=None): 
        """
        Mixup for categorical target (e.g., mixup and cutmix), do not mixup y and calculate loss of (pred_y, y_a) and (pred_y, y_b) separately. 
        However, in C-Mix, y is also mixed up. 
        """
        if y_b is not None and lam is not None:
            # mixup y 
            y_a, y_b = y_a[self.num_target], y_b[self.num_target]
            y = lam * y_a + (1 - lam) * y_b
            pred_y = pred_y[self.num_target]
            loss = self.criterion(pred_y.float(), y.float())
            if kl is not None: 
                loss += kl 
        else: 
            y = y_a[self.num_target].to(self.device)
            pred_y = pred_y[self.num_target]
            loss = self.criterion(pred_y.float(), y.float())
            if kl is not None: 
                loss += kl 
        self.loss_dict[self.num_target].append(loss.item())
        return loss, self.loss_dict        

