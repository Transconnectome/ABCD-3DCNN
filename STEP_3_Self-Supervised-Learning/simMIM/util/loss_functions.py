import torch 
from sklearn.metrics import roc_auc_score


class loss_forward(torch.nn.Module): 
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        #self.criterion = torch.nn.SmoothL1Loss() if num_classes == 1 else torch.nn.CrossEntropyLoss(label_smoothing=0.1)
        self.criterion = torch.nn.SmoothL1Loss() if num_classes == 1 else torch.nn.CrossEntropyLoss()


    def forward(self, pred_y, true_y):
        assert pred_y.shape[-1] == self.num_classes 
        if self.num_classes > 1:
            #pred_y = torch.nn.functional.softmax(pred_y, dim=1)
            loss = self.criterion(pred_y, true_y.long()) 
        elif self.num_classes == 1: 
            loss = self.criterion(pred_y, true_y.unsqueeze(-1)) 
        return loss


class mixup_loss(torch.nn.Module): 
    def __init__(self, num_classes):
        super().__init__()
        assert num_classes > 1 
        self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    def forward(self, pred_y, y_a, y_b, lam): 
        #pred_y = torch.nn.functional.softmax(pred_y, dim=1)
        return lam * self.criterion(pred_y, y_a.long()) + (1 - lam) * self.criterion(pred_y, y_b.long()) 



class calculating_eval_metrics(torch.nn.Module):
    def __init__(self, num_classes, is_DDP=True):
        """
        if num_classes > 1, calculating ACCURACY.
        if num_classes == 1, calculating L1 loss, L2 loss, and r-square 

        """
        super().__init__()
        self.num_classes = num_classes
        self.is_DDP = is_DDP
        if self.num_classes > 1: 
            self.total = torch.tensor([])
            self.correct = torch.tensor([])
            self.true = torch.tensor([])
            self.pred = torch.tensor([])
        elif self.num_classes == 1:
            self.true = torch.tensor([])
            self.pred = torch.tensor([])
        
        if self.is_DDP:
            self.world_size = torch.distributed.get_world_size()

    def store(self, pred_y, true_y):
        true_y, pred_y = true_y.cpu(), pred_y.cpu()
        if self.num_classes > 1:
            assert pred_y.shape[-1] > 1
            batch_size = true_y.shape[0]
            _, predicted = torch.max(pred_y, 1)
            self.total = torch.cat([self.total, torch.tensor([batch_size])])
            self.correct = torch.cat([self.correct, (predicted == true_y).sum().unsqueeze(0)])
            self.true = torch.cat([self.true, true_y])
            self.pred = torch.cat([self.pred, pred_y])
        elif self.num_classes == 1:
            true_y = true_y.unsqueeze(-1)
            assert true_y.shape[-1] == pred_y.shape[-1] == 1  
            self.true = torch.cat([self.true, true_y])
            self.pred = torch.cat([self.pred, pred_y])
    
    def get_result(self):
        result = {}
        if self.num_classes > 1: 
            if self.is_DDP:
                # gathering the results from each processes
                total, correct  = [torch.zeros(self.total.size()).cuda() for r in range(self.world_size)], [torch.zeros(self.correct.size()).cuda() for r in range(self.world_size)]
                true, pred  = [torch.zeros(self.true.size()).cuda() for r in range(self.world_size)], [torch.zeros(self.pred.size()).cuda() for r in range(self.world_size)]
                torch.distributed.all_gather(total, self.total.cuda())
                torch.distributed.all_gather(correct, self.correct.cuda())
                torch.distributed.all_gather(true, self.true.cuda())
                torch.distributed.all_gather(pred, self.pred.cuda())
                total, correct = sum(total), sum(correct)
                total, correct = total.sum().detach().cpu(), correct.sum().detach().cpu() 
                result['total'], result['correct'] = total.item(), correct.item() 
                result['ACC'] = 100 * correct.item() / total.item() 
                if self.num_classes == 2:
                    result['AUROC'] = roc_auc_score(true.detach().cpu(), pred[:1].detach().cpu())
            else: 
                result['total'] = self.total.sum().item()
                result['correct'] = self.correct.sum().item()
                result['ACC'] = 100 * self.correct.sum().item() /self.total.sum().item() 
                if self.num_classes ==2: 
                    result['AUROC'] = roc_auc_score(self.true, self.pred[:, 1])
             
        elif self.num_classes == 1:
            if self.is_DDP:
                # gathering the results from each processes
                true_tmp, pred_tmp = [torch.zeros(self.true.size(), dtype=torch.float64).cuda() for r in range(self.world_size)], [torch.zeros(self.pred.size(), dtype=torch.float64).cuda() for r in range(self.world_size)]
                torch.distributed.all_gather(true_tmp, self.true.cuda())
                torch.distributed.all_gather(pred_tmp, self.pred.cuda())
                true, pred = torch.tensor([]), torch.tensor([])
                for i in range(len(true_tmp)):
                    true, pred = torch.cat([true, true_tmp[i].detach().cpu()]), torch.cat([pred, pred_tmp[i].detach().cpu()])
                abs_loss = torch.nn.functional.l1_loss(true, pred)
                std_true, std_pred = self.standardization(true, pred) 
                mse_loss = torch.nn.functional.mse_loss(std_true, std_pred)
                y_var = torch.var(true)
                r_square = 1 - (mse_loss / y_var)
                result['abs_loss'] = abs_loss.item()
                result['mse_loss'] = mse_loss.item() 
                result['r_square'] = r_square.item()
            else: 
                abs_loss_fn = torch.nn.L1Loss() 
                mse_loss_fn = torch.nn.MSELoss()
                abs_loss = abs_loss_fn(self.true, self.pred)
                std_true, std_pred = self.standardization(self.true, self.pred)
                mse_loss = mse_loss_fn(std_true, std_pred)
                y_var = torch.var(self.true)
                r_square = 1 - (mse_loss / y_var)
                result['abs_loss'] = abs_loss.item()
                result['mse_loss'] = mse_loss.item() 
                result['r_square'] = r_square.item()

        return result 
 
    def standardization(self, x, y):
        mean_x = x.mean()
        stdev_x = x.var()**0.5
        mean_y = y.mean()
        stdev_y = y.var()**0.5

        return (x - mean_x) / stdev_x, (y - mean_y) / stdev_y


