import numpy as np 

import torch 

def mixup_data(x, y, alpha=0.15):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.shape[0]
    index = torch.randperm(batch_size).cuda()
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam 