# checkpoint saving(only if current model shows better mean validation accuracy across tasks than previous epochs)
import numpy as np
import torch

def checkpoint_saving(net, optimizer, val_accs, base_dir):
    val_accs = list(val_accs.values())
    val_accs = np.mean(val_accs, axis=0)
    
    current_acc = val_accs[-1]
    previous_accs = val_accs[:-1] # len(previous_accs) = epoch-1

    if len(previous_accs) == 0: # when the first epoch
        torch.save({'model_state_dict': net.backbone.state_dict(),'optimizer_state_dict': optimizer.state_dict()}, str(base_dir+'/model/AdaShare_ResNet.pt'))
    else: 
        for i in range(len(previous_accs)):
            if current_acc > previous_accs[i]:
                torch.save({'model_state_dict': net.backbone.state_dict(),'optimizer_state_dict': optimizer.state_dict()}, str(base_dir+'/model/AdaShare_ResNet.pt'))
