
import numpy as np
import pandas as pd

def CLIreporter(targets, train_loss, train_acc, val_loss, val_acc):
    '''command line interface reporter per every epoch during experiments'''
    var_column = []
    visual_report = {}
    visual_report['Loss (train/val)'] = []
    visual_report['MSE or ACC (train/val)'] = []

    for label_name in targets:
        var_column.append(label_name)
        loss_value = '{:2.2f} / {:2.2f}'.format(train_loss[label_name],val_loss[label_name])
        acc_value = '{:2.2f} / {:2.2f}'.format(train_acc[label_name],val_acc[label_name])
        visual_report['Loss (train/val)'].append(loss_value)
        visual_report['MSE or ACC (train/val)'].append(acc_value)

    print(pd.DataFrame(visual_report, index=var_column))

def CLIblockdropping(targets, policys):
    visual_report = {}

    for t_id in range(len(targets)):
        visual_report['skip layer_%s [yes, no]' % targets[t_id]] = policys[t_id]

    print(pd.DataFrame(visual_report))
## ============================================ ##
