import numpy as np


## categorical
def preprocessing_cat(subject_data, args):
    for cat_target in args.cat_target:
        if not 0 in list(subject_data.loc[:,cat_target]):
            subject_data[cat_target] = subject_data[cat_target] - 1
        else:
            continue
    return subject_data 


## numeric
def preprocessing_num(subject_data, args):
    for num_target in args.num_target:
        mean = np.mean(subject_data[num_target],axis=0)
        std = np.std(subject_data[num_target],axis=0)
        subject_data[num_target] = (subject_data[num_target]-mean)/std
    return subject_data
