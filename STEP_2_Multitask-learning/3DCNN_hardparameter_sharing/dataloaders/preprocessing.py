import numpy as np


## categorical
def preprocessing_cat(subject_data, args):
    if args.cat_target:
        for cat in args.cat_target:
            if not 0 in list(subject_data.loc[:,cat]):
                subject_data[cat] = subject_data[cat] - 1
            else:
                continue
        return subject_data 
    else:
        return subject_data

## numeric
def preprocessing_num(subject_data, args):
    if args.num_target:
        for num in args.num_target:
            mean = np.mean(subject_data[num],axis=0)
            std = np.std(subject_data[num],axis=0)
            subject_data[num] = (subject_data[num]-mean)/std
        return subject_data
    else:
        return subject_data
    




