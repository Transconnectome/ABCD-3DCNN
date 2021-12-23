import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from tqdm.auto import tqdm ##progress

## ========= define functinos for data preprocesing categorical variable and numerical variables ========= ##
## categorical
def preprocessing_cat(subject_data, args):
    for cat in args.cat_target:
        if not 0 in list(subject_data.loc[:,cat]):
            subject_data[cat] = subject_data[cat] - 1
        else:
            continue

## numeric
def preprocessing_num(subject_data, args):
    for num in args.num_target:
        mean = np.mean(subject_data[num],axis=0)
        std = np.std(subject_data[num],axis=0)
        subject_data[num] = (subject_data[num]-mean)/std

## combine categorical + numeric
def combining_image_target(image_files, subject_data, args):
    targets = args.cat_target + args.num_target

    col_list = targets + ['subjectkey']

    imageFiles_labels = []

    for subjectID in tqdm(image_files):
        subjectID = subjectID[:-4] #removing '.npy' for comparing
        #print(subjectID)
        for i in range(len(subject_data)):
            if subjectID == subject_data['subjectkey'][i]:
                imageFile_label = {}
                imageFile_label['subjectkey'] = subjectID+'.npy'

                # combine all target variables in dictionary type.
                for j in range(len(col_list)-1):
                    imageFile_label[subject_data.columns[j]] = subject_data[subject_data.columns[j]][i]


                imageFiles_labels.append(imageFile_label)

    return imageFiles_labels
## ====================================== ##