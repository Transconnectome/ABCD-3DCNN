import sys

from matplotlib.pyplot import get
sys.path.insert(0, '..')
import glob
import argparse

import os
import time
import yaml

import numpy as np
import pandas as pd


from utils.util import makedir, print_separator, read_yaml, create_path, print_yaml
import numpy as np
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True, type=str, help="Path for the config file")
parser.add_argument("--val_size", required=True, type=float, help='')
parser.add_argument("--test_size", required=True, type=float, help='')
args = parser.parse_args()


# read yaml file
with open(args.config) as f:
    opt = yaml.load(f, Loader = yaml.FullLoader)


# get names of image files
if opt['dataload']['dataset'] == 'ABCD':
    os.chdir(opt['dataload']['img_dataroot'])
    image_files = glob.glob('*.npy')
    image_files = sorted(image_files)

num_total = len(image_files)
num_val = int(args.val_size * num_total)
num_test = int(args.test_size * num_total)

image_val = image_files[:num_val]
image_test = image_files[num_val:num_val+num_test]
image_train = image_files[num_val+num_test:]

image_val_file = os.path.join(opt['dataload']['img_dataroot'], 'image_val_SubjectList.txt')
image_test_file = os.path.join(opt['dataload']['img_dataroot'], 'image_test_SubjectList.txt')
image_train_file = os.path.join(opt['dataload']['img_dataroot'], 'image_train_SubjectList.txt')


# saving file names of images as text file 
with open(image_val_file, 'w', encoding='UTF-8') as f:
    for subject in image_val:
        f.write(subject+'\n')

with open(image_test_file, 'w', encoding='UTF-8') as f:
    for subject in image_test:
        f.write(subject+'\n')

with open(image_train_file, 'w', encoding='UTF-8') as f:
    for subject in image_train:
        f.write(subject+'\n')

print_separator("DATA SPLITTING IS DONE")


