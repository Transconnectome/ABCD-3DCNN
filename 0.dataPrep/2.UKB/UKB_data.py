import nibabel as nib
import numpy as np
import os 
import glob
from tqdm import tqdm

base_dir = '/master_ssd/3DCNN/data/2.UKB/1.sMRI/'
os.chdir(base_dir)

file_list = glob.glob('*.nii.gz')

for subj_file in tqdm(file_list):
    file_name = base_dir + subj_file[:-7] + '.npy'

    img = nib.load(subj_file)
    img = np.array(img.dataobj)
    np.save(file_name, img)
    os.remove(subj_file)
