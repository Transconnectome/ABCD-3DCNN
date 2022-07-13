from tabnanny import check
import model.model_MAE as MAE
import glob
import os 
import nibabel as nib 

import torch
import numpy as np 

import monai
from monai.transforms import Resize

"""
#print(MAE.__dict__['mae_vit_base_patch16_2D'](norm_pix_loss=False))
#print(MAE.__dict__['mae_vit_large_patch16_2D'](norm_pix_loss=False))
#print(MAE.__dict__['mae_vit_huge_patch14_2D'](norm_pix_loss=False))
print(MAE.__dict__['mae_vit_base_patch16_3D'](img_size = 96, norm_pix_loss=False))
#print(MAE.__dict__['mae_vit_large_patch16_3D'](norm_pix_loss=False))
#print(MAE.__dict__['mae_vit_huge_patch14_3D'](norm_pix_loss=False))
"""

os.chdir('/scratch/connectome/dhkdgmlghks/lesion_tract_pipeline/lesion_tract/TransUNet_training/data/Data_Training_655')

img_list = glob.glob("*.nii.gz")

img = nib.load(img_list[0])
header = img.header
affine = img.affine
img = torch.tensor(np.array(img.dataobj)).float()
img = img.unsqueeze(0)
img = Resize((96,96,96))(img)
img = img.unsqueeze(0)
print(img.shape)
net = MAE.__dict__['mae_vit_base_patch16_2D'](img_size = 96, norm_pix_loss=False)
net.to(f'cuda:2')
img = img.to(f'cuda:2')



loss, pred, mask = net(img)
loss.backward()
print(loss.item())





