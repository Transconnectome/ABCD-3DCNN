import torch
import monai
from monai.transforms import AddChannel, Compose, RandSpatialCrop, RandRotate, RandFlip,RandRotate90, RandAdjustContrast, RandGaussianSmooth, RandGibbsNoise, RandKSpaceSpikeNoise, Resize, ScaleIntensity, Flip, ToTensor
from typing import Tuple, List
from tqdm import tqdm

#original source code refers to monai.transforms.RandCoarseDropout. 
#In original code, when fill_value = None, randomly selected number from [min, max) of each images is filled into dropout patch
#I've changed it to "when fill_value = None, minimum number of each images is filled into dropout patch "
from monai_custom_augmentation.RandCoarseDropout import RandCoarseDropout

"""Flow of data augmentation is as follows. 
    intensity_crop_resize (together image set1 and image set2) at CPU -> cuda -> transform (sepreately applied to image set1 and image set2) at GPU. 
    By applying scale intensity, crop, and resize, it can resolve CPU -> GPU bottleneck problem which occur when too many augmentation techniques are sequentially operated at CPU.
    That's because we can apply augmentation technique to all of samples in mini-batches simulateously by tensor operation at GPU.
    However, GPUs have limitations in their RAM memory. Thus, too large matrix couldn't be attached. 
    So, in this code, crop and resizing operatiins are done at CPU, afterward other augmentations are applied at GPU."""


def applying_augmentation(img, args):
    augmentation_list = [] # basic transformation

    if 'RandRotate90' in args.augmentation:
        augmentation_list.append(RandRotate90())
    if 'RandRotate' in args.augmentation:
        augmentation_list.append(RandRotate(range_x = 90, range_y = 90, range_z = 90, prob = 1.0))
    if 'RandFlip' in args.augmentation:
        augmentation_list.append(RandFlip())
    if 'RandAdjustContrast' in args.augmentation: 
        augmentation_list.append(RandAdjustContrast(prob = 0.3, gamma = (0.5, 4.5)))
    if 'RandGaussianSmooth' in args.augmentation: 
        augmentation_list.append(RandGaussianSmooth(sigma_x = (0.25, 1.5), sigma_y = (0.25, 1.5), sigma_z = (0.25, 1.5), prob = 0.8))
    if 'RandGibbsNoise' in args.augmentation: 
        augmentation_list.append(RandGibbsNoise(prob = 0.8))
    #if 'RandKSpaceSpikeNoise' in args.augmentation: 
    #    augmentation_list.append(RandKSpaceSpikeNoise(prob = 0.3))
    if 'RandCoarseDropout' in args.augmentation:
        """If RandCoarseDropout(fill_value=None), the minimun number of image is filled into dropout patch""" 
        augmentation_list.append(RandCoarseDropout(holes=24, spatial_size = (7, 7, 7), fill_value=None , prob = 0.5))  # It is setting for dropping out 25% of patches     
    
    augmentation = Compose(augmentation_list)
    
    for batch in range(img.size()[0]):
        img[batch] = augmentation(img[batch])
    
    return  img



    

 

