import pickle
import torch 
import torch.nn.functional as F
import numpy as np 
from torch.utils.data import Dataset
from dataloaders.preprocessing import MaskGenerator

from monai.data import NibabelReader
from monai.transforms import LoadImage, Randomizable, apply_transform
from monai.utils import MAX_SEED, get_seed

class Image_Dataset(Dataset, Randomizable):
    def __init__(self, image_files=None, labels=None, transform=None): 
        self.image_files = image_files
        self.labels = labels 
        self.image_loader = LoadImage(reader=None, image_only=True, dtype=np.float32)    # use default reader of LoadImage
        self.transform = transform
        self.set_random_state(seed=get_seed())
        self._seed = 0

    def randomize(self, data=None) -> None:
        self._seed = self.R.randint(MAX_SEED, dtype="uint32")
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, index: int): 
        # load image and apply transform 
        self.randomize()
        image = self.image_loader(self.image_files[index])
        if self.transform is not None: 
            if isinstance(self.transform, Randomizable):
                self.transform.set_random_state(seed=self._seed)
            if isinstance(self.transform, MaskGenerator): 
                image, mask = apply_transform(self.transform, image, map_items=False)
                image, mask = torch.tensor(image), torch.tensor(mask)
            else: 
                image = apply_transform(self.transform, image, map_items=False)
                image = torch.tensor(image)
                mask = None 
        if self.labels is not None:
            y = self.labels[index]
            return (image, y)
        else: 
            if mask is not None:
                return image, mask 
            else: 
                return mask