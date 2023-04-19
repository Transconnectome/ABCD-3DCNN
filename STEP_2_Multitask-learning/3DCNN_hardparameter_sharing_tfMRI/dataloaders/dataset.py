import pickle
import torch 
import torch.nn.functional as F
import numpy as np 
from torch.utils.data import Dataset

from monai.data import NibabelReader
from monai.transforms import LoadImage, Randomizable, apply_transform
from monai.utils import MAX_SEED, get_seed


class Image_Dataset(Dataset, Randomizable):
    def __init__(self, image_files=None, labels=None, transform=None, padding=False): 
        self.image_files = image_files
        self.labels = labels 
        self.image_loader = LoadImage(reader=None, image_only=True, dtype=np.float32)    # use default reader of LoadImage
        self.transform = transform
        self.padding = padding
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
        image = torch.tensor(image)
        if self.padding: 
            background_value = image.flatten()[0]
            image = torch.nn.functional.pad(image, (3, 2, -7, -6, 3, 2), value=background_value) 
        if self.transform is not None: 
            if isinstance(self.transform, Randomizable):
                self.transform.set_random_state(seed=self._seed)
            image = apply_transform(self.transform, image, map_items=False)
            
        if self.labels is not None: 
            y = self.labels[index]
            return (image, y)
        else: 
            return image
        