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


class MaskGenerator:
    def __init__(self, transform, input_size=192, mask_patch_size=16, model_patch_size=4, mask_ratio=0.6):
        self.transform = transform
        self.input_size = input_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio

        if isinstance(mask_patch_size, tuple):
            assert mask_patch_size[0] == mask_patch_size[1] == mask_patch_size[2]
            self.mask_patch_size = mask_patch_size[0]
        elif isinstance(mask_patch_size, int): 
            self.mask_patch_size = mask_patch_size

        assert self.input_size % self.mask_patch_size == 0
        assert self.mask_patch_size % self.model_patch_size == 0
        
        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size
        
        self.token_count = self.rand_size ** 3
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))
    
    def update_config(self, model_patch_size):
        if isinstance(model_patch_size, tuple):
            assert model_patch_size[0] == model_patch_size[1] == model_patch_size[2]
            model_patch_size = model_patch_size[0]
        self.model_patch_size = model_patch_size
        self.scale = self.mask_patch_size // model_patch_size
        
    def __call__(self, img):
        mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1
        
        mask = mask.reshape((self.rand_size, self.rand_size, self.rand_size))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1).repeat(self.scale, axis=2)
        
        return (self.transform(img), mask)