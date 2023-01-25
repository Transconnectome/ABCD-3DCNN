from typing import Any, Callable, Optional, Sequence, Union
from itertools import repeat

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from monai.config import DtypeLike
from monai.data.image_reader import ImageReader
from monai.transforms import LoadImage, Randomizable, apply_transform
from monai.utils import MAX_SEED, get_seed

class MultiModalImageDataset(Dataset, Randomizable):
    """
    Loads image/segmentation pairs of files from the given filename lists. Transformations can be specified
    for the image and segmentation arrays separately.
    The difference between this dataset and `ArrayDataset` is that this dataset can apply transform chain to images
    and segs and return both the images and metadata, and no need to specify transform to load images from files.
    For more information, please see the image_dataset demo in the MONAI tutorial repo,
    https://github.com/Project-MONAI/tutorials/blob/master/modules/image_dataset.ipynb
    """
    def __init__(
        self,
        image_files: pd.DataFrame, # modified
        seg_files: Optional[Sequence[str]] = None,
        labels: pd.DataFrame = None,
        transform: Sequence[Optional[Callable]] = None, # modified
        seg_transform: Optional[Callable] = None,
        label_transform: Optional[Callable] = None,
        image_only: bool = True,
        transform_with_metadata: bool = False,
        dtype: DtypeLike = np.float32,
        reader: Optional[Union[ImageReader, str]] = None,
        *args,
        **kwargs,
    ) -> None:
        if seg_files is not None and len(image_files) != len(seg_files):
            raise ValueError(
                "Must have same the number of segmentation as image files: "
                f"images={len(image_files)}, segmentations={len(seg_files)}."
            )

        self.image_files = image_files
        self.seg_files = seg_files
        self.labels = labels
        self.transform = transform
        self.seg_transform = seg_transform
        self.label_transform = label_transform
        if image_only and transform_with_metadata:
            raise ValueError("transform_with_metadata=True requires image_only=False.")
        self.image_only = image_only
        self.transform_with_metadata = transform_with_metadata
        self.loader = LoadImage(reader, image_only, dtype, *args, **kwargs)
        self.set_random_state(seed=get_seed())
        self._seed = 0  # transform synchronization seed


    def __len__(self) -> int:
        return len(self.image_files) # modified

    def randomize(self, data: Optional[Any] = None) -> None:
        self._seed = self.R.randint(MAX_SEED, dtype="uint32")


    def __getitem__(self, index: int):
        self.randomize()
        meta_data, seg_meta_data, seg, label = None, None, None, None

        # load data and optionally meta
        if self.image_only:
            img = list(map(self.loader, self.image_files.iloc[index])) # modified
            if self.seg_files is not None:
                seg = self.loader(self.seg_files[index])
        else:
            img, meta_data = list(map(self.loader, self.image_files.iloc[index])) # modified
            if self.seg_files is not None:
                seg, seg_meta_data = self.loader(self.seg_files[index])

        # apply the transforms
        if self.transform is not None:
            if isinstance(self.transform, Randomizable):
                self.transform.set_random_state(seed=self._seed)

            if self.transform_with_metadata:
                img, meta_data = list(map(apply_transform, self.transform, img,
                                     repeat(False), repeat(True))) # modified
            else:
                img = list(map(apply_transform, self.transform, img, repeat(False))) # modified

        if self.seg_files is not None and self.seg_transform is not None:
            if isinstance(self.seg_transform, Randomizable):
                self.seg_transform.set_random_state(seed=self._seed)

            if self.transform_with_metadata:
                seg, seg_meta_data = apply_transform(
                    self.seg_transform, (seg, seg_meta_data), map_items=False, unpack_items=True
                )
            else:
                seg = apply_transform(self.seg_transform, seg, map_items=False)

        if self.labels is not None:
            label = self.labels[index]
            # label = self.labels.iloc[index].to_dict()
            if self.label_transform is not None:
                label = apply_transform([self.label_transform]*len(label), label, map_items=False)  # type: ignore

        # construct outputs
        data = [img]
        if seg is not None:
            data.append(seg)
        if label is not None:
            data.append(label)
        if not self.image_only and meta_data is not None:
            data.append(meta_data)
        if not self.image_only and seg_meta_data is not None:
            data.append(seg_meta_data)
        if len(data) == 1:
            return data[0]
        # use tuple instead of list as the default collate_fn callback of MONAI DataLoader flattens nested lists
        return tuple(data)
    
class MultiChannelImageDataset(Dataset, Randomizable):
    """
    Loads image/segmentation pairs of files from the given filename lists. Transformations can be specified
    for the image and segmentation arrays separately.
    The difference between this dataset and `ArrayDataset` is that this dataset can apply transform chain to images
    and segs and return both the images and metadata, and no need to specify transform to load images from files.
    For more information, please see the image_dataset demo in the MONAI tutorial repo,
    https://github.com/Project-MONAI/tutorials/blob/master/modules/image_dataset.ipynb
    """
    def __init__(
        self,
        image_files: pd.DataFrame, # modified
        seg_files: Optional[Sequence[str]] = None,
        labels: pd.DataFrame = None,
        transform: Sequence[Optional[Callable]] = None, # modified
        seg_transform: Optional[Callable] = None,
        label_transform: Optional[Callable] = None,
        image_only: bool = True,
        transform_with_metadata: bool = False,
        dtype: DtypeLike = np.float32,
        reader: Optional[Union[ImageReader, str]] = None,
        *args,
        **kwargs,
    ) -> None:
        if seg_files is not None and len(image_files) != len(seg_files):
            raise ValueError(
                "Must have same the number of segmentation as image files: "
                f"images={len(image_files)}, segmentations={len(seg_files)}."
            )

        self.image_files = image_files
        self.seg_files = seg_files
        self.labels = labels
        self.transform = transform
        self.seg_transform = seg_transform
        self.label_transform = label_transform
        if image_only and transform_with_metadata:
            raise ValueError("transform_with_metadata=True requires image_only=False.")
        self.image_only = image_only
        self.transform_with_metadata = transform_with_metadata
        self.loader = LoadImage(reader, image_only, dtype, *args, **kwargs)
        self.set_random_state(seed=get_seed())
        self._seed = 0  # transform synchronization seed


    def __len__(self) -> int:
        return len(self.image_files) # modified

    def randomize(self, data: Optional[Any] = None) -> None:
        self._seed = self.R.randint(MAX_SEED, dtype="uint32")

    def __getitem__(self, index: int):
        self.randomize()
        meta_data, seg_meta_data, seg, label = None, None, None, None

        # load data and optionally meta
        if self.image_only:
            img = list(map(self.loader, self.image_files.iloc[index])) # modified
            if self.seg_files is not None:
                seg = self.loader(self.seg_files[index])
        else:
            img, meta_data = list(map(self.loader, self.image_files.iloc[index])) # modified
            if self.seg_files is not None:
                seg, seg_meta_data = self.loader(self.seg_files[index])

        # apply the transforms
        if self.transform is not None:
            if isinstance(self.transform, Randomizable):
                self.transform.set_random_state(seed=self._seed)

            if self.transform_with_metadata:
                img, meta_data = list(map(apply_transform, self.transform, img,
                                     repeat(False), repeat(True))) # modified
            else:
                img = list(map(apply_transform, self.transform, img, repeat(False))) # modified

        if self.seg_files is not None and self.seg_transform is not None:
            if isinstance(self.seg_transform, Randomizable):
                self.seg_transform.set_random_state(seed=self._seed)

            if self.transform_with_metadata:
                seg, seg_meta_data = apply_transform(
                    self.seg_transform, (seg, seg_meta_data), map_items=False, unpack_items=True
                )
            else:
                seg = apply_transform(self.seg_transform, seg, map_items=False)

        if self.labels is not None:
            label = self.labels[index]
#            label = self.labels.iloc[index].to_dict()
            if self.label_transform is not None:
                label = apply_transform([self.label_transform]*len(label), label, map_items=False)  # type: ignore
          
        # construct outputs
        img = [torch.cat(img, dim=0)] # modified for making multi-channel input
        data = [img]
        if seg is not None:
            data.append(seg)
        if label is not None:
            data.append(label)
        if not self.image_only and meta_data is not None:
            data.append(meta_data)
        if not self.image_only and seg_meta_data is not None:
            data.append(seg_meta_data)
#         if len(data) == 1: # commented because multimodal model takes list of img as an input
#             return data[0]
        # use tuple instead of list as the default collate_fn callback of MONAI DataLoader flattens nested lists
        return tuple(data)
