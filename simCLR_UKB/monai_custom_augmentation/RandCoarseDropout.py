from abc import abstractmethod
from monai.transforms.transform import RandomizableTransform, Transform
from monai.utils.enums import TransformBackends
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

from monai.data.utils import get_random_patch, get_valid_patch_size
from monai.utils import (
    InvalidPyTorchVersionError,
    convert_data_type,
    convert_to_dst_type,
    ensure_tuple,
    ensure_tuple_rep,
    ensure_tuple_size,
    fall_back_tuple,
    pytorch_after,
)

import numpy as np 

class RandCoarseTransform(RandomizableTransform):
    """
    Randomly select coarse regions in the image, then execute transform operations for the regions.
    It's the base class of all kinds of region transforms.
    Refer to papers: https://arxiv.org/abs/1708.04552

    Args:
        holes: number of regions to dropout, if `max_holes` is not None, use this arg as the minimum number to
            randomly select the expected number of regions.
        spatial_size: spatial size of the regions to dropout, if `max_spatial_size` is not None, use this arg
            as the minimum spatial size to randomly select size for every region.
            if some components of the `spatial_size` are non-positive values, the transform will use the
            corresponding components of input img size. For example, `spatial_size=(32, -1)` will be adapted
            to `(32, 64)` if the second spatial dimension size of img is `64`.
        max_holes: if not None, define the maximum number to randomly select the expected number of regions.
        max_spatial_size: if not None, define the maximum spatial size to randomly select size for every region.
            if some components of the `max_spatial_size` are non-positive values, the transform will use the
            corresponding components of input img size. For example, `max_spatial_size=(32, -1)` will be adapted
            to `(32, 64)` if the second spatial dimension size of img is `64`.
        prob: probability of applying the transform.

    """

    backend = [TransformBackends.NUMPY]

    def __init__(
        self,
        holes: int,
        spatial_size: Union[Sequence[int], int],
        max_holes: Optional[int] = None,
        max_spatial_size: Optional[Union[Sequence[int], int]] = None,
        prob: float = 0.1,
    ) -> None:
        RandomizableTransform.__init__(self, prob)
        if holes < 1:
            raise ValueError("number of holes must be greater than 0.")
        self.holes = holes
        self.spatial_size = spatial_size
        self.max_holes = max_holes
        self.max_spatial_size = max_spatial_size
        self.hole_coords: List = []

    def randomize(self, img_size: Sequence[int]) -> None:
        super().randomize(None)
        if not self._do_transform:
            return None
        size = fall_back_tuple(self.spatial_size, img_size)
        self.hole_coords = []  # clear previously computed coords
        num_holes = self.holes if self.max_holes is None else self.R.randint(self.holes, self.max_holes + 1)
        for _ in range(num_holes):
            if self.max_spatial_size is not None:
                max_size = fall_back_tuple(self.max_spatial_size, img_size)
                size = tuple(self.R.randint(low=size[i], high=max_size[i] + 1) for i in range(len(img_size)))
            valid_size = get_valid_patch_size(img_size, size)
            self.hole_coords.append((slice(None),) + get_random_patch(img_size, valid_size, self.R))


    @abstractmethod
    def _transform_holes(self, img: np.ndarray) -> np.ndarray:
        """
        Transform the randomly selected `self.hole_coords` in input images.

        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")

    def __call__(self, img, randomize: bool = True):
        if randomize:
            self.randomize(img.shape[1:])

        if not self._do_transform:
            return img

        img_np: np.ndarray
        img_np, *_ = convert_data_type(img, np.ndarray)  # type: ignore
        out = self._transform_holes(img=img_np)
        ret, *_ = convert_to_dst_type(src=out, dst=img)
        return ret



class RandCoarseDropout(RandCoarseTransform):
    """
    Randomly coarse dropout regions in the image, then fill in the rectangular regions with specified value.
    Or keep the rectangular regions and fill in the other areas with specified value.
    Refer to papers: https://arxiv.org/abs/1708.04552, https://arxiv.org/pdf/1604.07379
    And other implementation: https://albumentations.ai/docs/api_reference/augmentations/transforms/
    #albumentations.augmentations.transforms.CoarseDropout.

    Args:
        holes: number of regions to dropout, if `max_holes` is not None, use this arg as the minimum number to
            randomly select the expected number of regions.
        spatial_size: spatial size of the regions to dropout, if `max_spatial_size` is not None, use this arg
            as the minimum spatial size to randomly select size for every region.
            if some components of the `spatial_size` are non-positive values, the transform will use the
            corresponding components of input img size. For example, `spatial_size=(32, -1)` will be adapted
            to `(32, 64)` if the second spatial dimension size of img is `64`.
        dropout_holes: if `True`, dropout the regions of holes and fill value, if `False`, keep the holes and
            dropout the outside and fill value. default to `True`.
        fill_value: target value to fill the dropout regions, if providing a number, will use it as constant
            value to fill all the regions. if providing a tuple for the `min` and `max`, will randomly select
            value for every pixel / voxel from the range `[min, max)`. if None, will compute the `min` and `max`
            value of input image then randomly select value to fill, default to None.
        max_holes: if not None, define the maximum number to randomly select the expected number of regions.
        max_spatial_size: if not None, define the maximum spatial size to randomly select size for every region.
            if some components of the `max_spatial_size` are non-positive values, the transform will use the
            corresponding components of input img size. For example, `max_spatial_size=(32, -1)` will be adapted
            to `(32, 64)` if the second spatial dimension size of img is `64`.
        prob: probability of applying the transform.

    """

    def __init__(
        self,
        holes: int,
        spatial_size: Union[Sequence[int], int],
        dropout_holes: bool = True,
        fill_value: Optional[Union[Tuple[float, float], float]] = None,
        max_holes: Optional[int] = None,
        max_spatial_size: Optional[Union[Sequence[int], int]] = None,
        prob: float = 0.1,
    ) -> None:
        super().__init__(
            holes=holes, spatial_size=spatial_size, max_holes=max_holes, max_spatial_size=max_spatial_size, prob=prob
        )
        self.dropout_holes = dropout_holes
        if isinstance(fill_value, (tuple, list)):
            if len(fill_value) != 2:
                raise ValueError("fill value should contain 2 numbers if providing the `min` and `max`.")
        self.fill_value = fill_value

    def _transform_holes(self, img: np.ndarray):
        """
        Fill the randomly selected `self.hole_coords` in input images.
        Please note that we usually only use `self.R` in `randomize()` method, here is a special case.

        """
        #fill_value = (img.min(), img.max()) if self.fill_value is None else self.fill_value
        fill_value = img.min() if self.fill_value is None else self.fill_value

        if self.dropout_holes:
            for h in self.hole_coords:
                if isinstance(fill_value, (tuple, list)):
                    img[h] = self.R.uniform(fill_value[0], fill_value[1], size=img[h].shape)
                else:
                    img[h] = fill_value
            ret = img
        else:
            if isinstance(fill_value, (tuple, list)):
                ret = self.R.uniform(fill_value[0], fill_value[1], size=img.shape).astype(img.dtype, copy=False)
            else:
                ret = np.full_like(img, fill_value)
            for h in self.hole_coords:
                ret[h] = img[h]
        return ret