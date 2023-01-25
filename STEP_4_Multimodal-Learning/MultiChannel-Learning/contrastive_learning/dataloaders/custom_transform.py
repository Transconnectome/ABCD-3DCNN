# custom_transform
from monai.transforms import MaskIntensity
# from monai.data.meta_obj import get_track_meta # doesn't work in monai 0.8.0
from monai.transforms.utils import Fourier, equalize_hist, is_positive, rescale_array
from monai.utils.type_conversion import convert_to_tensor, convert_to_dst_type
from monai.data import NibabelReader

class MaskTissue(MaskIntensity):
    def __init__(self, mask_data_array = None, tissue_type:int = None, select_fn = is_positive) -> None:
        self.mask_data_array = mask_data_array
        self.tissue_type = tissue_type
        self.select_fn = select_fn
        self.reader = NibabelReader()
        self.idx = 0

    def __call__(self, img, mask_data_array = None):
        """
                        < Modified Version of MaskIntensity>
        Args:
            mask_data_array: it should be an array of brain images. 
            tissue_type: five types of tissues which are ['cgm', 'scgm', 'wm', 'csf', 'pt'].
            select_fn: function to select valid values of the `mask_data`, default is
            to select `values > 0`.
        Raises:
            - ValueError: When both ``mask_data_array`` and ``self.mask_data_array`` are None.
            - ValueError: When ``tissue_type`` is None.
            - ValueError: When ``mask_data`` and ``img`` channels differ and ``mask_data`` is not single channel.

        """
        img = convert_to_tensor(img)
        mask_data_array = self.mask_data_array if mask_data_array is None else mask_data_array
        
        if mask_data_array is None:
            raise ValueError("must provide the mask_data_array when initializing the transform or at runtime.")
        if self.tissue_type is None:
            raise ValueError("must provide <tissue_type> when initializing the transform.")
        tissue_idx = ['cgm', 'scgm', 'wm', 'csf', 'pt'].index(self.tissue_type)
        
        mask_data = self.reader.read(mask_data_array.iloc[self.idx]).get_fdata()[:,:,:,tissue_idx]
        mask_data_, *_ = convert_to_dst_type(src=mask_data, dst=img)

        mask_data_ = self.select_fn(mask_data_)
        if mask_data_.shape[0] != 1 and mask_data_.shape[0] != img.shape[0]:
            raise ValueError(
                "When mask_data is not single channel, mask_data channels must match img, "
                f"got img channels={img.shape[0]} mask_data channels={mask_data_.shape[0]}."
            )
        self.idx += 1
        
        return convert_to_dst_type(img * mask_data_, dst=img)[0]