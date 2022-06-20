
import numpy as np 
from monai.transforms import AddChannel, Compose, RandRotate90, Resize, NormalizeIntensity, Flip, ToTensor, RandSpatialCrop


class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key"""
    
    def __init__(self, standard_n_views=2, low_n_views=4 ,resize=[80,80,80]):
        self.standard_view_transform = Compose([AddChannel(),
                               RandSpatialCrop(roi_size= [104, 124, 104],max_roi_size=[156, 186, 156], random_center=True, random_size=True),
                               Resize(tuple(resize)),
                               ToTensor()])
        self.low_view_transform = Compose([AddChannel(),
                               RandSpatialCrop(roi_size= [78, 93, 78],max_roi_size=[104, 124, 104], random_center=True, random_size=True),
                               Resize(tuple(self._low_resize(resize))),
                               ToTensor()])

        self.standard_n_views = standard_n_views
        self.low_n_views = low_n_views

    def _low_resize(self, standard_resize):
        low_resize = []
        for element in standard_resize:
            low_resize.append(int(element/2))

        return low_resize 
                

    def __call__(self, x):
        views={}
        large_views = [self.standard_view_transform(x) for i in range(self.standard_n_views)]
        low_views = [self.low_view_transform(x) for i in range(self.low_n_views)]
        views['standard'], views['low_resolution'] = large_views, low_views
        
        return views
    