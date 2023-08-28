import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import EfficientNet

from typing import Dict, List, NamedTuple, Optional, Tuple, Type, Union

efficientnet_params = {
    # model_name: (width_mult, depth_mult, image_size, dropout_rate, dropconnect_rate)
    "efficientnet-b0": (1.0, 1.0, 224, 0.2, 0.2),
    "efficientnet-b1": (1.0, 1.1, 240, 0.2, 0.2),
    "efficientnet-b2": (1.1, 1.2, 260, 0.3, 0.2),
    "efficientnet-b3": (1.2, 1.4, 300, 0.3, 0.2),
    "efficientnet-b4": (1.4, 1.8, 380, 0.4, 0.2),
    "efficientnet-b5": (1.6, 2.2, 456, 0.4, 0.2),
    "efficientnet-b6": (1.8, 2.6, 528, 0.5, 0.2),
    "efficientnet-b7": (2.0, 3.1, 600, 0.5, 0.2),
    "efficientnet-b8": (2.2, 3.6, 672, 0.5, 0.2),
    "efficientnet-l2": (4.3, 5.3, 800, 0.5, 0.2),
}



class EfficientNet3D(EfficientNet):
    def __init__(
        self,
        model_name: str,
        pretrained: bool = True,
        progress: bool = True,
        spatial_dims: int = 3,
        in_channels: int = 1,
        num_classes: int = 1000,
        norm: Union[str, tuple] = ("batch", {"eps": 1e-3, "momentum": 0.01}),
        adv_prop: bool = False,
        subject_data=None, 
        args=None
    ) -> None:
        """
        Generic wrapper around EfficientNet, used to initialize EfficientNet-B0 to EfficientNet-B7 models
        model_name is mandatory argument as there is no EfficientNetBN itself,
        it needs the N in [0, 1, 2, 3, 4, 5, 6, 7, 8] to be a model

        Args:
            model_name: name of model to initialize, can be from [efficientnet-b0, ..., efficientnet-b8, efficientnet-l2].
            pretrained: whether to initialize pretrained ImageNet weights, only available for spatial_dims=2 and batch
                norm is used.
            progress: whether to show download progress for pretrained weights download.
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            num_classes: number of output classes.
            norm: feature normalization type and arguments.
            adv_prop: whether to use weights trained with adversarial examples.
                This argument only works when `pretrained` is `True`.

        Examples::

            # for pretrained spatial 2D ImageNet
            >>> image_size = get_efficientnet_image_size("efficientnet-b0")
            >>> inputs = torch.rand(1, 3, image_size, image_size)
            >>> model = EfficientNetBN("efficientnet-b0", pretrained=True)
            >>> model.eval()
            >>> outputs = model(inputs)

            # create spatial 2D
            >>> model = EfficientNetBN("efficientnet-b0", spatial_dims=2)

            # create spatial 3D
            >>> model = EfficientNetBN("efficientnet-b0", spatial_dims=3)

            # create EfficientNetB7 for spatial 2D
            >>> model = EfficientNetBN("efficientnet-b7", spatial_dims=2)

        """
        # block args
        blocks_args_str = [
            "r1_k3_s11_e1_i32_o16_se0.25",
            "r2_k3_s22_e6_i16_o24_se0.25",
            "r2_k5_s22_e6_i24_o40_se0.25",
            "r3_k3_s22_e6_i40_o80_se0.25",
            "r3_k5_s11_e6_i80_o112_se0.25",
            "r4_k5_s22_e6_i112_o192_se0.25",
            "r1_k3_s11_e6_i192_o320_se0.25",
        ]

        # check if model_name is valid model
        if model_name not in efficientnet_params:
            model_name_string = ", ".join(efficientnet_params.keys())
            raise ValueError(f"invalid model_name {model_name} found, must be one of {model_name_string} ")

        # get network parameters
        weight_coeff, depth_coeff, image_size, dropout_rate, dropconnect_rate = efficientnet_params[model_name]

        # create model and initialize random weights
        super().__init__(
            blocks_args_str=blocks_args_str,
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            num_classes=num_classes,
            width_coefficient=weight_coeff,
            depth_coefficient=depth_coeff,
            dropout_rate=dropout_rate,
            image_size=image_size,
            drop_connect_rate=dropconnect_rate,
            norm=norm,
        )

        self.subject_data = subject_data
        self.cat_target = args.cat_target
        self.num_target = args.num_target 
        self.target = args.cat_target + args.num_target

        out_channels = self._fc.in_features      
        self._fc = self._make_fclayers(in_features=out_channels,softmax=True)      # replace the last FC layer of original source code of monai.networks.net.EfficientNet to multi-task version

    def _make_fclayers(self, in_features, softmax=False):
        FClayer = []
        
        for cat_label in self.cat_target:
            out_dim = len(self.subject_data[cat_label].value_counts())
            
            if softmax:                        
                FClayer.append(nn.Sequential(nn.Linear(in_features, out_dim), nn.Softmax(dim=1)))
            else: 
                FClayer.append(nn.Sequential(nn.Linear(in_features, out_dim)))

        for num_label in self.num_target:
            FClayer.append(nn.Sequential(nn.Linear(in_features, 1)))

        return nn.ModuleList(FClayer)

    def forward(self, inputs: torch.Tensor):
        """
        Args:
            inputs: input should have spatially N dimensions
            ``(Batch, in_channels, dim_0[, dim_1, ..., dim_N])``, N is defined by `dimensions`.

        Returns:
            a torch Tensor of classification prediction in shape ``(Batch, num_classes)``.
        """
        results = {}

        # Stem
        x = self._conv_stem(self._conv_stem_padding(inputs))
        x = self._swish(self._bn0(x))
        # Blocks
        x = self._blocks(x)
        # Head
        x = self._conv_head(self._conv_head_padding(x))
        x = self._swish(self._bn1(x))

        # Pooling and final linear layer
        x = self._avg_pooling(x)

        x = x.flatten(start_dim=1)
        x = self._dropout(x)
        for i in range(len(self._fc)):
            results[self.target[i]] = self._fc[i](x)
        return results



def generate_model(subject_data, args): 
    assert args.model.find('efficientnet3D') != -1 
    if args.model == 'efficientnet3D-b0': 
        model = EfficientNet3D(model_name="efficientnet-b0", subject_data=subject_data, args=args)
    elif args.model == 'efficientnet3D-b1': 
        model = EfficientNet3D(model_name="efficientnet-b1", subject_data=subject_data, args=args)
    elif args.model == 'efficientnet3D-b2': 
        model = EfficientNet3D(model_name="efficientnet-b2", subject_data=subject_data, args=args)
    elif args.model == 'efficientnet3D-b3': 
        model = EfficientNet3D(model_name="efficientnet-b3", subject_data=subject_data, args=args)
    elif args.model == 'efficientnet3D-b4': 
        model = EfficientNet3D(model_name="efficientnet-b4", subject_data=subject_data, args=args)
    elif args.model == 'efficientnet3D-b5': 
        model = EfficientNet3D(model_name="efficientnet-b5", subject_data=subject_data, args=args)
    elif args.model == 'efficientnet3D-b6': 
        model = EfficientNet3D(model_name="efficientnet-b6", subject_data=subject_data, args=args)
    elif args.model == 'efficientnet3D-b7': 
        model = EfficientNet3D(model_name="efficientnet-b7", subject_data=subject_data, args=args)
    return model 



def efficientnet3D(subject_data, args):
    model = generate_model(subject_data, args)
    return model
