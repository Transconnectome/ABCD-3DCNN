## =================================== ##
## ======= DenseNet ======= ##
## =================================== ##

import collections
# model
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

## ========= DenseNet Model ========= #
#(ref) explanation - https://wingnim.tistory.com/39
#(ref) densenet3d - https://github.com/pytorch/vision/blob/main/torchvision/models/densenet.py
#(ref) pytorch - https://pytorch.org/vision/0.8/_modules/torchvision/models/densenet.html

class _DenseLayer(nn.Module):

    def __init__(self, num_input_features, growth_rate, bn_size):
        super().__init__()

        ## DenseNet Composite function: BN -> relu -> 3x3 conv
        # 1
        self.add_module('norm1', nn.BatchNorm3d(num_input_features))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv1', nn.Conv3d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False))

        # 2
        self.add_module('norm2', nn.BatchNorm3d(bn_size * growth_rate))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module('conv2', nn.Conv3d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False))

        #self.memory_efficient = memory_efficient
    
    def bn_function(self, inputs) -> Tensor:
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        return bottleneck_output

    def forward(self, x):
        if isinstance(x, Tensor):
            prev_features = [x]
        else:
            prev_features = x

        bottleneck_output = self.bn_function(prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        return new_features  ## **

class _DenseBlock(nn.ModuleDict):
    # receive and concatenate the outputs of all previous blocks as inputs 
    # growth rate? the number of channel of feature map in each layer
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate):
        super().__init__()
        
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate,
                                growth_rate, bn_size)
            self.add_module("denselayer%d" % (i + 1), layer)
    
    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)

class _Transition(nn.Sequential):
    ## convolution + pooling between block
    # in paper: bach normalization -> 1x1 conv layer -> 2x2 average pooling layer

    def __init__(self, num_input_features, num_output_features):
        super().__init__()
        self.add_module('norm', nn.BatchNorm3d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv3d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool3d(kernel_size=2, stride=2))

class DenseNet(nn.Module):
    """Densenet-BC model class
    Args:
        growth_rate (int) - how many filters to add each layer (k in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self, subject_data, args,
                 n_input_channels=1,conv1_t_size=7,conv1_t_stride=2,no_max_pool=False,
                 growth_rate=32,block_config=(6, 12, 24, 16),num_init_features=64,
                 bn_size=4,drop_rate=0,num_classes=1000):

        super(DenseNet, self).__init__()
        self.subject_data = subject_data
        self.brain_dtypes = args.data_type
        self.cat_target = args.cat_target
        self.num_target = args.num_target 
        self.target = args.cat_target + args.num_target
        
        self.n_input_channels = n_input_channels
        self.multi_channel = (len(self.brain_dtypes) > 1 and self.n_input_channels > 1)
        self.multimodal = (len(self.brain_dtypes) > 1)
        self.len_feature_extractor = len(self.brain_dtypes) if not self.multi_channel else 1
        self.conv1_t_size = conv1_t_size
        self.conv1_t_stride = conv1_t_stride
        self.growth_rate = growth_rate
        self.block_config = block_config
        self.num_init_features = num_init_features
        self.bn_size = bn_size
        self.drop_rate = drop_rate

        self.feature_extractors = self._make_feature_extractors()

        # Linear layer
        self.FClayers = self._make_fclayers()

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
    
    def _make_feature_extractors(self):
        feature_extractors = []
        for _ in range(self.len_feature_extractor):
            # First convolution  
            feature_extractor = nn.Sequential(collections.OrderedDict([
                ('conv0',nn.Conv3d(self.n_input_channels,
                                   self.num_init_features,
                                   kernel_size=(self.conv1_t_size, 7, 7),
                                   stride=(self.conv1_t_stride, 2, 2),
                                   padding=(self.conv1_t_size // 2, 3, 3),
                                   bias=False)),
                 ('norm0', nn.BatchNorm3d(self.num_init_features)),
                 ('relu0', nn.ReLU(inplace=True)),
                 ('pool0', nn.MaxPool3d(kernel_size=3, stride=2, padding=1))
            ]))
            # Each denseblock
            num_features = self.num_init_features
            for i, num_layers in enumerate(self.block_config):
                block = _DenseBlock(num_layers=num_layers,
                                    num_input_features=num_features,
                                    bn_size=self.bn_size,
                                    growth_rate=self.growth_rate)
                feature_extractor.add_module(f'denseblock{i+1}', block)
                num_features = num_features + num_layers*self.growth_rate

                if i != len(self.block_config) - 1:
                    trans = _Transition(num_input_features = num_features,
                                        num_output_features = num_features // 2)
                    feature_extractor.add_module(f'transition{i+1}', trans)
                    num_features = num_features // 2

            # Final batch norm
            feature_extractor.add_module('norm5', nn.BatchNorm3d(num_features))
            
            feature_extractors.append(feature_extractor)
            
        self.num_features = num_features
        
        return nn.ModuleList(feature_extractors)
    
    def _make_fclayers(self):
        FClayer = []
        in_dim = self.num_features * len(self.brain_dtypes) # case when output of each CNN is concatenated. Should be modified for other options(i.e. cross attentino MLP)
        for cat_label in self.cat_target:
            self.out_dim = len(self.subject_data[cat_label].value_counts())                        
            FClayer.append(nn.Sequential(nn.Linear(in_dim, self.out_dim)))

        for num_label in self.num_target:
            FClayer.append(nn.Sequential(nn.Linear(in_dim, 1)))

        return nn.ModuleList(FClayer)


    def forward(self, images):            
        outs = []
        results = {}
        for i in range(self.len_feature_extractor): # feed each brain modality into its own CNN
            x = images[:,i]
            features = self.feature_extractors[i](x)
            out = F.adaptive_avg_pool3d(features, output_size=(1, 1, 1))
            out = F.relu(out, inplace=True)
            out = torch.flatten(out, 1)
            outs.append(out)
            
        out = torch.cat(outs,1) # dimension option is important for keeping a shape of (BATCH_SIZE, NUMS OF IMAGE)
        for i in range(len(self.FClayers)):
            results[self.target[i]] = self.FClayers[i](out)
            
        return results

# reference: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html
class MHAEncoderBlock(nn.Module):
    def __init__(self, input_dim=1024, num_heads=8, dim_feedforward=1024*2, dropout=0.0):
        """
        Inputs:
            input_dim - Dimensionality of the input
            num_heads - Number of heads to use in the attention block
            dim_feedforward - Dimensionality of the hidden layer in the MLP
            dropout - Dropout probability to use in the dropout layers
        """
        super().__init__()

        # Attention layer
        self.self_attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads,
                                               dropout=dropout, bias=True, batch_first=True)

        # Two-layer MLP
        self.linear_net = nn.Sequential(collections.OrderedDict([
            ('lin0', nn.Linear(input_dim, dim_feedforward)),
            ('drop', nn.Dropout(dropout)),
            ('relu0', nn.ReLU(inplace=True)),
            ('lin1', nn.Linear(dim_feedforward, input_dim))
        ]))

        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Attention part
        attn_out, attn_weights = self.self_attn(x,x,x,attn_mask=mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        # MLP part
        linear_out = self.linear_net(x)
        x = x + self.dropout(linear_out)
        x = self.norm2(x)

        return x


class DenseNetAttn(DenseNet):
    """Densenet-BC model class
    Args:
        growth_rate (int) - how many filters to add each layer (k in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - nu
                self.subject_data = subject_data
        self.brain_dtypes = args.data_type
        self.cat_target = args.cat_target
        self.num_target = args.num_target 
        self.target = args.cat_target + args.num_target
        
        self.n_input_channels = n_input_channels
        self.multi_channel = (len(self.brain_dtypes) > 1 and self.n_input_channels > 1)
        self.conv1_t_size = conv1_t_size
        self.conv1_t_stride = conv1_t_stride
        self.growth_rate = growth_rate
        self.block_config = block_config
        self.num_init_features = num_init_features
        self.bn_size = bn_size
        self.drop_rate = drop_ratember of classification classes
    """

    def __init__(self, subject_data, args,
                 n_input_channels=1,conv1_t_size=7,conv1_t_stride=2,no_max_pool=False,
                 growth_rate=32,block_config=(6, 12, 24, 16),num_init_features=64,
                 n_attn_blocks=3, bn_size=4,drop_rate=0,num_classes=1000):

        super(DenseNetAttn, self).__init__(subject_data, args)
#         self.subject_data = subject_data
#         self.brain_dtypes = args.data_type
#         self.cat_target = args.cat_target
#         self.num_target = args.num_target 
#         self.target = args.cat_target + args.num_target
        
#         self.n_input_channels = n_input_channels
#         self.multi_channel = (len(self.brain_dtypes) > 1 and self.n_input_channels > 1)
#         self.conv1_t_size = conv1_t_size
#         self.conv1_t_stride = conv1_t_stride
#         self.growth_rate = growth_rate
#         self.block_config = block_config
#         self.num_init_features = num_init_features
#         self.bn_size = bn_size
#         self.drop_rate = drop_rate
#         self.feature_extractors = self._make_feature_extractors()
        self.MHAEncoders = self._make_MHAEncoders(n_attn_blocks, input_dim=1024, num_heads=8,
                                                  dim_feedforward=1024*2, dropout=self.drop_rate)
#         # Linear layer
#         self.FClayers = self._make_fclayers()
        
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.LayerNorm):                 
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear) or isinstance(m, nn.modules.linear.NonDynamicallyQuantizableLinear):
                nn.init.constant_(m.bias, 0)

    def _make_MHAEncoders(self, n_attn_blocks, **block_args):
        MHA_encoders = []
        for _ in range(self.len_feature_extractor):
            MHA_blocks = nn.Sequential()
            for i in range(n_attn_blocks):
                MHA_blocks.add_module(f'MHAblock{i+1}', MHAEncoderBlock(**block_args))
            MHA_encoders.append(MHA_blocks)
            
        return nn.ModuleList(MHA_encoders)
            
    def forward(self, images):   
        results={}
        embeddings, outs = [], []
        for i in range(self.len_feature_extractor): # feed each brain modality into its own CNN
            x = images[:,i]
            feature = self.feature_extractors[i](x)
            embedding = feature.view(feature.shape[0], feature.shape[1], -1)
            embedding = embedding.permute(0,2,1)
            embeddings.append(embedding)
        
        for i, x in enumerate(embeddings):
            out = self.MHAEncoders[i](x)
            out = out.permute(0,2,1)
            out = F.adaptive_avg_pool1d(out, output_size=1)
            out = F.relu(out, inplace=True)
            out = torch.flatten(out, 1)
            outs.append(out)
            
        out = torch.cat(outs,1) # dimension option is important for keeping a shape of (BATCH_SIZE, NUMS OF IMAGE)
        
        for i in range(len(self.FClayers)):
            results[self.target[i]] = self.FClayers[i](out)
            
        return results    
    
class DenseNetMM(DenseNet):
    """Densenet-BC model class
    Args:
        growth_rate (int) - how many filters to add each layer (k in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self, subject_data, args,
                 n_input_channels=1,conv1_t_size=7,conv1_t_stride=2,no_max_pool=False,
                 growth_rate=32,block_config=(6, 12, 24, 16),num_init_features=64,
                 dim_proj=3,bn_size=4,drop_rate=0,num_classes=1000):

        super(DenseNetMM, self).__init__(subject_data, args)
        self.args = args
        self.proj_head, self.FClayers = None, None
        if args.mode == 'pretraining':
            self.dim_proj = dim_proj
            self.proj_heads = self._make_proj_heads()
        elif args.mode == 'finetuning':
            self.FClayers = self._make_fclayers()
        
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.LayerNorm):                 
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

                
    def _make_proj_heads(self):
        proj_heads = nn.Sequential()
        for i in range(len(self.brain_dtypes)):
            proj_heads.add_module(f'head{i}',nn.Linear(self.num_features, self.dim_proj))

        return nn.ModuleList(proj_heads)
            
        
    def forward(self, images):  
        outs = []
        for i in range(images.shape[1]): # feed each brain modality into its own CNN encoder to make representation vectors
            x = images[:,i]
            feature = self.feature_extractors[i](x)
            out = F.adaptive_avg_pool3d(feature, output_size=(1, 1, 1))
            out = F.relu(out, inplace=True)
            out = torch.flatten(out, 1)
            out = F.normalize(out, dim=1) # normalize vector for vector dimension.
            outs.append(out)
        
        if self.args.mode == 'pretraining':
            results=[]
            for i in range(len(self.proj_heads)): # projection heads of each multi-modal encoder
                rep_vec = outs[i]
                proj_vec = self.proj_heads[i](rep_vec)
                proj_vec = F.normalize(proj_vec, dim=1)
                results.append(proj_vec)
                
        elif self.args.mode == 'finetuning': # layers other than FClayers should be frozen !!
            results={}
            out = torch.cat(outs,1)
            for i in range(len(self.FClayers)): # multi-target
                results[self.target[i]] = self.FClayers[i](out)
            
        return results
    
    
def generate_model(model_depth, subject_data, args, **kwargs):
    assert model_depth in [121, 169, 201, 264]
    if 'MM' in args.model:
        chosen_net = DenseNetMM
    elif 'attn' in args.model:
        chosen_net = DenseNetAttn
    else:
        chosen_net = DenseNet

    if model_depth == 121:
        block_config=(6, 12, 24, 16)
    elif model_depth == 169:
        block_config=(6, 12, 32, 32)
    elif model_depth == 201:
        block_config=(6, 12, 48, 32)
    elif model_depth == 264:
         block_config=(6, 12, 64, 48)
        
    model = chosen_net(subject_data, args,
                     num_init_features=64,
                     growth_rate=32,
                     block_config=block_config,
                     **kwargs)
    return model


def densenet3D121(subject_data, args):
    model = generate_model(121, subject_data, args)
    return model

def densenet3D169(subject_data, args):
    model = generate_model(169, subject_data, args)
    return model

def densenet3D201(subject_data, args):
    model = generate_model(201, subject_data, args)
    return model

def densenet3D264(subject_data, args):
    model = generate_model(264, subject_data, args)
    return model
