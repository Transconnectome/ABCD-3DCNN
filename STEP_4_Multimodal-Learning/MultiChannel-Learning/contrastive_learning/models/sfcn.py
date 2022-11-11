from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

class SFCN(nn.Module):
    def __init__(self, subject_data, args, channel_number=[32, 64, 128, 256, 256, 64], output_dim=40, dropout=True):
        super(SFCN, self).__init__()
        # Setting experiment related variables
        self.subject_data = subject_data
        self.cat_target = args.cat_target
        self.num_target = args.num_target 
        self.target = args.cat_target + args.num_target
        
        # Setting model related variables
        self.n_layer = len(channel_number)
        self.channel_number = channel_number
        self.last_feature = channel_number[-1]
        self.output_dim = output_dim
        self.dropout = dropout
        
        self.brain_dtypes = args.data_type
        
        # make feature extractor
        self.feature_extractors = self._make_feature_extractors()
        
        # make classifier part
        self.FClayers = self._make_fclayers()
        
#         avg_shape = max(set(args.resize))//(2**len(channel_number))
#         self.classifier = self._make_classifier(avg_shape)
        
        # initialize trainable weights
        self._init_weights()

        
    @staticmethod
    def conv_layer(in_channel, out_channel, maxpool=True, kernel_size=3, padding=0, maxpool_stride=2):
        if maxpool is True:
            layer = nn.Sequential(
                nn.Conv3d(in_channel, out_channel, padding=padding, kernel_size=kernel_size),
                nn.BatchNorm3d(out_channel),
                nn.MaxPool3d(2, stride=maxpool_stride),
                nn.ReLU(),
            )
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channel, out_channel, padding=padding, kernel_size=kernel_size),
                nn.BatchNorm3d(out_channel),
                nn.ReLU()
            )
        return layer
    
    
    def _make_feature_extractors(self):
        feature_extractors = []
        
        for brain_dtype in self.brain_dtypes:
            feature_extractor = nn.Sequential()

            for i in range(self.n_layer):
                in_channel = 1 if i == 0 else self.channel_number[i-1]
                out_channel = self.channel_number[i]

                curr_kernel_size = 3 if i < self.n_layer-1 else 1
                curr_padding = 1 if i < self.n_layer-1 else 0

                feature_extractor.add_module('conv_%d' % i,
                                             self.conv_layer(
                                                 in_channel, out_channel,
                                                 maxpool=True, kernel_size=curr_kernel_size,
                                                 padding=curr_padding))
                
            feature_extractors.append(feature_extractor)
                
        return nn.ModuleList(feature_extractors)
    
    
    def _make_fclayers(self):
        FClayer = []
        in_dim = self.last_feature * len(self.brain_dtypes) # should be modified
        
        for target in self.target:
            out_dim = len(self.subject_data[target].value_counts()) if target in self.cat_target else 1
            FClayer.append(nn.Sequential(nn.Linear(in_dim, out_dim)))

        return nn.ModuleList(FClayer)
    
       
    def make_classifier(self, avg_shape):
        self.classifier = nn.Sequential()
        if avg_shape >1:
            self.classifier.add_module('average_pool', nn.AvgPool3d(avg_shape))
        if self.dropout is True:
            self.classifier.add_module('dropout', nn.Dropout(0.5))
        i = self.n_layer
        in_channel = self.channel_number[-1]
        out_channel = output_dim
        self.classifier.add_module('conv_%d' % i,
                                   nn.Conv3d(in_channel, out_channel, padding=0, kernel_size=1))

        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
        
    def forward(self, images):
#         out = list()
#         x_f = self.features(x)
#         x = self.classifier(x_f)
#         x = F.log_softmax(x, dim=1)
#         out.append(x)
#         return out

        outs = []
        results = {'embeddings':[]} if len(self.brain_dtypes) > 1 else {}
        for i, x in enumerate(images): # feed each brain modality into its own CNN
            features = self.feature_extractors[i](x)
            if len(self.brain_dtypes) > 1:
                results['embeddings'].append(torch.flatten(features, 1))
            out = F.adaptive_avg_pool3d(features, output_size=(1, 1, 1))
            out = F.relu(out, inplace=True)
            out = torch.flatten(out, 1)
            outs.append(out)
            
        out = torch.cat(outs,1) # should be modified
        
        for i in range(len(self.FClayers)):
            results[self.target[i]] = self.FClayers[i](out)
            
        return results
