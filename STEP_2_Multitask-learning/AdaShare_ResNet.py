## =================================== ##
## ======= ResNet ======= ##
## =================================== ##

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


## ========= ResNet Model ========= #
#(ref)https://github.com/ML4HPC/Brain_fMRI ##Brain_fMRI/resnet3d.py
#(ref)https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
def conv3x3_3d(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1_3d(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck3d(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck3d, self).__init__()
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        width = int(planes * (base_width / 64.)) * groups ## ***
        
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1_3d(inplanes, width)
        self.bn1 = norm_layer(width)
        
        self.conv2 = conv3x3_3d(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        
        self.conv3 = conv1x1_3d(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        
        self.relu = nn.ReLU(inplace=True)
        
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet3d(nn.Module):
    """여기에서는 fc layer 이전까지만 forwarding을 진행한다.
        policy 정보가 있어야지 task specific한 policy value를 구할 수 있기 때문에, 
        MTL_NAS의 class에서 fc layer들에 대한 forwarding을 진행하여야 한다. """
    def __init__(self, block, layers, subject_data, args,num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None): # parameters for hard parameter sharing model 
        
        super(ResNet3d, self).__init__()
        
        # attribute for configuration
        self.layers = layers
        self.cat_target = args.cat_target
        self.num_target = args.num_target 
        self.target = args.cat_target + args.num_target

        # attribute for building models
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1

        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv3d(1, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        

        # building layers
        strides = [1, 2, 1, 1]
        dilations = [1, 1, 2, 4]
        filt_sizes = [64, 128, 256, 512]
        self.blocks, self.ds = [], []

        for idx, (filt_size, num_blocks, stride, dilation) in enumerate(zip(filt_sizes, layers, strides, dilations)):
            blocks, ds = self._make_layer(block, filt_size, num_blocks, stride=stride, dilation=dilation)
            self.blocks.append(nn.ModuleList(blocks))
            self.ds.append(ds)

        self.blocks = nn.ModuleList(self.blocks)
        self.ds = nn.ModuleList(self.ds)        
        

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) or isinstance(m, Bottleneck3d):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
  

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                            conv1x1_3d(self.inplanes, planes * block.expansion, stride),
                            norm_layer(planes * block.expansion),
                            )
            

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=dilation,
                                norm_layer=norm_layer))

        return layers, downsample


        
    def forward(self, x, policy=None):
        """
        In this forward function, the operation is finished at the right before classifiers. 
        Average pooling and flatten are done here."""
        # stem layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # residual blocks
        if policy is None:
            # forward through the all blocks without dropping
            for segment, num_blocks in enumerate(self.layers):
                for b in range(num_blocks):
                    # apply the residual skip out of _make_layers_
                    residual = self.ds[segment](x) if b == 0 and self.ds[segment] is not None else x
                    x = self.relu(residual + self.blocks[segment][b](x))

        else:
            # do the block dropping
            t = 0
            for segment, num_blocks in enumerate(self.layers):
                for b in range(num_blocks):
                    p = policy[t,0]
                    residual = self.ds[segment](x) if b == 0 and self.ds[segment] is not None else x
                    if p == 1.:
                        x = self.relu(residual + self.blocks[segment][b](x))
                    elif p == 0.:
                        x = residual
                    else:
                        raise ValueError('p = %.2f is incorrect' % p)
                    t += 1

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        #for i in range(len(self.FClayer)):
        #    results[self.target[i]] = self.classifiers[i](x)

        return x
 

class MTL_Backbone(nn.Module):
    """Revising hard parameter sharing model as AdaShare version"""
    """MTL_Backbone은 curriculum learning의 첫번째 단계인 policy estimation을 마치고 난 다음에, policy값을 정해주고 학습을 할 때에 필요한 것이다. 
    policy는 모델 선언한 다음에 설정해주고, setattr()로 지정해주면 된다."""
    def __init__(self, block, layers, subject_data, args,init_method, init_neg_logits=None, skip_layer=0):
        super(MTL_NAS, self).__init__()
       
        # about target label names and the number of targets
        self.subject_data = subject_data
        self.cat_target = args.cat_target
        self.num_target = args.num_target
        self.targets = args.cat_target + args.num_target
        self.num_targets = len(self.targets)

        # about backbone architecture
        self.backbone = ResNet3d(block,layers,subject_data, args)
        self.layers = layers
        self.block_config = block
        self.classifiers = self._make_fclayers()
        
        # about NAS
        self.skip_layer = skip_layer
        self.init_method = init_method
        self.init_neg_logits = init_neg_logits

        self.reset_logits()

        self.policys = []
        for i in range(self.num_targets):
            self.policys.append(None)


    def arch_parameters(self):
        params = []
        for name, param in self.named_parameters():
            if 'task' in name and 'logits' in name:
                params.append(param)
        return params

    
    def backbone_parameters(self):
        params = []
        for name, param in self.named_parameters():
            if 'backbone' in name:
                params.append(param)
        return params

    
    def task_specific_parameters(self):
        params = []
        for name, param in self.named_parameters():
            if 'task' in name and 'fc' in name:
                params.append(param)
        return params
    

    def network_parameters(self):
        params = []
        for name, param in self.named_parameters():
            if not ('task' in name and 'logits' in name):
                params.append(param)
        return params


    def train_sample_policy(self, temperature, hard_sampling):
        policys = []
        for target in self.targets:
            policy = F.gumbel_softmax(getattr(self, 'task%s_logits' % target), temperature, hard=hard_sampling)
            policys.append(policy)
        return policys 

    
    def test_sample_policy(self, hard_sampling):
        self.policy = []
        if hard_sampling: # 이 조건문의 경우 아직 구현 제대로 안함. 따라서 일단은 hard_sampling=False로 진행
            cuda_device = self.task1_logits.get_device()
            logits1 = self.task1_logits.detach().cpu().numpy()
            policy1 = np.argmax(logits1, axis=1)
            policy1 = np.stack((1-policy1,policy1),dim=1)
            if cuda_device != -1:
                self.policy1 = torch.from_numpy(np.array(policy1)).to('cuda:%d' % cuda_device)
            else:
                self.policy1 = torch.from_numpy(np.array(policy1))
        else:
            for target in self.targets:
                task_logits = getattr(self,'task%s_logits' % target)
                cuda_device = task_logits.get_device()
                logits = task_logits.detach().cpu().numpy()
                distribution = softmax(logits, axis=-1)
                single_policys = []
                for tmp_d in distribution:
                    sampled = np.random.choice((1,0), p=tmp_d)
                    policy = [sampled, 1-sampled]
                    single_policys.append(policy)
                if cuda_device != -1:
                    policy = torch.from_numpy(np.array(single_policys)).to('cuda:%d' % cuda_device)
                else:
                    policy = torch.from_numpy(np.array(single_policys))
                    self.policys.append(policy)
        return self.policys
                
        
    def reset_logits(self):
        num_layers = sum(self.layers)
            
        for target in self.targets:
            if self.init_method =='all_chosen':
                assert(self.init_neg_logits is not None)
                task_logits = self.init_neg_logits * torch.ones(num_layers - self.skip_layer,2)
                task_logits[:,0] = 0
            elif self.init_method == 'random':
                task_logits = 1e-3 * torch.randm(num_layers - self.skip_layer, 2)
            elif self.init_method == 'equal':
                task_logits = 0.5* torch.ones(num_layers - self.skip_layer, 2)
            else:
                raise NotImplementedError("Init Method %s is not implemented" % self.init_method)

            self._arch_parameters = []
            self.register_parameter('task%s_logits' % target, nn.Parameter(task_logits, requires_grad = True))
            self._arch_parameters.append(getattr(self, 'task%s_logits' % target))
        
    def _make_fclayers(self):
        """forwarding classifiers are performed in this MTL_NAS class().
        Thus, the function of making fc layers is implemented here. 
        """     
        self.FClayer = []
        
        for cat_label in self.cat_target:
            self.out_dim = len(self.subject_data[cat_label].value_counts()) 
            #self.fc = nn.Sequential(nn.Linear(512 * block.expansion, self.out_dim),
            #                        nn.Softmax(dim=1))
            #self.FClayer.append(self.fc)
            setattr(self, 'task_%s_fc' % cat_label, nn.Sequential(nn.Linear(512 * self.block_config.expansion, self.out_dim), nn.Softmax(dim=1)))


        for num_label in self.num_target:
            #self.fc = nn.Sequential(nn.Linear(512 * block.expansion, 1))
            #self.FClayer.append(self.fc)
            setattr(self, 'task_%s_fc' % num_label, nn.Linear(512 * self.block_config.expansion, 1) )

        return nn.ModuleList(self.FClayer)

                 
    def forward(self, img, temperature, is_policy, num_train_layers=None, hard_sampling=False, mode='train'):
        """The reason why save final outputs from feed forwarding as dictionary data type is for easily calculating loss. """
#        if num_train_layers is None:
#            num_train_layers = sum(self.layers) - self.skip_layer

#        num_train_layers = min(sum(self.layers) - self.skip_layer, num_train_layers)
        
        # Generate features
#        cuda_device =img.get_device()
        if is_policy:
#            if mode == 'train':
#                self.policys = self.train_sample_policy(temperature, hard_sampling)
#            elif mode == 'eval':
#                self.policys = self.test_sample_policy(hard_sampling)
#            elif mode == 'fix_policy':
#                for p in self.policys:
#                    assert(p is not None)
#            else:
#                raise NotImplementedError('mode %s is not implemented' % mode)
#
#            for t_id in range(self.num_targets):
#                if cuda_device != -1:     
#                    self.policys[t_id] = self.policys[t_id].to(cuda_device)
#                else:
#                    self.policys[t_id] = self.policys[t_id].cpu()
#
#            skip_layer = sum(self.layers) - num_train_layers
#            if cuda_device != -1:
#                padding = torch.ones(skip_layer, 2).to(cuda_device)
#            else:
#                padding = torch.ones(skip_layer, 2)
#            padding[:,1] = 0
#
#            padding_policys = []
            feats = []                         
            for t_id in range(self.num_targets): # the order of tasks is cat_target~num_target
                #padding_policy = torch.cat((padding.float(),self.policys[t_id][-num_train_layers:].float()),dim=0) 
                #padding_policys.append(padding_policy)
                padding_policy = self.policys[t_id]
                print(self.policys)
                feats.append(self.backbone(img,padding_policy)) # forwarding images until reaching the right before classifier.
        else:
            feats = [self.backbone(img)] * self.num_targets

        results = {}
        for t_id in range(self.num_targets): # the order of tasks is cat_target~num_target
            results[self.targets[t_id]] = getattr(self, 'task_%s_fc' % self.targets[t_id])(feats[t_id])


        return results, self.policys                    



class MTL_NAS(nn.Module):
    """Revising hard parameter sharing model as AdaShare version"""
    """MTL_NAS는 curriculum learning의 첫번째 단계인 policy estimation을 위해서 필요한 단계이다. (원문 페이퍼의 MTL2 클래스를 참조)"""
    def __init__(self, block, layers, subject_data, args,init_method, init_neg_logits=None, skip_layer=0):
        super(MTL_NAS, self).__init__()
       
        # about target label names and the number of targets
        self.subject_data = subject_data
        self.cat_target = args.cat_target
        self.num_target = args.num_target
        self.targets = args.cat_target + args.num_target
        self.num_targets = len(self.targets)

        # about backbone architecture
        self.backbone = ResNet3d(block,layers,subject_data, args)
        self.layers = layers
        self.block_config = block
        self.classifiers = self._make_fclayers()
        
        # about NAS
        self.skip_layer = skip_layer
        self.init_method = init_method
        self.init_neg_logits = init_neg_logits

        self.reset_logits()

        self.policys = []
        for i in range(self.num_targets):
            self.policys.append(None)


    def arch_parameters(self):
        params = []
        for name, param in self.named_parameters():
            if 'task' in name and 'logits' in name:
                params.append(param)
        return params

    
    def backbone_parameters(self):
        params = []
        for name, param in self.named_parameters():
            if 'backbone' in name:
                params.append(param)
        return params

    
    def task_specific_parameters(self):
        params = []
        for name, param in self.named_parameters():
            if 'task' in name and 'fc' in name:
                params.append(param)
        return params
    

    def network_parameters(self):
        params = []
        for name, param in self.named_parameters():
            if not ('task' in name and 'logits' in name):
                params.append(param)
        return params


    def train_sample_policy(self, temperature, hard_sampling):
        policys = []
        for target in self.targets:
            policy = F.gumbel_softmax(getattr(self, 'task%s_logits' % target), temperature, hard=hard_sampling)
            policys.append(policy)
        return policys 

    
    def test_sample_policy(self, hard_sampling):
        self.policy = []
        if hard_sampling: # 이 조건문의 경우 아직 구현 제대로 안함. 따라서 일단은 hard_sampling=False로 진행
            cuda_device = self.task1_logits.get_device()
            logits1 = self.task1_logits.detach().cpu().numpy()
            policy1 = np.argmax(logits1, axis=1)
            policy1 = np.stack((1-policy1,policy1),dim=1)
            if cuda_device != -1:
                self.policy1 = torch.from_numpy(np.array(policy1)).to('cuda:%d' % cuda_device)
            else:
                self.policy1 = torch.from_numpy(np.array(policy1))
        else:
            for target in self.targets:
                task_logits = getattr(self,'task%s_logits' % target)
                cuda_device = task_logits.get_device()
                logits = task_logits.detach().cpu().numpy()
                distribution = softmax(logits, axis=-1)
                single_policys = []
                for tmp_d in distribution:
                    sampled = np.random.choice((1,0), p=tmp_d)
                    policy = [sampled, 1-sampled]
                    single_policys.append(policy)
                if cuda_device != -1:
                    policy = torch.from_numpy(np.array(single_policys)).to('cuda:%d' % cuda_device)
                else:
                    policy = torch.from_numpy(np.array(single_policys))
                    self.policys.append(policy)
        return self.policys
                
        
    def reset_logits(self):
        num_layers = sum(self.layers)
            
        for target in self.targets:
            if self.init_method =='all_chosen':
                assert(self.init_neg_logits is not None)
                task_logits = self.init_neg_logits * torch.ones(num_layers - self.skip_layer,2)
                task_logits[:,0] = 0
            elif self.init_method == 'random':
                task_logits = 1e-3 * torch.randm(num_layers - self.skip_layer, 2)
            elif self.init_method == 'equal':
                task_logits = 0.5* torch.ones(num_layers - self.skip_layer, 2)
            else:
                raise NotImplementedError("Init Method %s is not implemented" % self.init_method)

            self._arch_parameters = []
            self.register_parameter('task%s_logits' % target, nn.Parameter(task_logits, requires_grad = True))
            self._arch_parameters.append(getattr(self, 'task%s_logits' % target))
        
    def _make_fclayers(self):
        """forwarding classifiers are performed in this MTL_NAS class().
        Thus, the function of making fc layers is implemented here. 
        """     
        self.FClayer = []
        
        for cat_label in self.cat_target:
            self.out_dim = len(self.subject_data[cat_label].value_counts()) 
            #self.fc = nn.Sequential(nn.Linear(512 * block.expansion, self.out_dim),
            #                        nn.Softmax(dim=1))
            #self.FClayer.append(self.fc)
            setattr(self, 'task_%s_fc' % cat_label, nn.Sequential(nn.Linear(512 * self.block_config.expansion, self.out_dim), nn.Softmax(dim=1)))


        for num_label in self.num_target:
            #self.fc = nn.Sequential(nn.Linear(512 * block.expansion, 1))
            #self.FClayer.append(self.fc)
            setattr(self, 'task_%s_fc' % num_label, nn.Linear(512 * self.block_config.expansion, 1) )

        return nn.ModuleList(self.FClayer)

                 
    def forward(self, img, temperature, is_policy, num_train_layers=None, hard_sampling=False, mode='train'):
        """The reason why save final outputs from feed forwarding as dictionary data type is for easily calculating loss. """
#        if num_train_layers is None:
#            num_train_layers = sum(self.layers) - self.skip_layer

#        num_train_layers = min(sum(self.layers) - self.skip_layer, num_train_layers)
        
        # Generate features
#        cuda_device =img.get_device()
        if is_policy:
#            if mode == 'train':
#                self.policys = self.train_sample_policy(temperature, hard_sampling)
#            elif mode == 'eval':
#                self.policys = self.test_sample_policy(hard_sampling)
#            elif mode == 'fix_policy':
#                for p in self.policys:
#                    assert(p is not None)
#            else:
#                raise NotImplementedError('mode %s is not implemented' % mode)
#
#            for t_id in range(self.num_targets):
#                if cuda_device != -1:     
#                    self.policys[t_id] = self.policys[t_id].to(cuda_device)
#                else:
#                    self.policys[t_id] = self.policys[t_id].cpu()
#
#            skip_layer = sum(self.layers) - num_train_layers
#            if cuda_device != -1:
#                padding = torch.ones(skip_layer, 2).to(cuda_device)
#            else:
#                padding = torch.ones(skip_layer, 2)
#            padding[:,1] = 0
#
#            padding_policys = []
            feats = []                         
            for t_id in range(self.num_targets): # the order of tasks is cat_target~num_target
                #padding_policy = torch.cat((padding.float(),self.policys[t_id][-num_train_layers:].float()),dim=0) 
                #padding_policys.append(padding_policy)
                padding_policy = self.policys[t_id]
                print(self.policys)
                feats.append(self.backbone(img,padding_policy)) # forwarding images until reaching the right before classifier.
        else:
            feats = [self.backbone(img)] * self.num_targets

        results = {}
        for t_id in range(self.num_targets): # the order of tasks is cat_target~num_target
            results[self.targets[t_id]] = getattr(self, 'task_%s_fc' % self.targets[t_id])(feats[t_id])


        return results, self.policys                    

        


                





def resnet3D50(subject_data,args):
    #model = MTL_Backbone(Bottleneck3d, [3, 4, 6, 3], subject_data, args, 'equal')
    model = MTL_NAS(Bottleneck3d, [3, 4, 6, 3], subject_data, args, 'equal')

    return model

#def resnet3D101(**kwargs):
    #layers = [3, 4, 23, 3]
    #model = ResNet3d(Bottleneck3d, layers, **kwargs)
    #model = MTL(model, args.cat_target, args.num_target, subject_data)
    #return model
        

#def resnet3D152(**kwargs):
    #layers = [3, 8, 36, 3]
    #model = ResNet3d(Bottleneck3d, layers, **kwargs)
    #model = MTL(model, args.cat_target, args.num_target, subject_data)
    #return model
      

## ====================================== ##