import sys
sys.path.insert(0, '..')
from models.base import *
import torch.nn.functional as F
from scipy.special import softmax
from models.util import count_params, compute_flops
import torch
import tqdm
import time
import math

def get_shape(shape1, shape2):
    out_shape = []
    for d1, d2 in zip(shape1, shape2):
        out_shape.append(min(d1, d2))
    return out_shape


class Deeplab_ResNet_Backbone(nn.Module):
    def __init__(self, block, layers):
        self.inplanes = 64
        self.groups = 1 
        self.base_width = 64 

        super(Deeplab_ResNet_Backbone, self).__init__()
        self.conv1 = nn.Conv3d(1, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(self.inplanes, affine=affine_par)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1, ceil_mode=True) # change
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

        #
        # self.layer1 = self._make_layer(block, 64, layers[0])
        # self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
        self.layer_config = layers

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                conv1x1_3d(self.inplanes, planes * block.expansion, stride=stride),
                nn.BatchNorm3d(planes * block.expansion, affine = affine_par))
            # for i in downsample._modules['1'].parameters():
            #     i.requires_grad = False

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,base_width=self.base_width, dilation=dilation))

        return layers, downsample


    def forward(self, x, policy=None):
        """In this forward function, the operation is finished at the right before classifiers. 
        Average pooling and flatten are done here."""
        # stem layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        if policy is None:
            # forward through the all blocks without dropping
            for segment, num_blocks in enumerate(self.layer_config):
                for b in range(num_blocks):
                    # apply the residual skip out of _make_layers_

                    residual = self.ds[segment](x) if b == 0 and self.ds[segment] is not None else x
                    x = F.relu(residual + self.blocks[segment][b](x))

        else:
            # do the block dropping
            t = 0
            for segment, num_blocks in enumerate(self.layer_config):
                for b in range(num_blocks):
                    residual = self.ds[segment](x) if b == 0 and self.ds[segment] is not None else x
                    fx = F.relu(residual + self.blocks[segment][b](x))
                    if policy.ndimension() == 2:
                        x = fx * policy[t, 0] + residual * policy[t, 1]
                    elif policy.ndimension() == 3:
                        x = fx * policy[:, t, 0].contiguous().view(-1, 1, 1, 1) + residual * policy[:, t, 1].contiguous().view(-1, 1, 1, 1)
                    elif policy.ndimension() == 1:
                        x = fx * policy[t] + residual * (1-policy[t])
                    t += 1
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        
        return x


class MTL(nn.Module):
    def __init__(self, block, layers, init_method, init_neg_logits=None, opt=None):
        super(MTL, self).__init__()
        # about target label names and the number of targets
        self.cat_target = opt['task']['cat_target']
        self.num_target = opt['task']['num_target']
        self.tasks = opt['task']['targets']
        self.num_tasks = len(opt['task']['tasks_num_class'])
        self.opt = opt

        # model architecture
        self.backbone = Deeplab_ResNet_Backbone(block, layers)
        self.layers = layers
        self.block = block
        self._make_fclayers()
        
        # about NAS
        self.skip_layer = opt['skip_layer']
        self.init_method = init_method
        self.init_neg_logits = init_neg_logits

        self.reset_logits()
        self.policys = []
        for t_id in range(self.num_tasks):
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
        for t_id in range(self.num_tasks):
            policy = F.gumbel_softmax(getattr(self, 'task%d_logits' % (t_id + 1)), temperature, hard=hard_sampling)
            policys.append(policy)
        return policys

    def test_sample_policy(self, hard_sampling):
        self.policys = []
        if hard_sampling:
            cuda_device = self.task1_logits.get_device()
            logits1 = self.task1_logits.detach().cpu().numpy()
            policy1 = np.argmax(logits1, axis=1)
            policy1 = np.stack((1 - policy1, policy1), dim=1)
            if cuda_device != -1:
                self.policy1 = torch.from_numpy(np.array(policy1)).to('cuda:%d' % cuda_device)
            else:
                self.policy1 = torch.from_numpy(np.array(policy1))
            cuda_device = self.task2_logits.get_device()
            logits2 = self.task2_logits.detach().cpu().numpy()
            policy2 = np.argmax(logits2, axis=1)
            policy2 = np.stack((1 - policy2, policy2), dim=1)
            if cuda_device != -1:
                self.policy2 = torch.from_numpy(np.array(policy2)).to('cuda:%d' % cuda_device)
            else:
                self.policy2 = torch.from_numpy(np.array(policy2))
        else:
            for t_id in range(self.num_tasks):
                task_logits = getattr(self, 'task%d_logits' % (t_id + 1))
                cuda_device = task_logits.get_device()
                logits = task_logits.detach().cpu().numpy()
                distribution = softmax(logits, axis=-1)
                single_policys = []
                for tmp_d in distribution:
                    sampled = np.random.choice((1, 0), p=tmp_d)
                    policy = [sampled, 1 - sampled]
                    single_policys.append(policy)
                if cuda_device != -1:
                    policy = torch.from_numpy(np.array(single_policys)).to('cuda:%d' % cuda_device)
                else:
                    policy = torch.from_numpy(np.array(single_policys))
                # setattr(self, 'policy%d' % t_id, policy)
                self.policys.append(policy)

        return self.policys

    def reset_logits(self):
        num_layers = sum(self.layers)
        for t_id in range(self.num_tasks):
            if self.init_method == 'all_chosen':
                assert(self.init_neg_logits is not None)
                task_logits = self.init_neg_logits * torch.ones(num_layers - self.skip_layer, 2)
                task_logits[:, 0] = 0
            elif self.init_method == 'random':
                task_logits = 1e-3 * torch.randn(num_layers-self.skip_layer, 2)
            elif self.init_method == 'equal':
                task_logits = 0.5 * torch.ones(num_layers-self.skip_layer, 2)
            else:
                raise NotImplementedError('Init Method %s is not implemented' % self.init_method)
            
            self._arch_parameters = []
            self.register_parameter('task%d_logits' % (t_id + 1), nn.Parameter(task_logits, requires_grad=True))
            self._arch_parameters.append(getattr(self, 'task%d_logits' % (t_id + 1)))

    def _make_fclayers(self):
        """forwarding classifiers are performed in this MTL class().
        Thus, the function of making fc layers is implemented here. 
        """     
        for t_id in range(self.num_tasks):
                setattr(self, 'task%d_fc' % (t_id + 1), Classification_Module(512 * self.block.expansion, self.opt['task']['tasks_num_class'][t_id]))
        

    def forward(self, img, temperature, is_policy, num_train_layers=None, hard_sampling=False, mode='train'):
        """The reason why save final outputs from feed forwarding as dictionary data type is for easily calculating loss. """
        if num_train_layers is None:
            num_train_layers = sum(self.layers) - self.skip_layer

        num_train_layers = min(sum(self.layers) - self.skip_layer, num_train_layers)

        # Generate features
        cuda_device =img.get_device()
        if is_policy:
            if mode == 'train':
                self.policys = self.train_sample_policy(temperature, hard_sampling)
            elif mode == 'eval':
                self.policys = self.test_sample_policy(hard_sampling)
            elif mode == 'fix_policy':
                for p in self.policys:
                    assert(p is not None)

            else:
                raise NotImplementedError('mode %s is not implemented' % mode)
            

            skip_layer = sum(self.layers) - num_train_layers
            if cuda_device != -1:
                padding = torch.ones(skip_layer, 2).to(cuda_device)
            else:
                padding = torch.ones(skip_layer, 2)
            padding[:,1] = 0

            padding_policys = []
            feats = []


            for t_id in range(self.num_tasks):
                if cuda_device != -1:     
                    #self.policys[t_id] = self.policys[t_id].to(cuda_device)
                    padding_policy = torch.cat((padding.float().to(cuda_device),self.policys[t_id][-num_train_layers:].float().to(cuda_device)),dim=0) 
                    padding_policys.append(padding_policy)
                    feats.append(self.backbone(img,padding_policy))
                else:
                    self.policys[t_id] = self.policys[t_id].cpu()            
                    padding_policy = torch.cat((padding.float(),self.policys[t_id][-num_train_layers:].float()),dim=0) 
                    padding_policys.append(padding_policy)
                    feats.append(self.backbone(img,padding_policy))

        else:
            feats = [self.backbone(img)] * self.num_tasks

        # Get the output
        outputs = []
        for t_id in range(self.num_tasks):
            output = getattr(self, 'task%d_fc' % (t_id + 1))(feats[t_id])
            outputs.append(output)

        return outputs, self.policys, [None] * self.num_tasks

