import os
import time
import numpy as np
import torch
from torch import device, nn
import torch.nn.functional as F
#from utils.util import print_current_errors
# from data_utils.image_decoder import inv_preprocess, decode_labels2


class BaseEnv():
    """
    The environment to train a simple classification model
    """

    def __init__(self, log_dir, checkpoint_dir, exp_name, tasks_num_class, device=0, is_train=True, opt=None):
        """
        :param log_dir: str, the path to save logs
        :param checkpoint_dir: str, the path to save checkpoints
        :param lr: float, the learning rate
        :param is_train: bool, specify during the training
        """
        print(self.name())
        self.checkpoint_dir = os.path.join(checkpoint_dir, exp_name)
        self.log_dir = os.path.join(log_dir, exp_name)
        self.is_train = is_train
        self.tasks_num_class = tasks_num_class
        self.device_id = device
        self.opt = opt
        self.dataset = self.opt['dataload']['dataset']
        self.tasks = self.opt['task']['targets'] 
        if torch.cuda.is_available():
            self.device = torch.device("cuda:%d" % device)

        self.networks = {}
        self.define_networks(tasks_num_class)

        self.define_loss()
        self.losses = {}    # This dictionary used for backpropagation of models

        # This dictitionary used for reporting results while training models
        self.results = {}   
        if opt['task']['cat_target']:
            for cat_target in opt['task']['cat_target']:
                self.results[cat_target] = {'loss':[], 'ACC or R2':[]}
        if opt['task']['num_target']:
            for num_target in opt['task']['num_target']:
                self.results[num_target] = {'loss':[], 'ACC or R2':[]}
        
        self.optimizers = {}
        self.schedulers = {}
        if is_train:
            # define optimizer
            self.define_optimizer()
            self.define_scheduler()
            # define summary writer
            #self.writer = SummaryWriter(log_dir=self.log_dir)

    # ##################### define networks / optimizers / losses ####################################

    def define_loss(self):
        self.cross_entropy = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()
        self.cross_entropy_sparsity = nn.CrossEntropyLoss()
        #self.cosine_similiarity = nn.CosineSimilarity()
        #self.l1_loss = nn.L1Loss()
        #self.l1_loss2 = nn.L1Loss(reduction='none')
        #if self.dataset == 'NYU_v2':
        #    self.cross_entropy = nn.CrossEntropyLoss(ignore_index=255)
        #    self.cross_entropy_sparsity = nn.CrossEntropyLoss(ignore_index=255)

        #    self.cross_entropy2 = nn.CrossEntropyLoss(ignore_index=255, reduction='none')
        #elif self.dataset == 'Taskonomy':
        #    dataroot = self.opt['dataload']['dataroot']
        #    weight = torch.from_numpy(np.load(os.path.join(dataroot, 'semseg_prior_factor.npy'))).to(self.device).float()
        #    self.cross_entropy = nn.CrossEntropyLoss(weight=weight, ignore_index=255)
        #    self.cross_entropy2 = nn.CrossEntropyLoss(ignore_index=255, reduction='none')
        #    self.cross_entropy_sparisty = nn.CrossEntropyLoss(ignore_index=255)
        #elif self.dataset == 'CityScapes':
        #    self.cross_entropy = nn.CrossEntropyLoss(ignore_index=-1)
        #    self.cross_entropy2 = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')
        #    self.cross_entropy_sparsity = nn.CrossEntropyLoss(ignore_index=-1)
        #else:
        #    raise NotImplementedError('Dataset %s is not implemented' % self.dataset)

    def define_networks(self, tasks_num_class):
        pass

    def define_optimizer(self):
        pass

    def define_scheduler(self):
        pass
    
    
    # ##################### train / test ####################################
    def set_inputs(self, batch):
        """
        :param batch: [images (a tensor [batch_size, d, h, w]), {'categories'1: np.ndarray [batch_size,], 'categories'1: np.ndarray [batch_size,], ...}]
        """
        self.img = batch[0]
        if torch.cuda.is_available():
            self.img = self.img.to(self.device)

        for tasks in self.opt['task']['targets']:
            if torch.cuda.is_available():
                setattr(self,'ground_truth_%s' % tasks, batch[1][tasks].to(self.device))


    def calculating_loss_acc(self):
         # calculating task-specific loss for backpropagation in each mini-batch

        if self.opt['task']['cat_target']:
            for cat_target in self.opt['task']['cat_target']:
                self.losses[cat_target] = {}
                predicted = getattr(self, '%s_pred' % cat_target)
                ground_truth = getattr(self, 'ground_truth_%s' % cat_target)
                loss = self.cross_entropy(predicted.float(), ground_truth.long())
                
                _, pred_class = torch.max(predicted.data, 1)
                correct = (pred_class == ground_truth).sum().item()
                total = ground_truth.size(0)

                self.results[cat_target]['ACC or R2'] = 100 * correct / total
                self.results[cat_target]['loss'] = loss.item()
                self.losses[cat_target]['total'] = loss
                
        
        if self.opt['task']['num_target']: 
            for num_target in self.opt['task']['num_target']:
                self.losses[num_target] = {}
                predicted = getattr(self, '%s_pred' % num_target)
                groud_truth = getattr(self,'ground_truth_%s' % num_target)
                loss = self.mse(predicted.float(), groud_truth.float().unsqueeze(1))
                
                y_var = torch.var(ground_truth)
                r_square = 1 - (loss / y_var) 
                self.results[num_target]['ACC or R2'] = r_square.item()
                self.results[num_target]['loss'] = loss.item()
                self.losses[num_target]['total'] = loss
        

    # ##################### change the state of each module ####################################
    def get_current_state(self, current_iter):
        current_state = {}
        for k, v in self.networks.items():
            if isinstance(v, nn.DataParallel):
                current_state[k] = v.module.state_dict()
            else:
                current_state[k] = v.state_dict()
        for k, v in self.optimizers.items():
            current_state[k] = v.state_dict()
        current_state['iter'] = current_iter
        return current_state

    def save(self, label, current_iter):
        """
        Save the current checkpoint
        :param label: str, the label for the loading checkpoint
        :param current_iter: int, the current iteration
        """
        current_state = self.get_current_state(current_iter)
        save_filename = '%s_model.pth.tar' % str(label)
        save_path = os.path.join(self.checkpoint_dir, save_filename)
        torch.save(current_state, save_path)

    def load_snapshot(self, snapshot):
        for k, v in self.networks.items():
            if k in snapshot.keys():
                # loading values for the existed keys
                model_dict = v.state_dict()
                pretrained_dict = {}
                for kk, vv in snapshot[k].items():
                    if kk in model_dict.keys() and model_dict[kk].shape == vv.shape:
                        pretrained_dict[kk] = vv
                    else:
                        print('skipping %s' % kk)
                model_dict.update(pretrained_dict)
                self.networks[k].load_state_dict(model_dict)
                # self.networks[k].load_state_dict(snapshot[k])
        if self.is_train:
            for k, v in self.optimizers.items():
                if k in snapshot.keys():
                    self.optimizers[k].load_state_dict(snapshot[k])
        return snapshot['iter']

    def load(self, label, path=None):
        """
        load the checkpoint
        :param label: str, the label for the loading checkpoint
        :param path: str, specify if knowing the checkpoint path
        """
        if path is None:
            save_filename = '%s_model.pth.tar' % label
            save_path = os.path.join(self.checkpoint_dir, save_filename)
        else:
            save_path = path
        if os.path.isfile(save_path):
            print('=> loading snapshot from {}'.format(save_path))
            snapshot = torch.load(save_path, map_location='cuda:%d' % self.device_id)
            return self.load_snapshot(snapshot)
        else:
            raise ValueError('snapshot %s does not exist' % save_path)

    # ##################### visualize #######################
    # def visualize(self):
    #     # TODO: implement the visualization of depth
    #     save_results = {}
    #     if 'seg' in self.tasks:
    #         num_seg_class = self.tasks_num_class[self.tasks.index('seg')]
    #         self.save_seg = decode_labels2(torch.argmax(self.seg_output, dim=1).unsqueeze(dim=1), num_seg_class, 'seg', self.seg)
    #         self.save_gt_seg = decode_labels2(self.seg, num_seg_class, 'seg', self.seg)
    #         save_results['save_seg'] = self.save_seg
    #         save_results['save_gt_seg'] = self.save_gt_seg
    #     if 'sn' in self.tasks:
    #         self.save_normal = decode_labels2(F.normalize(self.sn_output) * 255, None, 'normal', F.normalize(self.normal.float()) * 255)
    #         self.save_gt_normal = decode_labels2(F.normalize(self.normal.float()) * 255, None, 'normal', F.normalize(self.normal.float()) * 255,)
    #         save_results['save_sn'] = self.save_normal
    #         save_results['save_gt_sn'] = self.save_gt_normal
    #     if 'depth' in self.tasks:
    #         self.save_depth = decode_labels2(self.depth_output, None, 'depth', self.depth.float())
    #         self.save_gt_depth = decode_labels2(self.depth.float(), None, 'depth', self.depth.float())
    #         save_results['save_depth'] = self.save_depth
    #         save_results['save_gt_depth'] = self.save_gt_depth
    #     self.save_img = inv_preprocess(self.img)
    #     save_results['save_img'] = self.save_img
    #     return save_results

    # ##################### change the state of each module ####################################

    def train(self):
        """
        Change to the training mode
        """
        for k, v in self.networks.items():
            v.train()

    def eval(self):
        """
        Change to the eval mode
        """
        for k, v in self.networks.items():
            v.eval()

    def cuda(self, gpu_ids):
        if len(gpu_ids) == 1:
            for k, v in self.networks.items():
                v.to(self.device)
        else:
            for k, v in self.networks.items():
                self.networks[k] = nn.DataParallel(v, device_ids=gpu_ids)
                self.networks[k].to(self.device)

    def cpu(self):
        for k, v in self.networks.items():
            v.cpu()

    def name(self):
        return 'BaseEnv'