import numpy as np 
import torch 

from util.loss_functions  import loss_forward, mixup_loss, calculating_eval_metrics
from util.augmentation import mixup_data

from util.utils import CLIreporter, save_exp_result, checkpoint_save, checkpoint_load

import model.model_Swin as Swin
import model.model_ViT as ViT

from tqdm import tqdm


def inference(net, partition, num_classes, args):
    testloader = torch.utils.data.DataLoader(partition['test'],
                                            batch_size=args.batch_size,
                                            shuffle=False,
                                            num_workers=16)

    net.eval()
    
    losses = []
    loss_fn = loss_forward(num_classes)

    eval_metrics = calculating_eval_metrics(num_classes=num_classes, is_DDP=False)    
    with torch.no_grad():
        for i, data in enumerate(tqdm(testloader),0):
            #images = images.to(f'cuda:{net.device_ids[0]}')
            images, labels = data 
            if args.use_gpu:
                labels = labels.cuda()
                images = images.cuda()
            with torch.cuda.amp.autocast():
                pred = net(images)
                loss = loss_fn(pred, labels)
            losses.append(loss.item())
            eval_metrics.store(pred, labels)
                           
    return net, np.mean(losses), eval_metrics.get_result() 



def inference_engine(partition, num_classes, save_dir, args): #in_channels,out_dim
    pretrained_weight = None
    pretrained2d = False 
    simMIM_pretrained = False
    checkpoint_dir = args.checkpoint_dir
    assert checkpoint_dir is not None 
        
    # setting network 
    ###TODO###
    """
    save all of the essential information about experimental settings as checkpoint when finetuning, 
    and load the checkpoint and do the inference 
    """
    if args.model.find('swin') != -1:
        net = Swin.__dict__[args.model](pretrained=pretrained_weight, pretrained2d=pretrained2d, simMIM_pretrained=simMIM_pretrained, window_size=args.window_size, drop_rate=args.projection_drop, num_classes=num_classes)        # change an attribute of mask generator
    elif args.model.find('vit') != -1:
        assert args.model_patch_size == args.mask_patch_size
        net = ViT.__dict__[args.model](pretrained=pretrained_weight, pretrained2d=pretrained2d, simMIM_pretrained=simMIM_pretrained, img_size = args.img_size, patch_size=args.model_patch_size, attn_drop=args.attention_drop, drop=args.projection_drop, drop_path=args.path_drop, global_pool=args.global_pool, num_classes=num_classes, use_rel_pos_bias=args.use_rel_pos_bias, use_sincos_pos=args.use_sincos_pos)
        # change an attribute of mask generator
        print('The size of Patch is %i and the size of Mask Patch is %i' % (args.model_patch_size, args.mask_patch_size))


    # loading last checkpoint 
    if checkpoint_dir is not None: 
        net = checkpoint_load(net=net, checkpoint_dir=checkpoint_dir, mode='inference')

    # change the network module as torchscript module
    if args.torchscript:
        torch._C._jit_set_autocast_mode(True)
        net = torch.jit.script(net)

    # whether using gpu for inference or not
    if args.use_gpu:
        net.cuda()

    # do inference
    result = {}
    net, test_loss, test_performance = inference(net, partition, num_classes, args)
    if args.use_gpu:
        torch.cuda.empty_cache()

    result['test_loss'] = test_loss
    result.update(test_performance)

    return vars(args), result