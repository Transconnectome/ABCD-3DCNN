## ======= load module ======= ##
import os
import argparse
from copy import deepcopy
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
from utils.utils import set_random_seed,save_emb_result, checkpoint_load, save_emb_result
from utils.lr_scheduler import *
from dataloaders.dataloaders import check_study_sample, loading_images, loading_phenotype, combining_image_target, partition_dataset, partition_dataset_predefined
from envs.experiments import extract_embedding
import hashlib
import datetime

def argument_setting():
    parser = argparse.ArgumentParser()
    # arguments for dataset 
    parser.add_argument("--study_sample",default='UKB',type=str,required=False,help='')
    parser.add_argument('--partitioned_dataset_number', default=None, type=int, required=False)
    parser.add_argument("--train_size",default=0.8,type=float,required=False,help='')
    parser.add_argument("--val_size",default=0.1,type=float,required=False,help='')
    parser.add_argument("--test_size",default=0.1,type=float,required=False,help='')
    parser.add_argument("--resize",default=[96, 96, 96],type=int,nargs="*",required=False,help='')
    # arguments for model 
    parser.add_argument("--in_channels",default=1,type=int,required=False,help='')
    parser.add_argument("--model",required=True,type=str,help='',choices=[
                                                                        'resnet3D50', 'resnet3D101','resnet3D152', 
                                                                        'densenet3D121', 'densenet3D169','densenet3D201','densenet3D264', 
                                                                        'efficientnet3D-b0','efficientnet3D-b1','efficientnet3D-b2','efficientnet3D-b3','efficientnet3D-b4','efficientnet3D-b5','efficientnet3D-b6','efficientnet3D-b7'
                                                                        ])
    # arguments for inference
    parser.add_argument("--batch_size",default=16,type=int,required=False,help='')
    parser.add_argument("--cat_target", type=str, nargs='*', required=False, help='')
    parser.add_argument("--num_target", type=str,nargs='*', required=False, help='')
    parser.add_argument("--extracting_dataset", default='train', type=str, choices=['train', 'val', 'test'], help='Choose dataset to be used for extracting embeddings')
    # arguments for loading pre-trained model 
    parser.add_argument("--checkpoint_dir", type=str, default=None,required=False)
    # arguments for others 
    parser.add_argument("--exp_name",type=str,required=True,help='')
    parser.add_argument("--confusion_matrix", type=str, nargs='*',required=False, help='')

    args = parser.parse_args()
    print("Categorical target labels are {} and Numerical target labels are {}".format(args.cat_target, args.num_target))

    if not args.cat_target:
        args.cat_target = []
    elif not args.num_target:
        args.num_target = []
    elif not args.cat_target and args.num_target:
        raise ValueError('YOU SHOULD SELECT THE TARGET!')

    return args

## ========= Experiment =============== ##
def extract_engine(partition, subject_data, save_dir, args): #in_channels,out_dim
    targets = args.cat_target + args.num_target
    # ResNet 
    if args.model == 'resnet3D50':
        import models.resnet3d as resnet3d #model script
        net = resnet3d.resnet3D50(subject_data, args)
    elif args.model == 'resnet3D101':
        import models.resnet3d as resnet3d #model script
        net = resnet3d.resnet3D101(subject_data, args)
    elif args.model == 'resnet3D152':
        import models.resnet3d as resnet3d #model script
        net = resnet3d.resnet3D152(subject_data, args)
    # DenseNet
    elif args.model == 'densenet3D121':
        import models.densenet3d as densenet3d #model script
        net = densenet3d.densenet3D121(subject_data, args)
    elif args.model == 'densenet3D169':
        import models.densenet3d as densenet3d #model script
        net = densenet3d.densenet3D169(subject_data, args) 
    elif args.model == 'densenet3D201':
        import models.densenet3d as densenet3d #model script
        net = densenet3d.densenet3D201(subject_data, args)
    # EfficientNet V1 
    elif args.model.find('efficientnet3D') != -1: 
        import models.efficientnet3d as efficientnet3d
        net = efficientnet3d.efficientnet3D(subject_data,args)
    # Swin Transformer V1 & V2
    elif args.model.find('swinV1') != -1: 
        import models.swinV1 as swinv1 
        net = swinv1.__dict__[args.model](subject_data=subject_data, args=args)
    elif args.model.find('swinV2') != -1: 
        import models.swinV2 as swinv2
        net = swinv2.__dict__[args.model](subject_data=subject_data, args=args, img_size=args.resize)


    # test
    net.to('cpu')
    torch.cuda.empty_cache()

    assert args.checkpoint_dir
    net = checkpoint_load(net, args.checkpoint_dir)

    # setting DataParallel
    devices = []
    for d in range(torch.cuda.device_count()):
        devices.append(d)
    net = nn.DataParallel(net, device_ids = devices)
    net.cuda()

    embeddings = extract_embedding(net, partition[args.extracting_dataset], args)
    return embeddings
## ==================================== ##



if __name__ == "__main__":

    ## ========= Setting ========= ##
    args = argument_setting()
    current_dir = os.getcwd()
    image_dir, phenotype_dir = check_study_sample(study_sample=args.study_sample)
    image_files = loading_images(image_dir=image_dir, args=args, study_sample=args.study_sample)

    if args.partitioned_dataset_number is not None: 
        subject_data, target_list = loading_phenotype(phenotype_dir=phenotype_dir, args=args, study_sample=args.study_sample, partitioned_dataset_number=args.partitioned_dataset_number)
        ## data preprocesing categorical variable and numerical variables
        imageFiles_labels = combining_image_target(subject_data, image_files, target_list, partitioned_dataset_number=args.partitioned_dataset_number, study_sample=args.study_sample)
        partition = partition_dataset_predefined(imageFiles_labels, target_list, partitioned_dataset_number=args.partitioned_dataset_number, args=args)
    else:
        subject_data, target_list = loading_phenotype(phenotype_dir=phenotype_dir, args=args, study_sample=args.study_sample)
        ## data preprocesing categorical variable and numerical variables
        imageFiles_labels = combining_image_target(subject_data, image_files, target_list, study_sample=args.study_sample)
        partition = partition_dataset(imageFiles_labels, target_list, args=args)


    ## ========= Run Experiment and saving result ========= ##
    # seed number
    seed = 1234
    set_random_seed(seed)
    save_dir = current_dir + '/result'
    
    time_hash = datetime.datetime.now().time()
    hash_key = hashlib.sha1(str(time_hash).encode()).hexdigest()[:6]
    args.exp_name = args.exp_name + f'_{hash_key}'


    # Run Experiment
    embeddings = extract_engine(partition, subject_data, save_dir, deepcopy(args))

    # Save result
    save_emb_result(os.path.join(save_dir,'embedding'), args.exp_name, embeddings)
    ## ====================================== ##
