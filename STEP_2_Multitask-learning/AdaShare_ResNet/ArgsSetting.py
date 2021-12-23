import argparse 

def ArgsSetting():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",required=True,type=str,help='',choices=['resnet3D50','resnet3D101','resnet3D152'])
    parser.add_argument("--val_size",default=0.1,type=float,required=False,help='')
    parser.add_argument("--test_size",default=0.1,type=float,required=False,help='')
    parser.add_argument("--warmup_size",default=0.3,type=float,required=False,help='')
    parser.add_argument("--train_batch_size",default=1,type=int,required=False,help='')
    parser.add_argument("--val_batch_size",default=1,type=int,required=False,help='')
    parser.add_argument("--test_batch_size",default=1,type=int,required=False,help='')

    parser.add_argument("--resize",default=(96,96,96),required=False,help='')
    parser.add_argument("--in_channels",default=1,type=int,required=False,help='')
    parser.add_argument("--optim",type=str,required=True,help='', choices=['Adam','SGD'])
    parser.add_argument("--lr", default=0.01,type=float,required=False,help='')
    parser.add_argument("--weight_decay",default=0.001,type=float,required=False,help='')

    parser.add_argument("--epoch",type=int,required=True,help='')
    parser.add_argument("--warmup_itr_ratio",default=0.2,type=int,help='')
    parser.add_argument("--exp_name",type=str,required=True,help='')
    parser.add_argument("--cat_target", type=str, nargs='*', required=False, help='')
    parser.add_argument("--num_target", type=str,nargs='*', required=False, help='')


    args = parser.parse_args()

    print("Categorical target labels are {} and Numerical target labels are {}".format(args.cat_target, args.num_target))
    
    return args 