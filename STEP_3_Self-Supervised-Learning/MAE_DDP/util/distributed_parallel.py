import torch 
import os 



def init_distributed(args):
    assert torch.cuda.is_available(), "Distributed mode requires CUDA."
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])

        args.distributed = True
        torch.cuda.set_device(args.gpu)
        torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size)
        torch.distributed.barrier()
    else:
        print('Not using distributed mode')
        args.gpu = 0
        args.distributed = False
        return 

    """
    assert torch.cuda.is_available(), "Distributed mode requires CUDA."
    is_distributed = args.world_size > 1 
    ngpus_per_node = torch.cuda.device_count()

    if is_distributed == True:
        if args.local_rank != -1: # for torch.distributed.launch
            args.rank = args.local_rank
            args.gpus = args.local_rank
        elif 'SLURM_PROCID' in os.environ: # for slurm scheduler
            args.rank = int(os.environ['SLURM_PROCID'])
            args.gpus = args.rank % torch.cuda.device_count()

        torch.distributed.init_process_group(backend='nccl', init_method='env://',
                            world_size=args.world_size, rank=args.rank)
    
    else: 
        args.rank = 0 
        args.gpus = 0 

    args.batch_size = args.batch_size // ngpus_per_node
    """






"""

    torch.cuda.set_device(rank % torch.cuda.device_count())
    
    n_nodes = int(os.getenv('N_NODES'))
    
    # Initialize distributed communication
    args.world_size = int(n_nodes * args.ngpus_per_node)
    args.rank = 0
    
    torch.distributed.init_process_group(
            backend='nccl', init_method='env://',
            world_size=args.world_size, rank=args.rank)


    # set batch size. Total batch size / n_gpus_per_node
    args.batch_size = args.batch_size // len(args.gpus)

    print("Done initializing distributed")

"""
