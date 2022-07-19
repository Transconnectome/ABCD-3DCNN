When you want to run this code with specific gpu devices, **you should set gpu device ids as shell global environment first**. 
```--nproc_per_node``` is the number of gpus per node.   
```
export CUDA_VISIBLE_DEVICES=2,3
torchrun --standalone --nnodes=1 --nproc_per_node=2 pretrain_MAE.py --model mae_vit_base_patch16_3D --optim Adam --lr 1e-3 --epoch 400 --exp_name vitLARGE_MAE_MaskRatio0.75_Batch1024  --sbatch  --batch_size 64  --accumulation_steps 4 --norm_pix_loss
```   
or  
```
CUDA_VISIBLE_DEVICES=2,3 torchrun --standalone --nnodes=1 --nproc_per_node=2 pretrain_MAE.py --model mae_vit_base_patch16_3D --optim Adam --lr 1e-3 --epoch 400 --exp_name vitLARGE_MAE_MaskRatio0.75_Batch1024  --sbatch  --batch_size 64  --accumulation_steps 4 --norm_pix_loss
``` 
   
When you want to run this code with slurm, **you don't need to set gpu device ids** as shell global environment.  
```
torchrun --standalone --nnodes=1 --nproc_per_node=2 pretrain_MAE.py --model mae_vit_base_patch16_3D --optim Adam --lr 1e-3 --epoch 400 --exp_name vitLARGE_MAE_MaskRatio0.75_Batch1024  --sbatch  --batch_size 64  --accumulation_steps 4 --norm_pix_loss
```   

To avoid overload of CPU multiprocessing, it would be recommended to set ```OMP_NUM_THREADS={}```. 
```
export OMP_NUM_THREADS=14 
torchrun --standalone --nnodes=1 --nproc_per_node=2 pretrain_MAE.py --model mae_vit_base_patch16_3D --optim Adam --lr 1e-3 --epoch 400 --exp_name vitLARGE_MAE_MaskRatio0.75_Batch1024  --sbatch  --batch_size 64  --accumulation_steps 4 --norm_pix_loss
``` 