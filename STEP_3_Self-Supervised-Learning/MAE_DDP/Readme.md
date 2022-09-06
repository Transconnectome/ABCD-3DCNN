## Cautions
When you want to run this code with specific gpu devices, **you should set gpu device ids as shell global environment first**. 
```--nproc_per_node``` is the number of gpus per node.   
```
export CUDA_VISIBLE_DEVICES=2,3
torchrun --standalone --nnodes=1 --nproc_per_node=2 pretrain_MAE.py --model mae_vit_base_patch16_3D --optim AdamW --lr 1e-4 --epoch 400 --exp_name vitLARGE_MAE_MaskRatio0.75_Batch1024  --sbatch  --batch_size 64  --accumulation_steps 4 --norm_pix_loss
```   
or  
```
CUDA_VISIBLE_DEVICES=2,3 torchrun --standalone --nnodes=1 --nproc_per_node=2 pretrain_MAE.py --model mae_vit_base_patch16_3D --optim AdamW --lr 1e-4 --epoch 400 --exp_name vitLARGE_MAE_MaskRatio0.75_Batch1024  --sbatch  --batch_size 64  --accumulation_steps 4 --norm_pix_loss
``` 
   
When you want to run this code with slurm, **you don't need to set gpu device ids** as shell global environment.  
```
torchrun --standalone --nnodes=1 --nproc_per_node=2 pretrain_MAE.py --model mae_vit_base_patch16_3D --optim AdamW --lr 1e-4 --epoch 400 --exp_name vitLARGE_MAE_MaskRatio0.75_Batch1024  --sbatch  --batch_size 64  --accumulation_steps 4 --norm_pix_loss
```   

To avoid overload of CPU multiprocessing, it would be recommended to set ```OMP_NUM_THREADS={}```. 
```
export OMP_NUM_THREADS=14 
torchrun --standalone --nnodes=1 --nproc_per_node=2 pretrain_MAE.py --model mae_vit_base_patch16_3D --optim AdamW --lr 1e-4 --epoch 400 --exp_name vitLARGE_MAE_MaskRatio0.75_Batch1024  --sbatch  --batch_size 64  --accumulation_steps 4 --norm_pix_loss
``` 

## Pretraining 
Pretraining Vision Transformer in a self-supervised way by maksed autoencoder.  
```
torchrun --standalone --nnodes=1 --nproc_per_node=2 pretrain_MAE.py --model mae_vit_base_patch16_3D --optim AdamW --lr 1e-4 --epoch 400 --exp_name vitLARGE_MAE_MaskRatio0.75_Batch1024  --sbatch  --batch_size 64  --accumulation_steps 4 --norm_pix_loss
```  
  
## Finetuning
Finetuning Vision Transformer for downstream tasks.  
**You should set ```--pretrained_model {/dir/to/model/pth}```.**  
**Furthermore, you should set either ```--cat_target``` or ```--num_target```.**   
**If you want to predict categorical variable, you should set ```--cat_target```.**   
**If you want to predict continuous variable, you should set ```--num_target```.**  
```
torchrun --standalone --nnodes=1 --nproc_per_node=2 --model vit_base_patch16_3D --optim AdamW --lr 1e-4 --epoch 1000 --exp_name finetuning_test  --sbatch  --batch_size 32  --accumulation_steps 32 --pretrained_model /scratch/connectome/dhkdgmlghks/3DCNN_test/MAE_DDP/result/model/mae_vit_base_patch16_3D_vitLARGE_MAE_MaskRatio0.75_Batch1024_8cfcfa.pth --num_target age
```