## Cautions
When you want to run this code with specific gpu devices, **you should set gpu device ids as shell global environment first**. 
```--nproc_per_node``` is the number of gpus per node.   
```
export CUDA_VISIBLE_DEVICES=2,3
torchrun --standalone --nnodes=1 --nproc_per_node=2 pretrain_MAE.py --model mae_vit_base_patch16_3D --optim AdamW --lr 1e-4 --epoch 400 --exp_name vitBASE_MAE_MaskRatio0.75_Batch1024  --sbatch  --batch_size 64  --accumulation_steps 4 --norm_pix_loss --gradient_clipping
```   
or  
```
CUDA_VISIBLE_DEVICES=2,3 torchrun --standalone --nnodes=1 --nproc_per_node=2 pretrain_MAE.py --model mae_vit_base_patch16_3D --optim AdamW --lr 1e-4 --epoch 400 --exp_name vitBASE_MAE_MaskRatio0.75_Batch1024  --sbatch  --batch_size 64  --accumulation_steps 4 --norm_pix_loss --gradient_clipping
``` 
   
When you want to run this code with slurm, **you don't need to set gpu device ids** as shell global environment.  
```
torchrun --standalone --nnodes=1 --nproc_per_node=2 pretrain_MAE.py --model mae_vit_base_patch16_3D --optim AdamW --lr 1e-4 --epoch 400 --exp_name vitBASE_MAE_MaskRatio0.75_Batch1024  --sbatch  --batch_size 64  --accumulation_steps 4 --norm_pix_loss --gradient_clipping
```   

To avoid overload of CPU multiprocessing, it would be recommended to set ```OMP_NUM_THREADS={}```. 
```
export OMP_NUM_THREADS=14 
torchrun --standalone --nnodes=1 --nproc_per_node=2 pretrain_MAE.py --model mae_vit_base_patch16_3D --optim AdamW --lr 1e-4 --epoch 400 --exp_name vitBASE_MAE_MaskRatio0.75_Batch1024  --sbatch  --batch_size 64  --accumulation_steps 4 --norm_pix_loss --gradient_clipping
``` 

## Pretraining 
Pretraining Vision Transformer in a self-supervised way by maksed autoencoder.  
```
torchrun --standalone --nnodes=1 --nproc_per_node=2 pretrain_MAE.py --model mae_vit_base_patch16_3D --optim AdamW --lr 1e-4 --epoch 400 --exp_name vitBASE_MAE_MaskRatio0.75_Batch1024  --sbatch  --batch_size 64  --accumulation_steps 4 --norm_pix_loss --gradient_clipping
```  
You can choose data by setting ```--study_sample=UKB``` or ```--study_sample=ABCD```.  
In default, ```--study_sample=UKB```  
  
If you want to use absolute sin-cos positional encoding, add argument ```--use_sincos_pos```.  
In default, positional encoding is the zero-filled parameters update during training.  

If you want to use relative positional bias **only for encoder**, add argument ```--use_rel_pos_bias```.  
In default, positional encoding is the zero-filled parameters update during training.  

If you add both argument ```--use_sincos_pos``` and ```--use_rel_pos_bias```, **relative positional bias is used only for encoder** and **absolute sin-cos positional encoding is used only for decoder**.

  
## Finetuning
Finetuning Vision Transformer for downstream tasks.  
**You should set ```--pretrained_model {/dir/to/model/pth}```.**  
**Furthermore, you should set either ```--cat_target``` or ```--num_target```.**   
**If you want to predict categorical variable, you should set ```--cat_target```.**   
**If you want to predict continuous variable, you should set ```--num_target```.**  
```
torchrun --standalone --nnodes=1 --nproc_per_node=2 --model vit_base_patch16_3D --optim AdamW --lr 1e-4 --epoch 1000 --exp_name finetuning_test  --sbatch  --batch_size 32  --accumulation_steps 32 --pretrained_model /scratch/connectome/dhkdgmlghks/3DCNN_test/MAE_DDP/result/model/mae_vit_base_patch16_3D_vitBASE_MAE_MaskRatio0.75_Batch1024_8cfcfa.pth --num_target age --gradient_clipping
```
  
**You can choose whether use cls token for classification (or regression) or use average pooled latent features for classification (or regression)**.  
**In default, using average pooled latent features for classification (or regression)**. Or you can explicitly set ```--global_pool```.  
If you set ```--cls_token```, then cls token would be used for classification (or regression).  
  
If you want to use absolute sin-cos positional encoding, add argument ```--use_sincos_pos```.  
If you want to use relative positional bias, add argument ```--use_rel_pos_bias```.  
In default, positional encoding is the zero-filled parameters update during training.
