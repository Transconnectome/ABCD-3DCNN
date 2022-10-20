## Cautions
When you want to run this code with specific gpu devices, **you should set gpu device ids as shell global environment first**. 
```--nproc_per_node``` is the number of gpus per node.   
```
export CUDA_VISIBLE_DEVICES=2,3
torchrun --standalone --nnodes=1 --nproc_per_node=2 pretrain_simMIM.py --model simMIM_swin_small_3D --optim AdamW --lr 2e-4 --epoch 100 --exp_name simmim_test --sbatch --batch_size 4 --accumulation_steps 4 --gradient_clipping
```   
or  
```
CUDA_VISIBLE_DEVICES=2,3 torchrun --standalone --nnodes=1 --nproc_per_node=2 pretrain_simMIM.py --model simMIM_swin_small_3D --optim AdamW --lr 2e-4 --epoch 100 --exp_name simmim_test --sbatch --batch_size 4 --accumulation_steps 4 --gradient_clipping
``` 
   
When you want to run this code with slurm, **you don't need to set gpu device ids** as shell global environment.  
```
torchrun --standalone --nnodes=1 --nproc_per_node=2 pretrain_simMIM.py --model simMIM_swin_small_3D --optim AdamW --lr 2e-4 --epoch 100 --exp_name simmim_test --sbatch --batch_size 4 --accumulation_steps 4 --gradient_clipping
```   

To avoid overload of CPU multiprocessing, it would be recommended to set ```OMP_NUM_THREADS={}```. 
```
export OMP_NUM_THREADS=14 
torchrun --standalone --nnodes=1 --nproc_per_node=2 pretrain_simMIM.py --model simMIM_swin_small_3D --optim AdamW --lr 2e-4 --epoch 100 --exp_name simmim_test --sbatch --batch_size 4 --accumulation_steps 4 --gradient_clipping
``` 

## Pretraining 
Pretraining Vision Transformer in a self-supervised way by simMIM.  
Two kinds of backbone architectures could be used: **Vision Transformer** and **Swin Transformer**. 

### Common 
You can choose data by setting ```--study_sample=UKB``` or ```--study_sample=ABCD```.  
In default, ```--study_sample=UKB``` 

You can specify the size of input image by setting ```--img_size 128 128 128```.
In default, ```--img_size 96 96 96```.  

You can specify the size of mask patch by setting ```---mask_patch_size 8```. 
**When you want to change the size of mask patch during training ViT, be cautious that you should also specify the size of ```--model_patch_size 8```, which is the same as the size of mask patch (training Swin Transformer is not the case)**.  
  
You can load ImageNet22k pretrained ViT and Swin by setting ```--load_imagenet_prertrained```.  
  
You can use kernel fusion by activate ```--torchscript``` (Kernel fusion need pytoch version >= 1.2).
   

### ViT specific parameters     
If you want to use absolute sin-cos positional encoding, add argument ```--use_sincos_pos```.  
In default, positional encoding is the zero-filled parameters update during training.  

If you want to use relative positional bias **only for encoder**, add argument ```--use_rel_pos_bias```.  
In default, positional encoding is the zero-filled parameters update during training.  

### Swin Transformer specific parameters 
If you want to change the size of window, setting ```--window_size {int}```. 

  
## Finetuning
Finetuning Vision Transformer for downstream tasks.  
  
### Common
**You should set ```--pretrained_model {/dir/to/model/pth}```.**  
**Furthermore, you should set either ```--cat_target``` or ```--num_target```.**   
**If you want to predict categorical variable, you should set ```--cat_target```.**   
**If you want to predict continuous variable, you should set ```--num_target```.**  
```
torchrun --standalone --nnodes=1 --nproc_per_node=2 --model vit_base_patch16_3D --optim AdamW --lr 1e-4 --epoch 1000 --exp_name finetuning_test  --sbatch  --batch_size 32  --accumulation_steps 32 --pretrained_model /scratch/connectome/dhkdgmlghks/3DCNN_test/MAE_DDP/result/model/mae_vit_base_patch16_3D_vitBASE_MAE_MaskRatio0.75_Batch1024_8cfcfa.pth --num_target age --gradient_clipping
```
  
You can use kernel fusion by activate ```--torchscript``` (Kernel fusion need pytoch version >= 1.2).  
  
### ViT specific parameters
**You can choose whether use cls token for classification (or regression) or use average pooled latent features for classification (or regression)**.  
**In default, using average pooled latent features for classification (or regression)**. Or you can explicitly set ```--global_pool```.  
If you set ```--cls_token```, then cls token would be used for classification (or regression).  
If you want to use absolute sin-cos positional encoding, add argument ```--use_sincos_pos```.  
If you want to use relative positional bias, add argument ```--use_rel_pos_bias```.  
In default, positional encoding is the zero-filled parameters update during training.

### Swin Transformer specific parameters 
If you want to change the size of window, setting ```--windw_size {int}```. 
  
  
## Inferece 
If you want to use gpu for inference, activate ```--use_gpu```. Otherwise, running inference on cpu.  
Please specify ```CUDA_VISIBLE_DEVICES``` for run inference on speicific gpu id.  
You should specify ```--checkpoint_dir``` for load pretrained model for which you want to run inference. 

```CUDA_VISIBLE_DEVICES=7 python3 inference_only.py --study_sample ABCD --img_size 128 128 128 --batch_size 4 --model swin_base_3D --exp_name testtestestsetst --checkpoint_dir /home/ubuntu/dhkdgmlghks/simMIM/result/model/swin_base_3D_swinBASE_UKBandABCD_pretrained_sex_weightdecay0.01_batch1024_d46210.pth --cat_target sex --use_gpu```