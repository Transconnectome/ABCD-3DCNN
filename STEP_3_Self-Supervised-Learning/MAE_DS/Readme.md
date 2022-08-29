The example line is as below.  
When you use multi-gpus, you should specify device number of gpu.  
**Optimizer should be the variants of Adam (Adam, AdamW)**  
```
CUDA_VISIBLE_DEVICES=6,7 deepspeed  pretrain_MAE.py --model mae_vit_base_patch16_3D --epoch 1000 --exp_name deepspeed --deepspeed --deepspeed_config /scratch/connectome/dhkdgmlghks/3DCNN_test/MAE_DS/deepspeed_config.json
```
  

