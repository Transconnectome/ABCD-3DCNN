The example line is as below.  
When you use multi-gpus, you should specify device number of gpu.  
```
python3 pretrain_MAE.py --model mae_vit_base_patch16_3D --optim LARS --lr 1e-2 --epoch 400 --exp_name test_MAE_MaskRatio0.75_Batch1024 --train_batch_size 128 --accumulation_steps 8 --norm_pix_loss --gpus 0 1
```
  
When you use slurm scripts, you should activate ```--sbatch``` as a below example line.  
```
python3 pretrain_MAE.py --model mae_vit_base_patch16_3D --optim LARS --lr 1e-2 --epoch 400 --exp_name test_MAE_MaskRatio0.75_Batch1024 --train_batch_size 128 --accumulation_steps 8 --norm_pix_loss --sbatch 
```

If you want to run this code as job chain, you should activate ```--resume``` as a below example line.  

```
python3 pretrain_MAE.py --model mae_vit_base_patch16_3D --optim LARS --lr 1e-2 --epoch 400 --exp_name test_MAE_MaskRatio0.75_Batch1024 --train_batch_size 128 --accumulation_steps 8 --norm_pix_loss --sbatch --resume
```
