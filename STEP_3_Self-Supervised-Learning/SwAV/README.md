## Before you run these code  
### dataset 
**In this codes, pretraining SwAV and finetuning it for prediction task is a single PIPLINE applied to a single dataset.  
Thus, a single dataset is partitioned into 4 parts: pretraining, finetuning, validation, test.  
If you want to pretrain simCLR with one dataset and fintune it to another independent dataset, only finetuning dataset should be partitioned into 3 parts.  
Pretraining dataset does not need to be partitioned.  
Revising this code according to your experiment setup.**


## Reference Code 


## About these code
The list of models are as follows 
- **ResNet** (50 layers, 101 layers, 152 layers)
- **DenseNet** (121 layers, 169 layers, 201 layers, 264 layers)

   
This python script could be used in both **pretraining self-supervised model** and **fine-tuning prediction model**  

### The example of pretraining self-supervised model
Another important thing is ```--train_batch_size```. If you set ```--train_batch_size N```, the number of mini-batches used for calculating loss is ***2N***  

When you train models first time, you should set ```--resume False```.  
If you resume training models, you should set ```--resume True``` and ```--checkpoint_dir {checkpoint_file}```.

For example, you should set command line as follows when you train a model first time.
```
python3 run_SwAV.py --model resnet3D50 --resize 80 80 80 --optim LARS --epoch 100 --exp_name UKB_SwAV_test --gpus 6 7 --augmentation RandRotate RandRotate90 RandCrop RandFlip RandAdjustContrast RandGaussianSmooth RandGibbsNoise RandCoarseDropout --train_batch_size 128 --resume False 
```

If you want to resume training the previous model, you should set command line as follows. 
```
python3 run_SwAV.py --model resnet3D50 --resize 80 80 80 --optim LARS --epoch 100 --exp_name UKB_SwAV_test --gpus 6 7 --augmentation RandRotate RandRotate90 RandCrop RandFlip RandAdjustContrast RandGaussianSmooth RandGibbsNoise RandCoarseDropout --train_batch_size 16 --resume True --checkpoint_dir {checkpoint_file}
```

  
### The example of fine-tuning prediction model
You can perform fine-tuning experiments in the same way as you perform multi task learning experiments.  
  
When you train models first time, you should set ```--resume False``` and ```--pretrained_model_dir {pretrained_model_file}```.  
If you resume training models, you should set ```--resume False``` and ```--checkpoint_dir {checkpoint_file}```.  
  
For example, you should set command line as follows when you train a model first time.  
```
python3 run_3DCNN_hardparameter_sharing.py --model resnet3D50 --resize 80 80 80 --optim Adam --epoch 100 --exp_name UKB_simCLR_test --cat_target sex --gpus 6 7  --resume False --pretrained_model_dir {pre_trained_model_file}
```   
If you want to resume training the previous model, you should set command line as follows.  
```
python3 run_3DCNN_hardparameter_sharing.py --model resnet3D50 --resize 80 80 80 --optim Adam --epoch 4 --exp_name UKB_simCLR_test --cat_target sex --gpus 6 7  --resume True --checkpoint_dir {checkpoint_file}
```  


### The example of using slurm 
If you want to using slurm SBATCH, add options ```--sbatch True```.  
In this case, you don't need to assign gpu device ids by using gpu option ```--gpus {device_id}```.


