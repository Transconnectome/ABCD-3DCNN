The list of models are as follows 
- **simple 3DCNN**
- **VGGNet** (11 layers, 13 layers, 16 layers, 19 layers)
- **RessNet** (50 layers, 101 layers, 152 layers)
- **DenseNet** (121 layers, 169 layers, 201 layers, 264 layers)

### Example command for single task transfer learning 
   
Include **--transfer age** or **--transfer sex** to command   
This option will load pretrained age prediction model or sex model prediction    
If you want to use ABCD dataset, you should set **--dataset ABCD** and **--data freesurfer or fmriprep**.  


<code> <b> python3 /scratch/connectome/jubin/ABCD-3DCNN-jub/transfer_learning/run_3DCNN_transfer_learning.py \
    --transfer age --num_target age --dataset ABCD --data freesurfer --val_size 0.25 --test_size 0.5 \
    --lr 1e-3 --optim Adam --resize 96 96 96 --scheduler on --train_batch_size 128 --val_batch_size 128 \
    --exp_name TL_age_UKB_ABCD_01 --model densenet3D121 --epoch 60 --epoch_FC 15 --unfrozen_layer 1 --sbatch True </code>




    
This python script could be used in both **single task learning** and **multi task learning**  

### The example of single task learning

```
python3 run_3DCNN_hard_parameter_sharing.py --cat_target sex  --optim SGD --lr 1e-3 --gpus 4 5 --exp_name sex_test --model {model_name} --epoch 300 --train_batch_size 32 --val_batch_size 32 
```
or
```
python3 run_3DCNN_hard_parameter_sharing.py --num_target age  --optim SGD --lr 1e-3 --gpus 4 5 --exp_name sex_test --model {model_name} --epoch 300 --train_batch_size 32 --val_batch_size 32 
```
  
### The example of multi task learning

```
python3 run_3DCNN_hard_parameter_sharing.py --cat_target sex income --optim SGD --lr 1e-3 --gpus 4 5 --exp_name sex_test --model {model_name} --epoch 300 --train_batch_size 32 --val_batch_size 32 
```  
or 

```
python3 run_3DCNN_hard_parameter_sharing.py --cat_target sex --num_target age --optim SGD --lr 1e-3 --gpus 4 5 --exp_name sex_test --model {model_name} --epoch 300 --train_batch_size 32 --val_batch_size 32 
``` 

### The example of using slurm 
If you want to using slurm SBATCH, add options ```--sbatch True```.  
In this case, you don't need to assign gpu device ids by using gpu option ```--gpus {device_id}```.


