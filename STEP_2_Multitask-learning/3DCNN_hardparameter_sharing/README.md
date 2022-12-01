The list of models are as follows 
- **simple 3DCNN**
- **VGGNet** (11 layers, 13 layers, 16 layers, 19 layers)
- **ResNet** (50 layers, 101 layers, 152 layers)
- **DenseNet** (121 layers, 169 layers, 201 layers, 264 layers)

  
This python script could be used in both **single task learning** and **multi task learning**  

### The example of single task learning

```
python3 run_3DCNN_hard_parameter_sharing.py --cat_target sex  --optim SGD --lr 1e-3 --gpus 4 5 --exp_name sex_test --model {model_name} --epoch 300 --batch_size 32 --accumulation_steps 32
```
or
```
python3 run_3DCNN_hard_parameter_sharing.py --num_target age  --optim SGD --lr 1e-3 --gpus 4 5 --exp_name sex_test --model {model_name} --epoch 300 --batch_size 32 --accumulation_steps 32
```
  
### The example of multi task learning

```
python3 run_3DCNN_hard_parameter_sharing.py --cat_target sex income --optim SGD --lr 1e-3 --gpus 4 5 --exp_name sex_test --model {model_name} --epoch 300 --batch_size 32 --accumulation_steps 32
```  
or 

```
python3 run_3DCNN_hard_parameter_sharing.py --cat_target sex --num_target age --optim SGD --lr 1e-3 --gpus 4 5 --exp_name sex_test --model {model_name} --epoch 300 --batch_size 32 --accumulation_steps 32
``` 

### The example of using slurm 
If you want to using slurm SBATCH, add options ```--sbatch True```.  
In this case, you don't need to assign gpu device ids by using gpu option ```--gpus {device_id}```.
