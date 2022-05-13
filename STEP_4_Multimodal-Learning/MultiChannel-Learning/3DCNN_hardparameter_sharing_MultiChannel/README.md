## The flow of data load 
### Data Structure
The data structure is as follow.
```
DATA
|-- image data
|    |-- Fractional Anisotropy 
|    |-- Mean Diffusivity
|    |-- Radial Diffusivity
|
|-- phenotype data
     |--- phenotype.csv

```
### the flow of data loading
1. Making dataset as follow
```
Dataset = {'FA':['sub-01_FA.nii.gz', 'sub-02_FA.nii.gz',...],  
           'MD':['sub-01_MD.nii.gz', 'sub-02_MD.nii.gz',...],  
           'RD':['sub-01_RD.nii.gz', 'sub-02_RD.nii.gz',...],
           'label':{'sex':[1, 0,...], 'age':[121, 124,...]}}
```

2. Loading images, Transform (scale intensity, resize,...) and concatenate FA, MD, RD each images.
This process is done by custom collate_fn. 
collate_fn define how to each images is fed into mini-batches.
Thus, it could be said that mini-batches are made by torch.utils.data.DataLoader with this collate_fn.
```
def collate_fn(batch):
  images = []
  labels = []
  for i in batch:
    #batch[i] = {'FA': 'FA_image.nii.gz', 'MD': 'MD_image.nii.gz',RD:' RD_image.nii.gz',label:{'sex':1, 'age':121}}
    FA = Loading(batch[i]['FA'])
    FA = Transform(FA)
    MD = Loading(batch[i]['MD'])
    MD = Transform(FA)
    RD = Loading(batch[i]['RD'])
    RD = Transform(FA)
    image = concatenate([FA, MD, RD])
    images.append(image)
    labels.append(batch[i]['label'])
    
   return images, labels
```
 
4. With custom collate_fn, mini-batches are made by torch.utils.data.DataLoader.  
 




## Models 
The list of models are as follows 
- **simple 3DCNN**
- **VGGNet** (11 layers, 13 layers, 16 layers, 19 layers)
- **ResNet** (50 layers, 101 layers, 152 layers)
- **DenseNet** (121 layers, 169 layers, 201 layers, 264 layers)

  
This python script could be used in both **single task learning** and **multi task learning**  


## Running codes 
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
