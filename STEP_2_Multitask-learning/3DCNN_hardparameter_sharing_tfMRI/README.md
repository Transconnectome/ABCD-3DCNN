The list of models are as follows 
- **simple 3DCNN**
- **VGGNet** (11 layers, 13 layers, 16 layers, 19 layers)
- **ResNet** (50 layers, 101 layers, 152 layers)
- **DenseNet** (121 layers, 169 layers, 201 layers, 264 layers)

  
This python script could be used in both **single task learning** and **multi task learning**  
  
## Examples
### The example of single task learning

```
python3 run_3DCNN_hard_parameter_sharing.py --cat_target sex  --optim AdamW --lr 1e-3 --exp_name UKB_tfMRI_emotion_zstat2_sex --study_sample UKB_Emotion_tfMRI_zstat2 --model {model_name} --epoch 300 --batch_size 32 --accumulation_steps 32
```
or
```
python3 run_3DCNN_hard_parameter_sharing.py --num_target age  --optim AdamW --lr 1e-3 --exp_name UKB_tfMRI_emotion_zstat2_age --study_sample UKB_Emotion_tfMRI_zstat2 --model {model_name} --epoch 300 --batch_size 32 --accumulation_steps 32
```
  
### The example of multi task learning

```
python3 run_3DCNN_hard_parameter_sharing.py --cat_target sex income --optim AdamW --lr 1e-3 --exp_name UKB_tfMRI_emotion_zstat2_sex --study_sample UKB_Emotion_tfMRI_zstat2 --model {model_name} --epoch 300 --batch_size 32 --accumulation_steps 32
```  
or 

```
python3 run_3DCNN_hard_parameter_sharing.py --cat_target sex --num_target age --optim AdamW --lr 1e-3 --exp_name --exp_name UKB_tfMRI_emotion_zstat2_sex_age --study_sample UKB_Emotion_tfMRI_zstat2  --model {model_name} --epoch 300 --batch_size 32 --accumulation_steps 32
``` 
  
## Note 
### standardizing continuous labels
If you want to standardize ```---num_target {[the list of targets]}```, activate ```--scaling_num_target```.
For example,
```
python3 run_3DCNN_hard_parameter_sharing.py  --model densenet3D121 --epoch 200  --batch_size 128 --accumulation_steps 2 --optim AdamW --lr 1e-4 --study_sample UKB_Emotion_tfMRI_zstat2 --num_target income --exp_name UKB_tfMRI_emotion_zstat2_income
```

### Training with multi-gpu
In default, the code run in the multi-gpu setting.  
The code will automatically detect all available GPUs and use them for training.  
If you want to specify GPU ids, try the follow example.  
```
CUDA_VISIBLE_DEVICES=0,1 python3 run_3DCNN_hard_parameter_sharing.py --cat_target sex income --optim SGD --lr 1e-3 --exp_name sex_test --model {model_name} --epoch 300 --batch_size 32 --accumulation_steps 32
```
