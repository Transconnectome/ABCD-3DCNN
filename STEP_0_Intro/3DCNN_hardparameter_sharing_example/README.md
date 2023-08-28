The list of models are as follows 
- **simple 3DCNN**
- **VGGNet** (11 layers, 13 layers, 16 layers, 19 layers)
- **ResNet** (50 layers, 101 layers, 152 layers)
- **DenseNet** (121 layers, 169 layers, 201 layers, 264 layers)
- **EfficientNet** (b0, b1, b2, b3, b4, b5, b6, b7)
- **Swin Transformer V1 & V2** (tiny, small, base, large)

  
This python script could be used in both **single task learning** and **multi task learning**  

### The example of single task learning

```
python3 run_3DCNN_hard_parameter_sharing.py --cat_target sex  --optim AdamW --lr 1e-3 --exp_name sex_test --model {model_name} --epoch 300 --batch_size 32 --accumulation_steps 32
```
or
```
python3 run_3DCNN_hard_parameter_sharing.py --num_target age  --optim AdamW --lr 1e-3 --exp_name sex_test --model {model_name} --epoch 300 --batch_size 32 --accumulation_steps 32
```
  
### The example of multi task learning

```
python3 run_3DCNN_hard_parameter_sharing.py --cat_target sex income --optim AdamW --lr 1e-3 --exp_name sex_test --model {model_name} --epoch 300 --batch_size 32 --accumulation_steps 32
```  
or 

```
python3 run_3DCNN_hard_parameter_sharing.py --cat_target sex --num_target age --optim AdamW --lr 1e-3 --exp_name sex_test --model {model_name} --epoch 300 --batch_size 32 --accumulation_steps 32
``` 

