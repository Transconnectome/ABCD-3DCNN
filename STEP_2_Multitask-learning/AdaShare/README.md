# AdaShare: Learning What To Share For Efficient Deep Multi-Task Learning (NeurIPS 2020)

## Introduction


AdaShare is a **novel** and **differentiable** approach for efficient multi-task
learning that learns the feature sharing pattern to achieve the best recognition accuracy, while
restricting the memory footprint as much as possible. Our main idea is to learn the sharing pattern
through a task-specific policy that selectively chooses which layers to execute for a given task in
the multi-task network. In other words, we aim to obtain a single network for multi-task learning
that supports separate execution paths for different tasks.

Here is [the link](https://arxiv.org/pdf/1911.12423.pdf) for our arxiv version. 

Welcome to cite our work if you find it is helpful to your research.
```
@article{sun2020adashare,
  title={Adashare: Learning what to share for efficient deep multi-task learning},
  author={Sun, Ximeng and Panda, Rameswar and Feris, Rogerio and Saenko, Kate},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  year={2020}
}
```

# Dataset 
**Please split train/val/test image dataset for running this code.**
This model adopt **curriculum learning framework**.  
Thus, the overall workflow is also splitted as  **"Policy Learning Phase"**, **"Retraing Phase"**, and **"Test Phase"**.  
Thus, for valid data loading process, please split the dataset.
Please execute 'data_split.py' before starting experiments. 
This command will make text file containing file names of images in the directory of 'img_dataroot'

```
python3 data_split.py --config <yaml_file_name> --test_size <ratio of test sample size to total sample size> --val_size <ratio of validation sample size to total sample size>
``` 
For, example, `python3 ~/3DCNN/AdaShare/data_split.py --config ~/3DCNN/AdaShare/yamls/adashare/ABCD.yml --test_size 0.1 --val_size 0.1`

# Training
## Policy Learning Phase
Please execute `train.py` for policy learning, using the command 
```
python3 train.py --config <yaml_file_name> --gpus <gpu ids> --cat_target <categorical target variable> --num_target <numerical target variable> --exp_name <experiment name>
```
For example, `python3 train.py --config yamls/adashare/ABCD.yml --gpus 0 --cat_target sex race.ethnicity --num_target age BMI --exp_name ABCD_multitask`.

If you want to do experiments with only categorical target variables or numerical target variables: `python3 train.py --config yamls/adashare/ABCD.yml --gpus 2 --cat_target sex race.ethnicity --exp_name ABCD_multitask`  
or  `python3 train.py --config yamls/adashare/ABCD.yml --gpus 0 --num_target age BMI --exp_name ABCD_multitask`. 
  
If you want to do single task learning, just type one variable :  `python3 train.py --config yamls/adashare/ABCD.yml --gpus 2 --cat_target sex --exp_name ABCD_multitask`  
or ` python3 train.py --config yamls/adashare/ABCD.yml --gpus 0 --num_target age --exp_name ABCD_multitask`.  
  
If you want to do Data Parallelism, type ID number of cuda device: ` python3 train.py --config yamls/adashare/ABCD.yml --gpus 0 1 2 --cat_target sex race.ethnicity --num_target age BMI --exp_name ABCD_multitask`.
  
  
## Retrain Phase
After Policy Learning Phase, we sample 8 different architectures and execute `re-train.py` for retraining.
```
python re-train.py --config <yaml_file_name> --gpus <gpu ids> --exp_name <experiment name>
```
**`--exp_name <experiment name>` should be same as in Policy learning phase**.  

**(optional)** where we use different `--exp_ids` to specify different random seeds and generate different architectures. The best performance of all 8 runs is reported in the paper.

For example, `python3 re-train.py --config yamls/adashare/ABCD.yml --gpus 0 --cat_target sex race.ethnicity --num_target age BMI --exp_name ABCD_multitask`.

If you want to do experiments with only categorical target variables or numerical target variables: `python3 re-train.py --config yamls/adashare/ABCD.yml --gpus 2 --cat_target sex race.ethnicity`  
or  `python3 re-train.py --config yamls/adashare/ABCD.yml --gpus 0 --num_target age BMI --exp_name ABCD_multitask`. 
  
If you want to do single task learning, just type one variable :  `python3 re-train.py --config yamls/adashare/ABCD.yml --gpus 2 --cat_target sex --exp_name ABCD_multitask`  
or ` python3 re-train.py --config yamls/adashare/ABCD.yml --gpus 0 --num_target age --exp_name ABCD_multitask`.  
  
If you want to do Data Parallelism, type ID number of cuda device: ` python3 re-train.py --config yamls/adashare/ABCD.yml --gpus 0 1 2 --cat_target sex race.ethnicity --num_target age BMI --exp_name ABCD_multitask`.




# Test/Inference
After Retraining Phase, execute `test.py` for get the quantitative results on the test set. 
```
python test.py --config <yaml_file_name> --gpus <gpu ids> --exp_name <experiment name>
```
**`--exp_name <experiment name>` should be same as in Policy learning phase and Retrain phase**.  

For example, `python3 test.py --config yamls/adashare/ABCD.yml --gpus 0 --cat_target sex race.ethnicity --num_target age BMI --exp_name ABCD_multitask`.

If you want to do experiments with only categorical target variables or numerical target variables: `python3 test.py --config yamls/adashare/ABCD.yml --gpus 2 --cat_target sex race.ethnicity`  
or  `python3 test.py --config yamls/adashare/ABCD.yml --gpus 0 --num_target age BMI --exp_name ABCD_multitask`. 
  
If you want to do single task learning, just type one variable :  `python3 train.py --config yamls/adashare/ABCD.yml --gpus 2 --cat_target sex --exp_name ABCD_multitask`  
or ` python3 est.py --config yamls/adashare/ABCD.yml --gpus 0 --num_target age --exp_name ABCD_multitask`.  
  
If you want to do Data Parallelism, type ID number of cuda device: ` python3 est.py --config yamls/adashare/ABCD.yml --gpus 0 1 2 --cat_target sex race.ethnicity --num_target age BMI --exp_name ABCD_multitask`.




