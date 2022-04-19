## Explanation for Occlusion Sensitivity
Occlusion Sensitivity follows this flow. 
 * 1. masking a patch for original images 
 * 2. concatenating every masked images (every masked images are fed into Neural Network as mini batches)
 * 3. getting scores from all of masked images
 * 4. drawing heatmap by this scores
It's like data augmentation techniques (or Maksed Auto Encoder).
In other words, Occlusion Sensitivity is like "sliding NxN size window through images and caculating occlusion sensitivity scores". If this score is high, it means "when this NxN size part of image is occluded, accuracy of model drops dramatically". 

## Explanation for UKB_sex_OcclusionSensitivity.py
This code making individual Occlusion Sensitivity heatmap. 
The results from **mask_size=7 and stride=5** are visualized. 
Befor you run the code, directory should the same as follw.  

```
|--CODE RUNNING DIR
    |-- male_upper_0.75
    |-- male_lower_0.75
    |-- female_upper_0.75
    |-- female_lower_0.75

```
  
example command line
```
python3 UKB_sex_OcclusionSensitivity.py --target sex --save_dir /scratch/connectome/dhkdgmlghks/UKB_interpretation/sex/OcclusionSensitivity --mask_size 7 --stride 5 --gpus 7 5 --batch_size 512
```
If you want to use this code with SLURM JOB SUBMISSION(sbatch), use the following command line 
```
python3 UKB_sex_OcclusionSensitivity.py --target sex --save_dir /scratch/connectome/dhkdgmlghks/UKB_interpretation/sex/OcclusionSensitivity --mask_size 7 --stride 5 --gpus 7 5 --batch_size 512 --sbatch True
```

## Explanation for UKB_sex_OcclusionSensitivity.ipynb
This code making mean heatmap according to category. (i.e., male_upper_0.75, female_upper_0.75...)
Visualize mid-sagittal, mid-coronal, mid-horizontal slice of mean heatmap. 
To, easy understanding, open an individual T1 image and overlay it over heatmap
