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
python3 UKB_sex_OcclusionSensitivity.py --target sex --save_dir /scratch/connectome/dhkdgmlghks/UKB_interpretation/sex/OcclusionSensitivity --mask_size 7 --stride 5
```

## Explanation for UKB_sex_OcclusionSensitivity.ipynb
This code making mean heatmap according to category. (i.e., male_upper_0.75, female_upper_0.75...)
Visualize mid-sagittal, mid-coronal, mid-horizontal slice of mean heatmap. 
To, easy understanding, open an individual T1 image and overlay it over heatmap
