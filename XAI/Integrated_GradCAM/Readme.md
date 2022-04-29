# flow of pipeline 
At first, run **UKB_sex_IntegratedGradCAM.py** file to obtain individual heatmap. 
Second, mean template of every individual heatmap **UKB_sex_IntegratedGradCAM.ipynb** and visualize it. 

## Explanation for UKB_sex_IntegratedGradCAM.py
This code making individual IntegratedGradCAM heatmap. 
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
python3 UKB_sex_GradCAM.py --target sex --save_dir /scratch/connectome/dhkdgmlghks/UKB_interpretation/sex/GradCAM
```

## Explanation for UKB_sex_IntegratedGradCAM.ipynb
This code making mean heatmap according to category. (i.e., male_upper_0.75, female_upper_0.75...)
Visualize mid-sagittal, mid-coronal, mid-horizontal slice of mean heatmap. 
To, easy understanding, open an individual T1 image and overlay it over heatmap

