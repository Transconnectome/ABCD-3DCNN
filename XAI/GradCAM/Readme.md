## Explanation for UKB_sex_GradCAM.py
This code making individual GradCAM heatmap. 
The results from **the last convolution layer of densenet** are visualized. 
Befor you run the code, directory should the same as follw.  

```
|--CODE RUNNING DIR
    |-- male_upper_0.75
    |-- male_lower_0.75
    |-- female_upper_0.75
    |-- female_lower_0.75

```

## Explanation for UKB_sex_GradCAM.ipynb
This code making mean heatmap according to category. (i.e., male_upper_0.75, female_upper_0.75...)
Visualize mid-sagittal, mid-coronal, mid-horizontal slice of mean heatmap. 
To, easy understanding, open an individual T1 image and overlay it over heatmap