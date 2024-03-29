# Update 
```custom_noise_tunnel.py``` is a customized python script originated from captum's noise_tunnel.py.  
This script will prevent adding noise to background of T1w images.  
You should use **captum.__version__ ==0.6.0 for ```custom_noise_tunnel.py```**.  

# flow of pipeline 
At first, run **UKB_sex_IntegratedGrad.py** file to obtain individual heatmap. 
Second, mean template of every individual heatmap **UKB_sex_IntegratedGrad.ipynb** and visualize it. 

## Explanation for UKB_sex_IntegratedGrad.py
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
python3 UKB_sex_GradCAM.py --target sex --save_dir /scratch/connectome/dhkdgmlghks/UKB_interpretation/sex/GradCAM --model_dir {model_name.pth}
```

## Explanation for UKB_sex_IntegratedGrad.ipynb
This code making mean heatmap according to category. (i.e., male_upper_0.75, female_upper_0.75...)
Visualize mid-sagittal, mid-coronal, mid-horizontal slice of mean heatmap. 
To, easy understanding, open an individual T1 image and overlay it over heatmap
