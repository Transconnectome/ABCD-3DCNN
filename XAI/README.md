* **Tutorial_GradCAM_and_Occlusion Sensitivity.ipynb** file contains every detail description to do XAI
* **Grad CAM** and **Occlusion Sensitivity** directory contains python script and ipython notebook for analysis pipeline 

## What is Grad CAM?
Grad CAM measures gradient between label (class of image) and pixels of input image. 
When you think of the process of back propagation, gradient can be said as "Total amount of effect of input K on output class C". 
If some pixels have high gradient according to output class, it could be said that these pixels have great impact on model's prediciton.

## What is Occlusion Sensitivity?
Occlusion Sensitivity follows this flow. 
 * 1. masking a patch for original images 
 * 2. concatenating every masked images (every masked images are fed into Neural Network as mini batches)
 * 3. getting scores from all of masked images
 * 4. drawing heatmap by this scores
It's like data augmentation techniques (or Maksed Auto Encoder).
In other words, Occlusion Sensitivity is like "sliding NxN size window through images and caculating occlusion sensitivity scores". If this score is high, it means "when this NxN size part of image is occluded, accuracy of model drops dramatically". 
