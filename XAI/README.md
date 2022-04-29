## Description about codes 
 * **Tutorial_GradCAM_and_Occlusion Sensitivity.ipynb** file contains every detail description to do XAI
 * **Grad CAM** and **Occlusion Sensitivity** directory contains python script and ipython notebook for analysis pipeline 

## What is Grad CAM?
Grad CAM measures gradient between output of neural networks and pixels of input image. **(saliency := ∂f(x)/∂x)** 
When you think of the process of back propagation, gradient can be said as "Total amount of effect of input K on output class C". 
If some pixels have high gradient according to output class, it could be said that these pixels have great impact on model's prediciton.
Grad CAM++ is a variation of Grad CAM 
 * [참고 링크](https://tyami.github.io/deep%20learning/CNN-visualization-Grad-CAM/)
 * [원문 링크](https://arxiv.org/abs/1610.02391)

## What is Integrated Grad CAM?
Integrated Grad CAM is an extension of Grad CAM.  
As same as Grad CAM, integrated Grad CAM measures **∂f(x)/∂x**.  
Building on the same measures, furthermore, Integrated Grad CAM integrates **SmoothGrad and Interior Gradient**

  * **What is SmoothGrad?**  
  DNN uses piece-wise linear functions, so gradients of intermediate layers are not continuous but discontinuous.  
  SmoothGrad is a solution to overcome this problem by smoothing the gradient.   
  SmoothGrad add noise to pixel values of original input images, and get means of several results from saliency map measured with noise-added images.
  In equation, **SmoothGrad := 1/n(integral(∂f(x_hat)/∂x_hat)), x_hat = x + N(0,sigma^2)**  
  
  * **What is Interior Gradient?**  
  A feature may have a strong effect globally, but with small derivative locally.  
  In other words, gradient could be saturate, so that saliency map become noisy. (not detect object well)
  Interior Gradient adjust (reduce) pixel values of original input images, and get saliency map measured with adjusted images.  
  In equation, **Interior Gradient := ∂f(x_hat)/∂x_hat, x_hat = ax, 0 < a =< 1**  

**Be aware whether you want use noise tunneling or not. 
If you don't want to use noise tunneling, just deactivate the line assigning class "custom_noise_tunnel"**

* [참고 링크](https://www.youtube.com/watch?v=5fIy19GXAxI&list=PLypiXJdtIca5sxV7aE3-PS9fYX3vUdIOX&index=8)
* [원문 링크](https://arxiv.org/abs/1703.01365)

## What is Occlusion Sensitivity?
Occlusion Sensitivity follows this flow. 
 * 1. masking a patch for original images 
 * 2. concatenating every masked images (every masked images are fed into Neural Network as mini batches)
 * 3. getting scores from all of masked images
 * 4. drawing heatmap by this scores
It's like data augmentation techniques (or Maksed Auto Encoder).
In other words, Occlusion Sensitivity is like "sliding NxN size window through images and caculating occlusion sensitivity scores". If this score is high, it means "when this NxN size part of image is occluded, accuracy of model drops dramatically". 
 * [참고 링크](https://www.kaggle.com/code/blargl/simple-occlusion-and-saliency-maps/notebook)
 * [원문 링크](https://arxiv.org/pdf/1311.2901.pdf)
 * [코드 관련](https://docs.monai.io/en/stable/_modules/monai/visualize/occlusion_sensitivity.html)
 * [more details](https://stackoverflow.com/questions/59411239/how-does-the-occlusion-sensitivity-and-predicted-class-map-works-in-the-given-li)
