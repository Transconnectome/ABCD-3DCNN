# Latent Brain Structural Features Associated with Abnormal Developmental Trajectory 

## contents 
  * [objective](#objective)
  * [data](#data)
  * [project_step](#project_step)
  * [참고자료](#참고자료)
  * [test_model](#test_model)
  * [결과](#결과)
  
  
## objective  
정신 질환, 그 중에서도 abnormal trajectory에 영향을 미치는 정신 질환과 연관된 Brain의 latent something을 Neural Net이 represent 할 수 있을까?  
뇌-행동을 한번 mapping 해보고자 하는 것.  
Brain에서 질병으로 가는 ‘latent something’을 알 수 있지 않을까?  
이를 multi task learning을 통해서, 서로 heterogeneous한 것으로 보이는 정신질환들에서, ‘abnormal developmental trajectory’를 뇌의 수준에서 표상하는 structural feature를 뽑아낼 수 있을까?

## data 
ABCD release 2.0 T1w data (n_subject = 7088)

## project_step 
  * **step 1**: 다양한 Neural Net architecture (DenseNet, ResNet, vision transformer=optional)로 가장 간단한 변수들(sex, age 등)이 학습이 되는지를 확인해서 **모델 확인**      
  * **step 2**: 목적으로 하는 target variable들로 실험을 해보기. **Multi task learning.** 
  * **step 3**: **interpretation(grad-CAM**) 과정을 통해서 이미지의 어떤 부분에 attention이 있는 지 히트맵으로 확인하거나 **전체 heat map의 distribution을 살펴보면서 insight 끌어 내기**.  예를 들면, ASD를 target으로 예측하고 heat map을 봤는데 다른 cluster가 생기는지 확인. 목적은 Neural Network를 scientific tool로써 사용해보는 것! (아마 Bayesian approach가 해석의 측면에서도 도움이 될 것 같다! 하지만 학습이 오래 걸린다는 단점은 존재)
  * **step 4**: **Transfer learning** 시도 해보기.


## 참고자료
  * [optimizer](https://dacon.io/codeshare/2370): 모델 상관없이 붙일 수 있는 optimizer. model의 generalizability를 증가시키는 방법. 간단한 ResNet에만 붙여도 성능 향상 및 generalizability 증가  
  * [FixRes (Fixing the test-train discrepancy)](https://arxiv.org/pdf/2003.08237v5.pdf): 모델 상관없이 붙일 수 있음. train과 test 시의 input 이미지의 해상도를 바꾸는 기법.  
  * [multi task learning](https://arxiv.org/pdf/2003.08237v5.pdf)
  * [Recent version survey of multi task learning](https://arxiv.org/pdf/2009.09796.pdf) (NDDR-CNN은 확장성이 좋음) 


## test_model
  * **Simple CNN architecture**
  * **VGGNet (11, 13, 16, 19)**
  * **ResNet** 
  * **DenseNet**
  * **EfficientNet**


## 결과 
### STEP 1. Sex Classification ([paper link](https://github.com/Transconnectome/ABCD-3DCNN/blob/main/paper/3DCNN_________.pdf ))
### Abstract  (300 words)
 * **Background**  
Sex differences in human cognition and behaviors originate from the differences of brain structure. Conventional neuroscience studies rarely support a clear-cut sex difference in the human brain. On the stark contrast, recent computational neuroimaging (e.g., convolutional neural networks (CNN)) shows an accurate classification of sex based on the brain anatomical images (e.g., T1-weighted MRI), supporting a strong case for the brain sex differences. Since these studies are based on adults, an important question is the developmental window in which the sex differences in the brain emerge. I.e., can neural networks classify a sex of a preadolescent brain of which the organizational effect of sex steroids have yet to be full blown?
To test this, we trained multiple neural network architectures to learn sex differences of the brain with T1-weighted brain images in prepubertal children.  

 * **Methods**  
We used T1w images of 7,088 (male: 3,694, female: 3,356) children ages 9 to 10 from the Adolescent Brain Cognitive Development study. Each image was skull-stripped and spatially normalized to the ICBM pediatric asymmetrical template. The input size of images was resized for (96, 96, 96). We built simple CNN architecture as a baseline model and compared two different CNN architectures: VGGNet-19 and DenseNet-121.

 * **Results**  
VGGNet-19 outperformed to predict sex (Accuracy=0.94; F1-score=0.95). Baseline model and DesNet-121 also show over 90% accuracy. Of note, our models show a higher accuracy with the smaller samples than the prior CNN studies with adults.

 * **Conclusions**  
In summary, this study shows CNN can predict sex from T1w images. The results not only support sex differences in brain structure in preadolescent, but also show the capability of CNN to learn latent features from brain images. Considering the important role of sex differences in human, pre-trained CNN models with sex could be utilized for predicting more complex cognition and behavior by transfer learning.



  
 

### STEP 2. Multitask Learning


### STEP 3. Interpretation 


### STEP 4. Transfer Learning
