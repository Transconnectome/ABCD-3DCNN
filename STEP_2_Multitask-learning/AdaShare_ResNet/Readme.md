## 테스트 커맨드  
  ```{.python} 
python3 AdaShare_ResNet/AdaShare_ResNet_run.py --model resnet3D50 --train_batch_size 8 --optim Adam --epoch 10 --exp_name AdaShare_test --cat_target sex --num_target age 
 ```
  
## script flow 
  
데이터 로딩 
-> preprocessing하고 train / val / test 데이터 나누기  
-> training dataset 중에서 warming up용 learning용 데이터셋 나누기  
-> warming up 데이터셋으로 warming up 모델 돌리기 (모델의 학습 자체보다는 policy estimation이 중점이다. 따라서 iteration은 최대한 많이 가져가도 될 듯) 단, 이때 validation set을 가지고 scheduler도 적용하고, val acc 가장 높은 모델은 저장하도록 하기   
-> policy에서 argmax로 테스크별 layer에 대한 policy 추출하기  
-> warming up fix policy로 하여서 training 중에서 learning용 데이터셋으로 모델 돌리기. 이때 모델은 warming up 과정에서 pre-train 된 모델을 갖고 오기  
-> validation set으로 역시 scheduler 및 모델 저장하면서 학습시키기  
-> test set에서의 결과 보고.  
  
## 주의할점 
1. DataParallel의 경우 MTL 클래스가 아니라 Resnet3d 클래스에서 정의해주어야 한다. 즉, MTL 클래스의
  ```{.python} 
  self.backbone = nn.DataParallel(Resnet3d(),device_ids)
  ```
  가 되어야 한다.

  따라서, input과 label 또한
  ```{.python} 
  net.to(f'cuda:{net.device_ids[0]}')
  ```
  가 아니라  
  ```{.python} 
  net.to(f'cuda:{net.backbone.device_ids[0]}'}')
  ```
  가 되어야 한다.  
  
2. warming up phase에서는 warmup() 함수와 validate() 함수만 사용하며, learning phase에서는 train(), validate(), test() 함수를 사용해야 한다.  
   특히 warming up phase에서는 input이 skip하는 layer block없이, 모든 layer block을 forward 하면서 policy logit distribution을 추정해야하기 때문에 'mode'가 train 혹은 eval이어야 한다.  
   warming up phase가 끝난 다음에는 학습한 policy logit distribution을 근거로 하여, 0 또는 1로 구성된 policy, 즉 fix_policy로 learning phase에 넘어가야하기 떄문에 'mode'는 fix_policy가 되어야 한다.  
   이는 하이퍼 파라미터 변수들을 dictionary 형태로 만든 객체인 input_dict를 다음과 같이 MTL 클래스에 넣어줌으로써 기능하게 된다. 
   ```{.python}
   net(image, **input_dict) 
   ```
       
   또한 mode가 중요한 이유는, validate() 함수의 경우 warming up phase와 learning phase 모두에서 활용되는데, learning phase에서는 굳이 policy를 MTL의 forward()의 결과로 뱉을 이유가 없기 때문에, mode에 따라서 MTL의 forward()의 return 값이 달라지게 된다. (warming up phase에서는 tuple, learning phase에서는 single object) 
  
