# Strong baseline for viusal question answering

This is a re-implementation of Vahid Kazemi and Ali Elqursh's paper [Show, Ask, Attend, and Answer: A Strong Baseline For Visual Question Answering in Pytorch](http://arxiv.org/abs/1704.03162) based on [Cyanogenoid's code](http://github.com/Cyanogenoid/pytorch-vqa).

### preprocess-images.py
Change deprecated usage and rebuilt custom ResNet152 Loader without 'Pycaffe'

### model.py
Since Cyanogenoid focused on upgrading model's performance, I rebuilt model.py based on the paper. 

### train.py
Now tensorboard is available, so tracker for loss and accuracy is unneeded.
Also, use gradient clipping to prevent result from gradient exploding.

### How to run
1. install CoCo Datasets and set config.py's json and image file routes.
2. preprocess images with 'python preprocess-images.py' command
3. preprocess vocabulary (questions, answers) with 'python preprocss-vocab.py' command
4. Run training and evaluating steps with 'python train.py' command


### Accuracy and Train loss

### Sample Result###
![VQA_Baseline_result_1](https://user-images.githubusercontent.com/48676255/156728446-444ceafc-81e8-4a27-8d9e-f687cb1c8ed9.PNG)
![VQA_Baseline_result_2](https://user-images.githubusercontent.com/48676255/156728454-ff802123-926b-4e13-ba9b-c6c5c804e938.PNG)
![VQA_Baseline_result_3](https://user-images.githubusercontent.com/48676255/156728464-ddbd360a-378d-4291-99ba-2fb7e926b904.PNG)
