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
![VQA_Baseline_Acc_Loss](https://user-images.githubusercontent.com/48676255/156728840-32d58692-cb90-4bf5-9f6f-10a8033ecd7a.png)


### Sample Result

![VQA_Baseline_result_1](https://user-images.githubusercontent.com/48676255/156728952-f658e878-083c-41aa-9663-f1d8ec7f4331.PNG)

left-above : question 1,2,3
left-below : question 4

![VQA_Baseline_result_2](https://user-images.githubusercontent.com/48676255/156729083-64255102-71c8-4aea-b31e-f1a9dbb43fe8.PNG)

left-above : question 1,2
left-below : question 3,4

![VQA_Baseline_result_3](https://user-images.githubusercontent.com/48676255/156729124-0286a265-667b-43ad-9163-81bf03c5a7e7.PNG)

left-above : question 1,2
left-below : question 3,4
