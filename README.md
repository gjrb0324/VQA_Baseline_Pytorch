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


