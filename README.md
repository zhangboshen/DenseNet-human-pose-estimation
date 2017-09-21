# DenseNet-human-pose-estimation
This is a simply and raw code using DenseNet to do human pose estimation task.

This url(https://resbyte.github.io/posts/2017/05/tf-densenet/) is a DenseNet model stucture definition on Tensorflow.

I add two fully_connected layers to do keypoints regression task.

DenseNet_Kinect_train.py is training code, including model stucture definition, data_processing and training phase with GradientDescentOptimizer.

P.S. This code is coarse, raw, I didn't consider too much about time consuming and runing efficiency.

## Data

*Input*:  Batch of 200×200 depth images, three channels, and of course it could be RGB image, containing one single person body.

*Output*: n joints(in my code,n = 11) coordinate， a n×2 vector.

These two dataset might help to provide training data:  
ITOP by Feifei Li et al, https://www.albert.cm/projects/viewpoint_3d_pose/ ,   
and NTU-RGBD by Jun Liu et al, http://rose1.ntu.edu.sg/datasets/actionrecognition.asp

## Results
Here are some results,

![Results](https://github.com/zhangboshen/DenseNet-human-pose-estimation/blob/master/results.jpg)



And these images are aquired  by Kinect V2, normalized to 0~255, copy paste to 3 channels.
