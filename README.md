# DenseNet-human-pose-estimation
This is a simply and raw code using DenseNet to do human pose estimation task.

This url(https://resbyte.github.io/posts/2017/05/tf-densenet/) is a DenseNet model stucture definition on Tensorflow.

I add two fully_connected layers to do keypoints regression task.

DenseNet_Kinect_train.py is training code, including model stucture definition, data_processing and training phase with GradientDescentOptimizer.

P.S. This code is coarse, raw, I didn't consider too much about time consuming and runing efficiency.

# Data
Input: 200×200 depth images, containing one single person body.

Output: n joints(in my code,n = 11) coordinate， a n×2 vector.





