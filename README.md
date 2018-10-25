# Paper Title: 
Deep-Learning-for-Spatio-Temporal-Modeling-of-Dynamic-Spontaneous-Emotions

# Paper Link:
https://ieeexplore.ieee.org/document/8481451

# Abstract:
Facial expressions involve dynamic morphological changes in a face, conveying information about the expresser's feelings. Each emotion has a specific spatial deformation over the face and temporal profile with distinct time segments. We aim at modeling the human dynamic emotional behavior by taking into consideration the visual content of the face and its evolution. But emotions can both speed-up or slow-down, therefore it is important to incorporate information from the local neighborhood frames (short-term dependencies) and the global setting (long-term dependencies) to summarize the segment context despite of its time variations. A 3D-Convolutional Neural Networks (3D-CNN) is used to learn early local spatiotemporal features. The 3D-CNN is designed to capture subtle spatiotemporal changes that may occur on the face. Then a Convolutional-Long-Short-Term-Memory (ConvLSTM) network is designed to learn semantic information by taking into account longer spatiotemporal dependencies. The ConvLSTM network helps considering the global visual saliency of the expression. That is locating and learning features in space and time that stand out from their local neighbors in order to signify distinctive facial expression features along the entire sequence. Non-variant representations based on aggregating global spatiotemporal features at increasingly fine resolutions are then done using a weighted Spatial Pyramid Pooling layer.


# Citation:
@ARTICLE{8481451, 
author={D. A. AL CHANTI and A. Caplier}, 
journal={IEEE Transactions on Affective Computing}, 
title={Deep Learning for Spatio-Temporal Modeling of Dynamic Spontaneous Emotions}, 
year={2018}, 
volume={}, 
number={}, 
pages={1-1}, 
keywords={Spatiotemporal phenomena;Visualization;Face recognition;Face;Videos;Machine learning;Computational modeling;3D-CNN;ConvLSTM;Deep Learning;Dynamic Emotion;Facial Expression;SPP-net;Spatiotemporal Features}, 
doi={10.1109/TAFFC.2018.2873600}, 
ISSN={1949-3045}, 
month={},}


# Code Description:

> This code is the implementation of the main model provided in the paper: Deep Learning for Spatio-Temporal Modeling of Dynamic Spontaneous Emotions DOI: 10.1109/TAFFC.2018.2873600

> The code here is provided as a set of functions, where debugging is very easy to track.
It is straight forward and simple to modify or track or replicate.


## Data parser and Augmenation
The first part of the code provide an intuition on:
1. How to prepare your data.
2. How to sample from you data during the session without storing all data in memory.
3. How to perform Data Augmentation to increase the variability and the size of your data.

## Main Modules for running the code correctly
The second part of the code provide the main model:
1. ConvLayer Module: is an implementation of:
    1.1. 2D Convolutional Neural Network
    1.2. 3D Convolutional Neural Network
    1.3. 2D max pooling 
    1.4. 3D max pooling
    1.5. 2D Spatial Pyramid
    1.6. 3D Spatial Pyramid
    1.7. Fully Connected Layer
    1.8. Read Out Layer
    1.9. Other helpful functions
    
    
2. EfficientConvLSTMUnit Module: is the main implementation of ConvLSTM.

3. Utilities Module: is a set of helpful function to display, show and replicate the figures provided in the paper.
    
# Main Model: Defining the Graph"
1. SpatioTemporal_Feature_Learning : is an implementation of the model
2. Its running on the dataset
3. How to plot and replicate the figure.
4. It is simple and straight forward. Easy to manage and repreduce over any sequential dataset.

  

## Environment and Dependencies
  1. Python version 3.5.2
  2. TensorFlow
  3. OpenCV
  4. matplotlib
  5. Many other python packages.

