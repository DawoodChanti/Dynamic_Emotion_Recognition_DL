
# coding: utf-8

# In[ ]:

__author__='Dawood Al Chanti'


# In[ ]:

import tensorflow as tf
import numpy as np 
from tensorflow.contrib.layers.python import layers as tf_layers
from tensorflow.contrib.layers.python.layers import initializers


# In[ ]:

def xrange(x):
    return iter(range(x))

def Tshape(tensor):
    s = tensor.get_shape()
    return tuple([s[i].value for i in range(0, len(s))])


def variable_summaries(var,scopename):
    with tf.name_scope('summaries_'+scopename):
        tf.summary.histogram('histogram', var)

        
def mean_summaries(var,scopename):
    '''
    Average Mean of activation
    '''
    with tf.name_scope('summaries_'+scopename):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)     
        
# In[ ]:


def weight_variable(shape):
    weights_initializer=initializers.xavier_initializer(uniform=True,dtype=tf.float32)
    Weight = tf.Variable(weights_initializer(shape=shape))    
    return Weight


def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    Bias = tf.Variable(initial)
    return Bias


# In[ ]:

def conv3d(layer,kernel,filter_depth_stride,filter_spatial_stride):
     return tf.nn.conv3d(layer, kernel, [1, 
                                         filter_depth_stride,
                                         filter_spatial_stride,
                                         filter_spatial_stride,
                                         1], padding='SAME')

def max_pool3D(Input,kdepth,ksize,kdepth_stride,spatial_stride,scopename,WriteSummary=0):
    '''
    Arguments:
    Input: 5D Way Tensor
    kdepth: time pooling kernel
    ksize: Spatial pooling kernel
    Time stride or depth stride
    spatial stride
    scopename
    WriteSummary if 1 write to tensorboard a summary
    '''
    with tf.name_scope(scopename):
        pool= tf.nn.max_pool3d(Input, 
                          ksize=[1, kdepth, ksize, ksize, 1],
                          strides=[1, kdepth_stride, spatial_stride,spatial_stride, 1], 
                          padding='SAME')
    return pool


# In[ ]:  
def conv2d(Input, Filter,stride_val):
    return tf.nn.conv2d(Input, 
                        Filter, 
                        strides=[1, stride_val, stride_val, 1],
                        padding='SAME')

def max_pool(Input,kernelSize,stride_val,scopename):
    with tf.name_scope(scopename):
        pool= tf.nn.max_pool(Input, 
                          ksize=[1, kernelSize, kernelSize, 1],
                          strides=[1, stride_val, stride_val, 1], 
                          padding='SAME')
    return pool

def avg_pool(Input,kernelSize,stride_val,scopename,WriteSummary=0):
    with tf.name_scope(scopename):
        avg_pool= tf.nn.avg_pool(Input, 
                          ksize=[1, kernelSize, kernelSize, 1],
                          strides=[1, stride_val, stride_val, 1], 
                          padding='SAME')
    return avg_pool


def SPL2D(layer,k1,k2,k3,scopename):
    
    '''Spatial Pyramid Layer Composed of 3 Level:
    layer: Input 4D way tensor
    k1 k2 and k3 are the pool kernel and strides
    scopename: scope name of the tower
    WriteSummary if 1 , write summary to tensorboard
    
  
    TODO: Not Weighted Yet
    Apply 3 Level Spatial Pyramid and Concat their final output.
    It aim at normalizing the spatio temporal features
    Max poolingg take an input, Kernel Pooling Size and Spatial Stride Size.
    
    Weight Associated: 
    Level 0 : 1/2^(L-1) : where L is the pyramid level =4
    Level 1 to L : 1/2^(L-l+), where l is the current level and L goes from 1 to L

    '''
    
    Flatten = tf.contrib.layers.flatten    
    with tf.name_scope(scopename):
        SP1 = Flatten(tf.nn.max_pool(layer,[1,k1,k1,1],[1,k1,k1,1],padding='SAME',name='SP1'+scopename))
        SP2 = Flatten(tf.nn.max_pool(layer,[1,k2,k2,1],[1,k2,k2,1],padding='SAME',name='SP2'+scopename))
        SP3 = Flatten(tf.nn.max_pool(layer,[1,k3,k3,1],[1,k3,k3,1],padding='SAME',name='SP3'+scopename))
        SPP = tf.concat(1,[SP1,SP2,SP3],name='SPP2D'+scopename)
    return SPP




def SPL3D(layer,k1,k2,k3,k4,scopename):
    
    '''Spatial Pyramid Layer Composed of 3 Level:
    layer: Input 5D way tensor
    k1 k2 and k3 are the pool kernel and strides
    scopename: scope name of the tower
    WriteSummary if 1 , write summary to tensorboard
    '''

    Flatten = tf.contrib.layers.flatten    
    with tf.name_scope(scopename):
        SP1 = Flatten(tf.nn.max_pool3d(layer,[1,k1,k1,k1,1],[1,k1,k1,k1,1],padding='SAME',name='SP1'+scopename))
        SP2 = Flatten(tf.nn.max_pool3d(layer,[1,k2,k2,k2,1],[1,k2,k2,k2,1],padding='SAME',name='SP2'+scopename))
        SP3 = Flatten(tf.nn.max_pool3d(layer,[1,k3,k3,k3,1],[1,k3,k3,k3,1],padding='SAME',name='SP3'+scopename))
        SP4 = Flatten(tf.nn.max_pool3d(layer,[1,k4,k4,k4,1],[1,k4,k4,k4,1],padding='SAME',name='SP3'+scopename))
        SPP3D = tf.concat(1,[SP1,SP2,SP3,SP4],name='SPP3D'+scopename)
    return SPP3D


# In[ ]:

def lrn(layer):
    return tf.nn.lrn(layer, 
                     4,
                     bias=1.0, 
                     alpha=0.001 / 9.0, 
                     beta=0.75)
def ln(layer):
     return tf_layers.layer_norm(layer)
           
def dropout(input_layer_to_drop, keep_prob):
    droped_layer = tf.nn.dropout(input_layer_to_drop, keep_prob)
    return droped_layer

# Add Gaussian Noise
def gaussian_noise_layer(input_layer, std, isTrainPhase):
    """
    input_layer: layer which you would like to add the noise over
    std: noise variance ration
    isTrainPhase: Only add noise in training phase, while in test and validation, ignore the noise
    if the condition is ture, it will add noise, otherwise, noise will be 0
    
    tf.cond works as follows:
    ----------------------------
    def f1(): return tf.add(x, 1)
    def f2(): return tf.identity(x)
    r = tf.cond(train_phase, f1, f2)
    if train_phase is true, r=f1 otherewise r=f2
    """
    # Get the shape of the layer
    #tensor_shape = input_layer.get_shape().as_list()
    noise_layer = tf.cond(isTrainPhase, 
                          lambda: input_layer + tf.truncated_normal(shape=tf.shape(input_layer), 
                                                                    mean=0.0, stddev=std, 
                                                                    dtype=tf.float32), 
                          lambda: input_layer)
    return noise_layer

# In[ ]:

def ConvLayer2D(Input,
                out_channels,
                filter_size,
                strid_val,
                scopename,
                keep_prob_drop=None,
                IsActivated=True,
                WriteSummary=0):
    
    
    '''
    __author__='Dawood Al Chanti'
    
    A 2D Convolutional Layer that accept a 4D-way Tensor.
    
    Mandatory Input:
    ----------------
    Input: Layer or you input data of shape 4D" [Batch, spatial_x, spatial_y,In_channel]
    out_channels: Number of feature map
    filter_size: the height and the width of your filter: example: 3 
    
    : scalar striding value: example: 2 or 1
    scopename: The name of your scope 
    
    Optional:
    --------
    keep_prob_drop: drop probability: if None, consider it 1.0 : no drop out, which is the deafult value
    IsActivated: if true which is the default value: apply Relu as activation Function
    WriteSummary: if 0 which is the default value, don't write statistics otherwise do.
    Return: either activated or not activated conv layer.
    
    
    Example: How to use:
    
    scopename='Input'
    keep_prob_in = tf.placeholder(tf.float32, name='dropout_probability_'+scopename)
    H=20
    W=20
    num_channels=1
    x = tf.placeholder(tf.float32, [None, H, W, num_channels], name='Input')3
    ConvLayer2D(x,3,3,2,'Conv1',keep_prob_in,True,0)
    
    or :  ConvLayer2D(x,3,3,2,'Conv1')
    or:   ConvLayer2D(x,3,3,2,'Conv1',keep_prob_in,False,1)
    ....

    '''
    
    if keep_prob_drop==None:
        keep_prob_drop=1.0
    
    if WriteSummary==1:
        print('Please switch to tensorboard to visualize the variables statistics')
        
    with tf.name_scope(scopename):
        in_ch= Tshape(Input)[-1]
        out_ch= out_channels
        filter_s =filter_size
        stri = strid_val
        
        with tf.name_scope(scopename+'Weights'):
            kernel = weight_variable([filter_s, filter_s, in_ch, out_ch])
            if WriteSummary==1:
                variable_summaries(kernel,scopename)

        with tf.name_scope(scopename+'Bias'):
            bias = bias_variable([out_ch])

        with tf.name_scope(scopename+'Conv'):
            Conv = conv2d(Input, kernel,strid_val)
            Conv = tf.nn.bias_add(Conv, bias)
        
        if IsActivated:
            with tf.name_scope(scopename+'Activation'):
                Conv_activated = tf.nn.relu(Conv)
                with tf.name_scope(scopename+'activated_normalization'):
                    Conv_activated = ln(Conv_activated)                    
                with tf.name_scope(scopename+'activated_dropout'):
                        Conv_activated = dropout(Conv_activated,keep_prob_drop)
            return Conv_activated, kernel, bias
        
        else:
            with tf.name_scope(scopename+'not_activated_normalization'):
                Conv_not_activated = ln(Conv)
                with tf.name_scope(scopename+'not_activated_dropout'):
                    Conv_not_activated = dropout(Conv_not_activated,keep_prob_drop)
            return Conv_not_activated, kernel, bias


# In[ ]:

def weight_variable_mlp(shape):
    weights_initializer=initializers.xavier_initializer(uniform=True,dtype=tf.float32)
    Weight = tf.Variable(weights_initializer(shape=shape))    
    return Weight


def MLP(Input,
        num_hidden,
        scopename,
        keep_prob_drop=None,
        IsActivated=True,
        WriteSummary=0):
    '''Multi Layer Perceptron Layer
    Input Arguments:
    Input: Layer
    num_hidden: Number of hidden units
    scopename
    keep_prob_drop: dropout probability %
    IsActivated: is it relue?
    WriteSummary: 1 if u want to add summary to tensorboard
    '''
    if keep_prob_drop==None:
        keep_prob_drop=1.0
    
    if WriteSummary==1:
        print('Please switch to tensorboard to visualize the variables statistics')
        
    with tf.name_scope(scopename):
    
        in_ch= Tshape(Input)[-1]

        with tf.name_scope(scopename+'Weights'):
            Weights = weight_variable_mlp([in_ch, num_hidden])
            if WriteSummary==1:
                variable_summaries(Weights,scopename)

        with tf.name_scope(scopename+'Bias'):
            bias = bias_variable([num_hidden])

        with tf.name_scope('Fully_Connected'+scopename):
            FullyLayer = tf.matmul(Input, Weights) 
            FullyLayer = tf.nn.bias_add(FullyLayer, bias)
        
        if IsActivated:
            with tf.name_scope(scopename+'Activation'):
                FullyLayer_activated = tf.nn.relu(FullyLayer)
                with tf.name_scope(scopename+'activated_normalization'):
                    FullyLayer_activated = ln(FullyLayer_activated)
                with tf.name_scope(scopename+'activated_dropout'):
                    FullyLayer_activated = dropout(FullyLayer_activated,keep_prob_drop)    
            return FullyLayer_activated, Weights, bias
        
        else:
            with tf.name_scope(scopename+'not_activated_normalization'):
                FullyLayer_not_activated = ln(FullyLayer)
            with tf.name_scope(scopename+'not_activated_dropout'):
                FullyLayer_not_activated = dropout(FullyLayer_not_activated,keep_prob_drop)            
            return FullyLayer_not_activated, Weights, bias

# In[ ]:

def MLP_Out_Layer(Input,
        num_classes,
        scopename,
        WriteSummary=0):

    if WriteSummary==1:
        print('Please switch to tensorboard to visualize the variables statistics')
        
    with tf.name_scope(scopename):
        in_ch= Tshape(Input)[-1]
        with tf.name_scope(scopename+'Weights'):
            Weights = weight_variable_mlp([in_ch, num_classes])
            if WriteSummary==1:
                variable_summaries(Weights,scopename)
        with tf.name_scope(scopename+'Bias'):
            bias = bias_variable([num_classes])

        with tf.name_scope('Output_Layer'+scopename):
            FullyLayer = tf.matmul(Input, Weights) 
            FullyLayer = tf.nn.bias_add(FullyLayer, bias)
        return FullyLayer, Weights, bias

# In[ ]:
    
def ConvLayer3D(Input,
                Feature_Map,
                filter_depth,
                filter_size,
                filter_depth_stride,
                filter_spatial_stride,
                scopename,
                keep_prob_drop=None,
                IsActivated=True,
                WriteSummary=0):
    
    
    '''
    A 3D Convolutional Layer that accept a 5D-way Tensor: Batch, in_depth, in_hieght, in_width, in_channels
    
    Mandatory Input:
    ----------------
    Input:
    Feature_Map: Number of feature maps
    filter_depth: time step
    filter_size: the height and the width of your filter
    scopename: The name of your scope 
    
    Optional:
    --------
    keep_prob_drop: drop probability: if None, consider it 1.0 : no drop out, which is the deafult value
    IsActivated: if true which is the default value: apply Relu as activation Function
    WriteSummary: if 0 which is the default value, don't write statistics otherwise do.
    Return: either activated or not activated conv3d layer.
    
    '''

    if keep_prob_drop==None:
        keep_prob_drop=1.0
    
    if WriteSummary==1:
        print('Please switch to tensorboard to visualize the variables statistics')
        
    with tf.name_scope(scopename):
        in_ch= Tshape(Input)[-1]
        with tf.name_scope(scopename+'Weights'):
            kernel = weight_variable([filter_depth, filter_size, filter_size, in_ch, Feature_Map])
            if WriteSummary==1:
                variable_summaries(kernel,scopename)

        with tf.name_scope(scopename+'Bias'):
            bias = bias_variable([Feature_Map])

        with tf.name_scope(scopename+'Conv3D'):
            Conv3D =  conv3d(Input, kernel, filter_depth_stride,filter_spatial_stride)
            Conv3D = tf.nn.bias_add(Conv3D, bias)
        
        if IsActivated:
            with tf.name_scope(scopename+'Activation'):
                Conv_activated = tf.nn.relu(Conv3D)
                with tf.name_scope(scopename+'activated_normalization'):
                    Conv_activated = ln(Conv_activated) 
                with tf.name_scope(scopename+'activated_dropout'):
                        Conv_activated = dropout(Conv_activated,keep_prob_drop)
            return Conv_activated, kernel, bias
        
        else:
            with tf.name_scope(scopename+'not_activated_normalization'):
                Conv_not_activated = ln(Conv3D) 
                with tf.name_scope(scopename+'not_activated_dropout'):
                    Conv_not_activated = dropout(Conv_not_activated,keep_prob_drop)
            return Conv_not_activated, kernel, bias    
    