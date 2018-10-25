
# coding: utf-8

# In[1]:

__author__ = 'Dawood Al Chanti'
__Lab__ = 'Gipsa-Lab'
__Corresponding_Paper__ = 'https://ieeexplore.ieee.org/document/8481451'
__Journal__ = 'IEEE Transactions on Affective Computing'
__Model__ = 'Corresponding Implementation of the model'
__license__ = 'Usage of the code require citation of the paper, DOI: 10.1109/TAFFC.2018.2873600'


# # GIPSA-Lab / DIS Department
# ## Spatio Temporal Feature Extraction : Dynamic Model

# In[2]:

import tensorflow as tf
import numpy as np 
import math
import pickle
import pylab as pl
from IPython import display
import matplotlib.pyplot as plt
import os
import time
import sys
#import cv2
import random
#from scipy.misc import imresize
from tensorflow.contrib.layers.python import layers as tf_layers
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python.layers import initializers
import csv
get_ipython().magic(u'matplotlib inline')


# ### Import the main functions for 3D-Conv, ConvLSTM, 2D-Conv, SPP and the rest of the functions.

# In[3]:

from ConvLayer import *
from Utilities import *
from EfficientConvLSTMUnit import conv_lstm_cell


# ### Data path to sample from disc

# In[4]:

DataPath = 'E:/.....'


# ### Define the data preferences

# In[5]:

# Data Characteristics:
H=128
W=128
num_channels=1
FV= H * W
num_classes = 6

seq_length = 75
batch_size = None


# ### Keep Holder for Drop out and the corresponding summary in tensorboard

# In[6]:

with tf.name_scope('dropout_Input'):
    keep_prob_in = tf.placeholder(tf.float32)
    tf.summary.scalar('keep_prob_In', keep_prob_in)

with tf.name_scope('dropout_Output'):
    keep_prob_out = tf.placeholder(tf.float32)
    tf.summary.scalar('keep_prob_out', keep_prob_out)
    
with tf.name_scope('dropout_lstm'):
    keep_prob_lstm = tf.placeholder(tf.float32)
    tf.summary.scalar('keep_prob_lstm', keep_prob_lstm)
    
with tf.name_scope('dropout_convolution'):
    keep_prob_conv = tf.placeholder(tf.float32)
    tf.summary.scalar('keep_prob_conv', keep_prob_conv)
       
with tf.name_scope('dropout_mlp'):
    keep_prob_mlp = tf.placeholder(tf.float32)
    tf.summary.scalar('keep_prob_mlp', keep_prob_mlp)


# In[7]:

# A place holder as a pool to distinguish training phase from test phase
isTrain = tf.placeholder(tf.bool, name='training_phase')


# In[8]:

# Store and manipulate the shape of the list
x_unwrap_lstm = []
x_unwrap_conv = []


# ### Define the Model

# In[9]:

print('Build the Graph that represent the model')

with tf.name_scope('Input_Layer'):
    x = tf.placeholder(tf.float32, [None, seq_length, H, W, num_channels])
with tf.name_scope('Target'):
    y_ = tf.placeholder(tf.float32, shape=[None, num_classes])


# ### Drop some of the input features as a method of introducing Noise.

# In[ ]:

with tf.name_scope('Drop_In'):
    print('Perform Dropout over the Input Signal')
    x_drop = dropout(x, keep_prob_in)
    
print('The shape of the Input is: ')
print('---------------------------')
print(x) 


# ### Local and Global Spatio Temporal Feature Extraction Via 3D Convolutional Network

# In[11]:

lstm_func = conv_lstm_cell
lstm_state1, lstm_state2   = None, None


# ##### Performs truncated backpropagation over the time series frames: first pass initiate the variable and after first time step use them.
# ##### if the sequence length is 75 and we chose seq_batch=25, then we go through 3 times. Think about it as 25 frame per second.

# In[12]:

#--------------------- Define the Convolutional LSTM Network Block---------------------------------#
# Performs truncated backpropagation over the time series frames: first pass initiate the variable and after first time step use them.
varreuse_conv=0
varreuse_lstm=0
varreuse_2dconv=0
seq_batch = 25
looprange = int(seq_length/seq_batch)
for i in range(0,looprange): 
    reuseflagConv = bool(varreuse_conv)
    offset_seq = (i * seq_batch) % ((seq_batch+seq_length) - seq_batch)
    # Take a batch of 10 seq len from the data.
    new_current_input = x_drop[:,offset_seq:(offset_seq + seq_batch), :,:,:]
    
    # Local Feature Extraction over neighborhood frames
    with tf.variable_scope("Short_Term_Dependencies_Block",reuse=reuseflagConv) as scope:
        # Input Layer,Feature Maps, Kenrel Depth, Kernel Size, Depth Stride, Spatial Stride, ...
        conv3d_1,W_Conv3d_1, b_conv3d_1 = ConvLayer3D(new_current_input,60,3,3,2,2,'Conv3D_1',
                                                     keep_prob_conv,IsActivated=False,WriteSummary=1)
        
        print(conv3d_1)
        
        conv3d_2,W_Conv3d_2, b_conv3d_2 = ConvLayer3D(conv3d_1,60,3,3,2,2,'Conv3D_2',
                                                     keep_prob_conv,IsActivated=True,WriteSummary=1)
      

        print(conv3d_2)
        
        
        conv3d_3,W_Conv3d_3, b_conv3d_3 = ConvLayer3D(conv3d_2,90,3,3,2,2,'Conv3D_3',
                                                     keep_prob_conv,IsActivated=True,WriteSummary=1)
        

        print(conv3d_3)
        

    
        #--------------------- Define the Convolutional LSTM Network Block---------------------------------#
        #-- Here consider creating a loop since the input is going to be sequential: xt1, xt2, .... xtn
        Lstm_seq_length = Tshape(conv3d_3)[1] 

        for j in xrange(Lstm_seq_length): 
            print('Step: ', j)
            print('---------')
            reuseflag_lstm = bool(varreuse_lstm)
            reuseflag_conv2D = bool(varreuse_2dconv)
            
            current_input_lstm = conv3d_3[:,j,:,:,:]
            
            
            #Transition state for downsampling
            with tf.variable_scope("2D_ConvBlock",reuse=reuseflag_conv2D) as scope:
                
                conv2d_1,W_Conv2d_1, b_conv2d_1 = ConvLayer2D(current_input_lstm,90,3,1,
                                                              scopename='Conv2D_1',
                                                              keep_prob_drop=keep_prob_conv,
                                                              IsActivated=True,
                                                              WriteSummary=1)

                Pool_1 = max_pool(conv2d_1,3,2,scopename='Pool_1')
                
                print(Pool_1)
                
                print('----------------------------------------------------------')
                
          
            #Global Feature Extraction Accross the Entire Sequence
            with slim.arg_scope([lstm_func, 
                                 slim.layers.conv2d, 
                                 slim.layers.fully_connected,tf_layers.layer_norm],
                                reuse=reuseflag_lstm):

                                                      
                print('----------------------------------------------------------')
                
                Hidden_out1, lstm_state1 = lstm_func(Pool_1, lstm_state1,45,filter_size=3,scope='Conv_LSTM_Cell_1')
                lstm_state1 = tf_layers.layer_norm(lstm_state1,scope='layer_norm_H1_State')
                lstm_state1 = dropout(lstm_state1, keep_prob_lstm)
                print('Hidden State 1: ', lstm_state1)
                
                # pass the cell state alonge the hidden state of the previous cell.
                Hidden_out2, lstm_state2 = lstm_func(lstm_state1, lstm_state2,45,filter_size=3,scope='Conv_LSTM_Cell_2')
                lstm_state2 = tf_layers.layer_norm(lstm_state2,scope='layer_norm_H2_State')
                lstm_state2 = dropout(lstm_state2, keep_prob_lstm)      
                print('Hidden State 2: ', lstm_state2)
                

                next_state = tf.concat(3, [Hidden_out1,Hidden_out2])
                next_state = tf_layers.layer_norm(next_state,scope='Output_layer_Norm')
                print('Out State 2: ', next_state)
                
            # use the parameter after the first pass    
            if (j == 0) & (i ==0):
                varreuse_lstm=1 
                varreuse_conv=1 
                varreuse_2dconv=1
            
            # at the last time step of the sequential process: consider taking the last state as it represent the contextual content of the 
            # entire sequence
            if (i == looprange-1) & (j==Lstm_seq_length-1):    
                print('Appending: ', next_state)
                print('-------------------------------------------------------------------------------')
                x_unwrap_lstm.append(next_state)
                print('Appending output shape: ', len(x_unwrap_lstm))



# ### Retrieve the correct shape of the Tensor

# In[20]:

x_unwrap_lstm = tf.pack(x_unwrap_lstm)  # stack or pack have the same function, preferable stack it is compatible with new version
print('output shape all together: ', x_unwrap_lstm)

x_unwrap_lstm = tf.transpose(x_unwrap_lstm,perm=[1,0,2,3,4])
print('output shape all together after shifting batch with time step: ', x_unwrap_lstm)



# In[21]:

# Get the shape of the last layer so we can reshape it
_,seq,h_pool_image_size_H,h_pool_image_size_W,LastLayer_num_filter = Tshape(x_unwrap_lstm)
x_unwrap_lstm = tf.reshape(x_unwrap_lstm, [-1, h_pool_image_size_H , h_pool_image_size_W , LastLayer_num_filter])

print('Reshaped Layer ', x_unwrap_lstm)


# ## Perform Spatial Pyramid Layer

# In[28]:

Flatten = tf.contrib.layers.flatten 
k1=8
k2=4
k3=2
W_adaptative = tf.nn.softmax(weight_variable([3]))
with tf.variable_scope("SPPBlock") as scope:
        SP1 = Flatten(tf.nn.max_pool(x_unwrap_lstm,[1,k1,k1,1],[1,k1,k1,1],padding='SAME',name='SP1'+'SPPBlock'))
        SP2 = Flatten(tf.nn.max_pool(x_unwrap_lstm,[1,k2,k2,1],[1,k2,k2,1],padding='SAME',name='SP2'+'SPPBlock'))
        SP3 = Flatten(tf.nn.max_pool(x_unwrap_lstm,[1,k3,k3,1],[1,k3,k3,1],padding='SAME',name='SP3'+'SPPBlock'))
        ## Apply either Adaptative Weightining or based on the formula introduced in the paper
        SPL = tf.concat(1,[W_adaptative[0]*SP1,W_adaptative[1]*SP2,W_adaptative[2]*SP3],name='SPP2D'+'SPPBlock')


# ### perform dropout over the obtained Feature Vector

# In[30]:

SPL = dropout(SPL,keep_prob_out)
print(SPL)


# ### Fully Connected Layer

# In[31]:

# FC1 hidden node numbers 
h1_num = 2024
print('hidden unit of FC1:, ',h1_num )

with tf.variable_scope("FC_Block") as scope:
    mlp,W_mlp,B_mlp = MLP(SPL,h1_num,
            'Local_1',
            keep_prob_drop=keep_prob_mlp,
            IsActivated=True,
            WriteSummary=1)
    
    mlp = tf_layers.layer_norm(mlp,scope='mmlpnorm')

    mlp = dropout(mlp,keep_prob_mlp)
    print(mlp)


# ### Read Out Layer

# In[35]:

with tf.variable_scope("Output_Block") as scope:
    y_conv, W_out, B_out = MLP_Out_Layer(mlp,6,'Scores',WriteSummary=0)
    print(y_conv)


# ### Optimizer

# In[ ]:

# Schaduale Learning rate
global_step = tf.Variable(0, trainable=False)
boundaries = [70000]
values = [0.0001,0.00001]
learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)

with tf.name_scope('loss'):
    #cross_entropy = tf_layers.losses.hinge_loss(y_conv, y_)
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))  
tf.summary.scalar("loss", cross_entropy)

    
with tf.name_scope('Optimizer'):
    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy,global_step=global_step)
    #train_step = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cross_entropy,global_step=global_step)

with tf.name_scope('correct_prediction'):
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))

with tf.name_scope('accuracy'):    
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar ("accuracy", accuracy)


# In[ ]:

saver = tf.train.Saver()
model_path = '/home/alchantd/MMI Exp/MMI Experiment/version_1_tensorboard/Model/'
file_path = model_path + 'SpatioTemporalModel'


# In[ ]:

if isRestor:
    print("Model will be restored.")
else:
    print('Model will be tained from Scratch')   


# ### Read the Data:

# In[ ]:

TrainPath = DataPath + 'Train/'


# In[ ]:

ValPath = DataPath + 'Val/'


# In[ ]:

TestPath = DataPath + 'Test/'


# ### Load the Batch Set from Disc

# In[ ]:

bsize = 32
print('Read batch of the dataset')
train_dataset,train_labels = loadBatchData(DataPath+'Train/',bsize)
valid_dataset,valid_labels = loadBatchData(DataPath+'Val/',16)
test_dataset,test_labels = loadBatchData(DataPath+'Test/',16)

print('Training set:', train_dataset.shape)
print('Validation set:', valid_dataset.shape)
print('Test set:', test_dataset.shape)


# ### Run the Session

# In[ ]:

init_o = tf.global_variables_initializer()
Modelsummary={}
with tf.Session() as sess:
    if isRestor:
        sess.run(init_o) 
        saver.restore(sess,model_path)
        print("Model restored.")
    else:
        sess.run(init_o) 
        print('Model Training from Scratch')
        
    #-------------------------------------------------------------------------------------------#        
    # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
    merged = tf.summary.merge_all()
    summaries_dirtrain = '/home/alchantd/MMI Exp/MMI Experiment/version_1_tensorboard/logdata/train'
    train_writer = tf.summary.FileWriter(summaries_dirtrain,sess.graph)
    summaries_dirval =  '/home/alchantd/MMI Exp/MMI Experiment/version_1_tensorboard/logdata/val'
    val_writer = tf.summary.FileWriter(summaries_dirval)  
    summaries_dirtest = '/home/alchantd/MMI Exp/MMI Experiment/version_1_tensorboard/logdata/test'
    test_writer = tf.summary.FileWriter(summaries_dirtest) 
    #-------------------------------------------------------------------------------------------#
        
    start_time = time.process_time()
    for i in range(75000):
        # load data every bsize 
        if i%bsize == 0:
            train_dataset,train_labels = loadBatchData(DataPath+'Train/',bsize)
            valid_dataset,valid_labels = loadBatchData(DataPath+'Val/',16)
            test_dataset,test_labels = loadBatchData(DataPath+'Test/',16)


        # For each iteration, compute the elapsed time
        batchtime = time.process_time()
        _, loss_val,summary_train,train_accuracy = sess.run([train_step, cross_entropy, merged, accuracy],
                                                            feed_dict={x: train_dataset, 
                                                                          y_: train_labels,
                                                                          keep_prob_in: 0.65,
                                                                          keep_prob_conv: 0.95,
                                                                          keep_prob_lstm: 0.35,
                                                                          keep_prob_out: 0.50,
                                                                          keep_prob_mlp: 0.50}) 

        train_writer.add_summary(summary_train, i)       
    
        if i%100 == 0:
            _, loss_val_Valid,summary_val,val_accuracy = sess.run([train_step, cross_entropy,merged,accuracy],
                                                                  feed_dict={x: valid_dataset, 
                                                                      y_: valid_labels,
                                                                      keep_prob_in: 0.95,
                                                                      keep_prob_conv: 1.0,
                                                                      keep_prob_lstm: 1.0,
                                                                      keep_prob_out: 1.0,
                                                                      keep_prob_mlp: 1.0}) 
            
            
            _, loss_val_Test,summary_test,test_accuracy = sess.run([train_step, cross_entropy,merged,accuracy],
                                                                  feed_dict={x: test_dataset, 
                                                                      y_: test_labels,
                                                                      keep_prob_in: 0.95,
                                                                      keep_prob_conv: 1.0,
                                                                      keep_prob_lstm: 1.0,
                                                                      keep_prob_out: 1.0,
                                                                      keep_prob_mlp: 1.0}) 
            
            val_writer.add_summary(summary_val, i)      
            test_writer.add_summary(summary_test, i)  
            

            pl.plot([i],loss_val,'b.',)
            pl.plot([i],train_accuracy,'r*')
            pl.plot([i],val_accuracy,'k*')
            pl.plot([i],loss_val_Valid,'g.')
            
            pl.plot([i],test_accuracy,'co')
            pl.plot([i],loss_val_Test,'mv')

            display.clear_output(wait=True)
            display.display(pl.gcf())   

            sys.stdout.flush()
            print("\rIteration: %s , Training Loss: %s , Val Loss: %s, Val2 Loss: %s"%(i,loss_val,loss_val_Valid,loss_val_Test))
            print("\rIteration: %s , Training Acc: %s , Val Acc: %s, Val2 Acc: %s"%(i,train_accuracy*100,val_accuracy*100,test_accuracy*100))
            print('---------------------------')
            
            Batchtime_elapsed = (time.process_time()- batchtime)
            print("Batch Time:  %s minutes ---\n" % Batchtime_elapsed) 

            # Save the model results for future better figures displays.
            Modelsummary[i]=[loss_val,loss_val_Valid,loss_val_Test,train_accuracy,val_accuracy,test_accuracy]

        if i%5000 == 0:
            #saver.save(sess, file_path)
            print('>> Model Saved and can be restored during another session!')
            saver.save(sess, file_path, global_step=i) 
            



    print('>> optimization Process Finished\n')
    time_elapsed = (time.process_time()/60.0 - start_time/60.)
    print("--- %s in minutes ---" % time_elapsed)
   

    #saver.save(sess, file_path, global_step=i) 
    #print('>> Model Saved!')


# In[ ]:

print('Saving the Summary Curve')
w = csv.writer(open(model_path + 'ModelSummary.csv', 'w'))
for key, val in Modelsummary.items():
    w.writerow([key, val])
print('Done')


# ## Model Performance Metrics

# In[ ]:

#----------- Extract the Predicted Label, True Label and the Scores from the Model------------#
Predicted_Label = tf.argmax(y_conv,1)
True_Label = tf.argmax(y_,1)
scores = tf.nn.softmax(y_conv) # Which is the softmax of the logits

"""
1:'anger',   -->0
2:'contempt', -->1 Ignored
3:'disgust', -->2
4:'fear', -->3
5:'happy', -->4
6:'sadness', -->5
7:'surprise'-->6
"""
# #### Model Performance over the Test Set

# In[ ]:

#----------------------- Restore the Model-----------------------------------# 
isRestor = True
init_o = tf.global_variables_initializer()
with tf.Session() as sess:
    if isRestor:
        sess.run(init_o) 
        saver.restore(sess,file_path+'-70000')
        
        print("Model restored.")

    print('-------------------------------------------------------------------------------')
    
    
     #----------------------- Compute the Accuracy Rate -----------------------------------#        
    val_accuracy = accuracy.eval(feed_dict={x: valid_dataset, 
                                            y_:valid_labels, 
                                             keep_prob_in: 0.65,
                                                                      keep_prob_conv: 1.0,
                                                                      keep_prob_lstm: 0.97,
                                                                      keep_prob_out: 1.0,
                                                                      keep_prob_mlp: 1.0}) 
  
    print("Valid Accuracy: %s "%(val_accuracy*100))
    print('-------------------------------------------------------------------------------')
       
    print('Computing the True Label, the Predicted Label and the scores of Validation Set')
    val_predicted_label= Predicted_Label.eval(feed_dict={x:valid_dataset,
                                                          y_: valid_labels,
                                           keep_prob_in:  0.70,
                                                                      keep_prob_conv: 1.0,
                                                                      keep_prob_lstm: 0.97,
                                                                      keep_prob_out: 1.0,
                                                                      keep_prob_mlp: 1.0}) 
    val_true_label= True_Label.eval(feed_dict={x:valid_dataset, 
                                                y_: valid_labels,
                                           keep_prob_in: 0.70,
                                                                      keep_prob_conv: 1.0,
                                                                      keep_prob_lstm: 0.97,
                                                                      keep_prob_out: 1.0,
                                                                      keep_prob_mlp: 1.0}) 
    
    val_scores = scores.eval(feed_dict={x:valid_dataset,
                                         y_: valid_labels,
                                            keep_prob_in: 0.70,
                                                                      keep_prob_conv: 1.0,
                                                                      keep_prob_lstm: 0.97,
                                                                      keep_prob_out: 1.0,
                                                                      keep_prob_mlp: 1.0}) 
    
    val_correct_prediction = correct_prediction.eval(feed_dict={x:valid_dataset,
                                         y_: valid_labels,
                                             keep_prob_in: 0.70,
                                                                      keep_prob_conv: 1.0,
                                                                      keep_prob_lstm: 0.97,
                                                                      keep_prob_out: 1.0,
                                                                      keep_prob_mlp: 1.0}) 
    
   
    print('Done')
    print('-------------------------------------------------------------------------------')


# In[ ]:

print('Test Set Metric Reports')
print('-----------------------------')
FullMetrics('Test Set',val_true_label,val_predicted_label,'MMICK+')

#print('ROC Curve for validation Set over Class index 1, which corresponds to AN')
#Single_ROC(valid_labels, val_scores,3)

print('ROC Curve for Test Set over all Classes')
ROC_Curve_Multi_Class(valid_labels, val_scores,'MMIck+')


# ## Plot a clean version of the learning Curve

# In[ ]:

lists = sorted(Modelsummary.items()) # sorted by key, return a list of tuples

x, y = zip(*lists) # unpack a list of pairs into two tuples
plt.xlabel('Iterations')
plt.ylabel('Cross Entropy Loss')
plt.title('Loss Function')

plt.plot(x, y,label='Model length')

fig1 = plt.gcf()
plt.show()
fig1.savefig('SummaryGraph.png', dpi=200)


# In[ ]:

## Lets Plot the same Curve But Smoother
from scipy.interpolate import spline


## Smoothing Function
x_sm = np.array(x)
y_sm = np.array(y)
x_smooth = np.linspace(x_sm.min(), x_sm.max(), 100)
y_smooth = spline(x, y, x_smooth)


# Curve
plt.xlabel('Iterations')
plt.ylabel('Cross Entropy Loss')
#plt.title('Loss Function')

plt.plot(x_smooth, y_smooth)

plt.legend(loc='best')
plt.savefig('MMI-LearningCurve-ALL-IN',dpi=200)


# ### Pefrome TSNE to visualize the manifold

# In[ ]:

from sklearn.manifold import TSNE
import matplotlib.patheffects as PathEffects
import seaborn as sns

def scatter(x, colors):
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", 10))

    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40,
                    c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # We add the labels for each digit.
    txts = []
    for i in range(1,7):
        # Position of each label.
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts


# In[ ]:

bsize = 20
print('Read batch of the dataset')
valid_dataset,valid_labels = loadBatchData(DataPath+'Train/',bsize)
print('Validation set:', valid_dataset.shape)


# In[ ]:

# get the bottle neck layer feature vector
activation_mlp = getActivations(mlp,valid_dataset,valid_labels)


# In[ ]:

# Project to int labels
stim_labels =Inverse_hot_one_encoder(valid_labels,6)
# We first reorder the data points according to the handwritten numbers.
activation_mlp = np.vstack([activation_mlp[stim_labels==i]for i in range(1,7)])
stim_labels = np.hstack([stim_labels[stim_labels==i]for i in range(1,7)])


# In[ ]:

print("Running t-SNE ...")
X_proj = TSNE(verbose=1).fit_transform(activation_mlp)


# ## Visualize some Feature Maps

# In[ ]:


def plotNNFilter_3D(units,frameToVisualize,FigureName):
    '''
    unit have the shape: 5D tensor: 1, remained frames, H,W,M
    We have to select one of the remained framed to visualize: [0,0,:,:,i] or [0,5,:,:,i] ...
    '''
    filters = units.shape[4]
    plt.figure(1, figsize=(20,20))
    n_columns = 6
    n_rows = math.ceil(filters / n_columns) + 1
    for i in range(filters):
        plt.subplot(n_rows, n_columns, i+1)
        #plt.title(str(i))
        plt.imshow(units[0,frameToVisualize,:,:,i], interpolation="nearest", cmap="gray")    
        plt.axis('off')
    fig1 = plt.gcf()
    #plt.show()
    fig1.savefig(FigureName, dpi=200)

def plotNNFilter_2D(units,FigureName):
    '''
    unit have the shape: 5D tensor: 1, remained frames, H,W,M
    We have to select one of the remained framed to visualize: [0,0,:,:,i] or [0,5,:,:,i] ...
    '''
    filters = units.shape[3]
    plt.figure(1, figsize=(20,20))
    n_columns = 6
    n_rows = math.ceil(filters / n_columns) + 1
    for i in range(filters):
        plt.subplot(n_rows, n_columns, i+1)
        #plt.title(str(i))
        plt.imshow(units[0,:,:,i], interpolation="nearest", cmap="gray")    
        plt.axis('off')
    fig1 = plt.gcf()
    plt.show()
    fig1.savefig(FigureName, dpi=200)


# In[ ]:

Conv3DFM = getActivations(conv3d_3,valid_dataset[3:4,:,:,:,:],valid_labels[3:4])
print(Conv3DFM.shape)
plotNNFilter_3D(Conv3DFM,0,'AN-Layer-3-F0')

plotNNFilter_3D(Conv3DFM,3,'AN-Layer-3-F3')

