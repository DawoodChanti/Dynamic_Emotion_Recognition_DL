
# coding: utf-8

# In[1]:

__author__ = 'Dawood Al Chanti'


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
import csv
import random
from tensorflow.contrib.layers.python import layers as tf_layers
import tensorflow.contrib.slim as sli
from tensorflow.contrib.layers.python.layers import initializers
get_ipython().magic('matplotlib inline')


# In[3]:

# In[4]:

#------------------------Data ReFormate to have the 5D Tensor Shape-------------------------------------------------#
def reformat(dataset,labels,num_labels):
    """
    Reshape Input into 5 way D Tensor: Sub Index, Seq Index, W,H,Channels
    Reshape the Label into 1-Hot Encoding vectors
    """
    Sub,Seq,W,H =dataset.shape
    dataset = dataset.reshape((Sub,Seq,W, H, 1)).astype(np.float32)
    labels = hotoneEncoder(labels,num_labels) #(np.arange(num_labels) == labels[:,None]).astype(np.float32)
    return dataset, labels

# Plot a Sample from the Train data
def disp(im):
    fig = plt.imshow(im);
    fig.set_cmap('gray');
    plt.axis('off');
    plt.show()

#------------------------------------- One hot Encoding and Its Inverse------------------#    
from sklearn import preprocessing
## 000 is not accepted since we deal with probabilities, there shoud be 001 or 010 or 100 and not 000.
def hotoneEncoder(label, num_labels):
    lb = preprocessing.LabelBinarizer()
    classtofit = list(range(1,num_labels+1))
    lb.fit(classtofit) # Class numbers
    hot_one_encoded = lb.transform(label)
    return hot_one_encoded

def Inverse_hot_one_encoder(hot_one_encoded_label,num_labels):
    lb = preprocessing.LabelBinarizer()
    classtofit = list(range(1,num_labels+1))
    lb.fit(classtofit) # Class numbers
    labelasint = lb.inverse_transform(hot_one_encoded_label)
    return labelasint

#--------------------------- Data Shuffling and Randomization-------------------------------------------------------#
def randomize(dataset, labels):
    """
    Randomize the Dataset Both Data and Label keeping their original structure good.
    """
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels


# Normalize to have 0 mean and 1 variance
def Data_normalization(data):
    """
    Input: Data : 4-way Tensor : Subject index, Image Index, Image size W, Image size H.
    Output: Same shape as input, nomralized data with 0 mean and std = 1
    """
    data_size = data.shape[0]
    sequence_size = data.shape[1]
    image_size_W = data.shape[2]
    image_size_H = data.shape[3]
    # pre allocate memory for the data normalized
    normalized_data = np.empty([data_size,sequence_size,image_size_W*image_size_H])
    for subject in range(data_size):
        for im in range(sequence_size):
            vect = data[subject][im].ravel()
            normalized_data[subject][im] = np.divide(np.subtract(vect,vect.mean(axis=0)),vect.std(axis=0))
    return normalized_data.reshape(data_size,sequence_size,image_size_W,image_size_H)

def xrange(x):
    return iter(range(x))

def randomizeFV(dataset, labels):
    permutation = np.random.permutation(dataset.shape[0])
    shuffled_dataset = dataset[permutation]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels


# In[5]:

def getActivations(layer,stimuli,label_stimuli):
    init_o = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_o) 
        saver.restore(sess,file_path)
        print("Model restored.")
        print("Getting Activation.")
        activation_units = sess.run(layer,feed_dict={x: stimuli,
                                                y_: label_stimuli, 
                                                keep_prob_in: 1.0,
                                                keep_prob_conv: 1.0,
                                                keep_prob_lstm: 1.0,
                                                keep_prob_out: 1.0,
                                                keep_prob_mlp: 1.0,
                                                alpha: 0.0001,
                                                isTrain: False})
    print('Done')
    return activation_units
#--------------------------------------------------------------------#
def plotNNFilter_3D(units,frameToVisualize):
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
    plt.show()
    
def plotNNFilter_2D(units):
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
    plt.show()
  


# In[ ]:

from sklearn.metrics import classification_report, confusion_matrix,  roc_curve, auc
import pandas as pd
import seaborn as sns
from scipy import interp
def get_confusion_matrix(trulabel, predictedlabel,name):
    #use skilearn to compute the confusion matrix
    cm = confusion_matrix(trulabel,predictedlabel)
    
    # Normalize the heatMap
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
    # Assign a new index name for each column and transform numpy to pandas table
    cm= pd.DataFrame(cm, columns=['AN', 'DI', 'FE', 'HA','SA','SU'] ) # ['AF', 'AM', 'CU', 'DI','FR','SAT']
    # Assign new index name for each row
    cm.index =  ['AN', 'DI', 'FE', 'HA','SA','SU'] #['AN', 'DI', 'FE', 'HA','SA','SU']

    # Now USe Seaborn to plot the heatmap  
    ax = sns.heatmap(cm, annot=True, fmt=".2f", vmin=0, vmax=1) #cm.max().max()
    ax.get_figure().savefig('Confusion_Matrix'+name,dpi=200)
    
def FullMetrics(setname, trueLabel, predictedlabel,name):
    
    #-- Compute the Confusion Matrix --#
    print('Confusion matrix over : ' + setname)
    get_confusion_matrix(trueLabel,predictedlabel,name)
    
    print('Classification Report over the: ' + setname)
    print(classification_report(trueLabel, predictedlabel))


# In[ ]:

# Compute macro-average ROC curve and ROC area Multi Class

# For nice color Plot
def ROC_Curve_Multi_Class(y_test, y_scores,name):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    num_classes = y_test.shape[1]
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_scores.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= num_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()

    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)
    lw = 2
    labelindex = ['AN', 'DI', 'FE', 'HA','SA','SU']
    #labelindex = ['AF', 'AM', 'CU', 'DI','FR','SAT']
    colors = np.array(sns.color_palette("hls",num_classes))
    for i, color in zip(range(num_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(labelindex[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Multi-Classes')

    from matplotlib.font_manager import FontProperties
    fontP = FontProperties()
    fontP.set_size('small')
    lgd = plt.legend(bbox_to_anchor=(0.0, -0.2), loc='upper left',ncol=2,  borderaxespad=0 ,prop = fontP)
    #plt.legend(bbox_to_anchor=(0, -0.15, 1, 0), loc=2, ncol=2, ,,prop = fontP)
    #plt.show()
    #plt.savefig('ROC'+name) 
    fig1 = plt.gcf()
    plt.show()
    fig1.savefig('ROC'+name, bbox_extra_artists=(lgd,),bbox_inches='tight', dpi=200)
    


# In[ ]:

# Compute ROC curve and ROC area for Single class
def Single_ROC(y_test, y_scores, classIndex):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_scores.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    plt.figure()
    lw = 2

    print('ROC Curve for Class : ' + str(classIndex))

    plt.plot(fpr[classIndex], tpr[classIndex], color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic for single class')
    plt.legend(loc="lower right")
    plt.show()


# In[ ]:

# find the indecies of a label
def IndexOfLabel(labelset,labelindexSearchingfor):
    indexes = [i for i,x in enumerate(labelset.tolist()) if x == labelindexSearchingfor]
    return indexes

# delete the matrices corresponds to indecies
def DeleteMatrices(multiarray, indexes):
    masks = np.delete(multiarray, indexes,0)
    return masks

# Adjust the labels for example after deleting such a label it becomes 1 3 4 .. so me make it again 1 2 3 instead.
def labeladjustment(newlabel):
    adjustedLabel= np.zeros_like(newlabel)
    for i in range(newlabel.shape[0]):
        if newlabel[i]==1:
            adjustedLabel[i]=newlabel[i]
        else:
            adjustedLabel[i]=newlabel[i]-1
    return adjustedLabel


# In[ ]:

def loaddat(readpath):
    with open(readpath, 'rb') as handle:
        dat= pickle.load(handle,encoding='latin1')
    return dat

def savedat(datatosave,savpath):
    with open(savpath, 'wb') as handle:
        dat_toSave= pickle.dump(datatosave, handle, protocol=pickle.HIGHEST_PROTOCOL)


# In[ ]:

def GetData_From_Disc(Batch_Size,DataPath):
    listing = os.listdir(DataPath)  
    # Get the size of the original data inside the path
    DataSize  = len(listing)
    # Generate Batch_Size Indecies out of the original data size DataSize.
    for i in range(50):
        Sample_Indecies = random.sample(range(0, DataSize), Batch_Size)
    # Get their name and Full Path
    DataBatch_Path = map(lambda x: DataPath + listing[x] , Sample_Indecies)
    DataBatch_Path = [x for x in DataBatch_Path]

    Data = map(lambda x: loaddat(x) , DataBatch_Path)
    Data = [x for x in Data]
    Data = np.array(Data)
    
    label = map(lambda x: int(x.split('/')[-1].split('_')[1]) , DataBatch_Path)
    label = [x for x in label]
    label = np.array(label)
    return Data , label

def loadBatchData(batchpath,batchsize):
    '''
    Supply the path and the size of the desired data to load from disc.
    Return the data and the label, in 5D Tensor formate and binazrized labels
    '''
    batch_dataset,batch_labels = GetData_From_Disc(batchsize,batchpath)
    #print('>> Loading Done!')
    num_classes = 6
    batch_dataset, batch_labels = reformat(batch_dataset, batch_labels,num_classes)
    #print('>> Reformate into 5D tensor and Binarize the label Done!')
    return batch_dataset, batch_labels


# In[ ]:

def DisplaySequence(sequence,FigureName='myfigure'):
    '''
    Sequence Display for first few elements staring either from 0 or any define index
    '''
    # Define a figure, their axis and its characteristics: 2 rows , 4 columns, figure size 20 by 10
    fig, ((ax1, ax2,ax3,ax4), (ax11, ax22, ax33, ax44)) = plt.subplots(2,4,figsize=(20, 10))

    # Row 1 images
    ax1.imshow(sequence[10],interpolation="nearest", cmap="gray")    
    ax2.imshow(sequence[11],interpolation="nearest", cmap="gray") 
    ax3.imshow(sequence[12],interpolation="nearest", cmap="gray")    
    ax4.imshow(sequence[13],interpolation="nearest", cmap="gray") 
    # Row 2 images
    ax11.imshow(sequence[14],interpolation="nearest", cmap="gray")    
    ax22.imshow(sequence[15],interpolation="nearest", cmap="gray") 
    ax33.imshow(sequence[16],interpolation="nearest", cmap="gray")    
    ax44.imshow(sequence[17],interpolation="nearest", cmap="gray") 
    
    # Hide "spines" on axis
    ax1.set_axis_off()  
    ax2.set_axis_off() 
    ax3.set_axis_off() 
    ax4.set_axis_off() 
    ax11.set_axis_off()  
    ax22.set_axis_off() 
    ax33.set_axis_off() 
    ax44.set_axis_off() 
    
    # provide a tight display without big space
    fig.tight_layout()
    #wspace = 0.4   # the amount of width reserved for blank space between subplots
    #hspace = 0.5   # the amount of height reserved for white space between subplots
    #fig.subplots_adjust(hspace)

    plt.show()
    #fig.savefig(FigureName)


# In[ ]:
### Read CSV file as dictionary
def ReadModelSummary(CSVName):
    dict = {}
    for key, val in csv.reader(open(CSVName)):
        dict[key] = val
    return dict


# In[ ]:



