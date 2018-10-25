
# coding: utf-8

# In[1]:

__author__ = 'Dawood AL Chanti'
__Lab__ = 'GIPSA_Lab'
__idea__ = 'Online Method during training to load data from Disc randomly with replacment over a predefined Batch Size.'


# # Read Data Randomly from Disc Based on Batch Size

# In[2]:

import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image
import pickle
import random
import time


# In[19]:

# Should be ok for Python 3.5.2 64-bit otherwise, you may have problem with load data protocol, if so, use protocol 2 for python 2.7 version
import sys
print(sys.version)


# In[3]:

def write_data(filename,data):
    with open(filename, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_data(filename): 
    with open(filename, 'rb') as handle:
        loaded_data = pickle.load(handle, encoding='latin1') 
    return loaded_data

# In[4]:

# Plot a Sample from the Train data
def disp(im):
    '''
    Display a single image
    '''
    fig = plt.imshow(im);
    fig.set_cmap('gray');
    plt.axis('off');
    plt.show()


# In[5]:

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
    fig.savefig(FigureName)


# In[6]:

Batch_Size= 10
SeqLength=75
H=128
W=128
DataPath = 'E:/Puplication/IEEE Affective Computing 2018 Dynamic Emotion Code Offical/MMI_Data/'


# In[11]:

def GetData_From_Disc(Batch_Size,DataPath):
    listing = os.listdir(DataPath)  
    # Get the size of the original data inside the path
    DataSize  = len(listing)
    
    # Generate Batch_Size Indecies out of the original data size DataSize.
    for i in range(10):
        Sample_Indecies = random.sample(range(0, DataSize), Batch_Size)
    # Get their name and Full Path
    DataBatch_Path = map(lambda x: DataPath + listing[x] , Sample_Indecies)
    DataBatch_Path = [x for x in DataBatch_Path]
    
    Data = map(lambda x: load_data(x) , DataBatch_Path)
    Data = [x for x in Data]
    Data = np.array(Data)
    label = map(lambda x: int(x.split('/')[-1].split('_')[1]) , DataBatch_Path)
    label = [x for x in label]
    label = np.array(label)
    return Data, label


# ## Example of load a batch of data and displaying its sequence. To make sure every thing is ok

# In[12]:

start_time = time.time()
trainBatch, Batchlabel = GetData_From_Disc(10,DataPath)
elapsed_time = time.time() - start_time
print('Elapsed Time: ', elapsed_time)
DisplaySequence(trainBatch[0])


# In[13]:

trainBatch.shape


# In[14]:

Batchlabel[4]


# In[18]:

DisplaySequence(trainBatch[4],'FigureName_4')


# In[ ]:



