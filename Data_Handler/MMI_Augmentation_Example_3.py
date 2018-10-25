
# coding: utf-8

# In[ ]:

__author__ = 'Dawood AL Chanti'


# ### MMI Data Transformation: Data Augmentation

#  <style>
#   .bottom{
#      margin-bottom: 0.1cm;
#   }
# </style>
# 
# 
# <p class="bottom">
#    GIPSA-LAB / DIS
# </p>
# <p class="bottom">
#    Grenoble-INP
# </p>
# <p class="bottom">
#    15-June-2017
# </p>

# In[ ]:

# Import the dependencies
#import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image
from skimage.transform import resize
import pickle


# In[ ]:

def write_data(data,filename):
    with open(filename, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_data(filename): 
    with open(filename, 'rb') as handle:
        loaded_data = pickle.load(handle, encoding='latin1') 
    return loaded_data


# In[ ]:

# Plot a Sample from the Train data
def disp(im):
    '''
    Display a single image
    '''
    fig = plt.imshow(im);
    fig.set_cmap('gray');
    plt.axis('off');
    plt.show()


# In[ ]:

def pad(multiArray, SeqLength,H,W):
    '''
    Pad a sequence of images with zeros based on the maximun length sequence
    '''
    # Create a zero multi array matrix with the desired dimension
    arr = np.zeros((SeqLength,H,W))
    # Copy to it the element of the original Matrix
    arr[:len(multiArray)] = multiArray
    return arr


# In[ ]:

def DisplaySequence(sequence,FigureName='myfigure'):
    '''
    Sequence Display for first few elements staring either from 0 or any define index
    '''
    # Define a figure, their axis and its characteristics: 2 rows , 4 columns, figure size 20 by 10
    fig, ((ax1, ax2,ax3,ax4), (ax11, ax22, ax33, ax44)) = plt.subplots(2,4,figsize=(20, 10))

    # Row 1 images
    ax1.imshow(sequence[1],interpolation="nearest", cmap="gray")    
    ax2.imshow(sequence[2],interpolation="nearest", cmap="gray") 
    ax3.imshow(sequence[3],interpolation="nearest", cmap="gray")    
    ax4.imshow(sequence[4],interpolation="nearest", cmap="gray") 
    # Row 2 images
    ax11.imshow(sequence[10],interpolation="nearest", cmap="gray")    
    ax22.imshow(sequence[11],interpolation="nearest", cmap="gray") 
    ax33.imshow(sequence[12],interpolation="nearest", cmap="gray")    
    ax44.imshow(sequence[13],interpolation="nearest", cmap="gray") 
    
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


# ### Read the Evaluation  Set

# In[ ]:

# Dataset pickles Path : training Set
path1 = 'E:/Puplication/IEEE Affective Computing 2018 Dynamic Emotion Code Offical/MMI_Data/'
listing = os.listdir(path1)    

# Pre allocate the data size to be filled.
# has a shape: Subjects, 
SeqLength=75
H=128
W=128
# Since we already know that the maximum length is 75, we will pad with 0 the rest of empty frames
Data = np.empty((len(listing),SeqLength,H,W))
i=0
# Read the dataset and store it in a matrix
for dataname in listing:
    current = load_data(path1 + dataname)
    Data[i] =current 
    i=i+1
print('>> Done!')


# In[ ]:

DisplaySequence(Data[8],FigureName='TestSample')


# ### Get the label of Validation Set from the file name

# In[ ]:

# Dataset pickles Path : training Set
listing = os.listdir(path1)    
Label = np.zeros(len(listing),dtype=int)
i=0
for dataname in listing:
    '''
    Get the validation label from the file name
    '''
    Label[i] = int(dataname.split('_')[1])
    i+=1


# ### Save the label if preferable as txt file

# In[ ]:

path  = '/home/alchantd/MMI Exp/MMI Pickles/labels/'
np.savetxt(path + 'label.txt',Label,fmt='%d' )


# # Augmentation

# The Idea here, since the data size is large, we will save for each subject sequence session as a pickle file
# it will be normalized up to the original sequence length before padding with zeros, to reach the same final length.

# In[ ]:

from skimage.transform import rotate
from skimage.transform import warp


# #### Data Save Path

# In[ ]:

AugmentationSavePath = '/home/alchantd/MMI Exp/Augmented Pickles/'


# #### Transformation Functions

# In[ ]:

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
            # Since we append with zeros, we should ingnore them during normalization.
            vectormean = vect.mean()
            if vectormean !=0:
                normalized_data[subject][im] = np.divide(np.subtract(vect,vect.mean(axis=0)),vect.std(axis=0))
    return normalized_data.reshape(data_size,sequence_size,image_size_W,image_size_H)


# In[ ]:

# Function that perform Rotation over Images
def Rotate(Data,degree):
    """
    Inpute: Data of 4D Way Tensor: Subjects, Sequence , Image W, Image H
    degree: degree of desired rotation
    Output: Same as Input shape: 4D way tensor same as the input shape 
    """
    Rotate = np.empty(Data.shape)
    for SubjIndx in range(Data.shape[0]):
        for imIndx in range(Data.shape[1]):
            Rotate[SubjIndx][imIndx] = rotate(Data[SubjIndx][imIndx], degree, resize=False, preserve_range= True)
    # Return Normalized Version of the transformed data
    return Rotate #Data_normalization(Rotate)


# In[ ]:

def Translation(Data, ShiftMatrix):
    """
    Inpute: Data of 4D Way Tensor: Subjects, Sequence , Image W, Image H
    ShiftMatrix:  3x3 matrix where it define the axis of translation and the degree of translation.
    Output: Same as Input shape: 4D way tensor same as the input shape  
    """
    Shift = np.empty(Data.shape)
    for SubjIndx in range(Data.shape[0]):
        for imIndx in range(Data.shape[1]):
            Shift[SubjIndx][imIndx] = warp(Data[SubjIndx][imIndx], ShiftMatrix, preserve_range= True)
    # Return Normalized Version of the transformed data
    return Shift #Data_normalization(Shift) 


# #### Start Augmentation

# ### Augmentation over the validation Set

# ##### Rotation pos + 10

# In[ ]:

# ----------------------------------------Positive Rotation----------------------------------------#
RotateP10 = Rotate(Data,+10)
RotateP10 = Data_normalization(RotateP10)
disp(RotateP10[10][5])


# ### Save the Data into a folder :

# In[ ]:

path1 = '/home/alchantd/MMI Exp/MMI Pickles/Test/'
SavePathAug = AugmentationSavePath + 'Test/'
# Do this to get the original file name
listing = os.listdir(path1)    


# In[ ]:

# Save the augmented file in the same formate but add the transformation name
i=0
for dataname in listing:
    write_data(RotateP10[i] ,SavePathAug + dataname.split('_')[0]+ '_'+ dataname.split('_')[1]+ '_'                + 'RotateP10' + '_' +dataname.split('_')[2])
    i+=1
print('>> Done!')    


# In[ ]:

del(RotateP10)


# ##### Rotation pos + 15

# In[ ]:

#--------------------------------------Augment----------------------------------------------------#
RotateP15 = Rotate(Data,+15)
#--------------------------------------Normalize----------------------------------------------------#
RotateP15 = Data_normalization(RotateP15)
#--------------------------------------Display----------------------------------------------------#
disp(RotateP15[10][5])

#--------------------------------------Save to Disc----------------------------------------------------#
i=0
for dataname in listing:
    write_data(RotateP15[i] ,SavePathAug + dataname.split('_')[0]+ '_'+ dataname.split('_')[1]+ '_'                + 'RotateP15' + '_' +dataname.split('_')[2]) 
    i+=1
print('>> Done!')    


# In[ ]:

del(RotateP15)


# #### Rotation + 25

# In[ ]:

#--------------------------------------Augment----------------------------------------------------#
RotateP25 = Rotate(Data,+25)
#--------------------------------------Normalize----------------------------------------------------#
RotateP25 = Data_normalization(RotateP25)
#--------------------------------------Display----------------------------------------------------#
disp(RotateP25[10][5])
i=0
for dataname in listing:
    write_data(RotateP25[i] ,SavePathAug + dataname.split('_')[0]+ '_'+ dataname.split('_')[1]+ '_'                + 'RotateP25' + '_' +dataname.split('_')[2]) 
    i+=1
print('>> Done!')

del(RotateP25)


# #### Rotation with + 35

# In[ ]:

#--------------------------------------Augment----------------------------------------------------#
RotateP35 = Rotate(Data,+35)
#--------------------------------------Normalize----------------------------------------------------#
RotateP35 = Data_normalization(RotateP35)
#--------------------------------------Display----------------------------------------------------#
disp(RotateP35[10][5])
i=0
for dataname in listing:
    write_data(RotateP35[i] ,SavePathAug + dataname.split('_')[0]+ '_'+ dataname.split('_')[1]+ '_'                + 'RotateP35' + '_' +dataname.split('_')[2]) 
    i+=1
print('>> Done!')    


# In[ ]:

del(RotateP35)


# #### Rotation -10

# In[ ]:

#--------------------------------------Augment----------------------------------------------------#
RototateN10 = Rotate(Data,-10)
#--------------------------------------Normalize----------------------------------------------------#
RototateN10 = Data_normalization(RototateN10)
#--------------------------------------Display----------------------------------------------------#
disp(RototateN10[10][5])

i=0
for dataname in listing:
    write_data(RototateN10[i] ,SavePathAug + dataname.split('_')[0]+ '_'+ dataname.split('_')[1]+ '_'                + 'RototateN10' + '_' +dataname.split('_')[2]) 
    i+=1
print('>> Done!')    


# In[ ]:

del(RototateN10)


# #### Rotate -15

# In[ ]:

#--------------------------------------Augment----------------------------------------------------#
RototateN15 = Rotate(Data,-15)
#--------------------------------------Normalize----------------------------------------------------#
RototateN15 = Data_normalization(RototateN15)
#--------------------------------------Display----------------------------------------------------#
disp(RototateN15[10][5])

i=0
for dataname in listing:
    write_data(RototateN15[i] ,SavePathAug + dataname.split('_')[0]+ '_'+ dataname.split('_')[1]+ '_'                + 'RototateN15' + '_' +dataname.split('_')[2]) 
    i+=1
print('>> Done!')  
del(RototateN15)


# #### Rotate -25

# In[ ]:

#--------------------------------------Augment----------------------------------------------------#
RototateN25 = Rotate(Data,-25)
#--------------------------------------Normalize----------------------------------------------------#
RototateN25 = Data_normalization(RototateN25)
#--------------------------------------Display----------------------------------------------------#
disp(RototateN25[10][5])

i=0
for dataname in listing:
    write_data(RototateN25[i] ,SavePathAug + dataname.split('_')[0]+ '_'+ dataname.split('_')[1]+ '_'                + 'RototateN25' + '_' +dataname.split('_')[2]) 
    i+=1
print('>> Done!')    


# In[ ]:

del(RototateN25)


# #### Rotate -35

# In[ ]:

#--------------------------------------Augment----------------------------------------------------#
RototateN35 = Rotate(Data,-35)
#--------------------------------------Normalize----------------------------------------------------#
RototateN35 = Data_normalization(RototateN35)
#--------------------------------------Display----------------------------------------------------#
disp(RototateN35[10][5])

i=0
for dataname in listing:
    write_data(RototateN35[i] ,SavePathAug + dataname.split('_')[0]+ '_'+ dataname.split('_')[1]+ '_'                + 'RototateN35' + '_' +dataname.split('_')[2]) 
    i+=1
print('>> Done!')   

del(RototateN35)


# # Translation Transformation

# In[ ]:

#-----------------------Define the transformation Matrix------------------------------# 
matrixN30 = np.array([[1, 0, 0], [0, 1, -10], [0, 0, 1]])
matrixN40 = np.array([[1, 0, 0], [0, 1, -20], [0, 0, 1]])
matrixN50 = np.array([[1, 0, 0], [0, 1, -30], [0, 0, 1]])
matrixN60 = np.array([[1, 0, 0], [0, 1, -40], [0, 0, 1]])


# #### Shift 1

# In[ ]:

Shift1 = Translation(Data, matrixN30)
Shift1 = Data_normalization(Shift1)
disp(Shift1[10][5])

i=0
for dataname in listing:
    write_data(Shift1[i] ,SavePathAug + dataname.split('_')[0]+ '_'+ dataname.split('_')[1]+ '_'                + 'Shift1' + '_' +dataname.split('_')[2]) 
    i+=1
print('>> Done!')   

del(Shift1)


# #### Shift 2

# In[ ]:

Shift2 = Translation(Data, matrixN40)
Shift2 = Data_normalization(Shift2)
disp(Shift2[10][5])

#-------------------------------------Save to Disc----------------------------------------------------#
i=0
for dataname in listing:
    write_data(Shift2[i] ,SavePathAug + dataname.split('_')[0]+ '_'+ dataname.split('_')[1]+ '_'                + 'Shift2' + '_' +dataname.split('_')[2]) 
    i+=1
print('>> Done!')   

del(Shift2)


# ### Shift 3

# In[ ]:

Shift3 = Translation(Data, matrixN50)
Shift3 = Data_normalization(Shift3)
disp(Shift3[10][5])

i=0
for dataname in listing:
    write_data(Shift3[i] ,SavePathAug + dataname.split('_')[0]+ '_'+ dataname.split('_')[1]+ '_'                + 'Shift3' + '_' +dataname.split('_')[2]) 
    i+=1
print('>> Done!')   

del(Shift3)


# #### Shift 4

# In[ ]:

Shift4 = Translation(Data, matrixN60)
Shift4 = Data_normalization(Shift4)
disp(Shift4[10][5])

i=0
for dataname in listing:
    write_data(Shift4[i] ,SavePathAug + dataname.split('_')[0]+ '_'+ dataname.split('_')[1]+ '_'                + 'Shift4' + '_' +dataname.split('_')[2]) 
    i+=1
print('>> Done!')   

del(Shift4)


# # Vertical Shits

# In[ ]:

#----------------------------- +ve Shift-----------------------------------------------#
matrixP30 = np.array([[1, 0, 0], [0, 1, 10], [0, 0, 1]])
matrixP40 = np.array([[1, 0, 0], [0, 1, 20], [0, 0, 1]])
matrixP50 = np.array([[1, 0, 0], [0, 1, 30], [0, 0, 1]])
matrixP60 = np.array([[1, 0, 0], [0, 1, 40], [0, 0, 1]])


# In[ ]:

Shift5 = Translation(Data, matrixP30)
Shift5 = Data_normalization(Shift5)
disp(Shift5[10][5])
#-------------------------------------Save to Disc----------------------------------------------------#
i=0
for dataname in listing:
    write_data(Shift5[i] ,SavePathAug + dataname.split('_')[0]+ '_'+ dataname.split('_')[1]+ '_'                + 'Shift5' + '_' +dataname.split('_')[2]) 
    i+=1
print('>> Done!')   

del(Shift5)


# #### Shift 6

# In[ ]:

Shift6 = Translation(Data, matrixP40)
Shift6 = Data_normalization(Shift6)
disp(Shift6[10][5])
i=0
for dataname in listing:
    write_data(Shift6[i] ,SavePathAug + dataname.split('_')[0]+ '_'+ dataname.split('_')[1]+ '_'                + 'Shift6' + '_' +dataname.split('_')[2]) 
    i+=1
print('>> Done!')   
del(Shift6)


# #### Shift 7

# In[ ]:

Shift7 = Translation(Data, matrixP50)
Shift7 = Data_normalization(Shift7)
disp(Shift7[10][5])
#-----------------------------
i=0
for dataname in listing:
    write_data(Shift7[i] ,SavePathAug + dataname.split('_')[0]+ '_'+ dataname.split('_')[1]+ '_'                + 'Shift7' + '_' +dataname.split('_')[2]) 
    i+=1
print('>> Done!')   
del(Shift7)


# #### Shift 8

# In[ ]:

Shift8 = Translation(Data, matrixP60)
Shift8 = Data_normalization(Shift8)
disp(Shift8[10][5])
i=0
for dataname in listing:
    write_data(Shift8[i] ,SavePathAug + dataname.split('_')[0]+ '_'+ dataname.split('_')[1]+ '_'                + 'Shift8' + '_' +dataname.split('_')[2]) 
    i+=1
print('>> Done!')   
del(Shift8)


# ### Vertical Translation shifts

# In[ ]:

#----------------------------- Vertical Shift -ve-----------------------------------------------#
matrixVN30 = np.array([[1, 0, -10], [0, 1, 0], [0, 0, 1]])
matrixVN40 = np.array([[1, 0, -20], [0, 1, 0], [0, 0, 1]])
matrixVN50 = np.array([[1, 0, -30], [0, 1, 0], [0, 0, 1]])
matrixVN60 = np.array([[1, 0, -40], [0, 1, 0], [0, 0, 1]])


# In[ ]:

Shift9 = Translation(Data, matrixVN30)
Shift9 = Data_normalization(Shift9)
disp(Shift9[10][5])
#-----------------------------
# Do this to get the original file name
i=0
for dataname in listing:
    write_data(Shift9[i] ,SavePathAug + dataname.split('_')[0]+ '_'+ dataname.split('_')[1]+ '_'                + 'Shift9' + '_' +dataname.split('_')[2]) 
    i+=1
print('>> Done!')   
del(Shift9)


# In[ ]:

Shift10 = Translation(Data, matrixVN40)
Shift10 = Data_normalization(Shift10)
disp(Shift10[10][5])
#-----------------------------
i=0
for dataname in listing:
    write_data(Shift10[i] ,SavePathAug + dataname.split('_')[0]+ '_'+ dataname.split('_')[1]+ '_'                + 'Shift10' + '_' +dataname.split('_')[2]) 
    i+=1
print('>> Done!')   
del(Shift10)


# In[ ]:

Shift11 = Translation(Data, matrixVN50)
Shift11 = Data_normalization(Shift11)
disp(Shift11[10][5])
#-----------------------------
i=0
for dataname in listing:
    write_data(Shift11[i] ,SavePathAug + dataname.split('_')[0]+ '_'+ dataname.split('_')[1]+ '_'                + 'Shift11' + '_' +dataname.split('_')[2]) 
    i+=1
print('>> Done!')   
del(Shift11)


# In[ ]:

Shift12 = Translation(Data, matrixVN60)
Shift12 = Data_normalization(Shift12)
disp(Shift12[10][5])
#-----------------------------
i=0
for dataname in listing:
    write_data(Shift12[i] ,SavePathAug + dataname.split('_')[0]+ '_'+ dataname.split('_')[1]+ '_'                + 'Shift12' + '_' +dataname.split('_')[2]) 
    i+=1
print('>> Done!')   
del(Shift12)


# ### Vertical Translation in +ve Direction

# In[ ]:

matrixVP30 = np.array([[1, 0,10], [0, 1, 0], [0, 0, 1]])
matrixVP40 = np.array([[1, 0,20], [0, 1, 0], [0, 0, 1]])
matrixVP50 = np.array([[1, 0,30], [0, 1, 0], [0, 0, 1]])
matrixVP60 = np.array([[1, 0,40], [0, 1, 0], [0, 0, 1]])


# In[ ]:

Shift13 = Translation(Data, matrixVP30)
Shift13 = Data_normalization(Shift13)
disp(Shift13[10][5])
#-----------------------------
i=0
for dataname in listing:
    write_data(Shift13[i] ,SavePathAug + dataname.split('_')[0]+ '_'+ dataname.split('_')[1]+ '_'                + 'Shift13' + '_' +dataname.split('_')[2]) 
    i+=1
print('>> Done!')   
del(Shift13)


# In[ ]:

Shift14 = Translation(Data, matrixVP40)
Shift14 = Data_normalization(Shift14)
disp(Shift14[10][5])
#-----------------------------
i=0
for dataname in listing:
    write_data(Shift14[i] ,SavePathAug + dataname.split('_')[0]+ '_'+ dataname.split('_')[1]+ '_'                + 'Shift14' + '_' +dataname.split('_')[2]) 
    i+=1
print('>> Done!')   
del(Shift14)


# In[ ]:

Shift15 = Translation(Data, matrixVP50)
Shift15 = Data_normalization(Shift15)
disp(Shift15[10][5])
#-----------------------------
i=0
for dataname in listing:
    write_data(Shift15[i] ,SavePathAug + dataname.split('_')[0]+ '_'+ dataname.split('_')[1]+ '_'                + 'Shift15' + '_' +dataname.split('_')[2]) 
    i+=1
print('>> Done!')   
del(Shift15)


# In[ ]:

Shift16 = Translation(Data, matrixVP60)
Shift16 = Data_normalization(Shift16)
disp(Shift16[10][5])
#-----------------------------
i=0
for dataname in listing:
    write_data(Shift16[i] ,SavePathAug + dataname.split('_')[0]+ '_'+ dataname.split('_')[1]+ '_'                + 'Shift16' + '_' +dataname.split('_')[2]) 
    i+=1
print('>> Done!')   
del(Shift16)


# # Data flip

# In[ ]:

# ------------------------------Flip 180,90 and -90 ----------------#
Rotate180 = Rotate(Data,180)
Rotate180 = Data_normalization(Rotate180)
disp(Rotate180[10][5])
i=0
for dataname in listing:
    write_data(Rotate180[i] ,SavePathAug + dataname.split('_')[0]+ '_'+ dataname.split('_')[1]+ '_'                + 'Rotate180' + '_' +dataname.split('_')[2]) 
    i+=1
print('>> Done!')   
del(Rotate180)


# In[ ]:

# ------------------------------Flip 180,90 and -90 ----------------#
RotateN90 = Rotate(Data,-90)
RotateN90 = Data_normalization(RotateN90)
disp(RotateN90[10][5])

i=0
for dataname in listing:
    write_data(RotateN90[i] ,SavePathAug + dataname.split('_')[0]+ '_'+ dataname.split('_')[1]+ '_'                + 'RotateN90' + '_' +dataname.split('_')[2]) 
    i+=1
print('>> Done!')   
del(RotateN90)


# In[ ]:

# ------------------------------Flip 180,90 and -90 ----------------#
RotateP90 = Rotate(Data,+90)
RotateP90 = Data_normalization(RotateP90)
disp(RotateP90[10][5])
i=0
for dataname in listing:
    write_data(RotateP90[i] ,SavePathAug + dataname.split('_')[0]+ '_'+ dataname.split('_')[1]+ '_'                + 'RotateP90' + '_' +dataname.split('_')[2]) 
    i+=1
print('>> Done!')   
del(RotateP90)


# # Normalize the Original Data and save it

# In[ ]:

ValDataNormalized = Data_normalization(Data)
disp(ValDataNormalized[10][5])

i=0
for dataname in listing:
    write_data(ValDataNormalized[i] ,SavePathAug + dataname.split('_')[0]+ '_'+ dataname.split('_')[1]+ '_'                + 'Normalized' + '_' +dataname.split('_')[2]) 
    i+=1
print('>> Done!')   
del(ValDataNormalized)

