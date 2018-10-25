
# coding: utf-8

# In[ ]:

__Lab__ = 'Gipsa-Lab'
__Author__ = 'Dawood AL CHANTI'
__Idea__ = 'For each subject and session, read the whole sequence as Pickle file and save it to disc.'


# ### Import dependencies

# In[ ]:

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pickle
import cv2


# ##### Define the Data path

# In[ ]:

DataPath = 'G:/alchantd/Facial Expression Database Sequence and Video/MMI Facial Data Base/Fixed_Size_Seq/'   


# #### List the content of the directory, 

# In[ ]:

# List the folders inside the directory DataPath, 
# then tranfrom its content from its content from 
# sort its content: 1 2 3 ... 
MainfolderContent = sorted(list(map(int, os.listdir(DataPath))))

# Only consider the first 6 folder that corresponds to the first 6 expressions.
MainfolderContent=MainfolderContent[:6]


# In[ ]:

MainfolderContent


# In[ ]:

def write_data(filename,data):
    with open(filename, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_data(filename): 
    with open(filename, 'rb') as handle:
        loaded_data = pickle.load(handle)
    return loaded_data


# In[ ]:

# Define the Save Path
SavePath ='E:/Puplication/IEEE Affective Computing 2018 Dynamic Emotion Code Offical/MMI_Data/'


# In[ ]:

# Now, for each folder inside MainfolderContent, read its content, convert from RGB to Gray Level and then save the Sequence as pkl file


for session in MainfolderContent:
    sessionlabel = str(session)
    sessionFullPath = DataPath + str(session) + '/'

    # Access the new path and list its content
    SessionContent= sorted(list(map(int, os.listdir(sessionFullPath))))
    print('Session Number: %d is in Process', session)
    for subject in SessionContent:
        subjectFullPath = sessionFullPath + str(subject) + '/'
        SujectName = os.listdir(subjectFullPath)[0]
        
        # Append the label to the Subject Name
        subjectFullPath = subjectFullPath + SujectName + '/'
        Suject_Name_with_Label = SujectName + '_'+ str(session)
  
        #the image index is not well structured alonge time.
        imagesindex = os.listdir(subjectFullPath)
        # Align the correct input sequence over time.
        imagesindex2 = np.arange(1,len(imagesindex)+1)
        del(imagesindex)
        temp_data = map(lambda x: subjectFullPath + str(x)+'.jpg', imagesindex2)
        
        """
        Using H and W we can change the resize aspect ratio
        """
        # Allocate Empty Matrix to load the data: use the Subject name as a matrix
        H=128
        W=128
        sequence =  np.empty((len(imagesindex2),H,W))
        i = 0
        for imagename in temp_data:
            img = cv2.imread(imagename)
            img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
            img = cv2.resize(img, (H,W))
            sequence[i]= img
            i=i+1
        # Save each sequence as pkl file with its name and label and Subject Session: to prevent overwrite
        write_data(SavePath+Suject_Name_with_Label+'_'+str(subject)+'.pkl', sequence) 
        
    print('Processing the next Session')
    print('Done for the label %d', session )
print('Finished')


# In[ ]:

Plot_Figure(img)


# ### Load the Data and get the minimun and the maximum sequence length

# In[ ]:

# list the content of the directory in the save path
conentData = os.listdir(SavePath)


# In[ ]:

print(len(conentData))


# In[ ]:

Sample_1 = load_data(SavePath+ conentData[0])


# In[ ]:

def Plot_Figure(Im):
    fig = plt.imshow(Im,cmap="gray")
    plt.axis('off')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.show()


# In[ ]:

Plot_Figure(Sample_1[0])


# In[ ]:



