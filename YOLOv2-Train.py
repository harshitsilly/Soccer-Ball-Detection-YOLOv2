
# coding: utf-8

# # Introduction
# 
# This notebook shows how object detection can be done on your own dataset by training YOLOv2. I am going to use soccer playing images as training dataset as an example to detect soccer ball. After finishing this notebook, you will be able to train your own model, and detect objects that you are interested in.
# 
# It is really your choice which implementation to use. I happened to choose [darkflow](https://github.com/thtrieu/darkflow) which has been quite popular for Tensorflow support on Github open source hosting service. However, you can build the YOLO model on your own since the [official web-site](https://pjreddie.com/darknet/yolo/) provides its pre-trained weights as an external file.
# 
# Here is another choice of implementation called, [tensornets](https://github.com/taehoonlee/tensornets). Please check it out.

# ## Preparing the Data

# __1. Search for your interest__
# 
# __2. Collect media files for training (image, video)__
#   - Be aware of what your **problem domain** is
#     - For example, the wide camera angle shot for a soccer play, the soccer ball is very tiny. However, the ball size gets pretty bigger with the close-up camera shot. These two different types of photo mean the different domains. 
#     - In my case, I **collected pictures of only close-up camera shot**.
#     - Also, the collected pictures are **soley from 2018 Russia World Cup games**.
# 
# __3. Collect media files for testing (image, video)__
#   - It is a common sense not to use the same data from the training process for testing purpose. 
#   - I just chose some random pictures and a video from internet to check if the trained algorithm performs well with un-seen dataset
# 
# __4. If your media file is video, make a lot of screenshot__
#   - Here is how to take a bulk screenshot [LINK](https://pibby.livejournal.com/164326.html)
#   
# __5. Make annotations with drawing rectangles__
#   <div stlye="text-align:center;">
#       <img src="https://raw.githubusercontent.com/tzutalin/labelImg/master/demo/demo3.jpg"/ style="width:500px;">
#   </div>
#   
#   - Here is a tool to do it [LINK](https://github.com/tzutalin/labelImg)

# ## Downloading my dataset
# 
# I have collected dataset from several youtube videos of highlights in 2018 Russia World Cup. Here is a breif steps how I have done.
# 1. **downloaded several videos** of highlights in 2018 Russia World Cup from Youtube.
# 2. produced a bunch of **bulk screenshots** using VLC player.
# 3. **kept images** that I am **only interested** in.
# 4. used [LabelImg](https://github.com/tzutalin/labelImg) software to **make annotations** in each image.
# 5. **located images in './images/' directory and annotations in './annotations/' directory**.
# 
# Below is the link of my work. If you want to train on your own machine, please download **'soccer_ball_data.zip'** file and unzip it. You will see three separate folders. **Two of them(images, annotations)** are explained above, and **sample_video folder** contains a sample_video file to verify the result.
# 
# https://drive.google.com/drive/folders/1th1odcC5kjEnkZmwooETYlyLsz5AOTph?usp=sharing

# ## Importing Dependencies

# In[14]:


# %matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

from darkflow.net.build import TFNet
import cv2


# ## Making Your Own Configuration File (cfg)
# 
# You need to make a slight change on the configuration file (\*.cfg) to adjust existing model to fit into your own dataset. For example, YOLOv2's CNN model is trained on ImageNet dataset which contains a number of different object classes. However, I only want **one (soccer ball)** object class to detect.
# 
# You should do the following to make that change according to Darflow's official explanation.
# 
# 1. __Copy__ one of the pre-trained configuration file that you are going to use.
#   - In my case, I used yolo.cfg.
#   
# 2. __Rename__ the copied configuration file to your custom name
#   - In my case, I changed the name yolo.cfg to yolo_custom.cfg.
#   
# 3. Open the configuration file and __change classes in the \[region\] layer__ (the last layer) to the number of classes you are going to train for. 
#   - In my case, classes are set to 1.
#   
# 4. __change filters in the \[convolutional\] layer__ (the second to last layer) to num * (classes + 5). 
#   - In my case, num is 5 and classes are 1 so 5 * (1 + 5) = 30 therefore filters are set to 30.

# ## Specifying Classes of Your Interest
# 
# You also need to changes __'labels.txt'__ file included in the root directory. Just specify names of objects. In my case, I removed all text and left __'ball'__ instead.

# ## Downloading Pre-trained Weights
# 
# Please go [here](https://drive.google.com/drive/folders/0B1tW_VtY7onidEwyQ2FtQVplWEU) and **download a pre-trained weights** that you are interested. In my case, I downloaded **yolo.weights**. After downloading is done, **locate the file into 'bin'** directory in the root directory.
# 
# <div style="text-align:center;">
#     <img src="https://raw.githubusercontent.com/deep-diver/Object-Detection-YOLOv2-Darkflow/master/weights.PNG"/>
# </div>

# ## Defining Model Options & Build the Model
# 
# Before building the model, you need to define options. 
# 
# __1. model__: **configuration** file (\*.cfg) that you have defined for your situation
# 
# __2. load__: **pre-trained weight** file
# 
# __3. batch__: a number of data to train per a batch
#   - I set batch size 8 because my GPU card couldn't handle bigger size. My GPU card is **NVIDIA GTX 1080Ti**.
# 
# __4. epoch__: how many iterations to train
#   - I probably need to set this a bit smaller, but as the result shows I think 100 epochs are just fine
# 
# __5. gpu__: set **1.0** if you want to fully utilize your GPU hardware. Otherwise, remove this option.
# 
# __6. train__: set **'True'** if your purpose is training
# 
# __7. annotation__: **directory** where the annotation files are stored
# 
# __8. dataset__: **directory** where the image files are stored

# In[2]:


options = {"model": "cfg/yolo_custom.cfg", 
           "load": "yolo.weights",
           "batch": 8,
           "epoch": 100,
           "gpu": 0,
           "train": True,
           "annotation": "./images/images/ann2/",
           "dataset": "./images/images/img/"}


# In[3]:


tfnet = TFNet(options)


# ## Training the Model
# 
# Training took about **an hour** on NVIDIA GTX 1080Ti

# In[4]:


tfnet.train()


# In[6]:


# this line of code lets you save the built graph to a protobuf file (.pb)
# this step is unnecessary for this notebook
tfnet.savepb()


# ## Loading from the Latest Checkpoint
# 
# This cell shows how to load custom trained model from the checkpoint. It will easy your work.
# 
# __1. model__: the model configuration that you used (\*.cfg)
# 
# __2. load__: set -1. In this way, the latest checkpoint will be loaded.

# In[8]:


options = {"model": "cfg/yolo_custom.cfg",
           "load": -1,
           "gpu": 0}


# In[9]:


tfnet2 = TFNet(options)


# ## load_from_ckpt()
# 
# By calling load_from_ckpt() method in TFNet, it will load pre-trained parameters from the checkpoint that you just specified in options.

# In[10]:


tfnet2.load_from_ckpt()


# ## Predicting on a Image
# 
# This part is almost identical to my previous notebook [Object-Detection-YOLOv2-Darkflow](https://github.com/deep-diver/Object-Detection-YOLOv2-Darkflow). If you wonder some process, please check it out. I have also written some more information about the basic in [medium](https://towardsdatascience.com/yolov2-object-detection-using-darkflow-83db6aa5cf5f), so please check it out.

# In[12]:


import pprint as pp

original_img = cv2.imread("custom_dataset/maxresdefault.jpg")
original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
results = tfnet2.return_predict(original_img)
print(results)


# In[15]:


fig, ax = plt.subplots(figsize=(15, 15))
ax.imshow(original_img)


# In[16]:


def boxing(original_img , predictions):
    newImage = np.copy(original_img)

    for result in predictions:
        top_x = result['topleft']['x']
        top_y = result['topleft']['y']

        btm_x = result['bottomright']['x']
        btm_y = result['bottomright']['y']

        confidence = result['confidence']
        label = result['label'] + " " + str(round(confidence, 3))
        
        if confidence > 0.3:
            newImage = cv2.rectangle(newImage, (top_x, top_y), (btm_x, btm_y), (255,0,0), 3)
            newImage = cv2.putText(newImage, label, (top_x, top_y-5), cv2.FONT_HERSHEY_COMPLEX_SMALL , 0.8, (0, 230, 0), 1, cv2.LINE_AA)
        
    return newImage


# In[17]:


fig, ax = plt.subplots(figsize=(20, 10))
ax.imshow(boxing(original_img, results))


# In[18]:


from math import ceil

fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(20, 10))

for i in range(5):
    original_img = cv2.imread("sample_img/test_image" + str(i+1) + ".jpg")
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    results = tfnet2.return_predict(original_img)
    
    ax[ceil(i/3)-1, i%3].imshow(boxing(original_img, results))


# ## Predicting on a Video

# In[25]:


import time



cap = cv2.VideoCapture('./custom_dataset/Automated Number Plate Recognition (ANPR).mp4')
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) 
length = int(cap.get(cv2.CAP_PROP_FPS))
print( length )


# fourcc = cv2.VideoWriter_fourcc(*'DIVX')
# out = cv2.VideoWriter('./custom_dataset/1.mp4',fourcc, 20.0, (int(width), int(height)))

t0 = time.time()
ret, frame = cap.read()
frame = np.asarray(frame)      
results = tfnet2.return_predict(frame)

t1 = time.time()

total = t1-t0
print(total)
# while(True):
#     # Capture frame-by-frame
#     ret, frame = cap.read()
    
#     if ret == True:
#         frame = np.asarray(frame)      
#         results = tfnet2.return_predict(frame)
        
#         new_frame = boxing(frame, results)

#         # Display the resulting frame
#         out.write(new_frame)
# #         cv2.imshow('frame', new_frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     else:
#         break

# # When everything done, release the capture
# cap.release()
# out.release()
# cv2.destroyAllWindows()


# In[52]:


from IPython.display import YouTubeVideo
YouTubeVideo('1MwIVcni0P4')

