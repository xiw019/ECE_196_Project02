#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import cv2
import os
import csv
import glob
import pandas as pd
from PIL import Image
from datetime import date, datetime


# In[96]:


#read in the names from a csv
def read(path):
    userList = []
    with open(path, newline='') as f:
        reader = csv.reader(f)
        names = list(reader)
        for i in range(len(names)):
            userList.append(names[i][0])
#         userList = list(reader)
    return userList


# In[118]:


#write name to the csv file
def write(name):
    path = 'names.csv'
    new_name = []
    new_name.append(name)
    with open(path, 'a') as file:
        writer = csv.writer(file)
        writer.writerow(new_name)


# In[98]:


#check if user face already exist
#return true if exist
#return false and creat new user's temperature file
def check(name, userList):
    if name in userList:
        return True
    else:
        userList.append(name)
        write(name) #write to csv
        f= open('Temperature Recordings/' + name + '.txt', "w+")
        f.close()
        return False


# In[99]:


# open the camera and capture the face
# store the captured face in the dataset file
def capture(userList):
    #get the index of the user in the user list
    userID = 0; 
    if len(userList) == 0:
        userID = len(userList) 
    else:
        userID = len(userList) - 1
    #initialize and set camera
    cam = cv2.VideoCapture(0)
    cam.set(3, 640) # set video width
    cam.set(4, 480) # set video height

    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')    
    count = 0 #count the number of captured image
    while(True):
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)
        #show the face area
        for (x,y,w,h) in faces:
            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
            count += 1
            # Save the captured image into the datasets folder
            cv2.imwrite("dataset/User." + str(userID) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
            cv2.imshow('image', img)
        k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
        if k == 27:
            break
        elif count >= 30: # Take 30 face sample and stop video
            break


# In[100]:


# train the collected faces
def train():
    path = 'dataset' #path to the training dataset
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");
    # function to get the images and label data
    def getImagesAndLabels(path):
        imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
        faceSamples=[]
        ids = []
        for imagePath in imagePaths:
            PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
            img_numpy = np.array(PIL_img,'uint8')
            id = int(os.path.split(imagePath)[-1].split(".")[1])
            faces = detector.detectMultiScale(img_numpy)
            for (x,y,w,h) in faces:
                faceSamples.append(img_numpy[y:y+h,x:x+w])
                ids.append(id)
        return faceSamples,ids
    
    faces,ids = getImagesAndLabels(path)
    recognizer.train(faces, np.array(ids))
    # Save the model into trainer/trainer.yml
    recognizer.write('trainer/trainer.yml') 


# In[101]:


# clean all the data 
def clean():
    face_files = glob.glob('dataset/*.jpg')
    temp_files = glob.glob('Temperature Recordings/*.txt')
    #remove faces data
    for face in face_files:
        os.remove(face)
    #remove temperature data
    for temp in temp_files:
        os.remove(temp)
    #remove names data
    f = open('names.csv', "w+")
    f.truncate()
    f.close()     


# In[129]:





# In[ ]:




