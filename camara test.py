#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2


# In[ ]:


cap = cv2.VideoCapture(0)
 
while(True):
    ret, frame = cap.read()
#     frame = cv2.flip(frame, -1)    
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# In[ ]:




