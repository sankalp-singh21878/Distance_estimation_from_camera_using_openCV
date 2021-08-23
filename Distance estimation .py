#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from imutils import paths
import numpy as np
import imutils
import cv2


# In[ ]:


known_distance = 76.2
known_width = 14.3

GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

fonts = cv2.FONT_HERSHEY_COMPLEX


# In[ ]:


#Face detector object
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


# In[ ]:


def focal_distance(knownWidth, focalLength, perWidth):
    return (knownWidth * focalLength) / perWidth


# In[ ]:


def distance_finder(focal_length, real_face_width, face_width_in_frame):
    return (real_face_width * focal_length) / face_width_in_frame


# In[ ]:


def face_data(image):
    face_width = 0
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, h, w) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), GREEN, 2)
        face_width = w
        
    return face_width


# In[ ]:


ref_image = cv2.imread("ref_pic.jpg")

ref_image_face_width = face_data(ref_image)

focal_length = focal_distance(known_distance, known_width, ref_image_face_width)

cv2.imshow("ref_image", ref_image)
cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    face_width_in_frame = face_data(frame)
    
    if face_width_in_frame != 0:
        distance = distance_finder(focal_length, known_width, face_width_in_frame)
        
        cv2.line(frame, (30, 30), (230, 30), RED, 32)
        cv2.line(frame, (30, 30), (230, 30), BLACK, 28)
        
        cv2.putText(frame, f"Distance: {round(distance, 2)} CM", (30, 35), fonts, 0.6, GREEN, 2)
        
        cv2.imshow("frame", frame)
        
        if cv2.waitKey(1) == ord("q"):
            break
cap.release()
cv2.destroyAllWindows()

