import cv2 as cv

import numpy as np


img = np.zeros((512,512,3),np.uint8)

#print(img)
#img [200:300,200:300] = 255,0,0

cv.line(img,(0,0),(512,512),(0,255,2),3)
cv.rectangle(img,(200,200),(400,400),(0,0,255),2)
cv.circle(img,(200,200),50,(255,255,255),2)
cv.putText(img,"ibrahim abdelgwaad", (100,100),cv.FONT_HERSHEY_COMPLEX,1,(255,0,200),1)
cv.line(img,(110,110),(450,110),(250,0,250),1)
cv.imshow('image', img)

cv.waitKey(0)