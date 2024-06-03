import cv2 as cv 

import numpy as np

img = cv.imread('opencv_chapters/photos/ibra.png')

img_hori = np.hstack((img,img))
#img_ver =  np.vstack((img,img))



cv.imshow('image',img_hori)
#cv.imshow('image',img_ver)

cv.waitKey(0)