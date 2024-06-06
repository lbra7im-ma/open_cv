import cv2 as cv 
import numpy as np 


img = cv.imread('opencv_chapters/photos/ibra.jpg')
cv.imshow('img', img)

print(img.shape)
#================================================================

resized_img = cv.resize(img, (400,400))
cv.imshow('resized image', resized_img)

print(resized_img.shape)

################################################################################################
cropped_img = img[0:100,100:150]
cv.imshow('cropped image', cropped_img)


cv.waitKey(0)