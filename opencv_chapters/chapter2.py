import cv2 as cv
import numpy as np

kernel = np.ones((5,5),np.uint8)

img = cv.imread('opencv_chapters/photos/ibra.jpg')
cv.imshow('image', img)
#========================================================================================
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('gray', gray)
#========================================================================================
blur = cv.GaussianBlur(gray, (7,7), 0)
cv.imshow('blur', blur)
#========================================================================================
canny = cv.Canny(img,100,100)
cv.imshow('canny wedges', canny)
#========================================================================================
dailation = cv.dilate(canny, kernel, iterations=2)
cv.imshow('dailate', dailation)
#========================================================================================
erode = cv.erode(dailation,kernel,iterations=2)
cv.imshow('eroded', erode)
cv.waitKey(0)