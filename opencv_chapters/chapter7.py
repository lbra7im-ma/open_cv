import cv2 as cv

import numpy as np

def empty(a):
    pass
cv.namedWindow("trackbars")
cv.resizeWindow("trackbars",640,240)


cv.createTrackbar("hue min ","trackbars",0,179,empty)
cv.createTrackbar("hue max ","trackbars",179,179,empty)
cv.createTrackbar("sat min ","trackbars",0,255,empty)
cv.createTrackbar("sat max ","trackbars",255,255,empty)
cv.createTrackbar("val min ","trackbars",0,255,empty)
cv.createTrackbar("val max ","trackbars",255,255,empty)

while True:

    img = cv.imread('opencv_chapters/photos/ibra.jpg')

    imgHSV = cv.cvtColor(img,cv.COLOR_BGR2HSV)

    h_min = cv.getTrackbarPos("hue min ","trackbars")
    h_max = cv.getTrackbarPos("hue max ","trackbars")
    s_min = cv.getTrackbarPos("sat min ","trackbars")
    s_max = cv.getTrackbarPos("sat max ","trackbars")
    v_min = cv.getTrackbarPos("val min ","trackbars")
    v_max = cv.getTrackbarPos("val max ","trackbars")

    print(h_min,h_max,s_min,s_max,v_min,v_max)

    lower = np.array([h_min,s_min,v_min])
    upper = np.array([h_max,s_max,v_max])

    mask = cv.inRange(imgHSV,lower,upper)

    img_result = cv.bitwise_and(img,img,mask=mask) 

    cv.imshow('the pic', img)
    cv.imshow('the HSV', imgHSV)
    cv.imshow('MASK', mask) 
    cv.imshow('image result', img_result)



    cv.waitKey(1)