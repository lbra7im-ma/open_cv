import cv2 as cv
import numpy as np

img = cv.imread('card.jpg')

width,height = 250,350

pts1 = np.float32([[101,139],[203,53],[243,355],[321,240]])

pts2 = np.float32([[0,0],[width,0],[0,height],[width,height]])

matrix = cv.getPerspectiveTransform(pts1,pts2)

imgoutput = cv.warpPerspective(img,matrix,(width,height))



cv.imshow('card',img)
cv.imshow('warp image', imgoutput)
cv.waitKey(0)