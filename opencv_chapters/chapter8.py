import cv2 as cv 

import numpy as np 


path = 'opencv_chapters/photos/shape4.jpg'
img = cv.imread(path)

img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img_blur = cv.GaussianBlur(img_gray,(7,7),1)





cv.imshow("original", img)
cv.imshow(" gray image ", img_gray)
cv.imshow(" blur image ", img_blur)

cv.waitKey(0)