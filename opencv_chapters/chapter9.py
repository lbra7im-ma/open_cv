import cv2 as cv 

faceCascade = cv.CascadeClassifier('face detection and recognition on pictures in real-time/haarcascade_frontalface_default.xml')

img = cv.imread('opencv_chapters/photos/ibra3.jpg')

gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

faces = faceCascade.detectMultiScale(gray,1.1,4)

for (x,y,w,h) in faces:
    cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)



cv.imshow("result",img)
cv.waitKey(0)
