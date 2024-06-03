import cv2 as cv

img = cv.imread('opencv_chapters/ibra.jpg')
cv.imshow('the pic', img)

#==============================================================================================



video_capture = cv.VideoCapture(0)
#video_capture.set(3,640)
#video_capture.set(4,480)
#video_capture.set(10,100)


while True:

    succcess, img = video_capture.read()
    cv.imshow("video", img)
    if cv.waitKey(1) & 0xFF == ord('q'):
       break
       


