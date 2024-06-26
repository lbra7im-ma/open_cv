import cv2 as cv

##########################################
framewidth = 640
frameheight = 480

nplate = cv.CascadeClassifier("number plate detection\haarcascade_russian_plate_number.xml")
minarea = 100
color = (255,0,255)

video_capture = cv.VideoCapture(0)
video_capture.set(3,framewidth)
video_capture.set(4,frameheight)
video_capture.set(10,150)
count = 0
while True:

    succcess, img = video_capture.read()
    
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

    numberplates = nplate.detectMultiScale(gray,1.1,4)

    for (x,y,w,h) in numberplates:
        area = w*h
        if area > minarea:

            cv.rectangle(img,(x,y),(x+w,y+h),(255,0,255),2)
            cv.putText(img,"number plate",(x,y-5),
                       cv.FONT_HERSHEY_COMPLEX,1,color,2)
            imgroi = img[y:y+h,x:x+w]
            cv.imshow("roi", imgroi)


    cv.imshow("result", img)
    if cv.waitKey(1) & 0xFF == ord('s'):
       cv.imwrite("number plate detection/scanned/noplate_"+str(count)+".jpg",imgroi)
       cv.rectangle(img,(0,200),(640,300),(0,255,0),cv.FILLED)
       cv.putText(img,"scan saver",(150,256),cv.FONT_HERSHEY_DUPLEX,2,(0,0,255),2)
       cv.imshow("result",img)
       cv.waitKey(500)
       count += 1
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
