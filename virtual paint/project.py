import cv2 as cv
import numpy as np


framewidth = 640
frameheight = 480

video_capture = cv.VideoCapture(0)
video_capture.set(3,framewidth)
video_capture.set(4,frameheight)
video_capture.set(10,150)

mycolors = [
    [35, 100, 100, 85, 255, 255],  # الأخضر
    [0, 100, 100, 10, 255, 255],   # الأحمر (النطاق الأول)
    [170, 100, 100, 180, 255, 255], # الأحمر (النطاق الثاني)
   # [20, 100, 100, 30, 255, 255]   # الأصفر
]
mycolor_values = [
    (0, 255, 0),   # لون الرسم للكرة الخضراء (أخضر)
    (0, 0, 255),   # لون الرسم للكرة الحمراء (أحمر)
    (0, 0, 255),   # لون الرسم للكرة الحمراء (أحمر)
    #(0, 255, 255)  # لون الرسم للكرة الصفراء (أصفر)
]

mypoints =  []        #[x, y, colorid]


def findcolor(img,mycolors,mycolor_values):
        imgHSV = cv.cvtColor(img,cv.COLOR_BGR2HSV)
        count = 0
        newpoints = []
        for color in mycolors:
            lower = np.array(color[0:3])
            upper = np.array(color[3:6])
            mask = cv.inRange(imgHSV,lower,upper)
            x,y = getcontours(mask)
            cv.circle (imageresult,(x,y),10,mycolor_values[count],cv.FILLED)
            if x !=0 and y !=0:
                newpoints.append([x,y,count])
            count += 1
        return newpoints
            #cv.imshow(str(color[0]),mask)


def getcontours(img):
    contours, hierarchy = cv.findContours(img,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)  
    x,y,w,h = 0,0,0,0

    for cnt in contours:
        area = cv.contourArea(cnt)
        
        if area>500:

            cv.drawContours(imageresult,cnt,-1,(255,0,0),4)

            peri = cv.arcLength(cnt,True)
            
            approx = cv.approxPolyDP(cnt,0.02*peri,True)
        
            x,y,w,h = cv.boundingRect(approx)

    return x+w//2,y+h//2


def draw(mypoints,mycolor_values):
    for point in mypoints:
        cv.circle (imageresult,(point[0],point[1]),10,mycolor_values[point[2]],cv.FILLED)


    
while True:

    succcess, img = video_capture.read()
    
    imageresult = img.copy()

    newpoints = findcolor(img,mycolors,mycolor_values)
    if len(newpoints) !=0:
        for newp in newpoints:
            mypoints.append(newp)

    if len(mypoints)!=0:
        draw(mypoints,mycolor_values)

    cv.imshow("web-cam", imageresult)
  
    if cv.waitKey(1) & 0xFF == ord('q'):
       break
video_capture.release()
cv.destroyAllWindows()