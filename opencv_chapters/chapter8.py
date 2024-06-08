import cv2 as cv 

import numpy as np 


def getcontours(img):
    contours, hierarchy = cv.findContours(img,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)  
    for cnt in contours:
        area = cv.contourArea(cnt)
        print(area)
        if area>500:
            cv.drawContours(imgcontour,cnt,-1,(255,0,0),3)
            peri = cv.arcLength(cnt,True)
            print(peri)

            approx = cv.approxPolyDP(cnt,0.02*peri,True)
            print(len(approx))

            object = len(approx)

            x,y,w,h = cv.boundingRect(approx)

            if object == 3: objectType = "tri"
            elif object ==4:
                aspratio = w/float(h)
                if aspratio >0.95 and aspratio<1.05 : objectType = "squre"
                else: objectType = "rectangle"

            elif object >4: objectType = "circle"
            
            
            else: objectType = "none"

            cv.rectangle(imgcontour,(x,y),(x+w,y+h),(0,255,0),2)

            cv.putText(imgcontour,objectType,(x+(w//2)-10,y+(h//2)-10),cv.FONT_HERSHEY_COMPLEX,.5,(0,0,0),2)
        


path = 'opencv_chapters/photos/shape4.jpg'
img = cv.imread(path) 
imgcontour = img.copy()


img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img_blur = cv.GaussianBlur(img_gray,(7,7),1)
canny = cv.Canny(img_blur,50,50)

getcontours(canny)


cv.imshow("original", img)
cv.imshow(" gray image ", img_gray)
cv.imshow(" blur image ", img_blur)
cv.imshow(" canny edges image ", canny)
cv.imshow(" image contour ", imgcontour)



cv.waitKey(0)