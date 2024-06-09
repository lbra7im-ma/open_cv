import cv2 as cv
import numpy as np

def empty(a):
    pass

# إنشاء نافذة لأشرطة التمرير
cv.namedWindow("Trackbars")
cv.resizeWindow("Trackbars", 640, 240)
cv.createTrackbar("Hue Min", "Trackbars", 0, 179, empty)
cv.createTrackbar("Hue Max", "Trackbars", 179, 179, empty)
cv.createTrackbar("Sat Min", "Trackbars", 0, 255, empty)
cv.createTrackbar("Sat Max", "Trackbars", 255, 255, empty)
cv.createTrackbar("Val Min", "Trackbars", 0, 255, empty)
cv.createTrackbar("Val Max", "Trackbars", 255, 255, empty)

# فتح كاميرا الويب
cap = cv.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        break

    imgHSV = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    
    h_min = cv.getTrackbarPos("Hue Min", "Trackbars")
    h_max = cv.getTrackbarPos("Hue Max", "Trackbars")
    s_min = cv.getTrackbarPos("Sat Min", "Trackbars")
    s_max = cv.getTrackbarPos("Sat Max", "Trackbars")
    v_min = cv.getTrackbarPos("Val Min", "Trackbars")
    v_max = cv.getTrackbarPos("Val Max", "Trackbars")
    
    lower_orange = np.array([h_min, s_min, v_min])
    upper_orange = np.array([h_max, s_max, v_max])
    
    mask = cv.inRange(imgHSV, lower_orange, upper_orange)
    
    result = cv.bitwise_and(img, img, mask=mask)
    
    cv.imshow("Original", img)
    cv.imshow("Mask", mask)
    cv.imshow("Result", result)
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
