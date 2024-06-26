import cv2 as cv
import numpy as np

# إعدادات العرض
width_img = 320
height_img = 240

# إعداد الكاميرا
video_capture = cv.VideoCapture(0)
video_capture.set(3, width_img)
video_capture.set(4, height_img)
video_capture.set(10, 150)

def preprocessing(img):
    """تطبيق مراحل المعالجة المسبقة على الصورة"""
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5, 5), 1)
    canny = cv.Canny(blur, 200, 200)
    kernel = np.ones((5, 5))
    dial = cv.dilate(canny, kernel, iterations=2)
    img_thresh = cv.erode(dial, kernel, iterations=1)
    return gray, blur, canny, dial, img_thresh

def get_contours(img, img_count):
    """استخراج الحدود من الصورة"""
    biggest = np.array([])
    max_area = 0
    contours, _ = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area > 5000:
            peri = cv.arcLength(cnt, True)
            approx = cv.approxPolyDP(cnt, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    if biggest.size != 0:
        cv.drawContours(img_count, [biggest], -1, (255, 0, 0), 20)
    return biggest

def reorder(points):
    """إعادة ترتيب النقاط لأغراض تحويل المنظور"""
    points = points.reshape((4, 2))
    new_points = np.zeros((4, 2), np.int32)
    add = points.sum(1)
    new_points[0] = points[np.argmin(add)]
    new_points[3] = points[np.argmax(add)]
    diff = np.diff(points, axis=1)
    new_points[1] = points[np.argmin(diff)]
    new_points[2] = points[np.argmax(diff)]
    return new_points

def get_warp(img, biggest):
    """تطبيق تحويل المنظور على الصورة"""
    if biggest.size == 0:
        return img  # إذا لم يتم العثور على أي حدود، يتم إرجاع الصورة الأصلية
    biggest = reorder(biggest)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0, 0], [width_img, 0], [0, height_img], [width_img, height_img]])
    matrix = cv.getPerspectiveTransform(pts1, pts2)
    img_output = cv.warpPerspective(img, matrix, (width_img, height_img))
    img_cropped = img_output[20:img_output.shape[0] - 20, 20:img_output.shape[1] - 20]
    img_cropped = cv.resize(img_cropped, (width_img, height_img))

    return img_cropped

while True:
    success, img = video_capture.read()
    if not success:
        break

    img = cv.resize(img, (width_img, height_img))
    img_count = img.copy()

    gray, blur, canny, dial, img_thresh = preprocessing(img)
    biggest = get_contours(img_thresh, img_count)
    img_warp = get_warp(img, biggest)

    cv.imshow("Original", cv.resize(img, (width_img, height_img)))
    cv.imshow("Grayscale", cv.resize(gray, (width_img, height_img)))
    cv.imshow("Blurred", cv.resize(blur, (width_img, height_img)))
    cv.imshow("Canny Edges", cv.resize(canny, (width_img, height_img)))
    cv.imshow("Dilated", cv.resize(dial, (width_img, height_img)))
    cv.imshow("Thresholded", cv.resize(img_thresh, (width_img, height_img)))
    cv.imshow("Warped", cv.resize(img_warp, (400, 550)))

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# تحرير الموارد وإغلاق جميع النوافذ
video_capture.release()
cv.destroyAllWindows()
