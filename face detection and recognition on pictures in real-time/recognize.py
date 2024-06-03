import cv2 as cv 



def generate_dataset(img, id, img_id):
    cv.imwrite("d ata/user." + str(id) + "." + str(img_id) + ".jpg", img)



def draw_boundary(img, classfier, scaleFactoer, miNeighbors, color, text, clf):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    features = classfier.detectMultiScale(gray, scaleFactoer, miNeighbors)
# img = الصورة التي ترغب في رسم الإطارات عليها
# classifier = المصنف المستخدم للكشف عن المعالم في الصورة
# scaleFactor = العامل المستخدم لتقليل أو زيادة حجم المعالم المكتشفة
# minNeighbors = عدد  المعالم في عملية الكشف
# color = لون الإطارات التي سترسم حول المعالم
# text = النص الذي سيتم وضعه عند العثور على المعالم المحددة
# clf = المصنف المستخدم للتنبؤ بالمعالم الموجودة في الصور
    coords = []
    for (x, y, w, h) in features:
        cv.rectangle(img, (x, y), (x+w, y+h), color, 2)
        id, _ = clf.predict(gray[y:y+h, x:x+w])
        if id == 1:
             cv.putText(img, "ibrahim mahmoud - id: 1000214414 ", (x, y-8), cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv.LINE_AA)
        coords.append([x, y, w, h])  # Append coordinates to the list
    
        


    return gray, coords  # Return gray along with the coordinates


def reconnize(img, clf, faceCascade):
    color = {"blue":(0,0,255), "red":(0,0,255), "green":(0,255,0), "white":(0,0,255)}

    gray, coords = draw_boundary(img, faceCascade, 1.1, 10, color["blue"], "face", clf)
    
    # Check if no faces are detected
    if not coords:
        return img  # Return the original image if no faces are detected

    # Draw boundaries and text on the image
    for coord in coords:
        x, y, w, h = coord
        cv.rectangle(img, (x, y), (x+w, y+h), color["green"], 2)
        id, _ = clf.predict(gray[y:y+h, x:x+w])
        if id == 1:
            cv.putText(img, "ibrahim mahmoud - id: 1000214414 ", (x, y-8), cv.FONT_HERSHEY_SIMPLEX, 0.6, color["red"], 1, cv.LINE_AA)
        
    return img


def detect(img, faceCascade, eyeCascade, mouthCascade, noseCascade, img_id):
    color = {"blue":(255,0,0), "red":(0,255,0), "green":(0,0,255), "white":(255,255,255)}
    coords = draw_boundary(img, faceCascade, 1.1, 10, color["blue"], "face")

    if len(coords) == 4 :
        roi_img =  img[coords[1]:coords[1]+coords[3], coords[0]:coords[0]+coords[2]]

        user_id = 1
        generate_dataset(roi_img, user_id, img_id)

        
        #coords = draw_boundary(roi_img, eyeCascade, 1.1, 14, color["green"], "eyes")
        #coords = draw_boundary(roi_img, mouthCascade, 1.1, 20, color["red"], "mouth")
        #coords = draw_boundary(roi_img, noseCascade, 1.1, 5, color["white"], "nose")

    return img

     



faceCascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
#eyeCascade = cv.CascadeClassifier('haarcascade_eye.xml')
#mouthCascade = cv.CascadeClassifier('Mouth.xml')
#noseCascade = cv.CascadeClassifier('Nariz.xml')

clf = cv.face.LBPHFaceRecognizer_create()
clf.read("classifier.yml")



# الجزء الخاص يالكاميرا (streaming)----=-==================--==---------==================

video_capture = cv.VideoCapture(0)

img_id = 0

while True:
    _, img = video_capture.read()
    #img = detect(img, faceCascade, eyeCascade, mouthCascade, noseCascade, img_id)
    img = reconnize(img, clf, faceCascade)
    cv.imshow("face detection", img)
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    img_id += 1
video_capture.release()
cv.destroyAllWindows()