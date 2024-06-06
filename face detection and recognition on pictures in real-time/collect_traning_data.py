import cv2
import os

def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)
    coords = []
    for (x, y, w, h) in features:
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, text, (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        coords.append((x, y, w, h))
    return coords

def save_image(img, img_id, user_id, data_path):
    file_name = f"{data_path}/User.{user_id}.{img_id}.jpg"
    cv2.imwrite(file_name, img)
    print(f"Image saved as {file_name}")

def detect(img, faceCascade, img_id, user_id, data_path):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    coords = draw_boundary(img, faceCascade, 1.1, 10, (255, 0, 0), "Face")
    
    if len(coords) == 1:
        x, y, w, h = coords[0]
        roi_gray = gray_img[y:y+h, x:x+w]
        save_image(roi_gray, img_id, user_id, data_path)
    
    return img

faceCascade = cv2.CascadeClassifier('face detection and recognition on pictures in real-time/haarcascade_frontalface_default.xml')

data_path = 'face detection and recognition on pictures in real-time/data'
if not os.path.exists(data_path):
    os.makedirs(data_path)
    print(f"Created directory at {data_path}")

# هذا المثال يستخدم الكاميرا (يمكنك تعديله ليتوافق مع صورك)
cap = cv2.VideoCapture(0)
img_id = 0
user_id = 1  # يمكنك ضبط هذا بناءً على معرف المستخدم

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = detect(frame, faceCascade, img_id, user_id, data_path)
    cv2.imshow("Face Detection", frame)
    img_id += 1
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
