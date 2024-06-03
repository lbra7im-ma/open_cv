import cv2 as cv 



haar_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

people = ["Name Ahmed Ragheb Id 1201171Level 3", "Name Ali Ali Azmy Id 1201252 Level 3", "Name Bahi Eid Id 1201041 Level 3", "Name Mohamed Abdo Mady Id 1201355 Level 3", "Name Mohamed Abo Elkhair Id 1201356 Level 3","Prof.Dr. Mohammed Kamal"]

#features = np.load('features.npy')
#labels = np.load('labels.npy')

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')


img = cv.imread(r'C:\Users\Ibra\Documents\face recognition on pictures\faces\Name Bahi Eid Id 1201041 Level 3\c7.jpg')


gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('person', gray)

face_rect = haar_cascade.detectMultiScale(gray,1.1 ,20)

for (x,y,w,h) in face_rect:
    faces_roi = gray[y:y+h, x:x+w]

    label, confidence = face_recognizer.predict(faces_roi)
    print(f'label = {people[label]} with confidence of  {confidence}')

    cv.putText(img, str(people[label] ), (20,60), cv.FONT_HERSHEY_COMPLEX, 1.0, (255,255,0), thickness = 2 )
    cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2)

cv.imshow('detected face ', img)

cv.waitKey(0)