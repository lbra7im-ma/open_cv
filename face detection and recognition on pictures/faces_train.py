import cv2 as cv
import numpy as np
import os

people = ["Name Ahmed Ragheb Id 1201171Level 3", "Name Ali Ali Azmy Id 1201252 Level 3", "Name Bahi Eid Id 1201041 Level 3", "Name Mohamed Abdo Mady Id 1201355 Level 3", "Name Mohamed Abo Elkhair Id 1201356 Level 3","Prof.Dr. Mohammed Kamal"]
DIR = r"C:\Users\Ibra\Documents\face recognition on pictures\faces"

haar_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

features = []
labels = []

def create_train():
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)

        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            img = cv.imread(img_path)
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

            for (x, y, w, h) in faces_rect:
                faces_roi = gray[y:y+h, x:x+w]
                features.append(faces_roi)
                labels.append(label)

create_train()

print('training done =============================-------------------=================== ')

features = np.array(features, dtype='object')

face_recognizer = cv.face.LBPHFaceRecognizer_create()

face_recognizer.train(features, np.array(labels))

face_recognizer.save('face_trained.yml')

np.save('features.npy', features)
np.save('labels.npy', labels)
