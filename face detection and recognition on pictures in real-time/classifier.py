import numpy as np

from PIL import Image 

import os , cv2 as cv

def train_classifier(data_dir):
    path = [os.path.join (data_dir, f) for f in os.listdir(data_dir)]
    faces = []
    ids = []


    for image in path :
        img = Image.open(image).convert("L")
        imageNp = np.array(img, 'uint8')
        id = int(os.path.split(image) [1].split(".") [1])

        faces.append(imageNp)
        ids.append(id)

    ids = np.array(ids)

    clf = cv.face.LBPHFaceRecognizer_create()
    clf.train(faces, ids)
    clf.write("face detection and recognition on pictures in real-time/classifier.yml")

train_classifier("face detection and recognition on pictures in real-time/data")
print(f"trainnig the modele done succefully  ")