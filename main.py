import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

from keras.models import load_model
category = {0: 'Male',1:'Female'}

haar_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = load_model(r'D:\Python projects\Machine Learning\Male Female Detection\male_female_classifier.h5')


capture = cv2.VideoCapture(0)
data = []
font = cv2.FONT_HERSHEY_COMPLEX
while True:
    flag, img = capture.read()
    if flag:
        faces = haar_data.detectMultiScale(img)
        for x, y, w, h in faces:
            cv2.rectangle(img, (x, y), (x + w, y + w), (0, 255, 0), 4)
            face = img[y:y + h, x:x + w, :]
            face = cv2.resize(face, (100, 100))
            face = np.array(face).reshape(-1, 100, 100, 1)
            prediction = model.predict(face)
            pred = 0
            if prediction[0] > 0.5:
                pred = 1
            else:
                pred = 0

            n = category[pred]
            cv2.putText(img, n, (x, y), font, 1, (233, 244, 250), 2)

        cv2.imshow('result', img)
        if cv2.waitKey(2) == 27:
            break
capture.release()
cv2.destroyAllWindows()