import numpy as np
import cv2 as cv
import os

haar_cascade = cv.CascadeClassifier('haar_face.xml')

people = ['Benedict_Cumberbatch', 'PewDiePie', 'Tom_Holland']

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

img = cv.imread(r'D:\Coding\Python\color_detection\images_videos\faces\test_images\Cumberbatch_test.jpg')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Person', img)

faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)

for (x,y,w,h) in faces_rect:
    faces_roi = gray[y:y+h, x:x+w]

    label, confidence = face_recognizer.predict(faces_roi)

    cv.putText(img, f'{people[label]}, {round(confidence)}% sure.', (20,30), cv.FONT_HERSHEY_COMPLEX, 1.0, (255,0,0), 2)
    cv.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 4)
    cv.circle(img, (round(x+w/2), round(y+h/2)), 20, (0,0,0), 2)

cv.imshow('Detected Face', img)

cv.waitKey(0)