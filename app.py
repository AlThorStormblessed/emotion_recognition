import cv2
import os
import sys
import glob
import time
from deepface import DeepFace
import tensorflow as tf
from deepface.detectors import FaceDetector
import numpy as np
import matplotlib.pyplot as plt
# from flask import Flask, request, jsonify, render_template, Response
from PIL import Image

# app = Flask(__name__)

model = tf.keras.models.load_model('saved_model/model_t')

def get_class_arg(array):
    string = ""
    classes = ["Fear", "Disgust", "Angry", "Happy", "Sad", "Neutral", "Surprise"]
    for i in range(2, 7):
        string += f"{classes[i]} : {round(array[0][i]  * 100, 2)}\n"
    return string

def get_class(argument):
    return ["Fear", "Disgust",
                    "Angry", "Happy",
                    "Neutral", "Sad",
                    "Surprise"][argument]

# try:
#     name = sys.argv[1]
#     num_samples = int(sys.argv[2])
# except:
#     print("Arguments missing.")
#     print(desc)
#     exit(-1)

count = 0
start = False
cap = cv2.VideoCapture(0)
emotions = [[], [], [], [], [], [], []]

while count < 200:

    ret, frame = cap.read()
    if not ret:
        continue
    
    if start:
        cv2.imwrite("Face.jpg", frame)
        try:
            obj = DeepFace.analyze(img_path = "Face.jpg", actions = ["emotion"] ) 
            detector = FaceDetector.build_model('opencv')
            faces_1 = FaceDetector.detect_faces(detector, 'opencv', frame)
            dim = faces_1[0][1]
            cv2.rectangle(frame, (dim[0], dim[1]), (dim[0] + dim[2], dim[1] + dim[3]), (255, 255, 255), 2)
            roi = frame[dim[1]:dim[1] + dim[3], dim[0]:dim[0] + dim[2]]
            img = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (48, 48))
            img = img/255.0
            pred = model.predict(np.array([img]))
            pred[0][0] = 0
            pred[0][1] = 0
            pred[0][2] /= 1.2
            # pred[0][3] /= 2
            # pred[0][4] *= 2
            # pred[0][5] *= 1.4
            pred_sum = sum(pred[0])
            for i in range(7):
                pred[0][i] = pred[0][i]/pred_sum
            for i in range(2, 7):
                emotions[i].append(pred[0][i])

            pred_string = get_class_arg(pred)
            # emotion=get_class(pred)
            #cv2.imwrite(f"image_data/{emotion}/{name}{count}.jpg", roi)

            count += 1

        except Exception as e:
            pred_string = "No face"
            emotion = "No face"
            print(e)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f"{pred_string}",
                (5, 50), font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    k = cv2.waitKey(10)
    if k == ord('a'):
        start = not start
    if k ==ord('q'):
        break

    cv2.imshow("Emotion detected", frame)

for array in emotions[2:]:
    plt.plot(array)

plt.legend(["Angry", "Happy", "Sad", "Neutral", "Surprise"])
plt.grid()
plt.show()

cv2.destroyAllWindows()
