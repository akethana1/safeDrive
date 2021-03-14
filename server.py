from flask import Flask, request, jsonify, render_template, url_for, redirect
from flask_cors import CORS, cross_origin
from flask_googlemaps import GoogleMaps
from flask_googlemaps import Map
from keras.models import load_model
from datetime import datetime
import geocoder
import googlemaps
import pandas as pd
import requests
import json
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import os
import winsound
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['GOOGLEMAPS_KEY'] = "AIzaSyBpGtkmJSz7_PJSvT8LOGXVKZNmoPZsXCM"
GoogleMaps(app, key="8JZ7i18MjFuM35dJHq70n3Hx4")
flag = False
strike = 0


def model_func():
    global strike
    strike = 0
    ap = argparse.ArgumentParser()
    ap.add_argument("-w", "--webcam", type=int, default=1,
                    help="index of webcam on system")
    args = vars(ap.parse_args())
    EYE_AR_THRESH = 0.3
    EYE_AR_CONSEC_FRAMES = 30
    YAWN_THRESH = 25
    alarm_status = False
    alarm_status2 = False
    saying = False
    COUNTER = 0

    print("-> Loading the predictor and detector...")
    # detector = dlib.get_frontal_face_detector()
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")  # Faster but less accurate
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    print("-> Starting Video Stream")
    vs = VideoStream(src=args["webcam"]).start()
    # vs= VideoStream(usePiCamera=True).start()       //For Raspberry Pi
    time.sleep(1.0)



    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=450)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # rects = detector(gray, 0)
        rects = detector.detectMultiScale(gray, scaleFactor=1.1,
                                          minNeighbors=5, minSize=(30, 30),
                                          flags=cv2.CASCADE_SCALE_IMAGE)

        # for rect in rects:
        for (x, y, w, h) in rects:
            rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))

            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            eye = final_ear(shape)
            ear = eye[0]
            leftEye = eye[1]
            rightEye = eye[2]

            distance = lip_distance(shape)

            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            # cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            # cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            lip = shape[48:60]
            # cv2.drawContours(frame, [lip], -1, (0, 255, 0), 1)

            if ear < EYE_AR_THRESH:
                COUNTER += 1

                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    if alarm_status == False:
                        alarm_status = True
                        alarm()
                        strike +=1
                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            else:
                COUNTER = 0
                alarm_status = False

            if (distance > YAWN_THRESH):
                cv2.putText(frame, "Yawn Alert", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if alarm_status2 == False and saying == False:
                    alarm_status2 = True
                    strike += 1
                    alarm()
            else:
                alarm_status2 = False

            cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "YAWN: {:.2f}".format(distance), (300, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if strike >= 3:
            break
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    vs.stop()

def async_model():
    thr = Thread(target=model_func)
    thr.start()
    return thr

def get_instructions():
    g = geocoder.ip('me')
    print(g.latlng)
    key = 'AIzaSyBpGtkmJSz7_PJSvT8LOGXVKZNmoPZsXCM'
    gmaps = googlemaps.Client(key)
    lat = str(g.latlng[0])
    longi = str(g.latlng[1])
    url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json?location=" + lat + "," + longi + "&radius=5000&keyword=reststop&key=" + key
    response = requests.get(url).json()
    html_googlemaplink = response['results'][0]['photos'][0]['html_attributions'][0]

    address = [response['results'][0]['geometry']['location']['lat'],
               response['results'][0]['geometry']['location']['lng']]

    url = "https://maps.googleapis.com/maps/api/directions/json?origin=" + lat + "," + longi + "&destination=" + str(
        address[0]) + "," + str(address[1]) + "&key=" + key
    response = requests.get(url).json()

    distance = response['routes'][0]['legs'][0]['distance']['text']
    duration = response['routes'][0]['legs'][0]['duration']['text']
    end_address = response['routes'][0]['legs'][0]['end_address']
    start_address = response['routes'][0]['legs'][0]['start_address']
    print(distance)
    print(duration)
    print(end_address)
    print(start_address)
    distancelist = []
    durationlist = []
    html_instructions_list = []
    for i in response['routes'][0]['legs'][0]['steps']:
        distancelist.append(i['distance']['text'])
        durationlist.append(i['duration']['text'])
        html_instructions_list.append(i['html_instructions'])
    print(distancelist)
    print(durationlist)
    print(html_instructions_list)

def alarm():
    global alarm_status
    global alarm_status2
    global saying
    winsound.PlaySound(None, winsound.SND_PURGE)
    winsound.PlaySound("alarm.wav", winsound.SND_FILENAME)



def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    C = dist.euclidean(eye[0], eye[3])

    ear = (A + B) / (2.0 * C)

    return ear

def final_ear(shape):
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]

    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)

    ear = (leftEAR + rightEAR) / 2.0
    return (ear, leftEye, rightEye)

def lip_distance(shape):
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))

    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))

    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)

    distance = abs(top_mean[1] - low_mean[1])
    return distance

@app.route('/', methods=['GET', 'POST'])
@app.route('/home', methods=['GET', 'POST'])
def home():
    return render_template('home.html')

@app.route('/sleep', methods=['GET', 'POST'])
def sleep():
    async_model()
    return render_template('sleep.html', strike=strike)





if __name__ == "__main__":
  app.run(debug=True)
