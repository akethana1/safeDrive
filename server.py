from flask import Flask, request, jsonify
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
import numpy as np
import tensorflow as tf
import cv2
import base64
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['GOOGLEMAPS_KEY'] = "AIzaSyBpGtkmJSz7_PJSvT8LOGXVKZNmoPZsXCM"
GoogleMaps(app, key="8JZ7i18MjFuM35dJHq70n3Hx4")
model = load_model('model.h5')
@app.route('/')
def sleep():
  #code that takes in h5, runs webcam
  path = "haarcascade_frontalface_default.xml"
  faceCascade = cv2.CascadeClassifier(cv2.data.harrcascades + 'harrcascade_frontalface_default.xml')

  cap = cv2.VideoCapture(1)
  #Check if webcam oppened correctly
  if not cap.isOpened():
    cap = cv2.VideoCapture(0)
  if not cap.isOpened():
    raise IOError("Webcam error")

  while True:
    ret,frame = cap.read()
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray,1.1,4)
    for x,y,z,h in eyes:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (9, 255, 0), 2)
        eyess = eye_cascade.detectMultiScale(roi_gray)
        if len(eyess) == 0:
          print("No eyes detected")
        else:
          for (ex, ey, ew, eh) in eyess:
            eyes_roi = roi_color[ey: ey+eh, ex:ex + ew]
    final_image = cv2.resize(eyes_roi, (223, 223))
    final_image = np.expand_dims(final_image, axis = 0)
    final_image = final_image / 255.0

    Predictions = model
    if(Predictions < 0.1):
      sample = True
    else:
      sample = False

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    print(faceCascade.empty())
    faces = faceCascade.detectMultiScale(gray,1.1,4)

    for(x, y, w, h) in faces:
      cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    if(sample == True):
      g = geocoder.ip('me')
      print(g.latlng)
      key='AIzaSyBpGtkmJSz7_PJSvT8LOGXVKZNmoPZsXCM'
      gmaps = googlemaps.Client(key)
      lat=str(g.latlng[0])
      longi=str(g.latlng[1])
      url="https://maps.googleapis.com/maps/api/place/nearbysearch/json?location="+lat+","+longi+"&radius=5000&keyword=reststop&key="+key
      response = requests.get(url).json()
      html_googlemaplink=response['results'][0]['photos'][0]['html_attributions'][0]

      address=[response['results'][0]['geometry']['location']['lat'],response['results'][0]['geometry']['location']['lng']]

      url="https://maps.googleapis.com/maps/api/directions/json?origin="+lat+","+longi+"&destination="+str(address[0])+","+str(address[1])+"&key="+key
      response = requests.get(url).json()

      distance=response['routes'][0]['legs'][0]['distance']['text']
      duration=response['routes'][0]['legs'][0]['duration']['text']
      end_address=response['routes'][0]['legs'][0]['end_address']
      start_address=response['routes'][0]['legs'][0]['start_address']
      print(distance)
      print(duration)
      print(end_address)
      print(start_address)
      distancelist=[]
      durationlist=[]
      html_instructions_list=[]
      for i in response['routes'][0]['legs'][0]['steps']:
        distancelist.append(i['distance']['text'])
        durationlist.append(i['duration']['text'])
        html_instructions_list.append(i['html_instructions'])
      print(distancelist)
      print(durationlist)
      print(html_instructions_list)
