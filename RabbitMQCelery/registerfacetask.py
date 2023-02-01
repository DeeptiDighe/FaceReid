from collections import OrderedDict
from celery import Celery
import cv2
import base64
from imageio import imread
import io
from PIL import Image
import numpy as np
from Emotyx.Code.FaceReidentification.Complete.identify_faces_mongo import RegisterFaces
from Emotyx.TrainedModels.People_tracking.counting_people_yolov7 import Tracker
import argparse

app = Celery('registerfacetask', broker='amqp://myuser:password@localhost:5672/myvhost',
                    backend='mongodb://localhost:27017/celerydb')
faceid = RegisterFaces(0, r'C:\Users\Accubits\Downloads\osnet_ain_x1_0.pth', 'mongodb://localhost:27017')
trackers = OrderedDict()

@app.task()
def displayframe(frame, size):
    image_64_decoded = base64.decodebytes(frame.encode())
    decoded = cv2.imdecode(np.frombuffer(image_64_decoded, np.uint8), -1)
    print(decoded.shape)
    cv2.imshow('frame', decoded)
    cv2.waitKey(1)
    print('ALL OK!!')

@app.task()
def registerface(frame, size, camid):
    image_64_decoded = base64.decodebytes(frame.encode())
    decoded = cv2.imdecode(np.frombuffer(image_64_decoded, np.uint8), -1)
    decoded = decoded.copy()
    if trackers.get(camid):
        tracker = trackers[camid]
    else:
        tracker = Tracker()
        trackers[camid] = tracker
    response = faceid.register_face(decoded, camid, tracker)
    print(response)
    return True