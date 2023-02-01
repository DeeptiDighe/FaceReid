from collections import OrderedDict
from celery import Celery
import cv2
import base64
from imageio import imread
import io
from PIL import Image
import numpy as np
from Emotyx.Code.FaceReidentification.Complete.verify_faces_mongo import FaceReid
from Emotyx.TrainedModels.People_tracking.counting_people_yolov7 import Tracker
from Emotyx.Code.FaceReidentification.Complete.RabbitMQCelery.app import celery_app
import argparse

#app = Celery('verifyfacetask', broker='amqp://myuser:password@localhost:5672/myvhost',
#                    backend='mongodb://localhost:27017/celerydb')
faceid = FaceReid(r"C:\Projects\Emotyx\TrainedModels\People_tracking",0, r'C:\Users\Accubits\Downloads\osnet_ain_x1_0.pth', 'mongodb://localhost:27017')
trackers = OrderedDict()

@celery_app.task()
def displayframe(frame, size):
    image_64_decoded = base64.decodebytes(frame.encode())
    decoded = cv2.imdecode(np.frombuffer(image_64_decoded, np.uint8), -1)
    print(decoded.shape)
    cv2.imshow('frame', decoded)
    cv2.waitKey(1)
    print('ALL OK!!')

@celery_app.task()
def verifyface(frame, size, camid):
    image_64_decoded = base64.decodebytes(frame.encode())
    decoded = cv2.imdecode(np.frombuffer(image_64_decoded, np.uint8), -1)
    decoded = decoded.copy()
    response = faceid.verify_face_frame(decoded)
    cv2.imshow('Face Reidentification System', response)
    cv2.waitKey(1)
    return True