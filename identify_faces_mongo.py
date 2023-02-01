from time import time
import cv2
from retinaface import RetinaFace
import numpy as np
from People_tracking.counting_people_yolov7 import Tracker
from FeatureExtractor.feature_extractor import FeatureExtractor
from imquality import brisque
from deepface import DeepFace
from pymongo import MongoClient
from bson import Binary
import pickle
import argparse

def connect_mongodb(connectionstring):
    client = MongoClient(connectionstring)
    db = client['emotyx']
    all_tables = db.list_collection_names()
    if all(x in all_tables for x in ['person', 'embeddings']):
        return client
    if 'person' not in all_tables:
        db.create_collection('person')
    if 'embeddings' not in all_tables:
        db.create_collection('embeddings')
    return client

class RegisterFaces:

    def __init__(self, gpu, personmodel, mongoclient):
        if gpu:
            self.fe = FeatureExtractor('osnet_ain_x1_0', model_path=personmodel, device='gpu')
        else:
            self.fe = FeatureExtractor('osnet_ain_x1_0', model_path=personmodel, device='cpu')
        #self.tracker = Tracker()
        self.mongourl = mongoclient
        self.client = connect_mongodb(mongoclient)

    def npAngle(self, a, b, c):
        ba = a - b
        bc = c - b 
        
        cosine_angle = np.dot(ba, bc)/(np.linalg.norm(ba)*np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)
        
        return np.degrees(angle)

    def retinaface_detect(self, img, low=30, high=70):
        faces, identity = RetinaFace.extract_faces(img, align=True)
        if identity == None or identity['score'] < 0.9:
            return list(), list(), None
        landmarks = identity['landmarks']
        a=np.array(landmarks['left_eye'])
        b=np.array(landmarks['right_eye'])
        c=np.array(landmarks['nose'])
        angR = self.npAngle(a, b, c) # Calculate the right eye angle
        angL = self.npAngle(b, a, c)# Calculate the left eye angle
        aligned_faces = list()
        coords = list()
        for face in faces:
            x1,y1,x2,y2 = face[0]
            aligned_faces.append(face[1])
            coords.append((x1,y1,x2,y2))

        #if ((int(angR) in range(35, 57)) and (int(angL) in range(35, 58))):
        if ((int(angR) in range(low, high)) and (int(angL) in range(low, high))):
            predLabel='Frontal'
        else: 
            if angR < angL:
                predLabel='Left'
            else:
                predLabel='Right'

        return aligned_faces, coords, (predLabel,angL,angR)

    def get_objectIds(self, img, counter):
        registeredObjects = counter.process(img)
        return registeredObjects

    def get_face_objectId(self, face, registeredObjects):
        for object in registeredObjects:
            if self.contains(registeredObjects[object], face):
                return object, registeredObjects[object]
        return None, None

    def contains(self, r1, r2):
        r1x1,r1y1,r1x2,r1y2 = r1
        r2x1,r2y1,r2x2,r2y2 = r2
        return r1x1 < r2x1 < r2x2 < r1x2 and r1y1 < r2y1 < r2y2 < r1y2

    def register_face(self, frame, camid, tracker):
        try:
            oframe = frame.copy()
            registeredPeople = self.get_objectIds(frame, tracker)    #TODO: only process people which are coming towards camera        
            #TODO: instead of passing whole frame for face detection, pass half frame for faster detection as well as better faces
            #TODO: run face detection and person tracking parallely
            to_be_inserted = list()
            embeddings = list()
            for personId, personbb in registeredPeople.items():
                if (personbb[3]-personbb[1])*(personbb[2]-personbb[0]) < frame.size*0.0025:
                    continue
                #cv2.imwrite(os.path.join(temp,'tempcap.jpg'), frame[max(0,personbb[1]):personbb[3], max(0,personbb[0]):personbb[2]])
                score = brisque.score(frame[max(0,personbb[1]):personbb[3], max(0,personbb[0]):personbb[2]])
                if score > 40:  #lesser the better
                    continue
                try:
                    aligned_faces, coords, angle = self.retinaface_detect(frame[max(0,personbb[1]):personbb[3], max(0,personbb[0]):personbb[2]])
                    if len(aligned_faces)==0:
                        continue
                except:
                    continue
                if aligned_faces[0].shape[0] < 40 or aligned_faces[0].shape[1] < 30:
                    continue
                if angle[0]=='Frontal':
                    score = brisque.score(aligned_faces[0])
                    if score < 0 or score > 50:  #lesser the better
                        continue
                    i = np.random.randint(0, 1000)
                    row = dict()
                    row['_id'] = str(camid) + '_' + str(personId) + '_' + str(i)
                    row['person_img'] = Binary(oframe[max(0,personbb[1]):personbb[3],max(0,personbb[0]):personbb[2]])
                    emb, im = DeepFace.represent_face(aligned_faces[0][:,:,::-1], model_name='ArcFace', enforce_detection=False)
                    row['face_img'] = Binary(im)
                    row_emb = dict()
                    row_emb['_id'] = str(camid) + '_' + str(personId) + '_' + str(i)
                    row_emb['face_embedding'] = emb
                    row_emb['person_embedding'] = Binary(pickle.dumps(self.fe(oframe[max(0,personbb[1]):personbb[3],max(0,personbb[0]):personbb[2]]).cpu().numpy(),protocol=2))
                    cv2.imshow('person_img', oframe[max(0,personbb[1]):personbb[3],max(0,personbb[0]):personbb[2]])
                    cv2.imshow('face_img', im)
                    cv2.waitKey(1)
                    to_be_inserted.append(row)
                    embeddings.append(row_emb)

            if len(to_be_inserted) > 0:
                try:
                    self.client.emotyx.person.insert_many(to_be_inserted)
                    self.client.emotyx.embeddings.insert_many(embeddings)
                    print(f'**********************Inserted {str(camid) + '_' + str(personId) + '_' + str(i)} in the database')
                except:
                    self.client = self.connect_mongodb(self.mongourl)
                    self.client.emotyx.person.insert_many(to_be_inserted)
                    self.client.emotyx.embeddings.insert_many(embeddings)
                    print(f'**********************Inserted {str(camid) + '_' + str(personId) + '_' + str(i)} in the database')
            cv2.imshow('image', frame)
            k = cv2.waitKey(1)
        
        except Exception as e:
            return str(e)
        return 'ALL OK!'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('height', help='Fraction height of the frame measured from top to bottom for face identification. Ex: 0.5',
                        default=0.66)
    parser.add_argument('gpu', help='cpu if 0 else gpu', default=0)
    parser.add_argument('personmodel', help='Path to person features extractor', default=r'C:\Users\Accubits\Downloads\osnet_ain_x1_0.pth')
    parser.add_argument('mongoclient', help='MongoDB connection string', default='mongodb://localhost:27017')
    args = parser.parse_args()
    faceid = RegisterFaces(args)
    #faceid.register_face(some_frame)

def try_run():
    faceid = RegisterFaces(0, r'C:\Users\Accubits\Downloads\osnet_ain_x1_0.pth', 'mongodb://localhost:27017')
    vc = cv2.VideoCapture(r'C:\Projects\Emotyx\Code\FaceReidentification\Complete\videos\v1.mp4')
    while True:
        ret, frame = vc.read()
        res = faceid.register_face(frame)

#try_run()
