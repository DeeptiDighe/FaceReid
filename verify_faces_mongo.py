import os
from deepface import DeepFace
import cv2
import numpy as np
from FeatureExtractor.feature_extractor import FeatureExtractor
import pandas as pd
from scipy import spatial
from identify_faces_mongo import connect_mongodb
import pickle
from deepface.commons import distance as dst
from retinaface import RetinaFace
from imquality import brisque
import argparse
import base64

class FaceReid:

    def __init__(self, yolopath, gpu, personmodel, mongoclient):
        self.confThreshold = 0.6  #Confidence threshold
        self.nmsThreshold = 0.4   #Non-maximum suppression threshold

        # Give the configuration and weight files for the model and load the network using them.
        modelConfiguration = os.path.join(yolopath,"yolov3.cfg")
        modelWeights = os.path.join(yolopath,"yolov3.weights")

        print("[INFO] loading model...")
        self.net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
        self.ln = self.net.getLayerNames()
        self.ln = [self.ln[i - 1] for i in self.net.getUnconnectedOutLayers()]

        if gpu:
            self.fe = FeatureExtractor('osnet_ain_x1_0', model_path=personmodel, device='gpu')
        else:
            self.fe = FeatureExtractor('osnet_ain_x1_0', model_path=personmodel, device='cpu')
        #self.df = pd.read_csv(r'C:\Projects\Emotyx\Code\FaceReidentification\Complete\Embeddings\FileIds.csv')
        #elf.embeddings = np.load(r'C:\Projects\Emotyx\Code\FaceReidentification\Complete\Embeddings\embeddings.npy')

        client = connect_mongodb(mongoclient)
        self.df = pd.DataFrame(client['emotyx']['person'].find())
        #TODO: query the embeddings from database instead of loading the collection
        self.embeddings = pd.DataFrame(client['emotyx']['embeddings'].find())
        self.df = self.df.set_index('_id')
        self.embeddings = self.embeddings.set_index('_id')

    def run_yolo_on_frame(self, frame):
        blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), [0,0,0], 1, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.ln)
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]
        rects = []
        classIds = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > self.confThreshold:
                    box = detection[0:4] * np.array([frameWidth, frameHeight, frameWidth, frameHeight])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    if(classId == 0):
                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
                        classIds.append(classId)

        # Perform non maximum suppression to eliminate redundant overlapping boxes with
        # lower confidences.
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confThreshold, self.nmsThreshold)
        if len(indices) > 0:
            for i in indices.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                x = 0 if x < 0 else x
                y = 0 if y < 0 else y
                rects.append((x,y,x+w,y+h))

        return rects

    def verify_person(self, person):
        #create embedding for input person and check it with the already computed and stored embeddings
        embedding = self.fe(person)
        matches = list()
        similarity = list()
        for i in range(len(self.embeddings)):            
            cos_sim = 1 - spatial.distance.cosine(embedding, pickle.loads(self.embeddings.iloc[i]['person_embedding']))
            #eucledian = np.linalg.norm(embedding - self.embeddings[i])
            if cos_sim >= 0.6 and (self.df.index[i] not in matches):
                matches.append(self.df.index[i])
                similarity.append(cos_sim)

        if len(matches) > 0:
            df = pd.DataFrame({'file':matches, 'sim':similarity})
            df.sort_values(by='sim', ascending=False, inplace=True)
            df = df.head(5)

            return list(df['file'])
        else:
            return list()

    # Landmarks: [Left Eye], [Right eye], [nose], [left mouth], [right mouth]
    def npAngle(self, a, b, c):
        ba = a - b
        bc = c - b 
        
        cosine_angle = np.dot(ba, bc)/(np.linalg.norm(ba)*np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)
        
        return np.degrees(angle)

    def retinaface_detect(self, img, low=35, high=70):
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
            #cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            #return cropped and aligned face[1] as well
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

    def work_on_video(self):
        vc = cv2.VideoCapture(r'C:\Projects\Emotyx\Code\FaceReidentification\Complete\videos\v1.mp4')
        i = 0

        model = 'ArcFace'
        distance_metric = 'cosine'
        threshold = 0.1  #dst.findThreshold(model, distance_metric)
        results = list()
        try:
            while True:
                ret, frame = vc.read()
                if ret != True:
                    break
                i+=1
                if i%20 != 0:
                    continue

                #cv2.imshow('f', frame)
                #cv2.waitKey(1)
                rects = self.run_yolo_on_frame(frame)
                for rect in rects:
                    #cv2.imshow('f', frame[rect[1]:rect[3], rect[0]:rect[2]])
                    #cv2.waitKey()
                    matches = self.verify_person(frame[rect[1]:rect[3], rect[0]:rect[2]])
                    try:
                        #face = DeepFace.detectFaceImage(frame[rect[1]:rect[3], rect[0]:rect[2]], detector_backend='mtcnn', enforce_detection=True, target_size=(75,100))
                        aligned_faces, coords, angle = self.retinaface_detect(frame[rect[1]:rect[3], rect[0]:rect[2]], low=25, high=85)
                        if len(aligned_faces)==0:
                            continue
                    except:
                        continue
                    aligned_faces[0] = cv2.resize(aligned_faces[0],(75,100),interpolation = cv2.INTER_LINEAR)
                    if brisque.score(aligned_faces[0]) > 50 or angle[0]!='Frontal':
                        continue
                    target_embedding, _ = DeepFace.represent_face(aligned_faces[0], model)
                    matched_person = dict()
                    for matchPath in matches:
                        source_embedding = self.embeddings.loc[matchPath]['face_embedding']
                        distance = self.verify_face(source_embedding, target_embedding)
                        #if distance < threshold:
                        id = matchPath.split('_')[0] 
                        if id not in matched_person:
                            matched_person[id] = list()
                        matched_person[id].append(distance)
                    
                    if len(matched_person) > 0:
                        smallest = 1
                        matched = None
                        for id in matched_person.keys():
                            avg = sum(matched_person[id])/len(matched_person[id])
                            matched_person[id] = avg
                            if avg < smallest and avg <= threshold:
                                smallest = avg
                                matched = id

                        if matched is None:
                            res = 'The input image did not match with any face in the database!'
                            print(res)
                        else:
                            res = f'The input person matches with person id {matched} in the database!'
                            print(res)
                            #ln = len(os.listdir(path))
                            #cv2.imwrite(path+'\\'+str(ln)+'_'+str(matched)+'_matched'+'.jpg', face)
                            #cv2.imwrite(path+'\\'+str(ln)+'_'+str(matched)+'_original'+'.jpg', self.df.person.loc[matched]['face_img'])
                        results.append(res)
        except Exception as e:
            print(e)
            raise e
        finally:
            print(results)

    def verify_face_frame(self, frame, model='ArcFace', threshold=0.1):
        results = list()
        oframe = frame.copy()
        rects = self.run_yolo_on_frame(frame)
        for rect in rects:
            #cv2.imshow('f', frame[rect[1]:rect[3], rect[0]:rect[2]])
            #cv2.waitKey()
            matches = self.verify_person(frame[rect[1]:rect[3], rect[0]:rect[2]])
            try:
                #face = DeepFace.detectFaceImage(frame[rect[1]:rect[3], rect[0]:rect[2]], detector_backend='mtcnn', enforce_detection=True, target_size=(75,100))
                aligned_faces, coords, angle = self.retinaface_detect(frame[rect[1]:rect[3], rect[0]:rect[2]], low=25, high=85)
                if len(aligned_faces)==0:
                    continue
            except:
                continue
            aligned_faces[0] = cv2.resize(aligned_faces[0],(75,100),interpolation = cv2.INTER_LINEAR)
            if brisque.score(aligned_faces[0]) > 50 or angle[0]!='Frontal':
                continue
            target_embedding, _ = DeepFace.represent_face(aligned_faces[0], model)
            matched_person = dict()
            for matchPath in matches:
                source_embedding = self.embeddings.loc[matchPath]['face_embedding']
                distance = self.verify_face(source_embedding, target_embedding)
                #if distance < threshold:
                id = matchPath.split('_')[0]+'_'+ matchPath.split('_')[1]
                if id not in matched_person:
                    matched_person[id] = list()
                matched_person[id].append(distance)
            
            if len(matched_person) > 0:
                smallest = 1
                matched = None
                for id in matched_person.keys():
                    avg = sum(matched_person[id])/len(matched_person[id])
                    matched_person[id] = avg
                    if avg < smallest and avg <= threshold:
                        smallest = avg
                        matched = id

                #if matched is None:
                    #res = 'The input image did not match with any face in the database!'
                    #print(res)
                if matched is not None:
                    #res = f'The input person matches with person id {matched} in the database!'
                    #print(res)
                    cv2.putText(oframe, f'id:{matched}', (rect[0],rect[3]),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)
                    #ln = len(os.listdir(path))
                    #cv2.imwrite(path+'\\'+str(ln)+'_'+str(matched)+'_matched'+'.jpg', face)
                    #cv2.imwrite(path+'\\'+str(ln)+'_'+str(matched)+'_original'+'.jpg', self.df.person.loc[matched]['face_img'])
                #results.append(res)
                #cv2.putText(oframe, f'id:{matched}', (rect[0],rect[3]),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)
        return oframe

    def convert_frame_to_bino(frame):
        retval, buffer = cv2.imencode('.jpg', frame)
        return base64.b64encode(buffer).decode("utf-8")

    def verify_face(self, source, target):
        distance = dst.findCosineDistance(source, target)
        return distance

    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('yolopath', help='Path to yolo model directory', default=r"C:\Projects\Emotyx\TrainedModels\People_tracking")
    parser.add_argument('gpu', help='cpu if 0 else gpu', default=0)
    parser.add_argument('personmodel', help='Path to person features extractor', default=r'C:\Users\Accubits\Downloads\osnet_ain_x1_0.pth')
    parser.add_argument('mongoclient', help='MongoDB connection string', default='mongodb://localhost:27017')
    #args = parser.parse_args()
    reid = FaceReid(r"C:\Projects\Emotyx\TrainedModels\People_tracking",0,r'C:\Users\Accubits\Downloads\osnet_ain_x1_0.pth','mongodb://localhost:27017')
    reid.work_on_video()     


#main()   
