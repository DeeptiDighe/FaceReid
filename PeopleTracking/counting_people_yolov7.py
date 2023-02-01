from typing import OrderedDict
import cv2 as cv
import argparse
import sys
import numpy as np
import os.path
import os
import math
from utils.centroidtracker import CentroidTracker
from utils.trackableobject import TrackableObject

class Tracker:

    def __init__(self):
        # Initialize the parameters
        self.confThreshold = 0.6  #Confidence threshold
        self.nmsThreshold = 0.4   #Non-maximum suppression threshold

        # Give the configuration and weight files for the model and load the network using them.
        modelConfiguration = os.path.join(os.getcwd(), "yolov3.cfg")
        modelWeights = os.path.join(os.getcwd(), "yolov3.weights")
        # load our serialized model from disk
        print("[INFO] loading model...")
        self.net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
        #net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
        #net.setPreferableTarget(cv.dnn.DNN_TARGET_OPENCL)

        # Get the video writer initialized to save the output video
        #self.vid_writer = outputFile

        # instantiate our centroid tracker, then initialize a list to store
        # each of our dlib correlation trackers, followed by a dictionary to
        # map each unique object ID to a TrackableObject
        self.ct = CentroidTracker(maxDisappeared=5, maxDistance=50)

        # Process inputs
        #self.winName = 'People Tracking System'
        #cv.namedWindow(self.winName, cv.WINDOW_NORMAL)

        self.ln = self.net.getLayerNames()
        self.ln = [self.ln[i - 1] for i in self.net.getUnconnectedOutLayers()]

    # Remove the bounding boxes with low confidence using non-maxima suppression
    def postprocess(self, frame, outs):
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]

        rects = []

        # Scan through all the bounding boxes output from the network and keep only the
        # ones with high confidence scores. Assign the box's class label as the class with the highest score.
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
        indices = cv.dnn.NMSBoxes(boxes, confidences, self.confThreshold, self.nmsThreshold)
        registeredObjects = OrderedDict()
        if len(indices) > 0:
            for i in indices.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                rects.append((x,y,x+w,y+h))

        objects = self.ct.update(rects)
        for object in objects:
            x1,y1,x2,y2 = objects[object]
            cv.rectangle(frame, (x1,y1), (x2,y2), color=(255,0,0))
            registeredObjects[object] = objects[object]

        return registeredObjects

    def counting(self, frame, objects):
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]

        global totalDown
        global totalUp

        # loop over the tracked objects
        for (objectID, centroid) in objects.items():
            # check to see if a trackable object exists for the current
            # object ID
            to = self.trackableObjects.get(objectID, None)
    
            # if there is no existing trackable object, create one
            if to is None:
                to = TrackableObject(objectID, centroid)
    
            # otherwise, there is a trackable object so we can utilize it
            # to determine direction
            else:
                # the difference between the y-coordinate of the *current*
                # centroid and the mean of *previous* centroids will tell
                # us in which direction the object is moving (negative for
                # 'up' and positive for 'down')
                y = [c[1] for c in to.centroids]
                direction = centroid[1] - np.mean(y)
                to.centroids.append(centroid)
    
                # check to see if the object has been counted or not
                if not to.counted:
                    # if the direction is negative (indicating the object
                    # is moving up) AND the centroid is above the center
                    # line, count the object

                    if direction < 0 and centroid[1] in range(frameHeight//2 - 30, frameHeight//2 + 30):
                        totalUp += 1
                        to.counted = True
    
                    # if the direction is positive (indicating the object
                    # is moving down) AND the centroid is below the
                    # center line, count the object
                    elif direction > 0 and centroid[1] in range(frameHeight//2 - 30, frameHeight//2 + 30):
                        totalDown += 1
                        to.counted = True
    
            # store the trackable object in our dictionary
            self.trackableObjects[objectID] = to
            # draw both the ID of the object and the centroid of the
            # object on the output frame
            #text = "ID {}".format(objectID)
            #cv.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                #cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
        # construct a tuple of information we will be displaying on the
        # frame
        info = [
            ("Up", totalUp),
            ("Down", totalDown),
        ]

        # loop over the info tuples and draw them on our frame
        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv.putText(frame, text, (10, frameHeight - ((i * 20) + 20)),
                cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    def process(self, frame):
        oframe = frame.copy()
        # Create a 4D blob from a frame.
        blob = cv.dnn.blobFromImage(frame, 1/255, (416, 416), [0,0,0], 1, crop=False)

        # Sets the input to the network
        self.net.setInput(blob)

        # Runs the forward pass to get output of the output layers
        outs = self.net.forward(self.ln)

        # Remove the bounding boxes with low confidence
        registeredObjects = self.postprocess(oframe, outs)

        # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
        t, _ = self.net.getPerfProfile()
        label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
        cv.putText(oframe, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

        # Write the frame with the detection boxes

        #self.vid_writer.write(oframe.astype(np.uint8))

        #cv.imshow(self.winName, oframe)

        return registeredObjects
