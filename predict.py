from ultralytics import YOLO
import cv2
import torch
import pandas as pd

import numpy as np

# Load a model
#model = YOLO("yolov8m-pose.pt")  # load an official model
#model = torch.hub.load("ultralytics/yolov8", "yolov8m-seg")
model = YOLO("yolov8m-seg.pt")  # load an official model

#Add img or Video as a var to pass to cv2
img = cv2.imread("/Users/mcameron/yolo/20240904140832_MP4-0016.jpg")

# Saves to /runs/predict/img
#model.predict(img, save=True)
#results = model(img)

#run inference on the image
#https://docs.ultralytics.com/modes/predict/#inference-sources
#results = model(img, conf=0.25)
results = model.predict(img)

#print(results[0])
#xy_cords = masks.xy

#print(len(xy_cords))  # Number of masks
#print(xy_cords[0].shape)


# Convert results to a list of dictionaries and then to a DataFrame
#data = [{'xmin': box[0], 'ymin': box[1], 'xmax': box[2], 'ymax': box[3],
#         'confidence': box[4], 'class': box[5]} for box in results[0].masks.xyn]
#df = pd.DataFrame(data)
#print(df)


person = results[0].masks.xyn[0]
ball = results[0].masks.xyn[8]

#get predictions coords and do some math

#results.print()
## Get specific keypoint of leg and ball and see how we can track touches
## Maybe play around with these specific keypoints
## Track keypoint in relation to the ball
#can get the visual matched with the xyn coordinate array of each segmentation mask
#https://docs.ultralytics.com/reference/engine/results/#ultralytics.engine.results.Masks.xyn
#next step try to single out an array and visualize
