from ultralytics import YOLO
import cv2
import torch
import pandas as pd
from matplotlib import pyplot as plt

import numpy as np

# Load a model
#model = YOLO("yolov8m-pose.pt")  # load an official model
#model = torch.hub.load("ultralytics/yolov8", "yolov8m-seg")
model = YOLO("yolov8m-seg.pt")  # load an official model

#Add img or Video as a var to pass to cv2
img = cv2.imread("/home/gear/yolo/20240904140832_MP4-0016.jpg")

# Saves to /runs/predict/img
#model.predict(img, save=True)
#results = model(img)

#run inference on the image
#https://docs.ultralytics.com/modes/predict/#inference-sources
#results = model(img, conf=0.25)
results = model(img)
person_points = {}
person = results[0].masks.xy[0]
ball = results[0].masks.xy[8]

#print(person)
#print(ball)
plt.imshow(person)
plt.colorbar()
plt.show()

#next i need to figure out whats the best representation for a intersection of two points aka player and ball
## either masks or boxes classes respectively


#we have the box of the ball
#print(results[0].boxes.xyxy[0])

##(results[0].boxes.xyxy[8])


#for idx,point in enumerate(person):
#    print(idx,point)

#print(results[0])
#xy_cords = masks.xy

#print(len(xy_cords))  # Number of masks
#print(xy_cords[0].shape)


# Convert results to a list of dictionaries and then to a DataFrame
#data = [{'xmin': box[0], 'ymin': box[1], 'xmax': box[2], 'ymax': box[3],
#         'confidence': box[4], 'class': box[5]} for box in results[0].masks.xyn]
#df = pd.DataFrame(data)
#print(df)

