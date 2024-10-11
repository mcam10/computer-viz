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
img = cv2.imread("/Users/mcameron/yolo/20240904140832_MP4-0016.jpg")

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

#plotting

plt.title("Line graph")
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.plot(*zip(*person), marker='o', color='r', ls='')
plt.plot(*zip(*ball), marker='o', color='b', ls='')

plt.show()
