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

#person is 0
#sports ball is 32
results = model(img)

class_thresholds = {0: 0.75, 32:0.29} #Example thresholds for class 0 - person and 32 sports ball
filtered_boxes = []

boxes = results[0].boxes
classes = results[0].boxes.cls
confidences = results[0].boxes.conf

#Logic to filter down person in question - the main person and lets filter down the sports balls

for i, (cls, conf, box) in enumerate(zip(classes,confidences, boxes)):
    if conf > class_thresholds.get(int(cls),0.25):
        filtered_boxes.append(box)


for box in filtered_boxes:
    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(np.int32)
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color=(255,0,0), thickness=2)

cv2.imshow("Filtered Detections", img)
cv2.waitKey(0)
cv2.destroyAllWindows()



#person = results[0].masks.xy[0]
#ball = results[0].masks.xy[8]


#print("person:", person[62][0])
#print("ball:", ball[2][0])

#print(person[62][0] < ball[2][0] or ball[2][0] < person[62][0])

#print("person:", person[62][1])
#print("ball:", ball[2][1])



#if (person[62][1] < ball[2][1] or ball[2][1] < person[62][1]) == True:
#    print( "There is a touch")



#Need to find the best way to programmatically come up with these coordinates

#print("person xyxy boxes:", results[0].boxes.xyxy[0])
#print("ball xyxy boxes:", results[0].boxes.xyxy[8])

#print("person xy masks:",results[0].masks.xy[0])
#print("ball xy masks:", results[0].masks.xy[8])


#person_x = person[0]
#person_y = person[1]
#person_w = person[2]
#person_h = person[3]


#xmin= int(person_x - person_w / 2)
#xmax = int(person_x + person_w / 2)
#ymin = int(person_y - person_h / 2)
#ymax = int(person_y + person_h / 2)

#print(xmin)
#print(xmax)
#print(ymin)
#print(ymax)

#ball_x = ball[0]
#ball_y = ball[1]
#ball_w = ball[2]
#ball_h = ball[3]

#print(ball_x)
#print(ball_y)
#print(ball_w)
#print(ball_h)




#plt.title("Line graph")
#plt.xlabel("X axis")
#plt.ylabel("Y axis")
#plt.plot(*zip(*person), marker='o', color='r', ls='')
#plt.plot(*zip(*ball), marker='o', color='b', ls='')
#plt.show()

