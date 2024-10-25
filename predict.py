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

def check_box_overlap(box1, box2):
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    if x2_1 < x1_2 or x2_2 < x1_1:
        return False

    if y2_1 < y1_2 or y2_2 < y1_1:
        return False

    return True

## Person box
# Ball box

x = filtered_boxes[0].xywh[0][0]
y = filtered_boxes[0].xywh[0][1]
width = filtered_boxes[0].xywh[0][2]
height = filtered_boxes[0].xywh[0][3]

person = filtered_boxes[0].cls

x1_1 = int(x - width / 2)
x2_1 = int(x + width / 2)
y1_1 = int(y - height / 2)
y2_1 = int(y + height / 2)

x_1 = filtered_boxes[-1].xywh[0][0]
y_1 = filtered_boxes[-1].xywh[0][1]
width_2 = filtered_boxes[-1].xywh[0][2]
height_2 = filtered_boxes[-1].xywh[0][3]

ball = filtered_boxes[-1].cls
x1_2= int(x_1 - width_2 / 2)
x2_2 = int(x_1 + width_2 / 2)
y1_2 = int(y_1 - height_2 / 2)
y2_2 = int(y_1 + height_2 / 2)

if check_box_overlap((x1_1, y1_1, x2_1, y2_1),(x1_2, y1_2, x2_2, y2_2)) == True:
    print("Theres a touch")

collided = []


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

