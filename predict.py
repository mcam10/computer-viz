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
video_path = "/Users/mcameron/yolo/20240904140832.MP4"
cap = cv2.VideoCapture(video_path)


#Global objects                        
class_thresholds = {0: 0.75, 32:0.29} #Example thresholds for class 0 - person and 32 sports ball
filtered_boxes = []

def check_box_overlap(box1, box2):
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    if x2_1 < x1_2 or x2_2 < x1_1:
        return False

    if y2_1 < y1_2 or y2_2 < y1_1:
        return False

    return True

#loop through video frames                        
while cap.isOpened():
  ## Read a frame from the video
  success, frame = cap.read()

  if success:
     # Run YOLOv8 inference on the frame
     results = model(frame)
     boxes = results[0].boxes
     classes = results[0].boxes.cls
     confidences = results[0].boxes.conf
     annotated_frame = results[0].plot()
     collided = []
     # Display the annotated frame
     cv2.imshow("YOLOv8 Inference", annotated_frame)

     ## logic to filter down person and ball in question
     for i, (cls, conf, box) in enumerate(zip(classes,confidences, boxes)):
         if conf > class_thresholds.get(int(cls),0.25):
              filtered_boxes.append(box)
      
     for box in filtered_boxes:
         x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(np.int32)
         cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color=(255,0,0), thickness=2)

     x = filtered_boxes[0].xywh[0][0]
     y = filtered_boxes[0].xywh[0][1]
     width = filtered_boxes[0].xywh[0][2]
     height = filtered_boxes[0].xywh[0][3]
     person = filtered_boxes[0].cls

     x1_1 = int(x - width / 2)
     x2_1 = int(x + width / 2)
     y1_1 = int(y - height / 2)
     y2_1 = int(y + height / 2)
     #sports ball is the last item in the array hardcoding for now
     x_1 = filtered_boxes[-1].xywh[0][0]
     y_1 = filtered_boxes[-1].xywh[0][1]
     width_2 = filtered_boxes[-1].xywh[0][2]
     height_2 = filtered_boxes[-1].xywh[0][3]
     ball = filtered_boxes[-1].cls
     x1_2 = int(x_1 - width_2 / 2)
     x2_2 = int(x_1 + width_2 / 2)
     y1_2 = int(y_1 - height_2 / 2)
     y2_2 = int(y_1 + height_2 / 2)
     if check_box_overlap((x1_1, y1_1, x2_1, y2_1),(x1_2, y1_2, x2_2, y2_2)) == True:
         print("Theres a touch")

     if cv2.waitKey(1) & 0xFF == ord("q"):
         break
  else:
      # break the loop if the end of the video
      break
cap.release()
cv2.destroyAllWindows()

