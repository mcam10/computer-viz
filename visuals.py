from ultralytics import YOLO
import cv2
import torch
import pandas as pd
import numpy as np
from pathlib import Path

# Load a model
#model = YOLO("yolov8m-pose.pt")  # load an official model
#model = torch.hub.load("ultralytics/yolov8", "yolov8m-seg")
model = YOLO("yolov8m-seg.pt")  # load an official model

#Add img or Video as a var to pass to cv2
img = cv2.imread("/Users/mcameron/yolo/20240904140832_MP4-0016.jpg")

#results = model.predict(img)
results = model.predict(img, save_crop=True)

for result in results:
    orig = np.copy(result.orig_img)
    img_name = Path(result.path).stem

    #Iterate each pbject contour
    for ci, c in enumerate(result):
        label = c.names[c.boxes.cls.tolist().pop()]

        b_mask = np.zeros(orig.shape[:2], np.uint8)

        #create contour mask

        contour = c.masks.xy.pop().astype(np.int32).reshape(-1,1,2)
        _ = cv2.drawContours(b_mask, [contour], -2, (255, 255, 255), cv2.FILLED)

        mask3ch = cv2.cvtColor(b_mask, cv2.COLOR_GRAY2BGR)
        isolated = cv2.bitwise_and(mask3ch, orig)


        _ = cv2.imwrite(f"{img_name}_{label}-{ci}.png", isolated)



#df_result = results[0].to_df()

#print(df_result.loc[[0,8]])

#print(results.masks)

#orig = np.copy(results[0].orig_img)
#binary_mask = np.zeros(orig.shape[:2],np.uint8)
#contour = results[0].masks.xy[0].astype(np.int32).reshape(-1,1,2)
#cv2.drawContours(binary_mask, [contour], -1, (255, 255, 255), cv2.FILLED)

#cv2.imshow('binary_mask', binary_mask)


#results.print()
## Get specific keypoint of leg and ball and see how we can track touches
## Maybe play around with these specific keypoints
## Track keypoint in relation to the ball
