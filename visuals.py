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

results = model(img)

results[0].show()

df_result = results[0].to_df()

print(df_result.loc[[0,8]])


numpy_arr = np.copy(results[0].orig_img) 
b_mask = np.zeros(numpy_arr.shape[:2], np.uint8)
contour = results[0].masks.xy[0].astype(np.int32).reshape(-1, 1, 2)
_ = cv2.drawContours(b_mask, [contour], -1, (255, 255, 255), cv2.FILLED)

## isolate the object using the binary mask
mask3ch = cv2.cvtColor(b_mask, cv2.COLOR_GRAY2BGR)
isolated = cv2.bitwise_and(mask3ch, numpy_arr)

cv2.imwrite(f"{numpy_arr}.png", isolated)

# Convert results to a list of dictionaries and then to a DataFrame
#data = [{'xmin': box[0], 'ymin': box[1], 'xmax': box[2], 'ymax': box[3],
#         'confidence': box[4], 'class': box[5]} for box in results[0].masks.xyn]
#df = pd.DataFrame(data)
#print(df)

#for cords,elem in enumerate(results[0].masks.xyn):
#    print(cords, elem)

#results.print()
## Get specific keypoint of leg and ball and see how we can track touches
## Maybe play around with these specific keypoints
## Track keypoint in relation to the ball
