# import the necessary packages
import numpy as np
import time
import cv2
import os
import argparse
from dominant_color import *
'''ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-y", "--yolo", required=True,
	help="base path to YOLO directory")
args = vars(ap.parse_args())'''
labelsPath='/home/chandana/Desktop/yolo-object-detection/yolo-coco (copy)/coco.names'
LABELS = open(labelsPath).read().strip().split("\n")
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")
weightsPath='/home/chandana/Desktop/yolo-object-detection/yolo-coco (copy)/yolov3.weights'
configPath = '/home/chandana/Desktop/yolo-object-detection/yolo-coco (copy)/yolov3.cfg'
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
image = cv2.imread("human.jpg")	#load image#
(H, W) = image.shape[:2]
ln = net.getLayerNames()	#get output layers#
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
	swapRB=True, crop=False)
net.setInput(blob)
start = time.time()
layerOutputs = net.forward(ln)
end = time.time()
print("[INFO] YOLO took {:.6f} seconds".format(end - start))
boxes = []
confidences = []
classIDs = []
p = 0
for output in layerOutputs:
	for detection in output:
		scores = detection[5:]
		classID = np.argmax(scores)
		confidence = scores[classID]
		if confidence > 0.6:
			p = p+1
			box = detection[0:4] * np.array([W, H, W, H])
			(centerX, centerY, width, height) = box.astype("int")
			x = int(centerX - (width / 2))
			y = int(centerY - (height / 2))
			boxes.append([x, y, int(width), int(height)])
			confidences.append(float(confidence))
			classIDs.append(classID)
idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)
coordinates = []
#x1 = []
#y1 = []
#w1 = []
#h1 = []
if len(idxs) > 0:
	for i in idxs.flatten():
		(x, y) = (boxes[i][0], boxes[i][1])
		(w, h) = (boxes[i][2], boxes[i][3])
		#print(x, y, w, h)
		color_return = color_dominant(x, y, w, h, image)
		'''x1.append(x)
		y1.append(y)
		w1.append(w)
		h1.append(h)
		coordinates = list(zip(x1, y1, w1, h1))'''

		# draw a bounding box rectangle and label on the image
		color = [int(c) for c in COLORS[classIDs[i]]]
		cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
		text = "{}: {:.4f} {}".format(LABELS[classIDs[i]], confidences[i], color_return)
		cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
			0.5, color, 2)
	#print("coordinates in main",coordinates)
	#color_dominant(coordinates, image)
print("no of objects found", p)
cv2.imshow("Image", image)
cv2.waitKey(0)
