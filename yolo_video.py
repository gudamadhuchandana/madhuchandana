import numpy as np
import cv2
import time as t
import imutils
import os

labelsPath='/home/chandana/Desktop/yolo-object-detection/yolo-coco/coco.names'
LABELS = open(labelsPath).read().strip().split("\n")

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

weightsPath='/home/chandana/Desktop/yolo-object-detection/yolo-coco/yolov3.weights'
configPath = '/home/chandana/Desktop/yolo-object-detection/yolo-coco/yolov3.cfg'

print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

vs = cv2.VideoCapture('/home/chandana/Desktop/yolo-object-detection/videos/car_chase_01.mp4')
vs2 = 'car_chase_01.avi'
writer = None
(W, H) = (None, None)
try:
	prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
		else cv2.CAP_PROP_FRAME_COUNT
	total = int(vs.get(prop))
	print("[INFO] {} total frames in video".format(total))

except:
	print("[INFO] could not determine # of frames in video")
	print("[INFO] no approx. completion time can be provided")
	total = -1

while True:
	# read the next frame from the file
	(grabbed, frame) = vs.read()
	if not grabbed:
		break
	if W is None or H is None:
		(H, W) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)
	net.setInput(blob)
	start = t.time()
	layerOutputs = net.forward(ln)
	end = t.time()
	boxes = []
	confidences = []
	classIDs = []
	p = 0
	for output in layerOutputs:
		for detection in output:
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]
			if confidence > 0.5:
				if classID==2:
					p = p+1
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))
				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(classID)
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)
	if len(idxs) > 0:
		for i in idxs.flatten():
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])
			# draw a bounding box rectangle and label on the frame
			color = [int(c) for c in COLORS[classIDs[i]]]
			cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
			text = "{}: {:.4f}".format(LABELS[classIDs[i]],
				confidences[i])
			cv2.putText(frame, text, (x, y - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

	cv2.imshow("frame", frame)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break
	if writer is None:
		# initialize our video writer
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(vs2, fourcc, 30,
			(frame.shape[1], frame.shape[0]), True)
		if total > 0:
			elap = (end - start)
			print("[INFO] single frame took {:.4f} seconds".format(elap))
			print("[INFO] estimated total time to finish: {:.4f}".format(
				elap * total))
	writer.write(frame)
	time = vs.get(cv2.CAP_PROP_POS_MSEC)
	time = time / 1000
	frame_no = vs.get(cv2.CAP_PROP_POS_FRAMES)
	print("cars found", p, "frame no", frame_no, "at time", time)
print("[INFO] cleaning up...")
writer.release()
vs.release()