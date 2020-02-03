import cv2
import numpy as np

image = cv2.imread('image.jpg')
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
cv2.imshow('HSV Image',hsv)
cv2.waitKey(0)
lower_range = np.array([110, 50, 50])
upper_range = np.array([130, 255, 255])
mask = cv2.inRange(hsv, lower_range, upper_range)
cv2.imshow('image_window_name', image)
cv2.imshow('mask_window_name', mask)
cv2.waitKey(0)
cv2.destroyAllWindows()