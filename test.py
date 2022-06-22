import cv2
from cv2 import moments 
import numpy as np

image = cv2.imread('0.png')
lower = [120, 0, 0]
upper = [255, 50, 50]
lower = np.array(lower, dtype="uint8")
upper = np.array(upper, dtype="uint8")

mask = cv2.inRange(image, lower, upper)
m = cv2.moments(mask)
x = int(m["m10"] / m["m00"])
y = int(m["m01"] / m["m00"])
cv2.circle(image, (x, y), 1, (255, 255, 255), -1)
print(x, y)
cv2.imshow('', image)
cv2.waitKey(0)
