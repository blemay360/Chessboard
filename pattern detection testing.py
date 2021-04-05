import cv2
import numpy as np
from matplotlib import pyplot as plt

img_rgb = cv2.imread('CelticKnotTesting2Shrunk.jpg')
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
template = cv2.imread('CelticKnotTesting1Strip.jpg')
template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
w, h = template.shape[::-1]

res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF)
threshold = 3000000
loc = np.where( res >= threshold)
for pt in zip(*loc[::-1]):
    cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (255,0,0), 2)

cv2.imshow('Result',img_rgb)
cv2.waitKey(0)
