# Importing the OpenCV library 
import cv2
import numpy as np
# Reading the image using imread() function 
img = cv2.imread('chessboardAprilTag.jpg') 

import matplotlib.pyplot as plt 
#%matplotlib inline 
  
lt = [ 369.8694458 , 1383.19482422]
rt = [1764.80993652, 1366.29919434]
lb = [ 298.13302612, 2833.08911133]
rb = [1969.47949219, 2751.95361328]

mask = np.zeros(img.shape[0:2], dtype=np.uint8)

# convert image to grayscale 
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
  
# Shi-Tomasi corner detection function 
# We are detecting only 100 best corners here 
# You can change the number to get desired result. 
corners = cv2.goodFeaturesToTrack(gray_img, 100, 0.01, 10) 
  
# convert corners values to integer 
# So that we will be able to draw circles on them 
corners = np.int0(corners) 
  
# draw red color circles on all corners 
for i in corners: 
    x, y = i.ravel() 
    cv2.circle(img, (x, y), 20, (255, 0, 0), -1) 
  
# resulting image 
cv2.namedWindow('final', cv2.WINDOW_NORMAL)
cv2.resizeWindow('final', 800,1200)
cv2.imshow("final", img) 
  
# De-allocate any associated memory usage   
if cv2.waitKey(0) & 0xff == 27:  
    cv2.destroyAllWindows() 
