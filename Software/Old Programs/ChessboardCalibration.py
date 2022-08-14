# Importing the OpenCV library 
import cv2
import numpy as np
# Reading the image using imread() function 
image = cv2.imread('chessboardAprilTag.jpg') 

import matplotlib.pyplot as plt 
#%matplotlib inline 
 
lt = [ 367 , 1383]
rt = [1765, 1366]
lb = [ 298, 2833]
rb = [1969, 2752]

mask = np.zeros(image.shape[0:2], dtype=np.uint8)
points = np.array([[lt, rt, rb, lb]])

cv2.drawContours(mask, [points], -1, (255, 255, 255), -1, cv2.LINE_AA)
#cv2.fillPoly(mask, points, (255))

res = cv2.bitwise_and(image,image,mask = mask)
rect = cv2.boundingRect(points) # returns (x,y,w,h) of the rect
cropped = res[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]
## crate the white background of the same size of original image
wbg = np.ones_like(image, np.uint8)*255
cv2.bitwise_not(wbg,wbg, mask=mask)
# overlap the resulted cropped image on the white background
dst = wbg+res

#cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
#cv2.resizeWindow('Original', 800,1200)
#cv2.imshow('Original',image)

#cv2.namedWindow('Mask', cv2.WINDOW_NORMAL)
#cv2.resizeWindow('Mask', 800,1200)
#cv2.imshow("Mask",mask)

#cv2.namedWindow('Cropped', cv2.WINDOW_NORMAL)
#cv2.resizeWindow('Cropped', 800,1200)
#cv2.imshow("Cropped", cropped )

#cv2.namedWindow('Samed Size Black Image', cv2.WINDOW_NORMAL)
#cv2.resizeWindow('Samed Size Black Image', 800,1200)
#cv2.imshow("Samed Size Black Image", res)

#cv2.namedWindow('Samed Size White Image', cv2.WINDOW_NORMAL)
#cv2.resizeWindow('Samed Size White Image', 800,1200)
#cv2.imshow("Samed Size White Image", dst)

#cv2.waitKey(0)
#cv2.destroyAllWindows()

# convert image to grayscale 
gray_img = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY) 
  
# Shi-Tomasi corner detection function 
# We are detecting only 100 best corners here 
# You can change the number to get desired result. 
corners = cv2.goodFeaturesToTrack(gray_img, 81, 0.01, 10)
  
# convert corners values to integer 
# So that we will be able to draw circles on them 
corners = np.int0(corners)
  
# draw red color circles on all corners 
for i in corners: 
    x, y = i.ravel() 
    cv2.circle(cropped, (x, y), 20, (255, 0, 0), -1) 

# resulting image 
cv2.namedWindow('final', cv2.WINDOW_NORMAL)
cv2.resizeWindow('final', 800,1200)
cv2.imshow("final", cropped) 
  
# De-allocate any associated memory usage   
if cv2.waitKey(0) & 0xff == 27:  
    cv2.destroyAllWindows() 
