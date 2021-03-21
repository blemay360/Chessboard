import cv2, imutils
import numpy as np
from apriltag import apriltag

#------------------------------------------------SETUP-------------------------------------------------
  
detector = apriltag("tag36h11")  

# Red color in BGR
color = (0, 0, 255)

# Line thickness of 2 px
thickness = -1

#------------------------------------------------LIVE VIDEO-------------------------------------------------

# Radius of circle
radius = 3

#cap = cv2.VideoCapture(0)

print('Switch to images. Then press q key to stop')

while False:
    [ok, frame] = cap.read()

    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 

    detections = detector.detect(gray_img)
    
    output = frame
    
    for i in range(len(detections)):
        center_coordinates = (int(round(detections[i]["center"][0])), int(round(detections[i]["center"][1])))

        # Using cv2.circle() method
        # Draw a circle with blue line borders of thickness of 2 px
        cv2.circle(output, center_coordinates, radius, color, thickness)

    cv2.imshow('Original', frame)
    cv2.imshow('Gray', gray_img)
    cv2.imshow('Output', output)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()

#------------------------------------------------SINGLE IMAGE-------------------------------------------------

# Radius of circle
radius = 10

#imagepath = '/home/pi/Pictures/18in.jpg'
#imagepath = 'aprilTagImageBorders.jpg'
#imagepath = '/home/blemay360/1EB8-1359/on_four_wot.jpg'
imagepath = '/home/blemay360/Documents/chessboard-main/TestingImages/16bitApriltags_onBoard.jpg'
image = cv2.imread(imagepath)

#image = imutils.rotate(image, 180)

gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 

detector = apriltag("tag16h5")

detections = detector.detect(gray_img)

#print(detections[0]['lb-rb-rt-lt'][1])
print(detections)

colors = {0:(255, 0, 255), 1:(0, 0, 255), 2:(0, 255, 0), 3:(255, 0, 0)}

for i in range(len(detections)):
    if (detections[i]['margin'] > 50):
        center_coordinates = (int(round(detections[i]["lb-rb-rt-lt"][0][0])), int(round(detections[i]["lb-rb-rt-lt"][0][1])))
    
        # Using cv2.circle() method
        # Draw a circle with blue line borders of thickness of 2 px
        image = cv2.circle(image, center_coordinates, radius, colors[i], thickness)

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', 800,1200)
#cv2.resizeWindow('image', 200, 200)

cv2.imshow("image", image)

if cv2.waitKey(0) & 0xff == 27:  
    cv2.destroyAllWindows() 

