# Importing the OpenCV library 
import cv2, copy, math
import numpy as np
from apriltag import apriltag

def detect_apriltags(family, image):
    #Only needs to be done once, but for ease of coding we'll do it every function call
    detector = apriltag(family)
    
    #Convert input image to grayscale
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 

    #Look for apriltags
    detections = detector.detect(gray_img)
    
    #Return variable with detected apriltag info
    return detections

def parse_april_tag_coordinate(detections, tag_id, corner ='center'):
    if (corner == 'center'):
        x = detections[tag_id]['center'][0]
        y = detections[tag_id]['center'][1]
    else:
        corner_dict = { 'lb':0, 'rb':1, 'rt':2, 'lt':3 }
        x = detections[tag_id]['lb-rb-rt-lt'][corner_dict[corner]][0]
        y = detections[tag_id]['lb-rb-rt-lt'][corner_dict[corner]][1]
    output = (int(round(x)), int(round(y)))
    return output

def apriltag_area(detections, tag_id):
    positive = 0
    negative = 0
    for i in range(4):
        positive += detections[tag_id]['lb-rb-rt-lt'][i][0] * detections[tag_id]['lb-rb-rt-lt'][(i+1) % 4][1]
        negative += detections[tag_id]['lb-rb-rt-lt'][(i+1) % 4][0] * detections[tag_id]['lb-rb-rt-lt'][i][1]
    area = 0.5 * (positive - negative)
    return area

def array_to_tuple(array, num_arrays=1):
    if (num_arrays > 1):
        output = [0] * num_arrays
        for i in range(num_arrays):
            x = int(array[i][0])
            y = int(array[i][1])
            output[i] = (x, y)
        return output
    else:
        x = int(array[0])
        y = int(array[1])
        return (x, y)
    
def grab_inside_corners(detections):
    lt = parse_april_tag_coordinate(detections, 0, 'rb')
    rt = parse_april_tag_coordinate(detections, 1, 'lb')
    rb = parse_april_tag_coordinate(detections, 3, 'lt')
    lb = parse_april_tag_coordinate(detections, 2, 'rt')
    return lb, rb, rt, lt

def copy_image(image, num_copies):
    output = (copy.copy(image),)
    for i in range(num_copies - 1):
        output = output + (copy.copy(image),)
    return output

def crop_image(image, corners):
    mask = np.zeros(image.shape[0:2], dtype=np.uint8)
    points = np.array(corners)
    
    cv2.drawContours(mask, [points], -1, (255, 255, 255), -1, cv2.LINE_AA)
    #cv2.fillPoly(mask, points, (255))

    res = cv2.bitwise_and(image,image,mask = mask)
    rect = cv2.boundingRect(points) # returns (x,y,w,h) of the rect]
    
    cropped = res[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]
    ## crate the white background of the same size of original image
    wbg = np.ones_like(image, np.uint8)*255
    cv2.bitwise_not(wbg,wbg, mask=mask)
    # overlap the resulted cropped image on the white background
    dst = wbg+res
    
    return cropped

def project_corners_in_from_apriltags(detections):
    projected_corners = [0] * 4
    corner_dict = {0:('lt', 'rb'), 1:('rt', 'lb'), 2:('lb', 'rt'), 3:('rb', 'lt')}
    april_tag_order = [0, 1, 3, 2]
    for i in april_tag_order:
        outside_corner = parse_april_tag_coordinate(detections, i, corner_dict[i][0])
        inside_corner = parse_april_tag_coordinate(detections, i, corner_dict[i][1])
        distance_to_move = 1.125 * math.sqrt((outside_corner[0] - inside_corner[0])**2 + (outside_corner[1] - inside_corner[1])**2)

        if (outside_corner[0] == inside_corner[0]):
            if ((inside_corner[1] - outside_corner[1]) > 0):
                angle = math.pi/2
            else:
                angle = -math.pi/2
        else:
            angle = math.atan((inside_corner[1] - outside_corner[1])/(inside_corner[0] - outside_corner[0]))
            
        if (outside_corner[0] <= inside_corner[0]):
            x = int(round(outside_corner[0] + distance_to_move * math.cos(angle)))
            y = int(round(outside_corner[1] + distance_to_move * math.sin(angle)))
        else:
            x = int(round(outside_corner[0] - distance_to_move * math.cos(angle)))
            y = int(round(outside_corner[1] - distance_to_move * math.sin(angle)))
        projected_corners[april_tag_order[i]] = (x,y)
    return projected_corners

def measure_distance(pt1, pt2):
    return int(math.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2))

def perspective_shift(image, corners):
    distance = max(measure_distance(corners[0], corners[1]), measure_distance(corners[1], corners[2]), measure_distance(corners[2], corners[3]), measure_distance(corners[3], corners[0]))
        
    pts1 = np.float32(corners)
    pts2 = np.float32([[0,0],[distance,0],[0,distance],[distance,distance]])
    
    M = cv2.getPerspectiveTransform(pts1,pts2)
    image = cv2.warpPerspective(image, M, (distance,distance))
    
    return image

def poll_chessboard(image):
    color = 'green'
    ratio = 'picture'
    size = image.shape[0]
    pixel = size / 82
    corners = np.empty((9, 9), dtype=object)
    for y in range(9):
        for x in range(9):
            corners[y][x] = (int(round(10 * pixel * x + pixel)), int(round(10 * pixel * y + pixel)))
            #corners[y][x] = (x*2, y*4)
    
    blue = (255, 0, 0)
    thickness = 5
    pixel = int(round(pixel))
    harris_corners = np.zeros((8, 8), dtype=int)
    for y in range(8):
        for x in range(8):
            start_point = (corners[x][y][0] + pixel, corners[x][y][1] + pixel)
            end_point = (corners[x+1][y+1][0] - pixel, corners[x+1][y+1][1] - pixel)
            #image = cv2.rectangle(image, start_point, end_point, blue, thickness)
            #print(x, y, corners[y][x])
            #harris_corners[y][x] = average_colors(image[start_point[0]:end_point[0], start_point[1]:end_point[1]])
            #harris_corners[y][x] = corner_detection_Harris(image[start_point[0]:end_point[0], start_point[1]:end_point[1]])
            harris_corners[y][x] = edge_detection(image[start_point[0]:end_point[0], start_point[1]:end_point[1]])
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
            #corner_detection_Harris(image[start_point[0]:end_point[0], start_point[1]:end_point[1]])
        print(harris_corners[y])
    return image

def corner_detection_ShiTomasi(image, ratio):
    # convert image to grayscale 
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
  
    # Shi-Tomasi corner detection function 
    # We are detecting only 81 best corners here 
    corners = cv2.goodFeaturesToTrack(gray_img, 79, 0.01, 10)
  
    # convert corners values to integer 
    # So that we will be able to draw circles on them 
    corners = np.int0(corners)
  
    output = copy.copy(image)
  
    # draw blue color circles on all corners 
    for i in corners: 
        x, y = i.ravel()
        circle_image(output, (x, y), 'blue', ratio)
        #cv2.circle(output, (x, y), radius, blue, thickness)
        
    return output

def corner_detection_Harris(frame):
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray,11)
    gray = np.float32(gray)
    output = cv2.cornerHarris(gray,2,3,0.04)
    
    ##result is dilated for marking the corners, not important
    output = cv2.dilate(output,None)
    ## Threshold for an optimal value, it may vary depending on the image.
    frame[output>0.1*output.max()]=[255,0,0]
    
    cv2.imshow('Edges', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return len(output)

def average_colors(image):
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(img_hsv)
    #cv2.imshow("Hue", hue)
    #cv2.imshow("Saturation", saturation)
    #cv2.imshow("Value", value)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    average = 0
    for y in range(hue.shape[1]):
        for x in range(hue.shape[0]):
            average += hue[x, y]
    average = average // hue.size
    return average

def edge_detection(img):
    img = cv2.medianBlur(img,11)
    edges = cv2.Canny(img,80,100)
    return np.sum(edges)

def circle_image(image, coordinates, color, ratio):
    #Color definition dictionary
    color_dict = {'blue':(255, 0, 0), 'green':(0, 255, 0), 'red':(0, 0, 255)}
    
    # Line thickness of 2 px
    thickness = -1

    if (ratio == 'picture'):
        radius = 12
    else:
        radius = 3

    if (len(coordinates) > 2):
        for i in range(len(coordinates)):
            cv2.circle(image, coordinates[i], radius, color_dict[color], thickness)
    else:
        cv2.circle(image, coordinates, radius, color_dict[color], thickness)
            
def show_images(*arg):
    if (arg[0] == 'resize'):
        for i in range(1, len(arg)):
            cv2.namedWindow(arg[i][0], cv2.WINDOW_NORMAL)
            cv2.resizeWindow(arg[i][0], 800,1200)
            cv2.imshow(arg[i][0], arg[i][1])
    else:
        for i in range(len(arg)):
            cv2.imshow(arg[i][0], arg[i][1])

def video_loop_method():
    #Connect to camera
    #for i in range(100000):
    cam = cv2.VideoCapture(1)
    #if (cam.isOpened()):
        #print(i)
        #else:
            #print('Could not open video device')
    
    while True:
        #Reading a frame from the camera 
        [ok, frame] = cam.read()
        while(ok == False):
            pass
        
        #Run apriltag detection on image
        detections = detect_apriltags("tag36h11", frame)
        
        #Make copies of original image to keep the same for display 
        apriltagCorners, sectioned, cropped, final = copy_image(frame, 4)
            
        if(len(detections) == 4):
            #Get values for the inside corners of each 4 apriltags
            lb, rb, rt, lt = grab_inside_corners(detections)

            #Place circles on inside corners of each apriltag
            circle_image(apriltagCorners, (lt, rt, rb, lb), 'red', 'video')
            
            #Crop image using inside corners
            cropped = crop_image(frame, [[lt, rt, rb, lb]])
            
            section_image(sectioned, (lt, rt, rb, lb), 'video')
        
            #Run corner detection on cropped image
            final = corner_detection_ShiTomasi(cropped, 'video')

        show_images(('Original', frame), ('Sectioned', sectioned), ('Final', final))

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    cv2.destroyAllWindows()
    
    cam.release()

def picture_method(filename):
    #Reading the image using imread() function 
    image = cv2.imread(filename)
    
    #Run apriltag detection on image
    detections = detect_apriltags("tag36h11", image)
    
    #Make copies of original image to keep the same for display 
    apriltagCorners, shifted, cropped, final = copy_image(image, 4)
            
    #If 4 apriltags are seen (one in each 4 corners of chessboard), crop image and run detection
    if(len(detections) == 4):
        #Get values for the inside corners of each 4 apriltags
        lb, rb, rt, lt = grab_inside_corners(detections)
        
        #Place circles on inside corners of each apriltag
        circle_image(apriltagCorners, (lt, rt, rb, lb), 'red', 'picture')
        
        shifted = perspective_shift(shifted, (lt, rt, lb, rb))
                
        shifted = poll_chessboard(shifted)
        
        #Crop image using inside corners
        #cropped = crop_image(image, [[lt, rt, rb, lb]])
        
        #cropped = copy_image(perspectiveShift, 1)
                
        #Run corner detection on cropped image
        #final = corner_detection_ShiTomasi(shifted, 'picture')

    #Display output images
    show_images('resize', ('Original', image), ('Perspective shifted', shifted))

    #Wait for use to press key
    cv2.waitKey(0)
    
    #Delete all image windows
    cv2.destroyAllWindows()

#video_loop_method()
picture_method('chessboardAprilTagWhiteBorderOnePiece.jpg')
