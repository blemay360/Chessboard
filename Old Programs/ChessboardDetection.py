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
    labeled_image = copy.copy(image)
    
    Files = "ABCDEFGH"
    
    blue = (255, 0, 0)
    green = (0, 255, 0)
    red = (0, 0, 255)
    thickness = 5
    ratio = 'picture'
    size = labeled_image.shape[0]
    pixel = size / 82
    edge_count = np.zeros((8, 8), dtype=int)
    occupied = np.zeros((8, 8), dtype=int)
    for y in range(8):
        for x in range(8):
            start_point = (int(round(2 * pixel + 10 * pixel * x)), int(round(2 * pixel + 10 * pixel * y)))
            end_point = (int(round(10 * pixel * (x + 1))), int(round(10 * pixel * (y + 1))))

            #cv2.putText(labeled_image, Files[x]+str(8-y), (int(start_point[0] + pixel*2), int(start_point[1] + pixel*4.5)), cv2.FONT_HERSHEY_COMPLEX, 2.0, blue, 4)
            
            labeled_image[start_point[1]:end_point[1], start_point[0]:end_point[0]], edge_count[y][x] = edge_detection(labeled_image[start_point[1]:end_point[1], start_point[0]:end_point[0]])
            
            #cv2.putText(labeled_image, '('+str(x)+','+str(y)+')', (int(start_point[0] + pixel*2.5), start_point[1] + pixel*4), cv2.FONT_HERSHEY_COMPLEX, 1.0, blue, 4)
            
            #cv2.putText(labeled_image, str(start_point), (int(start_point[0] + pixel*0.5), start_point[1] + pixel*4), cv2.FONT_HERSHEY_COMPLEX, 1.0, blue, 4)
            
            if (edge_count[y][x] > 5000):
                labeled_image = cv2.rectangle(labeled_image, start_point, end_point, red, thickness)
                occupied[y][x] = 1
            else:
                labeled_image = cv2.rectangle(labeled_image, start_point, end_point, green, thickness)
                occupied[y][x] = 0
    #print(occupied)
    return labeled_image, occupied

def edge_detection(img):
    img = cv2.medianBlur(img,11)
    edges = cv2.Canny(img,80,100)
    #cv2.imshow(str(np.sum(edges)), edges)
    #cv2.imshow("Original", img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return np.stack((edges,)*3, axis=-1), np.sum(edges)

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
        apriltagCorners, shifted, final = copy_image(frame, 3)
            
        if(len(detections) == 4):
            #Get values for the inside corners of each 4 apriltags
            lb, rb, rt, lt = grab_inside_corners(detections)

            #Place circles on inside corners of each apriltag
            circle_image(apriltagCorners, (lt, rt, rb, lb), 'red', 'picture')
            
            shifted = perspective_shift(shifted, (lt, rt, lb, rb))
                    
            final = poll_chessboard(shifted)

        show_images(('Original', frame), ('Shifted', shifted), ('Final', final))

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
        
        #Shift perspective to make to make the inside corners of the apriltags the corners of the image
        shifted = perspective_shift(shifted, (lt, rt, lb, rb))
        
        #Go through each square of the chessboard to tell if square is populated with piece
        final, occupied = poll_chessboard(shifted)
        
    #Display output images
    #show_images('resize', ('Original', image), ('Perspective shifted', shifted), ('Labeled Image', final))
    show_images('resize', ('Final', final))

    #Wait for use to press key
    cv2.waitKey(0)
    
    #Delete all image windows
    cv2.destroyAllWindows()

#video_loop_method()
picture_method('TestingImages/ItalianGame1.jpg')
picture_method('TestingImages/ItalianGame2.jpg')
picture_method('TestingImages/ItalianGame3.jpg')
picture_method('TestingImages/ItalianGame4.jpg')
picture_method('TestingImages/ItalianGame5.jpg')
picture_method('TestingImages/ItalianGame6.jpg')

