#Variables to change stuff on a high level
#Whether to display the input image
display_input = False
#Whether to wait for a keypress on each image or not
wait = False
#Whether to play against a computer
vs_comp = False
#Which apriltags should be looked for
#Tags I've used in the project: 'tag16h5', 'tag36h11', 'tagStandard41h12'
tag_family = 'tag16h5'
#Default edge_count_threshold, gets updated on first frame to better suit current conditions
edge_count_threshold = 50000
#File directory to get images from
image_directory = 'TestingImages/PiImages/'

#Importing needed libraries
import contextlib
with contextlib.redirect_stdout(None):
    import pygame
import chess, time, pygame, chess.engine, cv2, copy, math, os, sys
import numpy as np
from apriltag import apriltag

#If run on my laptop, disable pi specific code
if (os.uname()[1] == "blemay360-Swift-SF314-53G"):
    pi = False
#Else enable pi code
else:
    pi = True

#Import pi specific libraries
if pi:
    from picamera.array import PiRGBArray
    from picamera import PiCamera

'''
-----------------------------------------TO DO--------------------------------------------------
Adapt color detection function to work with large board
Test large board with pi
Test error detection for get_detection_array 
    Take off two pieces at once
    Add an extra piece
Add adaptive thresholding to get_detection_array
?Make new function for processing first image
'''

def end_program():
    global engine
    engine.quit()

#-----------------------------------------APRILTAG FUNCTIONS
def detect_apriltags(family, image):
    '''
    Takes a family of apriltags to look for, as well as an image to look at, and returns an array with a dictionary of detection info for each apriltag detected in the image
    '''
    #Only needs to be done once, but for ease of coding we'll do it every function call
    detector = apriltag(family)
    
    #Convert input image to grayscale
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 

    #Look for apriltags
    detections = detector.detect(gray_img)
    
    #Return variable with detected apriltag info
    return detections

def parse_april_tag_coordinate(detections, tag_id, corner ='center'):
    '''
    Function to easily parse the apriltag detection array
    Takes in the detection array, the desired tag to get info for, as well as the desired vertex to get the coordinates for
    '''
    
    index = None
    for tag in range(len(detections)):
        if (detections[tag]['id'] == tag_id):
            index = tag
            
    if (index == None):
        print("Error finding apriltag " + str(tag_id))
        return None
    
    #If the corner variable isn't there or is 'center'
    if (corner == 'center'):
        #Set x value of the center coordinate
        x = detections[tag_id]['center'][0]
        #Set y value of the center coordinate
        y = detections[tag_id]['center'][1]
    #Otherwise if the desired coordinate is not the center
    else:
        #Dictionary for converting from text descriptions of the corners to the index of the corner coordinate in the apriltag array
        corner_dict = { 'lb':0, 'rb':1, 'rt':2, 'lt':3 }
        #Set x value of the corner coordinate
        x = detections[tag_id]['lb-rb-rt-lt'][corner_dict[corner]][0]
        #Set y value of the corner coordinate
        y = detections[tag_id]['lb-rb-rt-lt'][corner_dict[corner]][1]
    #Output is the x and y coordinates in a tuple, rounded and cast to ints to the nearest pixel
    output = (int(round(x)), int(round(y)))
    return output

def grab_inside_corners(detections):
    global run
    '''
    Function for returning the inner coordinates of the apriltags on the chessboard
    Takes the detection array in as an input, and uses the parse_april_tag_coordinate function to get the corner coordinates of each tag
    Returns each corner in a 4 element list
    '''
    lt = parse_april_tag_coordinate(detections, 0, 'rb')
    rt = parse_april_tag_coordinate(detections, 1, 'lb')
    rb = parse_april_tag_coordinate(detections, 3, 'lt')
    lb = parse_april_tag_coordinate(detections, 2, 'rt')
    output = (lb, rb, rt, lt)
    if (any(map(lambda elem: elem is None, output))):
        end_program()
        raise Exception("Error finding all four apriltags") 
        sys.exit()
    return output

def grab_outside_corners(detections):
    '''
    Function for returning the inner coordinates of the apriltags on the chessboard
    Takes the detection array in as an input, and uses the parse_april_tag_coordinate function to get the corner coordinates of each tag
    Returns each corner in a 4 element list
    '''
    lt = parse_april_tag_coordinate(detections, 0, 'lt')
    rt = parse_april_tag_coordinate(detections, 1, 'rt')
    rb = parse_april_tag_coordinate(detections, 3, 'rb')
    lb = parse_april_tag_coordinate(detections, 2, 'lb')
    return lb, rb, rt, lt

def return_tag_margin(input_list):
    return input_list['margin']

def return_tag_id(input_list):
    return input_list['id']

#-----------------------------------------IMAGE UTILITIES
def copy_image(image, num_copies):
    '''
    Function to return any number of copies of an image
    Takes in the image as an input, as well as the number of copies to make
    '''
    #Start the output as a one element tuple containing a copy of the input image
    output = (copy.copy(image),)
    
    #Iterate for num_copies - 1 times to copy the rest of the images needed
    for i in range(num_copies - 1):
        #Add the other copied images to the output tuple
        output = output + (copy.copy(image),)
        
    return output

def measure_distance(pt1, pt2):
    '''
    Function to clean up lines where distance measuring is needed
    Takes in two coordinates as input
    Uses pythagorean theorem to calculate distance
    Distance is returned as an int
    '''
    return int(math.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2))

def circle_image(image, coordinates, color, ratio):
    '''
    Add circles to a given image
    WON'T WORK FOR TWO COORDINATES!!
    Takes the image to add a circle to as an argument, as well as the coordinates at which to place the circle, the color to make the circle, and whether the image is from a picture or video, which changes the ratio and thus the desired radius of the circle
    Doesn't return anything
    '''
    #Color definition dictionary
    color_dict = {'blue':(255, 0, 0), 'green':(0, 255, 0), 'red':(0, 0, 255)}
    
    #Fill entire circle
    thickness = -1

    #If the image is from a picture, use a large radius circle to make the circle more visible on a large picture
    if (ratio == 'picture'):
        radius = 12
    elif (isinstance(ratio, int)):
        radius = ratio
    #If the image is from a video, use a small radius circle so as to not take up too much space
    else:
        radius = 3

    #If there are multiple circles to add to the picture
    if (len(coordinates) > 2):
        #Cycle through all the coordinates
        for i in range(len(coordinates)):
            #Add a circle to the image at coordinates[i]
            cv2.circle(image, coordinates[i], radius, color_dict[color], thickness)
    #If there is only one circle to add
    else:
        #Add circle to image
        cv2.circle(image, coordinates, radius, color_dict[color], thickness)
        
def show_images(*arg):
    '''
    Function to display any number of images at once
    If the images are very large and need to be resized, the first argument should be 'resize'
    Takes in tuples of the name of the window and the image to display
    Returns nothing
    '''
    #If the images are very large and need to be resized
    if (arg[0] == 'resize'):
        #If displaying on the pi
        if pi:
            #Iterate through each image, skipping the first argument
            for i in range(1, len(arg)):
                #Set the window to be able to be resized
                cv2.namedWindow(arg[i][0], cv2.WINDOW_NORMAL)
                #Resize the window to 200 by 200 pixels
                cv2.resizeWindow(arg[i][0], 200,200)
                #Show the image
                cv2.imshow(arg[i][0], arg[i][1])
        else:
            #Iterate through each image, skipping the first argument
            for i in range(1, len(arg)):
                #Set the window to be able to be resized
                cv2.namedWindow(arg[i][0], cv2.WINDOW_NORMAL)
                #Resize the window to 700 by 700 pixels
                cv2.resizeWindow(arg[i][0], 700,700)
                #Show the image
                cv2.imshow(arg[i][0], arg[i][1])
    #If the images should be displayed as is
    else:
        #Cycle through all the images
        for i in range(len(arg)):
            #Diplay the image
            cv2.imshow(arg[i][0], arg[i][1])

#-----------------------------------------IMAGE DETECTION FUNCTIONS
def perspective_shift(image, corners):
    '''
    Function to perform a perspectivve shift on an image
    Used to cut out unnecessary parts of the image and just focus on the chessboard, as well as make sectioning the chessboard easier
    Takes in the image to shift, as well as the coordinates of the image that should become the new corners of the image
    Returns the shifted image
    '''
    #Measure the distance of each size of the chessboard and keep the maximum length
    distance = max(measure_distance(corners[0], corners[1]), measure_distance(corners[1], corners[2]), measure_distance(corners[2], corners[3]), measure_distance(corners[3], corners[0]))
        
    #Convert the input corner coordinates to float
    pts1 = np.float32(corners)
    #Set the size of the output image to be the maximum chess side length of pixels from the input image and convert to float
    pts2 = np.float32([[0,0],[distance,0],[0,distance],[distance,distance]])
    
    #Get matrix with which to shift the image using the coordinates of corners from the input image and the coordinates of where those points should be in the output image
    M = cv2.getPerspectiveTransform(pts1,pts2)
    #Perform perspective shift
    shifted_image = cv2.warpPerspective(image, M, (distance,distance))
    
    return shifted_image

def get_detection_color_array(image, turn_background, first_frame=False):
    global edge_count_threshold
    '''
    Function to detect pieces on a chessboard
    Takes in an image of just the chessboard, perspective shifted, and the amount of pieces on the board in the previous frame to help with thresholding
    Returns the input image with each square replaced with its edge detection array and color coded squares showing which squares are deemed to have pieces in them
    The rectangles used for edge detection are the inside parts of the squares of the chessboard
    Each square was made to be 10 x 10 pixels before being scaled up, 10% of the square width is taken off when calculating corner coordinates, so the measured rectangle ends up being the inside 8 x 8 of the square
    A white rectange means a piece was detected, a black rectangle means the square is empty
    '''
    
    add_text_to_image = True
    
    #Make a copy of the input image to modify to show detection
    detection_image = copy.copy(image)
    color_image = cv2.cvtColor(copy.copy(image), cv2.COLOR_BGR2HSV)
    color_image = np.stack((color_image[:,:,0],) * 3, axis=-1)
            
    #Rectangle color for empty squares
    empty_color = (0, 0, 0)
    #Rectangle color for filled squares
    occupied_color = (255, 255, 255)
    
    if pi:
        #Size of text to add to image
        text_size = 0.25
        #Thickness of text to add to image
        text_thickness = 1
        #Thickness of lines to draw on image
        line_thickness = 1
    else:
        #Size of text to add to image
        text_size = 1.5
        #Thickness of text to add to image
        text_thickness = 4
        #Thickness of lines to draw on image
        line_thickness = 6
    
    #Size of the input image, assumed to be a square because the input image should be perspective shifted
    size = detection_image.shape[0]
    #Calculate the size of a pixel on the paper (NOT an actual pixel, the pixel here refers to creating the chessboard)
    #The apriltag is separated from the chessboard by one pixel, each square of the chessboard is 10 pixels wide, meaning 82 pixels for the width/height of the input image
    pixel = size / 82
    #Amount of space to move inwards from the corner of the square when setting up the region of interest
    margin = 2
    
    #Initialize array for storing the edge count of each square of the chessboard
    edge_count = np.zeros((8, 8), dtype=int)
    #Initialize array for storing the determination for whether each square is occupied
    detection_array = np.zeros((8, 8), dtype=int)
    #Initialize array for storing the color of each piece
    color_array = np.zeros((8, 8), dtype=int)
    
    if first_frame:
        #Initialize list to store the amount of edges detected in each occupied square
        occupied_edge_count = [0] * 32
    
    #Iterate through each rank of the chessboard
    for y in range(8):
        #Iterate through each file of the chessboard
        for x in range(8):
            #Calculate the coordinates of the upper left corner of the rectangle to be measured
            lt = (int(round(pixel + margin * pixel + 10 * pixel * x)), int(round(pixel + margin * pixel + 10 * pixel * y)))
            #Calculate the coordinates of the lower right corner off the rectange to be measured
            rb = (int(round(pixel +10 * pixel * (x + 1) - margin * pixel)), int(round(pixel + 10 * pixel * (y + 1) - margin * pixel)))
            
            #Crop out only the current square for analysis
            square = detection_image[lt[1]:rb[1], lt[0]:rb[0]]
            
            #Perform edge detection on the square, replacing the relevant part of the input image with the edge detection array, storing the edge count in the edge_count array, and taking in the center of mass of the detected edges as well as the standard deviation of points
            detection_image[lt[1]:rb[1], lt[0]:rb[0]], edge_count[y][x], x_average, y_average, std = edge_detection(square)            
            
            #If this is the first frame, all squares in the 0th, 1st, 6th, and 7th are known to be occupied
            if first_frame and ((y == 0) or (y == 1) or (y == 6) or (y == 7)):
                #Save edge count to use in setting the edge_count_threshold
                occupied_edge_count[(y % 4) * 8 + x] = edge_count[y][x]
                #If the edge_count is below the current edge_count_threshold, raise the edge_count to be above the threshold so it starts as a occupied square
                if (edge_count[y][x] <= edge_count_threshold):
                    edge_count[y][x] = edge_count_threshold + 1
            
            #If the edge count of the square exceeds edge_count_threshold the square is deemed occupied
            if (edge_count[y][x] > edge_count_threshold):
                #Add a rectangle of the appropriate color to the output image
                detection_image = cv2.rectangle(detection_image, lt, rb, occupied_color, line_thickness)
                #Mark the square as occupied in the output array
                detection_array[y][x] = 1
                #Measure the average color of the piece
                color_image[lt[1]:rb[1], lt[0]:rb[0]], color_array[y][x] = average_color(color_image[lt[1]:rb[1], lt[0]:rb[0]], x_average, y_average, std)
                cv2.putText(color_image, str(color_array[y][x]), (int(lt[0]), int(lt[1] + pixel*0)), cv2.FONT_HERSHEY_COMPLEX, text_size, (255, 255, 255), text_thickness)
            #If the edge cound of the square is or is under edge_count_threshold, then the square is empty
            else:
                #Add a rectangle of the appropriate color to the output image
                detection_image = cv2.rectangle(detection_image, lt, rb, empty_color, line_thickness)
                #Mark the square as occupied in the output array
                detection_array[y][x] = 0

    if first_frame:
        #Set new edge_count_threshold to be slightly smaller than the lowest edge count of an occupied square
        edge_count_threshold = int(min(occupied_edge_count) * 0.7)
    
    if (np.count_nonzero(detection_array) > (turn_background[0] + turn_background[1]) or np.count_nonzero(detection_array) < (turn_background[0] + turn_background[1] - 1)):
        print("Error detecting number of pieces on board")
        print("Counted " + str(np.count_nonzero(detection_array)) + " pieces")
        print("Expected " + str(turn_background[0] + turn_background[1]) + " or " + str((turn_background[0] + turn_background[1]) - 1) + " pieces")
        print("Edge count threshold is " + str(edge_count_threshold))
        print(detection_array)
        print(edge_count)
        show_images('resize', ("Piece Detection", detection_image))
        cv2.waitKey(1)
        
        #Maybe change the threshold here?
    
    #If one less piece is on the board in the current frame than the sum of pieces expected from both sides of the board
    if (np.count_nonzero(detection_array) + 1 == turn_background[0] + turn_background[1]):
        #Subtract one piece from whichever side didn't just move
        turn_background[turn_background[2]] -= 1

    #Flatten the color array to a 1D array to bbe able to sort it
    raveled_color_array = np.ravel(color_array)
    #Sort the nonzero values of the flattened color array
    color_array_sorted = np.sort(raveled_color_array[np.flatnonzero(color_array)])
    
    #Dictionary for what to display on each square of the output image depending on its color_array value
    #color_ref = {0:"Empty", 1:"Black", 2:"White"}
    color_ref = {0:("Empty", (255, 255, 255)), 1:("Black", (150, 150, 150)), 2:("White", (255, 255, 255))}
    
    #Iterate through each rank (horizontal row)
    for y in range(8):
        #Iterate through each file (vertical column)
        for x in range(8):
            #If the current square is occupied
            if (color_array[y][x] > 0):
                #If the average color of the current piece is in the bottom x number of average piece colors on the board, label it as white
                #x is gotten from the turn_background list, where turn_background[0] tells the number of white pieces expected on the board
                #turn_background is updated above based on when the detection array shows a piece disappeared
                if (color_array[y][x] in color_array_sorted[:turn_background[0]]):
                    #Label the current square as having a white piece
                    color_array[y][x] = 2
                #If the current piece color isn't determined to be white, label it black
                else:
                    #Label it black
                    color_array[y][x] = 1
                if add_text_to_image:
                    #Calculate the coordinates of the upper left corner of the rectangle to be measured
                    lt = (int(round(pixel + margin * pixel + 10 * pixel * x)), int(round(pixel + margin * pixel + 10 * pixel * y)))
                    #Label the current square with the determination of whether it is empty, or what color piece it has
                    #Add a rectangle of the appropriate color to the output image
                    #color_image = cv2.rectangle(color_image, lt, rb, color_ref[color_array[y][x]], int(pixel)*2)
                    cv2.putText(color_image, color_ref[color_array[y][x]][0], (int(lt[0]), int(lt[1] + pixel*7)), cv2.FONT_HERSHEY_COMPLEX, text_size, color_ref[color_array[y][x]][1], text_thickness)
    
    #print(edge_count)
    #print(detection_array)
    #print(color_array)
    return detection_image, detection_array, color_image, color_array

def edge_detection(image):
    '''
    Function to perform edge detection on an image, and return the array of edges and how many edge were measured, as well as the center of mass of the edges and the standard deviation of the edges
    Takes in the image to measure, and outputs the measured image and the number of edges
    '''
    #Blur the image to get rid of noise and bad edge measurements
    blurred = cv2.medianBlur(image,7)
    #Perform edge measurement
    edges = cv2.Canny(blurred,80,100)
    
    #If there are edges
    if np.count_nonzero(edges):
        #Find average y location of edges
        y_average = int(round(np.mean(np.nonzero(edges)[0])))
        #Find average x location of edges
        x_average = int(round(np.mean(np.nonzero(edges)[1])))
        #Find standard deviation of edges in y direction
        y_std = int(round(np.std(np.nonzero(edges)[0])))
        #Find standard deviation of edges in x direction
        x_std = int(round(np.std(np.nonzero(edges)[1])))
        #Average standard deviation values to get circle radius
        std = int(round(1.35 * (y_std + x_std) / 2))
    #If there are no edges
    else:
        #Set all statistics values to 0
        y_average, x_average, std = 0, 0, 0
        
    #Copy edge detection image 3 times so it can replace a section of a 3 channel image
    edges = np.stack((edges,)*3, axis=-1)
    #Find total amount of edges
    edge_count = np.sum(edges)
        
    return edges, edge_count, x_average, y_average, std

def get_color_array(image, detection_array, turn_background):
    '''
    Function to measure the color of each piece to tell whether it is black or white (or just light/dark in a grayscale image)
    Takes in the image to measure from, the detection array that tells which squares to measure, and the turn background list to see how many pieces of each color it is looking for
    The input image is assumed to be a square
    Returns a color array with 0s where no piece is, 1s where black pieces are, and 2s where white pieces are
    '''
    #Whether to add text to the color detection image
    add_text_to_image = True
    #Convert a copy of in input image to a grayscale image
    hsvImage = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(copy.copy(hsvImage), cv2.COLOR_BGR2GRAY)
    #Get the side length of the input image, assumed to be a square
    size = gray.shape[0]
    #Calculate the size of a scaled up pixel from the chessboard (not the size of a pixel in the image)
    pixel = size / 82
    #Amount of space to move inwards from the corner of the square when setting up the region of interest
    margin = 2
    
    if pi:
        #Size of text to add to image
        text_size = 0.25
        #Thickness of text to add to image
        text_thickness = 1
        #Thickness of lines to draw on image
        line_thickness = 1
    else:
        #Size of text to add to image
        text_size = 1.5
        #Thickness of text to add to image
        text_thickness = 4
        #Thickness of lines to draw on image
        line_thickness = 6
    
    #Initialize an array to store the average color value for each square
    color_array = np.zeros((8, 8), dtype=float)
    
    #Variable to store the average color of an empty white square for reference
    white_square_ref = 0
    #Variable to store the number of empty white squares
    empty_white_count = 0
    #Variable to store the average color of an empty black square for reference
    black_square_ref = 0
    #Variable to setore the number of empty black squares
    empty_black_count = 0
    
    #Iterate through each rank (horizontal row)
    for y in range(8):
        #Iterate through each file (vertical column)
        for x in range(8):
            #If the current square is empty
            if (detection_array[y][x] == 0):
                #Calculate the coordinates of the upper left corner of the rectangle to be measured
                start_point = (int(round(pixel + margin * pixel + 10 * pixel * x)), int(round(pixel + margin * pixel + 10 * pixel * y)))
                #Calculate the coordinates of the lower right corner off the rectange to be measured
                end_point = (int(round(pixel +10 * pixel * (x + 1) - margin * pixel)), int(round(pixel + 10 * pixel * (y + 1) - margin * pixel)))
                
                #Take the average color in the rectangle
                average = average_color(gray[start_point[1]:end_point[1], start_point[0]:end_point[0]])
                
                if add_text_to_image:
                    #Display the average color in the current square
                    cv2.putText(gray, str(round(average, 1)), (int(start_point[0]), int(start_point[1] + pixel*2.5)), cv2.FONT_HERSHEY_COMPLEX, text_size, 255, text_thickness)
                
                #Draw a rectangle around the rectangle in which color is measured for this square
                cv2.rectangle(gray, start_point, end_point, 255, line_thickness)
                
                #If the current square is white
                if ((x + y) % 2 == 0):
                    #Add the average color of the square to the running sum of white square colors
                    white_square_ref += average
                    #Increment the number of empty white squares found
                    empty_white_count += 1
                #If the current square is black
                else:
                    #Add the average color of the square to the running sum of white square colors
                    black_square_ref += average
                    #Add the average color of the square to the running sum of white square colors
                    empty_black_count += 1
    
    #If there are no empty white squares
    if (empty_white_count == 0):
        #Default value if no white squares are empty
        white_square_ref = 185
    #If there are empty white squares
    else:
        #Compute averate color value for empty white squares
        white_square_ref = white_square_ref / empty_white_count
    #If there are no empty black squares
    if (empty_black_count == 0):
        #Default value if no black squares are empty
        black_square_ref = 105
    #If there are empty black squares
    else:
        #Compute average color value for empty black squares
        black_square_ref = black_square_ref / empty_black_count
    
    #Percentage of color values to remove above and below the average square color before measuring piece color
    percentange = 0.125
    
    #Compute lower and upper limits of colors to remove from each square before measuring piece color
    color_refs = [[white_square_ref * (1 - percentange), white_square_ref * (1 + percentange)], 
                  [black_square_ref * (1 - percentange), black_square_ref * (1 + percentange)]]
    
    #Iterate through each rank (horizontal row)
    for y in range(8):
        #Iterate through each file (vertical column)
        for x in range(8):
            #If current square is occupied
            if (detection_array[y][x] == 1):
                #Calculate the coordinates of the upper left corner of the rectangle to be measured
                start_point = (int(round(pixel + margin * pixel + 10 * pixel * x)), int(round(pixel + margin * pixel + 10 * pixel * y)))
                #Calculate the coordinates of the lower right corner off the rectange to be measured
                end_point = (int(round(pixel +10 * pixel * (x + 1) - margin * pixel)), int(round(pixel + 10 * pixel * (y + 1) - margin * pixel)))
                
                #Section off square as just the part of the image to measure
                square = gray[start_point[1]:end_point[1], start_point[0]:end_point[0]]
                #Create a mask for the current square of colors that are close to the average color of the current square colors empty square color
                #If the current square is white, (x+y)%2 will be 0, referencing the first nested list, otherwise the second nested list will be used when (x+y)%2 is 1 for black squares
                square_mask = cv2.inRange(square, color_refs[(x + y) % 2][0], color_refs[(x + y) % 2][1])
                #Remove the empty square color from the current square
                output = cv2.bitwise_and(square, square, mask = cv2.bitwise_not(square_mask))
                #Replace the square in the grayscale image with the masked image
                gray[start_point[1]:end_point[1], start_point[0]:end_point[0]] = output
                
                #Take the average color of the piece, not considering the color of the empty square that was removed
                color_array[y][x] = average_color(output)
                
                if add_text_to_image:
                    #Print the average color of the piece on the grayscale image
                    cv2.putText(gray, str(round(average_color(output), 1)), (int(start_point[0]), int(start_point[1] + pixel*2.5)), cv2.FONT_HERSHEY_COMPLEX, text_size, 255, text_thickness)
    
    #Flatten the color array to a 1D array to bbe able to sort it
    raveled_color_array = np.ravel(color_array)
    #Sort the nonzero values of the flattened color array
    color_array_sorted = np.sort(raveled_color_array[np.flatnonzero(color_array)])
    
    #Dictionary for what to display on each square of the output image depending on its color_array value
    color_ref = {0:"Empty", 1:"Black", 2:"White"}
    
    #Iterate through each rank (horizontal row)
    for y in range(8):
        #Iterate through each file (vertical column)
        for x in range(8):
            #If the current square is occupied
            if (color_array[y][x] > 0):
                #If the average color of the current piece is in the bottom x number of average piece colors on the board, label it as black
                #x is gotten from the turn_background list, where turn_background[1] tells the number of black pieces expected on the board
                #turn_background is updated in the process_frame function based on when the get_detection_array shows a piece disappeared after whites move
                if (color_array[y][x] in color_array_sorted[:turn_background[1]]):
                    #Label the current square as having a black piece
                    color_array[y][x] = 1
                #If the current piece color isn't determined to be black, label it white
                else:
                    #Label it white
                    color_array[y][x] = 2
            #Calculate upper left corner of rectangle to put text in
            start_point = (int(round(pixel + margin * pixel + 10 * pixel * x)), int(round(pixel + margin * pixel + 10 * pixel * y)))
            if add_text_to_image:
                #Label the current square with the determination of whether it is empty, or what color piece it has
                cv2.putText(gray, color_ref[color_array[y][x]], (int(start_point[0]), int(start_point[1] + pixel*4.5)), cv2.FONT_HERSHEY_COMPLEX, text_size, 255, text_thickness)
    return gray, color_array.astype(np.int64)

def average_color(image, x, y, radius):
    '''
    Function to average the hue of a hsv image
    Takes in a hsv image and returns an average of the nonzero hues inside a given circle
    '''
    #Create an empty mask the size of the input image
    mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    #Place a white circle on the mask with the given dimensions
    cv2.circle(mask, (x, y), radius, (255, 255, 255), -1)
    #Create a new image called color_measure by only keeping the hue pixels that intersect with the white circle on the mask
    color_measure = np.stack((image[:,:,0]*mask,) * 3, axis=-1)
    #Measure average color from nonzero pixels
    color = round(np.average(color_measure[np.nonzero(mask)]))
    
    return color_measure, color

#-----------------------------------------IMAGE DETECTION MAIN FUNCTION
def process_frame(frame, turn_background, first_frame=False):
    #Whether to automaticallly focus on the middle of the image or not
    auto_crop = False
    
    if auto_crop:
        #If frame is taller than it is wide
        if (frame.shape[0] > frame.shape[1]):
            #Make a copy of frame in case cropping it doesn't work
            input_frame = frame
            #Create a square that is as wide as the photo and in the middle of the frame
            start_point = (0, (frame.shape[0] - frame.shape[1])//2)
            end_point = (frame.shape[1], (frame.shape[0] - frame.shape[1])//2 + frame.shape[1])
            #Crop the frame to just the middle square
            frame = frame[start_point[1]:end_point[1], start_point[0]:end_point[0]]
    
    #Run apriltag detection on image
    detections = detect_apriltags(tag_family, frame)
    
    if auto_crop:
        #If not all 4 apriltags are detected
        if (len(detections) != 4) and (input_frame.shape[0] > input_frame.shape[1]):
            print("Rerunning detection on entire image")
            #Revert back to the original frame
            frame = input_frame
            #Run apriltag detection again
            detections = detect_apriltags(tag_family, frame)
    
    #Make copies of original image to keep the same for display 
    apriltagCorners, shifted, color_detection, piece_detection = copy_image(frame, 4)            
    
    #If the detections are still under 4
    if (len(detections) < 4):
            print("Missing an apriltag")
            print("Apriltags found: " + str(len(detections)))
            print(detections)
            #Set the window to be able to be resized
            cv2.namedWindow("Input Image", cv2.WINDOW_NORMAL)
            #Resize the window
            if pi:
                cv2.resizeWindow("Input Image", 200, 200)
            else:
                cv2.resizeWindow("Input Image", 700, 700)
            #Show the image
            cv2.imshow("Input Image", frame)
            #Wait for a keypress
            cv2.waitKey(0)
    #If there are extra apriltags detected
    elif (len(detections) > 4):
        #print("Detected Extra Apriltag")
        #Make a copy of the detection tuple as a list for easier sorting and deleting
        detection_list = list(detections)
        #Sort the detection list by descending confidence in the tag detection
        detection_list.sort(reverse=True, key=return_tag_margin)
        #For all the detected tags over 4
        for i in range(len(detections) - 4):
            #Remove the last value
            detection_list.pop()
        #Sort the list by tag id to return it to the original order
        detection_list.sort(key=return_tag_id)
        #Replace the previous detection tuple by the modified one
        detections = tuple(detection_list)

    #If 4 apriltags are seen (one in each 4 corners of chessboard), crop image and run detection functions
    if(len(detections) == 4):
        #Get values for the inside corners of each 4 apriltags
        lb, rb, rt, lt = grab_inside_corners(detections)
        
        #Place circles on inside corners of each apriltag
        #circle_image(apriltagCorners, (lt, rt, rb, lb), 'red', 'picture')
        #show_images('resize', ('Apriltag Corners', apriltagCorners))
        #cv2.waitKey(0)
        
        #Shift perspective to make to make the inside corners of the apriltags the corners of the image
        shifted = perspective_shift(shifted, (lt, rt, lb, rb))
        
        #Go through each square of the chessboard to tell if square is populated with piece and measure piece color
        piece_detection, detection_array, color_detection, color_array = get_detection_color_array(shifted, turn_background, first_frame)
    
    return color_array, piece_detection, color_detection

#-----------------------------------------CHESS MOVE FUNCTIONS
def create_board_array():
    '''
    Function to create an array of chess pieces in the starting configuration
    Each element has an bmp file icon for the piece, the location of that icon on the chessboard, and the character representation of the piece
    Takes no input
    Returns the board array
    '''
    #List of filenames to use, the order corresponds to the piece_dict dictionary
    if pi:
        filenames = ['Icons/rook_black.bmp', 
                 'Icons/knight_black.bmp', 
                 'Icons/bishop_black.bmp', 
                 'Icons/queen_black.bmp', 
                 'Icons/king_black.bmp', 
                 'Icons/pawn_black.bmp', 
                 'Icons/pawn_white.bmp', 
                 'Icons/rook_white.bmp', 
                 'Icons/knight_white.bmp', 
                 'Icons/bishop_white.bmp', 
                 'Icons/queen_white.bmp', 
                 'Icons/king_white.bmp']
    else:
        filenames = ['Icons/rook_black.svg', 
                 'Icons/knight_black.svg', 
                 'Icons/bishop_black.svg', 
                 'Icons/queen_black.svg', 
                 'Icons/king_black.svg', 
                 'Icons/pawn_black.svg', 
                 'Icons/pawn_white.svg', 
                 'Icons/rook_white.svg', 
                 'Icons/knight_white.svg', 
                 'Icons/bishop_white.svg', 
                 'Icons/queen_white.svg', 
                 'Icons/king_white.svg']
    #Dictionary for converting the character representation of each piece to the filename of that piece icon
    piece_dict = {'r':0, 'n':1, 'b':2, 'q':3, 'k':4, 'p':5, 'P':6, 'R':7, 'N':8, 'B':9, 'Q':10, 'K':11}
    #Dictionary for correct sizing of each icon
    #Found by printing out the rectangle of each icon after importing then dividing by two
    size_dict = {'r':(17, 21), 'n':(17, 22), 'b':(17, 22), 'q':(21, 22), 'k':(20, 22), 'p':(17, 21), 'P':(17, 21), 'R':(17, 21), 'N':(17, 22), 'B':(17, 22), 'Q':(21, 22), 'K':(20, 22)}
    
    #Get the starting position of a chessboard in fen format, then at each '/', yielding a list of 8 elements for each rank of the board
    position = chess.STARTING_BOARD_FEN.split('/')

    #Initialize an 8 x 8 x 3 array for each square of the chessboard, and the 3 pieces of info for each piece
    board_array = np.empty(shape=(8, 8, 3),dtype='object')
    
    #For each element of the position list
    for y in range(8):
        #Empty squares are represented by a number for the amount of continuous emmpty squares in that section of the rank, so ranks 3-6 are just the string '8' and can be skipped
        if (position[y].isdigit() != True):
            #For each character in the filled ranks
            for x in range(8):
                #Load the icon into the first element by referencing the piece_dict to convert character to filename
                board_array[x][y][0] = pygame.image.load(filenames[piece_dict[position[y][x]]])
                #Scale icon to fit with pi screen
                if pi:
                    board_array[x][y][0] = pygame.transform.scale(board_array[x][y][0], size_dict[position[y][x]])
                #Place the rectangle location in the second element
                board_array[x][y][1] = board_array[x][y][0].get_rect()
                #Put the string representation of the piece in the third element
                board_array[x][y][2] = position[y][x]
    
    return board_array

def update_board_array_uci(board, board_array, uci):
    '''
    Updates the board_array from a uci notation string
    First the uci move is tested to see if it is a unique case that requires extra actions like:
        white or black castling on queenside or kingside
        a pawn promotion
        en passant capture
    If the move isn't any of the above, the code from the else statement is run
    '''
    #Dictionary for converting the file letter notation to the board_array x coordinate
    files = {'a':0, 'b':1, 'c':2, 'd':3, 'e':4, 'f':5, 'g':6, 'h':7}
    #Compute piece origin coordinate
    origin = (files[uci[0]], 8 - int(uci[1]))
    #Compute piece destination coordinate
    destination = (files[uci[2]], 8 - int(uci[3]))
        
    #White king-side castling
    if (uci == "e1g1"):
        #Move king
        board_array[destination[0]][destination[1]] = board_array[origin[0]][origin[1]]
        board_array[origin[0]][origin[1]] = (None, None, None)
        #Move rook
        board_array[5][7] = board_array[7][7]
        board_array[7][7] = (None, None, None)
    #White queen-side castling
    elif (uci == "e1c1"):
        #Move king
        board_array[destination[0]][destination[1]] = board_array[origin[0]][origin[1]]
        board_array[origin[0]][origin[1]] = (None, None, None)
        #Move rook
        board_array[3][7] = board_array[0][7]
        board_array[0][7] = (None, None, None)
    #Black king-side castling
    elif (uci == "e8g8"):
        #Move king
        board_array[destination[0]][destination[1]] = board_array[origin[0]][origin[1]]
        board_array[origin[0]][origin[1]] = (None, None, None)
        #Move rook
        board_array[5][0] = board_array[7][0]
        board_array[7][0] = (None, None, None)
    #Black queen-side castling
    elif (uci == "e8c8"):
        #Move king
        board_array[destination[0]][destination[1]] = board_array[origin[0]][origin[1]]
        board_array[origin[0]][origin[1]] = (None, None, None)
        #Move rook
        board_array[3][0] = board_array[0][0]
        board_array[0][0] = (None, None, None)
    #Promotion
    elif (len(uci) == 5):
        #List of filenames to use, the order corresponds to the piece_dict dictionary
        if pi:
            filenames = ['Icons/rook_black.bmp', 
                    'Icons/knight_black.bmp', 
                    'Icons/bishop_black.bmp', 
                    'Icons/queen_black.bmp', 
                    'Icons/king_black.bmp', 
                    'Icons/pawn_black.bmp', 
                    'Icons/pawn_white.bmp', 
                    'Icons/rook_white.bmp', 
                    'Icons/knight_white.bmp', 
                    'Icons/bishop_white.bmp', 
                    'Icons/queen_white.bmp', 
                    'Icons/king_white.bmp']
        else:
            filenames = ['Icons/rook_black.svg', 
                    'Icons/knight_black.svg', 
                    'Icons/bishop_black.svg', 
                    'Icons/queen_black.svg', 
                    'Icons/king_black.svg', 
                    'Icons/pawn_black.svg', 
                    'Icons/pawn_white.svg', 
                    'Icons/rook_white.svg', 
                    'Icons/knight_white.svg', 
                    'Icons/bishop_white.svg', 
                    'Icons/queen_white.svg', 
                    'Icons/king_white.svg']
        #Dictionary for converting the character representation of each piece to the filename of that piece icon
        piece_dict = {'r':0, 'n':1, 'b':2, 'q':3, 'k':4, 'p':5, 'P':6, 'R':7, 'N':8, 'B':9, 'Q':10, 'K':11}
        #Dictionary for correct sizing of each icon
        #Found by printing out the rectangle of each icon after importing then dividing by two
        size_dict = {'r':(17, 21), 'n':(17, 22), 'b':(17, 22), 'q':(21, 22), 'k':(20, 22), 'p':(17, 21), 'P':(17, 21), 'R':(17, 21), 'N':(17, 22), 'B':(17, 22), 'Q':(21, 22), 'K':(20, 22)}
        
        #White pieces are capitalized
        if (destination[1] == 0):
            piece = uci[4].upper
        #Black pieces are lowercase, just like the notation already is
        elif (destination[1] == 7):
            piece = uci[4]
        
        #Load in image for newly promoted piece
        board_array[destination[0]][destination[1]][0] = pygame.image.load(filenames[piece_dict[piece]])
        #Get size and location variable for the new image
        board_array[destination[0]][destination[1]][1] = board_array[destination[0]][destination[1]][0].get_rect()
        #Update the board_array character of the piece
        board_array[destination[0]][destination[1]][2] = piece
        #Replace move origin square with empty space
        board_array[origin[0]][origin[1]] = (None, None, None)
    #If the move is en passant
    #NOTE: the move has to be sent to this function before being pushed to the board, or is_en_passant won't recognize the move as an en passant capture
    elif (board.is_en_passant(chess.Move.from_uci(uci))):
        #Remove piece behind destination square in opposite direction of pawn travel
        board_array[destination[0]][destination[1] + (origin[1] - destination[1])] = (None, None, None)
        #Replace destination square with piece from move origin square
        board_array[destination[0]][destination[1]] = board_array[origin[0]][origin[1]]
        #Replace move origin square with empty space
        board_array[origin[0]][origin[1]] = (None, None, None)
    #Any other move
    else:            
        #Replace destination square with piece from move origin square
        board_array[destination[0]][destination[1]] = board_array[origin[0]][origin[1]]
        #Replace move origin square with empty space
        board_array[origin[0]][origin[1]] = (None, None, None)
    return board_array

def color_array_to_uci(old_array, new_array, board):
    '''
    Function to translate the difference in the previous and current color arrays to a chess move in uci format
    Takes in the previous color array, the current color array, and the current board variable to tell when a pawn reaches the back rank
    Works by taking the difference between the new array and old array, and assigning the move based on the nonzero elements of the resulting array
    Returns the move as a string in uci notation
    '''
    #String with the names of each file in order
    files = 'abcdefgh'
    #Initialize empty array to hold the names of each square
    square_names = np.empty((8, 8), dtype=object)
    #Iterate through each vertical file
    for y in range(8):
        #Iterate through each horizontal rank
        for x in range(8):
            #Assign square names with the file and the rank as a string
            square_names[y][x] = files[x] + str(8-y)
    #Take the difference between the new and old color arrays to see which squares changed
    difference = new_array - old_array
    #Empty list to easily store the origin and destination squares needed for uci notation
    move = ['', '']
    
    #If 3 squares were changed, that signals an en passant capture
    if (np.count_nonzero(difference) == 3):
        #Transform difference array to a 1D array by flattening it and take the absolute value element wise
        flat_abs_difference = np.abs(difference.flatten())
        #Count the occurence of each value in the flattened absolute value difference array, and return whichever number shows up least
        #0 will show up most in the first element, which is ignored, the important parts are the second and third elements, which show how many times ones and twos show up in the difference array, respectively
        min_val = np.argmin(np.bincount(flat_abs_difference))
        #Make an array with zeros where the squares weren't changed, and the min_val wherever squares were changed
        min_val = min_val * np.ones((8,8), dtype=int) * (difference != 0).tolist()
        #Subtract the min_val array from the difference array to remove the value that occurs the least in the difference array, this was the pawn that was captured in en passant
        #Since the captured pawn isn't given in uci notation, we just want the origin and destination square for the capturing pawn, which are the other two square changes
        difference = np.abs(difference) - min_val
        #After the difference array is replaced by the modified version without the least occuring value, the if statement for 2 changed squares will evaluate to be true and the uci notation can be found as normal
    #If 2 squares were changed, that signals either moving a piece to an empty square, or a conventional capture
    if (np.count_nonzero(difference) == 2):   
        #Create an empty modifier string variable in case a pawn is promoted and the promotion piece needs to be added to the end of the uci string
        modifier = ''
        #
        move[ min(new_array[np.nonzero(difference)][0], 1)] = square_names[np.nonzero(difference)][0]
        move[ min(new_array[np.nonzero(difference)][1], 1)] = square_names[np.nonzero(difference)][1]
        #If the piece came from the 7th rank and went to the 8th rank, or came from the 2nd rank and went to the 1st rank, check to see if it is a pawn that should be promoted
        if (int(move[0][1]) == 7 and int(move[1][1]) == 8) or (int(move[0][1]) == 2 and int(move[1][1]) == 1):
            #Get and split the current board fen notation into 8 strings for each rank
            position = board.fen().split()[0].split('/')
            #Set the x (file) notation to 0
            x = 0
            #If the piece being considered is at the back rank, it has to be a white pawn to be promoted
            if (int(move[1][1]) == 8):
                pawn = 'P'
            #Else if the considered piece is at the front rank, it has to be a black pawn to be promoted
            else:
                pawn = 'p'
            #For each character in the rank string
            for character in position[8 - int(move[0][1])]:
                #If the current character is a number, add that many to the x variable to skip the empty squares
                if (character.isdigit()):
                    x += int(character)
                #If the character is not a number
                else:
                    #Check if the square it came from previously held a pawn
                    if (files[x] == move[0][0] and character == pawn):
                        #Default the promotion type to a queen
                        modifier = 'q'
                    #Add one to the x variable to move onto the next square
                    x += 1
        #Assemble the move in uci notation by joining the origin square with the destination square, and the type of piece it is promoted to if necessary
        move = move[0] + move[1] + modifier
    #If 4 squares were changed, that signals that one of the sides castled
    elif (np.count_nonzero(difference) == 4):
        #If the castling happened in the front rank and the squares involved are on the right side of the board
        if (np.nonzero(difference)[0][0] == 7 and max(np.nonzero(difference)[1]) > 4):
            move = 'e1g1'
        #If the castling happened in the front rank and the squares involved are on the left side of the board
        elif (np.nonzero(difference)[0][0] == 7 and max(np.nonzero(difference)[1]) == 4):
            move = 'e1c1'
        #If the castling happened in the back rank and the squares involved are on the right side of the board
        elif (np.nonzero(difference)[0][0] == 0 and max(np.nonzero(difference)[1]) > 4):
            move = 'e8g8'
        #If the castling happened in the back rank and the squares involved are on the left side of the board
        elif (np.nonzero(difference)[0][0] == 0 and max(np.nonzero(difference)[1]) == 4):
            move = 'e8c8'
    return move
        
#-----------------------------------------CHESS GUI FUNCTIONS
def create_gui():
    '''
    Function to create the gui elements that are needed to display the chessboard
    No inputs are taken
    The gui window and the square size used are returned
    '''
    #Initialize an instance of pygame
    pygame.init()
    
    #The pi screen is smaller and needs different sizing
    if pi:
        #Set square size to 40 pixels
        square_size = 40
    else:
        #Set square size to 70 pixels
        square_size = 70
    #Set the size of the window to 8 squares by 8 squares
    window = pygame.display.set_mode((square_size * 8, square_size * 8))
    #Set the window name to Chessboard
    pygame.display.set_caption('Chessboard')
    #Load in an icon of a knight
    icon=pygame.image.load('Icons/icon.bmp')
    #Set the window icon
    pygame.display.set_icon(icon)
    #Draw a grid on the window
    window = draw_grid(window, square_size)
    #Show the window
    pygame.display.flip()
    return window, square_size

def draw_grid(window, square_size):
    '''
    Function to draw a chessboard grid in a pygame window
    Takes in the pygame window variable and the size of each square as inputs
    Returns the window with a grid on it
    '''
    #For each horizontal rank of the board
    for rank in range(8):
        #For each vertical file of the board
        for File in range(8):
            #Alternate the colors by using remainder functions to only color every other square tan, and switch whether even or odd squares are tan each rank
            if ((File + rank % 2) % 2 == 0):
                #Set color for this square to be tan
                color = (235, 236, 208) #tan
            else:
                #Set color for this square to be green
                color = (119, 149, 86) #green
            #Color the current square the selected color
            pygame.draw.rect(window, color, pygame.Rect(File * square_size, rank * square_size, square_size, square_size))
    return window

def print_board(window, board_array, square_size):
    '''
    Function to print the board_array on the pygame window
    Takes in the window to print on, the board array with piece icons, locations, and characters, and the size of the square to get the spacing right
    Returns nothing
    '''
    #Draw a grid on the window to overlay the current window and cover everything up
    draw_grid(window, square_size)
    #Iterate through each rank of the chessboard
    for y in range(8):
        #Iterate through each file of the chessboard
        for x in range(8):
            #If the current square of the board array isn't empty
            if (board_array[x][y][0] != None):
                #Set x coordinate to the middle of the current square
                board_array[x][y][1].centerx = (square_size // 2) + square_size * x
                #Set y coordinate to the middle of the corrent square
                board_array[x][y][1].centery = (square_size // 2) + square_size * y
                #Send updates to the window
                window.blit(board_array[x][y][0], board_array[x][y][1])
    #Display the window
    pygame.display.flip()

#-----------------------------------------CHESS FUNCTIONS
def open_engine(difficulty):
    if pi:
        #List of engine to use and their rankings
        engines = [[968, "feeks-master/main.py"], [1198, "belofte64-2.1.1"], [3717, "stockfish_13_linux_x64"]]
        #Sort engines by rating
        engines.sort(key = lambda engines: engines[0]) 
        #Sort to get closest ranking to the desired difficulty
        engine_to_use = min([(engines[i][0], engines[i][1], i) for i in range(len(engines))], key = lambda x: abs(x[0]-difficulty))
        print("Using engine " + engine_to_use[1] + " with a rating of " + str(engine_to_use[0]))
        #Open engine
        engine = chess.engine.SimpleEngine.popen_uci("/home/pi/Chessboard/Engines/" + engine_to_use[1])
    else:
        #List of engine to use and their rankings
        engines = [[968, "feeks-master/main.py"], [1198, "belofte64-2.1.1"], [3717, "stockfish_13_linux_x64"]]
        #Sort engines by rating
        engines.sort(key = lambda engines: engines[0]) 
        #Sort to get closest ranking to the desired difficulty
        engine_to_use = min([(engines[i][0], engines[i][1], i) for i in range(len(engines))], key = lambda x: abs(x[0]-difficulty))
        print("Using engine " + engine_to_use[1] + " with a rating of " + str(engine_to_use[0]))
        #Open engine
        engine = chess.engine.SimpleEngine.popen_uci("/home/blemay360/Documents/chessboard-main/Engines/" + engine_to_use[1])
    return engine

#-----------------------------------------PI FUNCTIONS
def setup_camera():
    #Initialize the camera and grab a reference to the raw camera capture
    camera = PiCamera()
    #Set resolution
    camera.resolution = (1024, 768)
    return camera

def pi_take_image(camera):
    #Get raw capture from camera
    #For some reason this works better calling this line everytime
    rawCapture = PiRGBArray(camera)
    #Allow the camera to warmup
    time.sleep(0.1)
    #Grab an image from the camera
    camera.capture(rawCapture, format="bgr")
    input_image = rawCapture.array
    return input_image
    
#-----------------------------------------MAIN FUNCTION
def main():
    global engine
    #Create a gui to display the state of the chessboard, saving the window it creates and the size of each square in the chessboard for later functions
    window, square_size = create_gui()

    #Create a board array to keep track of the images displayed on the gui chessboard
    board_array = create_board_array()

    #Create a board variable to keep track of the game
    board = chess.Board()

    #Print the chessboard to the window
    print_board(window, board_array, square_size)
    
    #Set desired difficulty of the computer
    #If playing computer, use desired difficulty
    if vs_comp:
        difficulty = 900
    #Otherwise use the max rated engine for best suggestions
    else:
        difficulty = 3000
     
    #Find and open the desired engine for the difficulty
    engine = open_engine(difficulty)
    
    #Initialize a variable to keep track of how many white and black pieces are on the board, and which side just moved
    turn_background = [16, 16, 0]
    #[# of white pieces on board, # of black pieces on board, 0=black just moved | 1=white just moved]
    
    if pi:
        #Set up camera
        camera = setup_camera()
        #Take an image from the camera
        input_image = pi_take_image(camera)
    else:
        #Read the first frame in
        input_image = cv2.imread(image_directory + '1.jpg')
        
    if display_input:
            #Show the input image
            show_images('resize', ("Input Image", input_image))
            #Pause until user presses a key
            cv2.waitKey(10)
        
    #Save the color array as old to compare with the second frame later
    old_color_array, piece_detection, gray = process_frame(input_image, turn_background, True)
    
    #Show the grayscale color detection image and piece detection image
    show_images('resize', ('Color Values', gray), ('Piece Detection', piece_detection))
        
    if wait:
        #Wait for a keypress while updating the chessboard gui
        while (cv2.waitKey(100) == -1):
            print_board(window, board_array, square_size)
    else:
        #Don't wait for a keypress
        cv2.waitKey(1)

    #-----------------------------------------LOOP
    run = True
    counter = 2
    right_move = 1
    while run:
        #Start a timer to measure processing time for the current frame
        start = time.time()
        #Update the variable of which side just went
        turn_background[2] = 1 - (counter % 2)
        
        #If the wrong move was made for a computer match, wait here for user to correct before reading in a frame
        if vs_comp:
            cv2.waitKey(right_move)
        
        #Read the current frame
        if pi:
            input_image = pi_take_image(camera)
        else:
            input_image = cv2.imread(image_directory + str(counter) + '.jpg')
        
        #Process the current frame
        new_color_array, piece_detection, gray = process_frame(input_image, turn_background)
        
        if display_input:
            #Show the input image
            show_images('resize', ("Input Image", input_image))
        
        #Show the grayscale color detection image and piece detection image for the current image
        show_images('resize', ('Color Values', gray), ('Piece Detection', piece_detection))
        
        #Compare the color detection array of the current image with the last image to deterime the move that was made
        if not np.array_equal(old_color_array, new_color_array):
            move = color_array_to_uci(old_color_array, new_color_array, board)
            #Compute the move variable using the chess library
            move = chess.Move.from_uci(move)
        else:
            move = None
            print("none")
        
        if vs_comp and (turn_background[2] == 0):
            if (move == result.move):
                print("Good job")
                right_move = 1
            else:
                print("Wrong move")
                right_move = 0
                continue
        
        #If the move wasn't legal
        if not (move in board.legal_moves):
            #Print the move wasn't legal
            print("Not legal")
            #continue
        
        #Update the board array to reflect the move that was just made
        board_array = update_board_array_uci(board, board_array, chess.Move.uci(move))
        #Push the move to the board
        board.push(move)
        #Update the chessboard gui
        print_board(window, board_array, square_size)
        #Replace the old color array with the current color array to prepare for the next frame
        old_color_array = new_color_array
        
        #Print the move that was made and the time it took to process the frame
        print(move, time.time() - start)
        
        if vs_comp and (turn_background[2] == 1):
            result = engine.play(board, chess.engine.Limit(time=0.5))
            print("Computer move: " + str(result.move))
            #board_array = update_board_array_uci(board, board_array, chess.Move.uci(result.move))
            #board.push(result.move)
        
        if wait:
            #Wait for a keypress while updating the chessboard gui
            while (cv2.waitKey(100) == -1):
                print_board(window, board_array, square_size)
        else:
            #Don't wait for a keypress
            cv2.waitKey(1)
        #Update counter to know we're on the next frame
        counter += 1
        
        if not pi:
            max_file_count = max([int(i.split('.')[0]) for i in os.listdir(image_directory) if i.split('.')[0].isdigit()]) + 1;
        else:
            max_file_count == 0
        
        #If it's running from images and finishes the last image
        if (counter == max_file_count) and not pi:
            #Stop running the program
            print("Reached end of files")
            run = False
    
    engine.quit()
    #When all images have been processed, wait 1 second to allow the user to decompress before ending
    time.sleep(1)

main()
