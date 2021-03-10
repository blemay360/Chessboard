# Importing the OpenCV library 
from os import environ
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import chess, time, pygame, chess.engine, cv2, copy, math
import numpy as np
from apriltag import apriltag

#-----------------------------------------DETECTION FUNCTIONS--------------------------------------------------

#-----------------------------------------APRILTAG FUNCTIONS
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

#-----------------------------------------IMAGE UTILITIES
def copy_image(image, num_copies):
    output = (copy.copy(image),)
    for i in range(num_copies - 1):
        output = output + (copy.copy(image),)
    return output

def measure_distance(pt1, pt2):
    return int(math.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2))

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

#-----------------------------------------IMAGE DETECTION FUNCTIONS
def perspective_shift(image, corners):
    distance = max(measure_distance(corners[0], corners[1]), measure_distance(corners[1], corners[2]), measure_distance(corners[2], corners[3]), measure_distance(corners[3], corners[0]))
        
    pts1 = np.float32(corners)
    pts2 = np.float32([[0,0],[distance,0],[0,distance],[distance,distance]])
    
    M = cv2.getPerspectiveTransform(pts1,pts2)
    image = cv2.warpPerspective(image, M, (distance,distance))
    
    return image

def get_detection_array(image):
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
    detection_array = np.zeros((8, 8), dtype=int)
    for y in range(8):
        for x in range(8):
            start_point = (int(round(2 * pixel + 10 * pixel * x)), int(round(2 * pixel + 10 * pixel * y)))
            end_point = (int(round(10 * pixel * (x + 1))), int(round(10 * pixel * (y + 1))))

            #cv2.putText(labeled_image, Files[x]+str(8-y), (int(start_point[0] + pixel*2), int(start_point[1] + pixel*4.5)), cv2.FONT_HERSHEY_COMPLEX, 2.0, blue, 4)
            
            
            labeled_image[start_point[1]:end_point[1], start_point[0]:end_point[0]], edge_count[y][x] = edge_detection(labeled_image[start_point[1]:end_point[1], start_point[0]:end_point[0]])

            #cv2.putText(labeled_image, '('+str(x)+','+str(y)+')', (int(start_point[0] + pixel*2.5), start_point[1] + pixel*4), cv2.FONT_HERSHEY_COMPLEX, 1.0, blue, 4)
            
            #cv2.putText(labeled_image, str(start_point), (int(start_point[0] + pixel*0.5), start_point[1] + pixel*4), cv2.FONT_HERSHEY_COMPLEX, 1.0, blue, 4)
            
            if (edge_count[y][x] > 50000):
                labeled_image = cv2.rectangle(labeled_image, start_point, end_point, red, thickness)
                detection_array[y][x] = 1
            else:
                labeled_image = cv2.rectangle(labeled_image, start_point, end_point, green, thickness)
                detection_array[y][x] = 0
    #print(edge_count)
    #print(detection_array)
    return labeled_image, detection_array

def edge_detection(image):
    image = cv2.medianBlur(image,11)
    edges = cv2.Canny(image,80,100)
    #cv2.imshow(str(np.sum(edges)), edges)
    #cv2.imshow("Original", image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return np.stack((edges,)*3, axis=-1), np.sum(edges)

def get_color_array(image, detection_array):
    blue = (255, 0, 0)
    green = (0, 255, 0)
    red = (0, 0, 255)
    thickness = 5
    gray = cv2.cvtColor(copy.copy(image), cv2.COLOR_BGR2GRAY)
    size = gray.shape[0]
    pixel = size / 82
    color_array = np.zeros((8, 8), dtype=int)
    average_empty_white = 0
    average_empty_black = 0
    for y in range(8):
        for x in range(8):
            start_point = (int(round(3 * pixel + 10 * pixel * x)), int(round(3 * pixel + 10 * pixel * y)))
            end_point = (int(round(10 * pixel * (x + 1) - pixel)), int(round(10 * pixel * (y + 1) - pixel)))
            
            average = average_color(gray[start_point[1]:end_point[1], start_point[0]:end_point[0]])
                        
            cv2.putText(gray, str(average), (int(start_point[0]), int(start_point[1] + pixel*4.5)), cv2.FONT_HERSHEY_COMPLEX, 1, blue, 4)
            
    return gray, color_array

def average_color(image):
    if (len(image.shape) > 2):
        average = [0] * 3
        for color in range(image.shape[2]):
            for y in range(image.shape[0]):
                for x in range(image.shape[1]):
                    average[color] += image[y][x][color]
            average[color] = average[color] // (image.shape[0] * image.shape[1])
    else:
        average = 0
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                    average += image[y][x]
            average = round(average / (image.shape[0] * image.shape[1]), 4)
    #print(average)
    return average
                

#-----------------------------------------IMAGE DETECTION MAIN FUNCTIONS
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

def process_frame(frame):
    #Run apriltag detection on image
    detections = detect_apriltags("tag36h11", frame)
    
    #Make copies of original image to keep the same for display 
    apriltagCorners, shifted, cropped, final = copy_image(frame, 4)
            
    #If 4 apriltags are seen (one in each 4 corners of chessboard), crop image and run detection
    if(len(detections) == 4):
        #Get values for the inside corners of each 4 apriltags
        lb, rb, rt, lt = grab_inside_corners(detections)
        
        #Place circles on inside corners of each apriltag
        circle_image(apriltagCorners, (lt, rt, rb, lb), 'red', 'picture')
        
        #Shift perspective to make to make the inside corners of the apriltags the corners of the image
        shifted = perspective_shift(shifted, (lt, rt, lb, rb))
        
        #Go through each square of the chessboard to tell if square is populated with piece
        final, detection_array = get_detection_array(shifted)
        
        gray, color_array = get_color_array(shifted, detection_array)
        
    #Display output images
    #show_images('resize', ('Original', frame), ('Perspective shifted', shifted), ('Labeled Image', final))
    #show_images('resize', ('Final', final))

    #Wait for use to press key
    #cv2.waitKey(0)
    
    #Delete all image windows
    #cv2.destroyAllWindows()
    
    return detection_array, final, gray

#-----------------------------------------CHESS MOVE FUNCTIONS
def create_board_array():
    filenames = ['Icons/rook_black.svg', 'Icons/knight_black.svg', 'Icons/bishop_black.svg', 'Icons/queen_black.svg', 'Icons/king_black.svg', 'Icons/pawn_black.svg', 'Icons/pawn_white.svg', 'Icons/rook_white.svg', 'Icons/knight_white.svg', 'Icons/bishop_white.svg', 'Icons/queen_white.svg', 'Icons/king_white.svg']
    piece_dict = {'r':0, 'n':1, 'b':2, 'q':3, 'k':4, 'p':5, 'P':6, 'R':7, 'N':8, 'B':9, 'Q':10, 'K':11}
    
    position = chess.STARTING_BOARD_FEN.split('/')

    board_array = np.empty(shape=(8, 8, 3),dtype='object')
    
    for y in range(8):
        if (position[y].isdigit() != True):
            for x in range(8):
                board_array[x][y][0] = pygame.image.load(filenames[piece_dict[position[y][x]]])
                board_array[x][y][1] = board_array[x][y][0].get_rect()
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
        #White pieces
        if (destination[1] == 0):
            #Queen promotion
            if (uci[4] == 'q'):
                board_array[destination[0]][destination[1]][0] = pygame.image.load('Icons/queen_white.svg')
                board_array[destination[0]][destination[1]][1] = board_array[destination[0]][destination[1]][0].get_rect()
                board_array[destination[0]][destination[1]][2] = 'Q'
            #Rook promotion
            elif (uci[4] == 'r'):
                board_array[destination[0]][destination[1]][0] = pygame.image.load('Icons/rook_white.svg')
                board_array[destination[0]][destination[1]][1] = board_array[destination[0]][destination[1]][0].get_rect()
                board_array[destination[0]][destination[1]][2] = 'R'
            #Knight promotion
            elif (uci[4] == 'n'):
                board_array[destination[0]][destination[1]][0] = pygame.image.load('Icons/knight_white.svg')
                board_array[destination[0]][destination[1]][1] = board_array[destination[0]][destination[1]][0].get_rect()
                board_array[destination[0]][destination[1]][2] = 'N'
            #Bishop promotion
            elif (uci[4] == 'b'):
                board_array[destination[0]][destination[1]][0] = pygame.image.load('Icons/bishop_white.svg')
                board_array[destination[0]][destination[1]][1] = board_array[destination[0]][destination[1]][0].get_rect()
                board_array[destination[0]][destination[1]][2] = 'B'
        #Black pieces
        elif (destination[1] == 7):
            #Queen promotion
            if (uci[4] == 'q'):
                board_array[destination[0]][destination[1]][0] = pygame.image.load('Icons/queen_black.svg')
                board_array[destination[0]][destination[1]][1] = board_array[destination[0]][destination[1]][0].get_rect()
                board_array[destination[0]][destination[1]][2] = 'q'
            #Rook promotion
            elif (uci[4] == 'r'):
                board_array[destination[0]][destination[1]][0] = pygame.image.load('Icons/rook_black.svg')
                board_array[destination[0]][destination[1]][1] = board_array[destination[0]][destination[1]][0].get_rect()
                board_array[destination[0]][destination[1]][2] = 'r'
            #Knight promotion
            elif (uci[4] == 'n'):
                board_array[destination[0]][destination[1]][0] = pygame.image.load('Icons/knight_black.svg')
                board_array[destination[0]][destination[1]][1] = board_array[destination[0]][destination[1]][0].get_rect()
                board_array[destination[0]][destination[1]][2] = 'n'
            #Bishop promotion
            elif (uci[4] == 'b'):
                board_array[destination[0]][destination[1]][0] = pygame.image.load('Icons/bishop_black.svg')
                board_array[destination[0]][destination[1]][1] = board_array[destination[0]][destination[1]][0].get_rect()
                board_array[destination[0]][destination[1]][2] = 'b'
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

def refresh_board_array(board):
    filenames = ['Icons/rook_black.svg', 'Icons/knight_black.svg', 'Icons/bishop_black.svg', 'Icons/queen_black.svg', 'Icons/king_black.svg', 'Icons/pawn_black.svg', 'Icons/pawn_white.svg', 'Icons/rook_white.svg', 'Icons/knight_white.svg', 'Icons/bishop_white.svg', 'Icons/queen_white.svg', 'Icons/king_white.svg']
    piece_dict = {'r':0, 'n':1, 'b':2, 'q':3, 'k':4, 'p':5, 'P':6, 'R':7, 'N':8, 'B':9, 'Q':10, 'K':11}
    position = board.fen().split()[0].split('/')
    print(position)

    board_array = np.empty(shape=(8, 8, 3),dtype='object')
    
    for y in range(8):
        x = 0
        index = 0
        for character in position[y]:
            if (character.isdigit()):
                
                for i in range(int(character)):
                    print(x,y)
                    board_array[x+i][y] = (None, None, None)
                x += int(character)
            else:
                print(x,y)
                board_array[x][y][0] = pygame.image.load(filenames[piece_dict[position[y][index]]])
                board_array[x][y][1] = board_array[x][y][0].get_rect()
                board_array[x][y][2] = position[y][index]
                x += 1
            index += 1
    return board_array

def detection_array_to_uci(old_array, new_array, board=None):
    #Reference string for printing out the square file
    files = 'abcdefgh'
    
    #If the same amount of pieces are on the board (no captures, just a piece moving)
    if (np.sum(old_array) == np.sum(new_array)):
        destination = ''
        source = ''
        for y in range(8):
            for x in range(8):
                if (old_array[y][x] != new_array[y][x]):
                    if (new_array[y][x]):
                        destination = str(files[x]) + str(8-y)
                    else:
                        source = str(files[x]) + str(8-y)
        move = source + destination
        return move
    if (np.sum(old_array) - 1 == np.sum(new_array)):
        print("Capture")
        

        
#-----------------------------------------CHESS GUI FUNCTIONS
def create_gui():
    pygame.init()
    square_size = 70
    window = pygame.display.set_mode((square_size * 8, square_size * 8))
    pygame.display.set_caption('Chessboard')
    icon=pygame.image.load('Icons/icon.png')
    pygame.display.set_icon(icon)
    window = draw_grid(window, square_size)
    pygame.display.flip()
    return window, square_size

def draw_grid(window, square_size):
    for rank in range(8):
        for File in range(8):
            if ((File + rank % 2) % 2 == 0):
                color = (235, 236, 208) #tan
            else:
                color = (119, 149, 86) #green
            pygame.draw.rect(window, color, pygame.Rect(File * square_size, rank * square_size, square_size, square_size))
    return window

def print_board(window, board_array, square_size):
    draw_grid(window, square_size)
    for y in range(8):
        for x in range(8):
            if (board_array[x][y][0] != None):
                board_array[x][y][1].centerx = (square_size // 2) + square_size * x
                board_array[x][y][1].centery = (square_size // 2) + square_size * y
                window.blit(board_array[x][y][0], board_array[x][y][1])
    pygame.display.flip()

window, square_size = create_gui()

board_array = create_board_array()

board = chess.Board()

print_board(window, board_array, square_size)

old_detection_array, final_image, gray = process_frame(cv2.imread('TestingImages/ItalianGame1.jpg'))
show_images('resize', ('Final Move 1', final_image), ('Color Values Move 1', gray))
cv2.waitKey(0)

for i in range(2, 7):
    new_detection_array, final_image, gray = process_frame(cv2.imread('TestingImages/ItalianGame' + str(i) + '.jpg'))
    cv2.destroyAllWindows()
    show_images('resize', ('Final Move '+str(i), final_image), ('Color Values Move '+str(i), gray))
    move = detection_array_to_uci(old_detection_array, new_detection_array)
    move = chess.Move.from_uci(move)
    board_array = update_board_array_uci(board, board_array, chess.Move.uci(move))
    board.push(move)
    print_board(window, board_array, square_size)
    old_detection_array = new_detection_array
    
    cv2.waitKey(0)

time.sleep(2)

#new_detection_array = picture_method('TestingImages/ItalianGame_CaptureTesting1.jpg')
#detection_array_to_uci(old_detection_array, new_detection_array)
#old_detection_array = new_detection_array

#new_detection_array = picture_method('TestingImages/ItalianGame_CaptureTesting2.jpg')
#detection_array_to_uci(old_detection_array, new_detection_array)
#old_detection_array = new_detection_array

#new_detection_array = picture_method('TestingImages/ItalianGame_CaptureTesting3.jpg')
#detection_array_to_uci(old_detection_array, new_detection_array)
#old_detection_array = new_detection_array

#new_detection_array = picture_method('TestingImages/ItalianGame_CaptureTesting4.jpg')
#detection_array_to_uci(old_detection_array, new_detection_array)
#old_detection_array = new_detection_array

#new_detection_array = picture_method('TestingImages/ItalianGame_CaptureTesting5.jpg')
#detection_array_to_uci(old_detection_array, new_detection_array)
#old_detection_array = new_detection_array

#new_detection_array = picture_method('TestingImages/ItalianGame_CaptureTesting6.jpg')
#detection_array_to_uci(old_detection_array, new_detection_array)
#old_detection_array = new_detection_array

#new_detection_array = picture_method('TestingImages/ItalianGame_CaptureTesting7.jpg')
#detection_array_to_uci(old_detection_array, new_detection_array)
#old_detection_array = new_detection_array
