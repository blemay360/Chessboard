from os import environ
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import chess, numpy, time, pygame

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

def create_board_array():
    board = chess.Board()
    
    position = board.fen().split()[0].split('/')

    board_array = numpy.empty(shape=(8, 8, 3),dtype='object')
    
    filenames = ['Icons/rook_black.svg', 'Icons/knight_black.svg', 'Icons/bishop_black.svg', 'Icons/queen_black.svg', 'Icons/king_black.svg', 'Icons/pawn_black.svg', 'Icons/pawn_white.svg', 'Icons/rook_white.svg', 'Icons/knight_white.svg', 'Icons/bishop_white.svg', 'Icons/queen_white.svg', 'Icons/king_white.svg']
    piece_dict = {'r':0, 'n':1, 'b':2, 'q':3, 'k':4, 'p':5, 'P':6, 'R':7, 'N':8, 'B':9, 'Q':10, 'K':11}
    
    for y in range(8):
        if (position[y].isdigit() != True):
            for x in range(8):
                board_array[x][y][0] = pygame.image.load(filenames[piece_dict[position[y][x]]])
                board_array[x][y][1] = board_array[x][y][0].get_rect()
                board_array[x][y][2] = position[y][x]
    return board, board_array

def update_board_array(board, board_array):
    #Break up the fen notation for the current chess board into a list with 8 strings detailing each rank of the board
    position = board.fen().split()[0].split('/')
    
    piece_moved_out = []
    piece_moved_in = []
    piece_replaced = []
    
    #Iterate through each y coordinate (rank) of the board
    for y in range(8):
        #Start a counter to keep track of the x coordinate (file)
        x = 0
        #Iterate through each character in the string representation of the file (can be variable length strings)
        for character in position[y]:
            #If the character is a number (empty square(s))
            if (character.isdigit()):
                #Start a for loop for the number of empty squares in this section
                for i in range(int(character)):
                    #If the board_array shows has populated square that is now empty
                    if (board_array[x + i][y][2] != None):
                        piece_moved_out.append((x+i, y))
                #Update x coordinate to skip the empty squares
                x += int(character)
            #If the character is a string (populated square)
            else:
                #If the board_array shows has a empty spot that now has a piece
                if (board_array[x][y][2] == None):
                    piece_moved_in.append((x,y))
                #If the board_array has the wrong piece in a spot
                elif (board_array[x][y][2] != character):
                    piece_replaced.append((x, y))
                #Increment x coordinate to move to next square
                x += 1
                
    if (len(piece_moved_out) != len(piece_moved_in) + len(piece_replaced)):
        print("Piece move mismatch")
      
    if (len(piece_moved_out) == 2):
        if (piece_moved_out[0][1] == 0):
            print("Black is castling")
        elif (piece_moved_out[0][1] == 7):
            print("White is castling")
    elif (piece_moved_in):
        board_array[piece_moved_in[0][0]][piece_moved_in[0][1]] = board_array[piece_moved_out[0][0]][piece_moved_out[0][1]]
        board_array[piece_moved_out[0][0]][piece_moved_out[0][1]] = (None, None, None)
    elif (piece_replaced):
        board_array[piece_replaced[0][0]][piece_replaced[0][1]] = board_array[piece_moved_out[0][0]][piece_moved_out[0][1]]
        board_array[piece_moved_out[0][0]][piece_moved_out[0][1]] = (None, None, None)
    return board_array

def print_board(window, board_array, square_size):
    draw_grid(window, square_size)
    for y in range(8):
        for x in range(8):
            if (board_array[x][y][0] != None):
                board_array[x][y][1].centerx = (square_size // 2) + square_size * x
                board_array[x][y][1].centery = (square_size // 2) + square_size * y
                window.blit(board_array[x][y][0], board_array[x][y][1])
    pygame.display.flip()

def get_position_difference(position1, position2):
    position1 = position1.split()[0].split('/')
    position2 = position2.split()[0].split('/')
    for rank in range(8):
        if (position1[rank] != position2[rank]):
            print(position1[rank], position2[rank])

window, square_size = create_gui()

board, board_array = create_board_array()

#for y in range(8):
    #for x in range(8):
        #if (board_array[x][y][2] != None):
            #print(board_array[x][y][2], end=' ')
    #print()

print_board(window, board_array, square_size)
time.sleep(2)


#position = last_position = board.fen()

#first_piece_position(window, position, pieces, piece_dict, square_size)

board.push_san("e4")

board_array = update_board_array(board, board_array)
print_board(window, board_array, square_size)
time.sleep(2)

board.push_san("e5")
board_array = update_board_array(board, board_array)
print_board(window, board_array, square_size)
time.sleep(2)

board.push_san("Qh5")
board_array = update_board_array(board, board_array)
print_board(window, board_array, square_size)
time.sleep(2)

board.push_san("Nc6")
board_array = update_board_array(board, board_array)
print_board(window, board_array, square_size)
time.sleep(2)

board.push_san("Bc4")
board_array = update_board_array(board, board_array)
print_board(window, board_array, square_size)
time.sleep(2)

board.push_san("Nf6")
board_array = update_board_array(board, board_array)
print_board(window, board_array, square_size)
time.sleep(2)

board.push_san("Qxf7")
board_array = update_board_array(board, board_array)
print_board(window, board_array, square_size)
time.sleep(2)


#done = False
#while not done:
    #for event in pygame.event.get():
        #if event.type == pygame.QUIT:
            #done = True
    #pygame.display.flip()
