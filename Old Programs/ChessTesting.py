from os import environ
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import chess, numpy, time, pygame, chess.engine

filenames = ['Icons/rook_black.svg', 'Icons/knight_black.svg', 'Icons/bishop_black.svg', 'Icons/queen_black.svg', 'Icons/king_black.svg', 'Icons/pawn_black.svg', 'Icons/pawn_white.svg', 'Icons/rook_white.svg', 'Icons/knight_white.svg', 'Icons/bishop_white.svg', 'Icons/queen_white.svg', 'Icons/king_white.svg']
piece_dict = {'r':0, 'n':1, 'b':2, 'q':3, 'k':4, 'p':5, 'P':6, 'R':7, 'N':8, 'B':9, 'Q':10, 'K':11}

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
    position = chess.STARTING_BOARD_FEN.split('/')

    board_array = numpy.empty(shape=(8, 8, 3),dtype='object')
    
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
    position = board.fen().split()[0].split('/')
    print(position)

    board_array = numpy.empty(shape=(8, 8, 3),dtype='object')
    
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

engine = chess.engine.SimpleEngine.popen_uci("/home/blemay360/Documents/stockfish_13_linux_x64/stockfish_13_linux_x64")

done = False

while not board.is_game_over() and not done:
    #white_move = chess.Move.from_uci(input())
    #valid = False
    #while not valid:
        #try:
            #white_move = board.parse_san(input())
        #except ValueError:
            #print('ValueError, try again')
        #else:
            #valid = True
    #board_array = update_board_array_uci(board, board_array, chess.Move.uci(white_move))
    #board.push(white_move)
    #print_board(window, board_array, square_size)
    
    result = engine.play(board, chess.engine.Limit(time=0.01))
    board_array = update_board_array_uci(board, board_array, chess.Move.uci(result.move))
    #board_array = refresh_board_array(board)
    board.push(result.move)
    i = 0
    while not done and i < 9:
        print_board(window, board_array, square_size)
        time.sleep(0.1)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        i += 1

engine.quit()

#done = False
#while not done:
    #for event in pygame.event.get():
        #if event.type == pygame.QUIT:
            #done = True
    #pygame.display.flip()
