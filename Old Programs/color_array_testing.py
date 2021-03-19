import numpy as np
array1 = np.array([[1, 1, 1, 1, 1, 1, 1, 1], 
                   [1, 1, 1, 0, 1, 0, 1, 1], 
                   [0, 0, 0, 0, 0, 0, 0, 0], 
                   [0, 0, 0, 1, 2, 1, 0, 0], 
                   [0, 0, 0, 0, 0, 0, 0, 0], 
                   [0, 0, 0, 0, 0, 0, 0, 0],
                   [2, 2, 2, 2, 0, 2, 2, 2],
                   [2, 0, 2, 2, 2, 2, 2, 2]])

array2 = np.array([[1, 1, 1, 1, 1, 1, 1, 1], 
                   [1, 1, 1, 0, 1, 0, 1, 1], 
                   [0, 0, 0, 2, 0, 0, 0, 0], 
                   [0, 0, 0, 0, 0, 1, 0, 0], 
                   [0, 0, 0, 0, 0, 0, 0, 0], 
                   [0, 0, 0, 0, 0, 0, 0, 0],
                   [2, 2, 2, 2, 0, 2, 2, 2],
                   [2, 0, 2, 2, 2, 2, 2, 2]])


def array_to_uci(old_array, new_array):
    files = 'abcdefgh'
    name_matrix = np.empty((8, 8), dtype=object)
    for y in range(8):
        for x in range(8):
            name_matrix[y][x] = files[x] + str(8-y)
    
    difference = new_array - old_array
            
    move = ['', '']
    
    if (np.count_nonzero(difference) == 3):
        flat_difference = difference.flatten()
        abs_difference = np.abs(flat_difference)
        min_val = np.argmin(np.bincount(abs_difference))
        min_val = min_val * np.ones((8,8), dtype=int) * (difference != 0).tolist()
        difference = np.abs(difference) - min_val
    if (np.count_nonzero(difference) == 2):   
        modifier = ''
        move[ min(new_array[np.nonzero(difference)][0], 1)] = name_matrix[np.nonzero(difference)][0]
        move[ min(new_array[np.nonzero(difference)][1], 1)] = name_matrix[np.nonzero(difference)][1]
        if (int(move[1][1]) == 8 or int(move[1][1]) == 0):
            print("Check for pawn promotion")
        move = move[0] + move[1] + modifier
    
    elif (np.count_nonzero(difference) == 4):
        if (np.nonzero(difference)[0][0] == 7 and max(np.nonzero(difference)[1]) > 4):
            move = 'e1g1'
        elif (np.nonzero(difference)[0][0] == 7 and max(np.nonzero(difference)[1]) == 4):
            move = 'e1c1'
        elif (np.nonzero(difference)[0][0] == 0 and max(np.nonzero(difference)[1]) > 4):
            move = 'e8g8'
        elif (np.nonzero(difference)[0][0] == 0 and max(np.nonzero(difference)[1]) == 4):
            move = 'e8c8'
    return move
            
print(array_to_uci(array1, array2))
