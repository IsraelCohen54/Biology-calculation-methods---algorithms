# test
import pygame
import random
import numpy as np
import copy


BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 175, 0)
RED = (200, 10, 10)
YELLOW = (200, 200, 10)
BLUE = (0, 32, 255)
WINDOW_HEIGHT = 900
WINDOW_WIDTH = 1760

# global variables:
healthy_cells = []  # list with location of healthy people. format: [[i,j],[i,j]...]
sick_cells_with_gen = []  # list with location of sick people, and the num of generations. format: [[i,j,gen],[i,j,gen]...]
immuned_cells = []  # list with location of immuned/recovered people. format: [[i,j],[i,j]...]


def drawGrid(matrix):  # given a matrix, draw matrix on screen
    blockSize = 3  # the size of cell
    i = 0  # start from row zero
    j = 0  # start from column zero
    for y in range(2, WINDOW_HEIGHT - 5, blockSize + 2):
        for x in range(2, WINDOW_WIDTH - 5, blockSize + 2):
            rect = pygame.Rect(x, y, blockSize, blockSize)
            try:

                if (matrix[i][j] == 0):
                    pygame.draw.rect(SCREEN, BLUE, rect)
                elif (matrix[i][j] == 1):
                    pygame.draw.rect(SCREEN, RED, rect)
                elif (matrix[i][j] == 2):
                    pygame.draw.rect(SCREEN, GREEN, rect)
                elif (matrix[i][j] == 3):
                    pygame.draw.rect(SCREEN, WHITE, rect)

            except:
                continue


            j = j + 1  # next column
        j = 0  # return to the start of the column
        i = i + 1  # next line


def random_matrix(num_of_healthy, num_of_sick,num_of_immuned, matrix_hight, matrix_width):  # create matrix from given parameters, cells are randomly spread
    matrix = []
    for i in range(matrix_hight):  # create a matrix with empty cells
        row = []
        for j in range(matrix_width):
            row.append(3)
        matrix.append(row)

    x = [*range(0, matrix_hight, 1)]  # create a list with all possible cells indexes
    y = [*range(0, matrix_width, 1)]
    all_cells = []
    for i in (x):
        for j in (y):
            all_cells.append([i, j])

    occupied = []  # create list with indexes of cell that are occupied

    for i in range(num_of_healthy):  # place healthy people in matrix and occupied list
        choice = random.choice(all_cells)  # choose random cell from list
        healthy_cells.append(choice)  # add to healthy list
        all_cells.remove(choice)  # remove from list to prevent choosing the same cell
        occupied.append(choice)  # add to occupied list
        matrix[choice[0]][choice[1]] = 0  # mark cell as healthy

    for i in range(num_of_sick):
        choice = random.choice(all_cells)
        occupied.append(choice)
        sick_cell = copy.deepcopy(choice)
        sick_cell.append(0)  # 0 for Generations counter
        sick_cells_with_gen.append(sick_cell)
        all_cells.remove(choice)
        matrix[choice[0]][choice[1]] = 1

    for i in range(num_of_immuned):
        choice = random.choice(all_cells)
        occupied.append(choice)
        immuned_cell = copy.deepcopy(choice)
        immuned_cells.append(immuned_cell)
        all_cells.remove(choice)
        matrix[choice[0]][choice[1]] = 2
    return [matrix, occupied]


def check_sick_at_every_direction(i, cell_type, list_type):
    if list_type ==0:
        a = healthy_cells[i][0]
        b = healthy_cells[i][1]
    else:
        a = immuned_cells[i][0]
        b = immuned_cells[i][1]




    # for wrap_around - meaning cell is withing matrix do check:
    if a != 0 and a != (matrix_height - 1) and b != 0 and b != (matrix_width - 1):  # TODO matrix len isnt= wind size(40)
        if matrix[a + 1][b] == cell_type or matrix[a][b + 1] == cell_type or matrix[a - 1][b] == cell_type or matrix[a][
            b - 1] == cell_type \
                or matrix[a + 1][b + 1] == cell_type or matrix[a - 1][b - 1] == cell_type or matrix[a - 1][
            b + 1] == cell_type or \
                matrix[a + 1][b - 1] == cell_type:  # meaning next to sick cell, somewhere in the middle of matrix
            return True
    elif a == matrix_height - 1 and b != 0 and b != matrix_width - 1:  # if cell is withing middle bottom bound check:
        if matrix[0][b] == cell_type or matrix[a][b + 1] == cell_type or matrix[a - 1][b] == cell_type or matrix[a][
            b - 1] == cell_type \
                or matrix[0][b + 1] == cell_type or matrix[a - 1][b - 1] == cell_type or matrix[a - 1][
            b + 1] == cell_type or matrix[0][
            b - 1] == cell_type:
            return True
    elif a == 0 and b != 0 and b != matrix_width - 1:  # if cell is withing middle upper bound check:
        if matrix[a + 1][b] == cell_type or matrix[a][b + 1] == cell_type or matrix[matrix_height - 1][b] == cell_type or \
                matrix[a][b - 1] == cell_type \
                or matrix[a + 1][b + 1] == cell_type or matrix[matrix_height - 1][b - 1] == cell_type or \
                matrix[matrix_height - 1][
                    b + 1] == cell_type or matrix[a + 1][b - 1] == cell_type:
            return True
    elif b == matrix_width - 1 and a != 0 and a != matrix_height - 1:  # if cell is withing middle right bound check:
        if matrix[a + 1][b] == cell_type or matrix[a][0] == cell_type or matrix[a - 1][b] == cell_type or matrix[a][
            b - 1] == cell_type \
                or matrix[a + 1][0] == cell_type or matrix[a - 1][b - 1] == cell_type or matrix[a - 1][
            0] == cell_type or matrix[a + 1][
            b - 1] == cell_type:
            return True
    elif b == 0 and a != 0 and a != matrix_height - 1:  # if cell is withing middle left bound check:
        if matrix[a + 1][b] == cell_type or matrix[a][b + 1] == cell_type or matrix[a - 1][b] == cell_type or matrix[a][
            matrix_width - 1] == cell_type \
                or matrix[a + 1][b + 1] == cell_type or matrix[a - 1][matrix_width - 1] == cell_type or matrix[a - 1][
            b + 1] == cell_type or \
                matrix[a + 1][matrix_width - 1] == cell_type:
            return True
    elif a == 0 and b == 0:  # if cell is top left bound check:
        if matrix[a + 1][b] == cell_type or matrix[a][b + 1] == cell_type or matrix[matrix_height - 1][b] == cell_type or \
                matrix[a][
                    matrix_width - 1] == cell_type \
                or matrix[a + 1][b + 1] == cell_type or matrix[matrix_height - 1][matrix_width - 1] == cell_type or \
                matrix[matrix_height - 1][b + 1] == cell_type or matrix[a + 1][matrix_width - 1] == cell_type:
            return True
    elif a == 0 and b == matrix_width - 1:  # if cell is top right bound check:
        if matrix[a + 1][b] == cell_type or matrix[a][0] == cell_type or matrix[matrix_height - 1][b] == cell_type or \
                matrix[a][b - 1] == cell_type \
                or matrix[a + 1][0] == cell_type or matrix[matrix_height - 1][b - 1] == cell_type or \
                matrix[matrix_height - 1][0] == cell_type or \
                matrix[a + 1][b - 1] == cell_type:
            return True
    elif a == matrix_height - 1 and b == 0:  # if cell is bottom left bound check:
        if matrix[0][b] == cell_type or matrix[a][b + 1] == cell_type or matrix[a - 1][b] == cell_type or matrix[a][
            matrix_width - 1] == cell_type \
                or matrix[0][b + 1] == cell_type or matrix[a - 1][matrix_width - 1] == cell_type or matrix[a - 1][
            b + 1] == cell_type or \
                matrix[0][matrix_width - 1] == cell_type:
            return True
    elif a == matrix_height - 1 and b == matrix_width - 1:  # if cell is bottom right bound check:
        if matrix[0][b] == cell_type or matrix[a][0] == cell_type or matrix[a - 1][b] == cell_type or matrix[a][
            b - 1] == cell_type \
                or matrix[0][0] == cell_type or matrix[a - 1][b - 1] == cell_type or matrix[a - 1][0] == cell_type or \
                matrix[0][b - 1] == cell_type:
            return True
    return False


def generation_passed():
    for i in range(len(sick_cells_with_gen)):
        sick_cells_with_gen[i][2] += 1


def check_if_got_immuned():  # by generations
    mat_immuned_indeces = []
    len_num = len(sick_cells_with_gen)
    for i in range(len_num):
        j = len_num - i - 1
        if sick_cells_with_gen[j][2] == generations_to_immunity:
            mat_immuned_indeces.append(sick_cells_with_gen[j][0])
            mat_immuned_indeces.append(sick_cells_with_gen[j][1])
            immuned_cells.append(mat_immuned_indeces)
            mat_immuned_indeces = []
            sick_cells_with_gen.remove(sick_cells_with_gen[j])


def update_cell_state():
    check = 0
    #changing cell state from healthy to sick:
    next_to_sick = False
    sick = 1
    healthy = 0
    healthy_list_len = len(healthy_cells)
    j = 0
    if len(healthy_cells) != 0:  # list isn't empty
        for i in range(healthy_list_len):
            j = healthy_list_len - 1 - i
            if matrix[healthy_cells[j][0]][healthy_cells[j][1]] == 0:  # the cell is healthy
                if (len(sick_cells_with_gen) != 0):  # list sicks isn't empty
                    next_to_sick = check_sick_at_every_direction(j, sick, healthy)
                    if next_to_sick:  # if cell indeed next to sick
                        check = np.random.choice(np.arange(0, 2), p=[1 - infection_prob_healthy,
                                                                     infection_prob_healthy])  # check if got infected by sick cell - 0.3 to 0, 0.7 to 1 (1 = infected)
                if check == 0:  # didn't got infected
                    continue
                else:  # change healthy and sick lists:
                    temp = copy.deepcopy(healthy_cells[j])
                    temp.append(0)  # add generations = 0
                    sick_cells_with_gen.append(temp)
                    healthy_cells.remove(healthy_cells[j])
                    check = 0
    # changing cell state from immuned to sick:
    next_to_sick = False
    immuned_cells_len = len(immuned_cells)
    j = 0
    immuned = 2
    if len(immuned_cells) != 0:  # list isn't empty
        for i in range(immuned_cells_len):
            j = immuned_cells_len - 1 - i
            if matrix[immuned_cells[j][0]][immuned_cells[j][1]] == 2:  # the cell is immuned
                if (len(sick_cells_with_gen) != 0):  # list sicks isn't empty
                    next_to_sick = check_sick_at_every_direction(j, sick, immuned)
                    if next_to_sick:  # if cell indeed next to sick
                        check = np.random.choice(np.arange(0, 2), p=[1 - infection_prob_immuned,
                                                                     infection_prob_immuned])  # check if got infected by less probability to 1 (1 = infected, 0 = not infected))
                if check == 0:  # didn't got infected
                    continue
                else:  # change healthy and sick lists:
                    temp = copy.deepcopy(immuned_cells[j])
                    temp.append(0)  # add generations = 0
                    sick_cells_with_gen.append(temp)
                    immuned_cells.remove(immuned_cells[j])
                    check = 0

    generation_passed()  # add +1 to generation off sick cells
    check_if_got_immuned()  # check if got immuned

    if len(sick_cells_with_gen) != 0:  # list isn't empty
        for i in range(len(sick_cells_with_gen)):
            matrix[sick_cells_with_gen[i][0]][sick_cells_with_gen[i][1]] = 1  # update sicks #TODO check dim
    if len(immuned_cells) != 0:  # list isn't empty
        for i in range(len(immuned_cells)):
            matrix[immuned_cells[i][0]][immuned_cells[i][1]] = 2  # update immunes #TODO check dim



def move_all_cells(matrix, occupied):  # move matrix in 1 generation


    sick_dic = {}   #make a dictionary from sick list for better performance
    for [i,j,k] in sick_cells_with_gen:
        sick_dic[i,j] = k

    matrix = np.array(matrix)

    row, col = matrix.shape
    next_occupied = []  # list of indexes that cells moved to these indexes. format: [[i,j],[i,j]]
    for [i, j] in occupied:  # for every person in matrix/occupied
        free_space = False  # flag that indicates if the next cell if a free cell and can be moved to
        options = ["stay", "up", "down", "left", "right", "left-up", "left-down", "right-up", "right-down"]
        while free_space == False: #while we havn't found free space to move to

            if len(options) == 0: # if there is no options move-stay in the same place
                next_occupied.append([i, j])
                break


            choice = random.choice(options) #choose random move
            options.remove(choice)  # remove from list to prevent rechoose


            next_i, next_j = next_cell(i, j, choice, row, col) #calculate next cell indexes


            if (matrix[next_i, next_j] == 3): # if there is no one in this cell

                next_occupied.append([next_i, next_j]) #append
                matrix[next_i, next_j] = matrix[i, j]  # move matrix
                matrix[i, j] = 3


                if matrix[next_i,next_j] == 1: # if the cell is infected, update infected list
                    gen = sick_dic[i,j]
                    sick_dic[next_i,next_j] = sick_dic[i,j]
                    del sick_dic[i,j]
                    index = sick_cells_with_gen.index([i,j,gen])
                    sick_cells_with_gen[index] = [next_i, next_j, gen]

                free_space = True


    healthy_cells.clear()   #update lists
    immuned_cells.clear()

    matrix = matrix.tolist()
    for i in range(row):
        for j in range(col):
            if (matrix[i][j] == 0):
                healthy_cells.append([i,j])
            if (matrix[i][j] == 2):
                immuned_cells.append([i,j])

    return matrix, next_occupied



def next_cell(i, j, choice, row, col):  # find next cell when choice and matrix size is given
    next_i = i
    next_j = j

    if (choice == "stay"):
        next_i = i
        next_j = j

    if (choice == "left-up"):
        next_i, next_j = next_cell(i, j, "left", row, col)
        next_i, next_j = next_cell(next_i, next_j, "up", row, col)

    if (choice == "left-down"):
        next_i, next_j = next_cell(i, j, "left", row, col)
        next_i, next_j = next_cell(next_i, next_j, "down", row, col)

    if (choice == "right-up"):
        next_i, next_j = next_cell(i, j, "right", row, col)
        next_i, next_j = next_cell(next_i, next_j, "up", row, col)

    if (choice == "right-down"):
        next_i, next_j = next_cell(i, j, "right", row, col)
        next_i, next_j = next_cell(next_i, next_j, "down", row, col)

    if (choice == "right"):
        if (j == col - 1):  # edge
            next_i = i
            next_j = 0
        else:
            next_i = i
            next_j = j + 1

    if (choice == "down"):
        if (i == row - 1):  # edge
            next_i = 0
            next_j = j
        else:
            next_i = i + 1
            next_j = j

    if (choice == "up"):

        if (i == 0): # edge
            next_i = row - 1
            next_j = j
        else:
            next_i = i - 1
            next_j = j

    if (choice == "left"):
        next_i = i
        next_j = j - 1

        if (j == 0): # edge
            next_i = i
            next_j = col - 1

    return [next_i, next_j]


def print_matrix(matrix): # print matrix for debugging
    matrix = np.array(matrix)
    row, col = matrix.shape
    for i in range(row):
        for j in range(col):
            if (matrix[i][j] != 3):
                print([i, j])
    print("\n")


from pygame.locals import *


def get_parameters_from_user(text_to_show): #get input from user
    X = 700
    Y = 360
    pygame.init()
    screen = pygame.display.set_mode((X, Y))
    pygame.display.set_caption('Show Text')
    font = pygame.font.Font(None, 30)
    text = font.render(text_to_show, True, WHITE, BLUE)
    textRect = text.get_rect()
    textRect.center = (250, 50)
    input = ''
    while True:
        for evt in pygame.event.get():
            if evt.type == KEYDOWN:
                if evt.key == pygame.K_RETURN:
                    return input
                elif evt.key == pygame.K_BACKSPACE:
                    input = input[:-1]
                else:
                    input += evt.unicode
            elif evt.type == QUIT:
                return
        screen.fill(BLUE)
        screen.blit(text, textRect)
        block = font.render(input, True, (255, 255, 255))
        rect = block.get_rect()
        rect.center = screen.get_rect().center
        screen.blit(block, rect)
        pygame.display.flip()


if __name__ == '__main__':
    matrix_height= 200
    matrix_width= 350

    # 0 healthy BLUE
    # 1 sick RED
    # 2 immuned/recovred GREEN
    # 3 empty WHITE

    # simulation parameters:

    infection_prob_immuned = 0.05
    infection_prob_healthy = 0.8

    generations_to_immunity = 7

    sick = 500
    healthy = 500
    immuned = 500

    user_choice = True
    choice = get_parameters_from_user('Would you like to use default parameters? (yes/no)')
    if (choice == 'yes'):
        user_choice = False
    if (choice == 'no'):
        user_choice = True

    if (user_choice== True):
        infection_prob_immuned = float(get_parameters_from_user('Enter p for infection of immuned person:'))
        infection_prob_healthy = float(get_parameters_from_user('Enter p for infection of healthy person:'))
        generations_to_immunity =  int(get_parameters_from_user('Enter generation for immunity'))

        sick = int(get_parameters_from_user('Enter number of infected people'))
        healthy = int(get_parameters_from_user('Enter number of healthy people'))
        immuned = int(get_parameters_from_user('Enter number of immuned people'))




    generations = 0


    # create random matrix
    matrix, occupied = random_matrix(num_of_sick=sick, num_of_healthy=healthy, num_of_immuned=immuned, matrix_hight= matrix_height, matrix_width=matrix_width)
    global SCREEN, CLOCK
    pygame.init()
    SCREEN = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    CLOCK = pygame.time.Clock()

    finish = False

    while not finish:
        drawGrid(matrix)  # draw matrix

        for event in pygame.event.get(): # find if window close button is pressed
            if event.type == pygame.QUIT:
                finish = True

        pygame.display.update()
        pressed = False
        while (pressed == False):
            for event in pygame.event.get():
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if (event.button == 3): #printing for debbuging
                        print_matrix(matrix)
                        print(occupied, '\n')
                    else:
                        pressed = True

        update_cell_state()  # update infection in all cells
        matrix, occupied = move_all_cells(matrix, occupied)  # move all cells
        generations +=1
        print("gen num:", generations)
        if (len(sick_cells_with_gen) == 0):
            print("the plague is gone")
            finish = True
        pygame.display.update()
