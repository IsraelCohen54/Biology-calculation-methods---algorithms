import numpy as np
import math
import sys
import random
from random import choice
from pandas import *
import itertools
import pandas as pd

# hyper params:
matrix_dim = 25
mat_num = 200
restricion_len = 8


# The func add black square (value 1) to row matrix by 8 restriction from (thr row restriction, sec half of) configuration
# than by choss to fill by percentage, if more black in line, than more likly to choose black, after black, put at least one space
# first row and first column generated to 0, to held correctness per row/column
# for more understanding, dig in! :)
def mat_construct_at_begin(matrix):
    matrix.append([0 for i in range(matrix_dim + 1)])  # to grade columns
    first = False
    row = []
    temp_black_list = []
    temp_space_list = []

    data_index = 0
    for i in range(matrix_dim + 1):  # create a matrix with cell according to rows params
        space_holder = 0
        num_black_square = 0  # how many black square are there
        num_black_parts = 0
        # grade row
        if first == False:
            first = True
            continue

        for z in range(restricion_len):  # check how many blanks are there to make random space, like 1101, or 1011 etc.
            if int(rows_data[data_index]) == 0:
                space_holder += 1
            else:
                num_black_square += int(rows_data[data_index])
                num_black_parts += 1
                temp_black_list.append(int(rows_data[data_index]))
            data_index += 1

        for i in range(matrix_dim - num_black_square):
            temp_space_list.append([1])  # num of spaces

        row.append(0)  # first column grade
        if (num_black_square == matrix_dim):
            row.append([1 for i in range(matrix_dim)])
            continue
        else:
            while len(temp_space_list) != 0 or len(temp_black_list) != 0:  # filliing single row
                # Bigger chances to choose from what we have more:

                if (num_black_parts >= len(temp_space_list) and num_black_parts != 1 and len(temp_black_list) != 0):
                    choose = 1
                else:
                    if len(temp_space_list) != 0:
                        choose = np.random.choice(np.arange(0, 2), p=[1 - (len(temp_black_list) / matrix_dim),
                                                                      (len(temp_black_list) / matrix_dim)])
                    else:
                        choose = np.random.choice(np.arange(0, 2), p=[(len(temp_space_list) / matrix_dim),
                                                                      1 - (len(temp_space_list) / matrix_dim)])

                if (choose == 0):
                    row.append(0)
                    temp_space_list.pop()
                    continue
                fill_black = int(temp_black_list.pop(0))
                for i in range(fill_black):
                    row.append(1)
                if len(temp_space_list) != 0:  # add minimum space between black sequence
                    row.append(0)
                    temp_space_list.pop()

            # as every seq of black have at least one space after it except the last
        temp_space_list.clear()
        temp_black_list.clear()

        matrix.append(row.copy())
        row.clear()


# calculating fitness by evaluating restriction and sum rows&cols success, the sum updated to (0,0) in each mat
def fitness_calc(dict, row_restric, column_restric):
    # Change to np.array
    for i in range(mat_num):
        dict[i] = np.array(dict[i])

    # ~~delete~~
    # print(dict[0][2,:]) #grab row 2
    # print(dict[0][:,2]) #column 2
    # run from 1 to 25, enter values in zero...
    # ~~delete~~

    sum_row_grade = 0
    sum_column_grade = 0
    row_state = -1  # while checking row - if failed or not.. 0 = failure, 1 = success
    lst_rest = []

    for key in dict:  # mat index
        row_res_cpy = row_restric.copy()
        jump_grade_row = 0
        for row_num, i in enumerate(dict[key]):  # specific mat rows
            if jump_grade_row == 0:
                jump_grade_row = 1
                continue
            jump_grade_num = 0
            if len(row_res_cpy) != 0:
                for j in range(restricion_len):  # get 8 restrictions to fulfill
                    lst_rest.append(int(row_res_cpy.pop(0)))
            else:
                break
            # strategy: check out only black square.
            row_temp = []
            row_temp = i.copy()
            row_temp = row_temp.tolist()
            row_temp.pop(0)  # take out grade num
            # run over restriction list & row data together:
            while (len(row_temp) != 0 and len(lst_rest) != 0 and row_state != 0):
                while len(lst_rest) != 0:
                    if int(lst_rest[0]) == 0:
                        lst_rest.pop(0)
                    else:
                        break
                if len(lst_rest) != 0:
                    sole_num = int(lst_rest.pop(0))  # get number that isnt 0
                # clear row non relevant zeroes
                while len(row_temp) != 0:
                    if int(row_temp[0]) == 0:
                        row_temp.pop(0)
                    else:
                        break
                # check black seq according to current restriction
                while sole_num != 0 and len(row_temp) != 0:
                    if int(row_temp[0]) == 1:
                        row_temp.pop(0)
                        sole_num -= 1
                    else:
                        row_state = 0
                        break
                if row_state == 0:
                    break
                if sole_num != 0:
                    row_state = 0
                    break
                # check space after black seq, if in restriction there is another number that isnt zero
                # check other black num exist:
                check = 0
                # check if restriction has another non zero number
                if len(lst_rest) != 0:
                    for i1 in lst_rest:
                        if int(i1) != 0:
                            check = i1
                    if len(row_temp) == 0 and check != 0:  # row ended, but still num non zero in restriction
                        row_state = 0
                        break
                    if len(row_temp) == 0 and check == 0:  # row ended, only zero in restriction
                        row_state = 1
                        break
                    if len(row_temp) != 0 and int(row_temp[0]) == 1:  # no space, not according to restriction
                        row_state = 0
                        break
                    if len(row_temp) != 0 and int(
                            row_temp[0]) == 0 and check != 0:  # space, restriction contain num
                        continue
                    if len(row_temp) != 0 and int(
                            row_temp[0]) == 0 and check != 0:  # space, restriction contain zeroes
                        for jjj in row_temp:  # check if no num in row ahead
                            if int(jjj) != 0:
                                row_state = 0
                                break
                        row_state = 1
                        break
                elif len(row_temp) != 0:  # len(lst_rest)==0
                    for zzz in row_temp:  # check if no num in row ahead
                        if int(zzz) != 0:
                            row_state = 0
                            break
                    if row_state != 0:
                        row_state = 1
                        break
                elif len(row_temp) == 0 and len(lst_rest) == 0:
                    row_state = 1
                else:
                    print("Sanity check, shouldn't happen :)")
                    exit()
            if row_state == -1:
                row_state = 1
            dict[key][row_num, 0] = str(row_state)
            sum_row_grade += row_state
            row_state = -1
            lst_rest.clear()
        dict[key][0, 0] = sum_row_grade
        sum_row_grade = 0

        # column graded:
        col_num = 0
        col_state = -1
        col_res_cpy = column_restric.copy()
        jump_grade_row = 0
        for col_num, i in enumerate(dict[key].T):  # specific mat columns
            if jump_grade_row == 0:
                jump_grade_row = 1
                continue
            jump_grade_num = 0
            if len(col_res_cpy) != 0:
                for j in range(restricion_len):  # get 8 restrictions to fulfill
                    lst_rest.append(int(col_res_cpy.pop(0)))
            else:
                break
            # strategy: check out only black square.
            col_temp = []
            col_temp = i.copy()
            col_temp = col_temp.tolist()
            col_temp.pop(0)  # take out grade num
            # run over restriction list & col data together:
            while (len(col_temp) != 0 and len(lst_rest) != 0 and col_state != 0):
                while len(lst_rest) != 0:
                    if int(lst_rest[0]) == 0:
                        lst_rest.pop(0)
                    else:
                        break
                if len(lst_rest) != 0:
                    sole_num = int(lst_rest.pop(0))  # get number that isnt 0
                # clear col non relevant zeroes
                while len(col_temp) != 0:
                    if int(col_temp[0]) == 0:
                        col_temp.pop(0)
                    else:
                        break
                # check black seq according to current restriction
                while sole_num != 0 and len(col_temp) != 0:
                    if int(col_temp[0]) == 1:
                        col_temp.pop(0)
                        sole_num -= 1
                    else:
                        col_state = 0
                        break
                if col_state == 0:
                    break
                if sole_num != 0:
                    col_state = 0
                    break
                # check space after black seq, if needed
                # check if there is another num in restriction
                check = 0
                if len(lst_rest) != 0:
                    for i1 in lst_rest:
                        if int(i1) != 0:
                            check = i1
                    if len(col_temp) == 0 and check != 0:  # row ended, but still num non zero in restriction
                        col_state = 0
                        break
                    if len(col_temp) == 0 and check == 0:  # row ended, only zero in restriction
                        col_state = 1
                        break
                    if len(col_temp) != 0 and int(col_temp[0]) == 1:  # no space, not according to restriction
                        col_state = 0
                        break
                    if len(col_temp) != 0 and int(
                            col_temp[0]) == 0 and check != 0:  # space, restriction contain num
                        continue
                    if len(col_temp) != 0 and int(
                            col_temp[0]) == 0 and check != 0:  # space, restriction contain zeroes
                        for jjj in col_temp:  # check if no num in row ahead
                            if int(jjj) != 0:
                                col_state = 0
                                break
                        col_state = 1
                        break
                elif len(col_temp) != 0:  # len(lst_rest)==0
                    for zzz in col_temp:  # check if no num in row ahead
                        if int(zzz) != 0:
                            col_state = 0
                            break
                    if col_state != 0:
                        col_state = 1
                    break
                elif len(col_temp) == 0 and len(lst_rest) == 0:
                    col_state = 1
                else:
                    print("Sanity check, shouldn't happen :)")
                    exit()
            if col_state == -1:
                col_state = 1
            # if col_state == 1:
            #    let_me_check = 0
            dict[key][0, col_num] = str(col_state)
            sum_column_grade += col_state
            col_state = -1
            lst_rest.clear()
        dict[key][0, 0] += sum_column_grade
        sum_column_grade = 0

    # change back to list
    for i in range(mat_num):
        dict[i] = (dict[i]).tolist()


# assuming mat rank placed at <(0,0)>
def print_best_mat_in_dct(dct):
    index = 0  # save best mat index
    highest_value = -1
    for i in dct:
        if dct[i][0][0] > highest_value:
            highest_value = dct[i][0][0]

            index = i
    # print highest mat value:
    for r in dct[index]:
        for c in r:
            print(c, end=" ")
        print()


def mutate(dct, chance_for_mutation):
    for key, matrix in dct.items():
        for row_index, row in enumerate(matrix):
            for col_index, row in enumerate(row):
                if random.random() < chance_for_mutation:
                    if (row_index != 0 and col_index != 0):
                        matrix[row_index][col_index] = 1 - matrix[row_index][col_index]


def cross_over(matrix_a, matrix_b):
    matrix_a = np.array(matrix_a)
    matrix_b = np.array(matrix_b)

    child = np.empty([matrix_dim + 1, matrix_dim + 1], dtype=int)
    child[0, :] = matrix_a[0, :]

    for i in range(1, matrix_dim + 1):

        if (matrix_a[i, :][0] == 25 and matrix_b[i, :][0] == 25):
            sum_scores = matrix_a[0][0] + matrix_b[0][0]
            lottery_num = random.randint(0, sum_scores)
            if lottery_num > matrix_a[0][0]:
                child[i, :] = matrix_b[i, :]
            if lottery_num < matrix_a[0][0]:
                child[i, :] = matrix_a[i, :]

        elif (matrix_a[i, :][0] == 25 and matrix_b[i, :][0] < 25):
            child[i, :] = matrix_a[i, :]
        elif (matrix_a[i, :][0] < 25 and matrix_b[i, :][0] == 25):
            child[i, :] = matrix_b[i, :]

        # elif (matrix_a[i, :][0] < 25 and matrix_b[i, :][0] < 25):
        #     rand_num = random.randint(0, matrix_dim)
        #     A1, A2 = np.split(matrix_a[i, :], [rand_num])
        #     B1, B2 = np.split(matrix_b[i, :], [rand_num])
        #     row = np.concatenate((A1, B2))
        #     """
        #     choice = random.randint(0, 1)
        #     if choice == 0:
        #         row = np.concatenate((A1, B2))
        #     else:
        #         row = np.concatenate((B1, A2))
        #     """
        #     child[i, :] = row



        elif (matrix_a[i, :][0] < 25  and  matrix_b[i, :][0] < 25):
            scores_diff = math.ceil((matrix_a[i, :][0] - matrix_b[i, :][0])/3)
            linear_roulette = []
            if scores_diff >= 0:
                for k in range (matrix_dim):
                    for j in range(max(scores_diff,1)):
                        linear_roulette.append(k)
                    scores_diff  = scores_diff -1

            if scores_diff < 0:
                scores_diff = abs(scores_diff)
                for k in reversed(range (matrix_dim)):
                    for j in range(max(scores_diff,1)):
                        linear_roulette.append(k)
                    scores_diff  = scores_diff -1
            rand_num = choice(linear_roulette)
            A1,A2 = np.split(matrix_a[i, :],[rand_num])
            B1,B2 = np.split(matrix_b[i, :],[rand_num])
            row = np.concatenate ((A1, B2))
            child[i, :] = row
    return child.tolist()


# calc fitness of matrix
def new_fitness(mat, row_restric, column_restric):
    mat = np.array(mat)
    mat[0, :].fill(0)
    mat[:, 0].fill(0)
    mat = mat.tolist()
    mat = new_fitness_rows(mat, row_restric)
    mat = np.array(mat)
    mat = np.matrix.transpose(mat)
    mat = mat.tolist()
    mat = new_fitness_rows(mat, column_restric)
    mat = np.array(mat)
    mat = np.matrix.transpose(mat)
    mat[0, 0] = np.sum(mat[0, :]) + np.sum(mat[:, 0])
    return mat.tolist()


# calc fitness of row
def new_fitness_one_row(mat, row_index, row_restric):
    resrict_index = (row_index-1) * 8
    restriction = row_restric[resrict_index:resrict_index + 8]
    mat_row = mat[row_index][:]
    del mat_row[0]
    # del mat[i][0]
    mat_row = calc_new_representation(mat_row)
    mat_row = list(filter((0).__ne__, mat_row))

    # restriction = row_restric[position:position + 8]
    restriction = list(map(int, restriction))
    restriction = list(filter((0).__ne__, restriction))
    mat_row, restriction = eq_list(mat_row, restriction)
    diff = sub(mat_row, restriction)
    return (25 - diff)


# calc rows fitness
def new_fitness_rows(mat, row_restric):
    position = 0
    # mat[0][:].insert(0, 0)
    for i in range(1, matrix_dim + 1):
        mat_row = mat[i][:]
        del mat_row[0]
        del mat[i][0]
        mat_row = calc_new_representation(mat_row)
        mat_row = list(filter((0).__ne__, mat_row))
        restriction = row_restric[position:position + 8]
        restriction = list(map(int, restriction))
        restriction = list(filter((0).__ne__, restriction))
        mat_row, restriction = eq_list(mat_row, restriction)
        diff = sub(mat_row, restriction)
        mat[i].insert(0, 25 - diff)  # max(50-diff*3,0))
        position += 8
    return mat


def calc_new_representation(mat_row):
    index = 0
    while (index != len(mat_row) - 1):
        if (mat_row[index] > 0 and mat_row[index + 1] > 0):
            mat_row[index + 1] = mat_row[index + 1] + mat_row[index]
            mat_row[index] = 0
        index += 1
    return mat_row


def eq_list(list1, list2):
    if (len(list1) == len(list2)):
        return list1, list2

    if (len(list1) > len(list2)):
        diff = len(list1) - len(list2)
        for i in range(diff):
            list2.append(0)

    if (len(list1) < len(list2)):
        diff = len(list2) - len(list1)
        for i in range(diff):
            list1.append(0)
    return list1, list2


def sub(list1, list2):
    difference = 0
    zip_object = zip(list1, list2)
    for list1_i, list2_i in zip_object:
        difference += abs(list1_i - list2_i)
    return difference


import statistics


def print_best_mat_score(dic):
    scores = []
    for key, value in dic.items():
        scores.append(value[0][0])
    max_score = max(scores)
    print("max score = ",max_score )
    print("average score = ", statistics.mean(scores))
    if (max_score == 1250):
        print_best_mat_in_dct(dic)
        exit()
    return max_score


def get_n_best_mats(n, dic):
    scores = {}
    for key, value in dic.items():
        scores[value] = dic[key][0][0]
    print(max(scores))


def create_random_matrix(rows_data):
    matrix = []
    for i in range(matrix_dim + 1):  # create a matrix with empty cells
        row = []
        for j in range(matrix_dim + 1):
            row.append(0)
        matrix.append(row)

    rows_data = list(map(int, rows_data))
    cells_num = sum(rows_data)
    x = [*range(1, matrix_dim + 1, 1)]  # create a list with all possible cells indexes
    y = [*range(1, matrix_dim + 1, 1)]
    all_cells = []
    for i in (x):
        for j in (y):
            all_cells.append([i, j])
    for i in range(cells_num):
        choice = random.choice(all_cells)  # choose random cell from list
        all_cells.remove(choice)
        matrix[choice[0]][choice[1]] = 1
    return matrix

from copy import copy, deepcopy
import datetime
import time

def optimization(dct, row_restric, column_restric):
    if (GA_type == "Darwinian"):
        original_dct = deepcopy(dct)

    x = [*range(1, matrix_dim + 1, 1)]  # create a list with all possible cells indices
    y = [*range(1, matrix_dim + 1, 1)]
    all_cells = []
    for i in x:
        for j in y:
            all_cells.append([i, j])
    total1= 0
    total2 = 0
    total3 = 0
    total4 = 0
    total5 = 0
    # only 50 optimization per generation:
    for mat_ind_chosen in range(mat_num):
        num_of_optimizations = 50
        all_mat_indices = deepcopy(all_cells)

        for j in range(num_of_optimizations):
            t0 = time.time()
            ta = time.time()

            choice = random.choice(all_mat_indices)  # choose random cell from list
            all_mat_indices.remove(choice)
            tb = time.time()
            total3+= tb-ta

            tc = time.time()
            curr_mat = deepcopy(np.array(dct[mat_ind_chosen]))
            curr_mat = curr_mat.tolist()
            td = time.time()
            total4+= td-tc

            te = time.time()

            val_row_before = curr_mat[choice[0]][0]
            val_column_before = curr_mat[0][choice[1]]
            sum_before = val_row_before + val_column_before
            tf = time.time()
            total5+= tf-te

            t1 = time.time()
            total1+=(t1-t0)

            t3 = time.time()

            curr_mat[choice[0]][choice[1]] = 1 - curr_mat[choice[0]][choice[1]]

            temp_mat = deepcopy(np.array(curr_mat))
            temp_mat = temp_mat.T
            temp_mat = temp_mat.tolist()

            val_row_after = new_fitness_one_row(curr_mat, choice[0], row_restric)
            val_column_after = new_fitness_one_row(temp_mat, choice[1], column_restric)
            sum_after = val_row_after + val_column_after

            if sum_after > sum_before:
                dct[mat_ind_chosen][0][choice[1]] = val_column_after
                dct[mat_ind_chosen][choice[0]][0] = val_row_after
                dct[mat_ind_chosen][choice[0]][choice[1]] = 1 - dct[mat_ind_chosen][choice[0]][choice[1]]
            t4 = time.time()
            total2+=(t4-t3)
    # print ("total3" , total3)
    # print ("total4" , total4)
    # print ("total5" , total5)
    if (GA_type == "Darwinian"):
        return dct,original_dct
    else:
        return dct

import matplotlib.pyplot as plt



if __name__ == "__main__":
    try:
        a = open('C:/Users/Israel/Desktop/Biological_calculation/ex2/Lamarckian_hard_better.xlsx', 'w')
        a.close()
    except IOError:
        print('file is already open')
        exit()

    generations = 1000
    # GA_type = 'Darwinian'
    #GA_type = 'Lamarckian'
    GA_type = 'Regular'


    #load data
    data = sys.argv[1:]
    len_data = len(data)
    row_filler = math.floor((len_data) / 2)
    columns_data = data[0:math.floor((len_data) / 2)]
    rows_data = data[math.floor((len_data) / 2):len_data]

# dict hold random matrices
    dct = {}
    for i in range(mat_num):
        dct[i] = create_random_matrix(rows_data)

# calculate fitness
    temp_dct = {}
    i = 0
    for key, value in dct.items():
        new_mat = new_fitness(value, rows_data, columns_data)
        temp_dct[i] = new_mat
        i += 1
    dct = temp_dct


    print("best_mat:")
    print_best_mat_score(dct)
    max_scores_list = []

    for i in range(generations):
        print("generation ", i)
    #mutate

        if (i>500 and i <600):
            mutate(dct, chance_for_mutation=0.01)

        else: mutate(dct, chance_for_mutation=0.001)




    # niching
        if (i % 60 == 0 and i != 0):
            print("---------------niching...")
            for t in range(20):
                rand = random.randint(0, mat_num)
                dct[rand] = create_random_matrix(rows_data)

    # transposing
        if (i % 5 == 0 and i != 0):
            print("----------------transpose...")
            for key, value in dct.items():
                transpose_value = np.transpose(np.array(value))
                dct[key] = transpose_value.tolist()
            temp = columns_data
            columns_data = rows_data
            rows_data = temp

    #calculate fitness
        temp_dct = {}
        i = 0
        for key, value in dct.items():
            new_mat = new_fitness(value, rows_data, columns_data)
            temp_dct[i] = new_mat
            i += 1
        dct = temp_dct

    # optimization
        if (GA_type != "Regular"):
            if (GA_type == "Darwinian"):
                optimized_dct,dct = optimization(dct, rows_data, columns_data)

            if (GA_type == "Lamarckian"):
                dct = optimization(dct, rows_data, columns_data)
                optimized_dct = dct
        else:
            optimized_dct = dct

    # make linear roulette for choosing parents for next generation
        linear_roulette = []
        for key, value in optimized_dct.items():
            score = value[0][0]
            for j in range(math.floor(((score + 1) / 200) + 0.5) ** 4):  ###was 4 instead of 7
                linear_roulette.append(key)



    # cross overs
        next_gen_dct = {}
        for k in range(mat_num):
            first_mat = choice(linear_roulette)
            second_mat = choice(linear_roulette)
            child = cross_over(dct[first_mat], dct[second_mat])
            next_gen_dct[k] = child
        dct = next_gen_dct



        max_score = print_best_mat_score(dct)
        max_scores_list.append(max_score)
    generations_list =  [*range(1, generations+1, 1)]

    all_list= []
    all_list.append(generations_list)
    all_list.append(max_scores_list)

    df = pd.DataFrame(all_list)
    df = df.transpose()
    writer = pd.ExcelWriter('C:/Users/Israel/Desktop/Biological_calculation/ex2/Generi_hard_better.xlsx', engine='xlsxwriter')
    df.to_excel(writer, sheet_name='welcome', index=False)
    writer.save()

    plt.plot( generations_list, max_scores_list)

    # naming the x axis
    plt.xlabel('generation')
    # naming the y axis
    plt.ylabel('max score')

    # giving a title to my graph
    if (GA_type == "Darwinian"):
        plt.title('Darwinian genetic algorithm')
    if (GA_type == "Lamarckian"):
        plt.title('Lamarckian genetic algorithm')
    if (GA_type == "Regular"):
        plt.title('Regular genetic algorithm')
    # function to show the plot
    plt.show()
    print_best_mat_in_dct(dct)



    # fitness_calc(dct, rows_data, columns_data)
# print_best_mat_in_dct(dct)
