import os
from matplotlib import cm
import random
import matplotlib.pyplot as plt
import sys
from random import choice
import numpy as np
import math

def compare_2_columns(col1, col2):
    """
    This function compare 2 columns, for example : col1 =[1,0,1], col2= [1,1,1],
    score = 1+0+1 = 2
    :param column 1:
    :param column 2:
    :return: score
    """
    score = 0
    length = col1.size
    for i in range(length):
        if (col1[i] == col2[i]):
            score += 1
        else:
            score -= 1
    return score


def create_weight_matrix(matrix):
    """
    Create weight_matrix
    :param np matrix:
    :return: np weight_matrix
    """
    #convert np vector (x,) to matrix (x,1)
    try:
        rows, cols = matrix.shape
    except:
        matrix = matrix.reshape(1, -1)
    length = np.size(matrix, 1)
    weight_matrix = np.empty([length, length], dtype=int)

    #For every 2 columns, calculate score and add to weight_matrix
    for i in range(length):
        for j in range(length):
            score = compare_2_columns(matrix[:, i], matrix[:, j])
            weight_matrix[i, j] = score
    return weight_matrix

def compare_2_digits(in_dig, out_dig):
    """
    compare two digits (100 len vector each)
    :param digit 1:
    :param digit 2:
    :return: score
    """
    result = 0
    for a,b in zip(in_dig,out_dig):
        if a==b:
            result += 1
    return result

#add mutation over one letter in vector. choose mutation_percentage% in vec at random place and change number
def mutation(vector, mutation_percentage):
    new_vec = vector.copy()

    all_cells = []
    for i in range(100):
            all_cells.append(i)

    for j in range(mutation_percentage):
        choice = random.choice(all_cells)  # choose random cell from list
        all_cells.remove(choice)
        new_vec[choice] = 1-new_vec[choice]
    return new_vec


def get_matrix_from_input(vector_len):
    """
    This func get matrix from input
    :param vector_len:
    :return: digits matrix
    """
    digits = np.empty([vector_len, vector_len], dtype=int)
    digits_file = sys.argv[1]
    counter = 0

    lines = open(digits_file)
    prev_line = ''
    single_digit = ''

    for index, line in enumerate(lines):
        line = line.strip()
        if (line == ''):
            digits[counter, :] = np.array(list(single_digit))
            single_digit = ''
            counter += 1

        else:
            single_digit = single_digit + line

    lines.close()
    return digits

def forward_input(weight_matrix,vector_len,input_str):
    """
    predict digit from input by weight_matrix
    :param weight_matrix:
    :param vector_len:
    :param input_str:
    :return: predicted digit
    """

    generation_counter = 0
    next_string = input_str.copy()
    flag = False

    while (flag == False):
        indexes_list = [*range(vector_len)]
        prev_str = next_string.copy()
        while (len (indexes_list) != 0):
            chosen_index = choice(indexes_list)
            indexes_list.remove(chosen_index)
            col = weight_matrix[:, chosen_index]
            sum_score = 0

            for i in range(vector_len):
                if (i != chosen_index):
                    sum_score += (next_string[i] * col[i])
            if (sum_score >=0):
                next_string[chosen_index] = 1
            else:
                next_string[chosen_index] = 0

        if (prev_str.tolist() == next_string.tolist()):
            flag = True
        else: generation_counter += 1
    return next_string,generation_counter

import operator
def check_differ_to_digits (predicted, i,digits_list):
    scores = {}
    for digit in range(i):
        score = compare_2_digits(digits_list[digit, :], predicted)
        scores[digit] = score
    maximum = max(scores, key=scores.get)
    if (scores[maximum] > 80):
        return maximum
    else: return -1
from PIL import Image
def res_letters(digits_list, weight_matrix, i, mutation_rate):
    """
    Calculate accuracy of i digits from weight_matrix that built from these digits
    :param digits matrix:
    :param weight_matrix:
    :param i: number of digits. i.e i = 3, digits = 0,1,2
    :param mutation_rate:
    :return:
    """
    #convert np vector (x,) to matrix (x,1)
    try:
        rows, cols = digits_list.shape
    except:
        digits_list = digits_list.reshape(1, -1)

    error_predicted_digit = 0
    error_predicted_nothing = 0
    total_error_predicted_digit = 0
    total_error_predicted_nothing = 0
    total_score = 0
    total_gens = 0
    #check every digit
    for digit in range (i+1):
        corrects = 0 #number of correct prediction in digit
        error_predicted_nothing = 0
        error_predicted_digit = 0
        predicted_image = np.zeros((10, 10), dtype=np.uint8)
        for k in range(100): #predict 100 mutated digis
            gens = 0
            mutated = mutation(digits_list[digit, :], mutation_rate)
            predicted,gens = forward_input(weight_matrix, 100, mutated)
            total_gens += gens

            predict_percentage = compare_2_digits(digits_list[digit, :], predicted)
            if (predict_percentage > 80):
                corrects += 1
            else: # it failed:
                #return digid num or -1 to wrong prediction
                temp_result = check_differ_to_digits(predicted, i,digits_list)
                if temp_result == -1:
                    error_predicted_nothing += 1
                    #if error_predicted_nothing>80:
                    #        we_are_proffessional = 1000
                    """
                    for m in range (10):
                        for j in range(10):
                            predicted_image[m, j] = predicted[j + m * 10]
                            print(predicted[j+m*10], end='')
                        print("")
                    np.multiply(predicted_image, 255)
                    """
                else:
                    error_predicted_digit += 1
        total_score += corrects
        total_error_predicted_digit += error_predicted_digit
        total_error_predicted_nothing += error_predicted_nothing
        # print("genss = ",gens)
        # print("total genss = ",total_gens)

        #a = 0
    total_score = total_score / ((i+1)*100)
    total_gens = total_gens/ ((i+1)*100)

    # print("total gens",total_gens)
    total_error_predicted_digit = total_error_predicted_digit / ((i+1)*100)
    total_error_predicted_nothing = total_error_predicted_nothing / ((i+1)*100)
    return total_score,total_error_predicted_digit, total_error_predicted_nothing,total_gens

from copy import deepcopy
def digits_update_zero_to_one(dig):
    dig_copied = deepcopy(dig)
    for i in dig_copied:
        for j in i:
            if j == 0:
                dig_copied[i,j]=-1
    return dig_copied

if __name__ == '__main__':
    digits = get_matrix_from_input(100)

    # weight_matrix = create_weight_matrix(digits[0, :])
    # corrects = 0
    # for i in range (100):
    #     mutated = mutation(digits[0, :])
    #     predicted = forward_input(weight_matrix,100,mutated)
    #     predict_percentage = compare_in_out_letter(digits[0], predicted)
    #     if (predict_percentage > 90):
    #         corrects += 1


    new_digits = np.empty([10, 100], dtype=int)
    new_digits[0,:] = digits[0,:].copy()
    new_digits[1, :] = digits[10, :].copy()
    new_digits[2, :] = digits[20, :].copy()
    new_digits[3, :] = digits[30, :].copy()
    new_digits[4, :] = digits[40, :].copy()
    new_digits[5, :] = digits[50, :].copy()
    new_digits[6, :] = digits[60, :].copy()
    new_digits[7, :] = digits[70, :].copy()
    new_digits[8, :] = digits[80, :].copy()
    new_digits[9, :] = digits[90, :].copy()

    # # Part 1:
    # for i in range(10):
    #     weight_matrix = create_weight_matrix(new_digits[0:i+1,:])
    #     res,error_predicted_digit_result,error_predicted_nothing_result = res_letters(new_digits[0:i+1,:],weight_matrix,i,10)
    #     print("digits:", [*range(i+1)],".accuracy= ", res)
    #     print("error predicted digits:", error_predicted_digit_result, ". error predicted nothing:", error_predicted_nothing_result)
    #     print("\n")

    # #Part 2:
    # x = np.array([0,0,0,1,1,1,2,2,2,3,3,3,4,4,4,5,5,5,6,6,6,7,7,7,8,8,8,9,9,9])
    # y = np.array([15,10,5,15,10,5,15,10,5,15,10,5,15,10,5,15,10,5,15,10,5,15,10,5,15,10,5,15,10,5])
    #
    # x2 = np.array([0,1,2,3,4,5,6,7,8,9])
    # y2 = np.array([15,10,5])
    #
    # res_dig = np.zeros(30)
    #
    # counter = 0
    # for i in x2:
    #     for j in y2:
    #         weight_matrix = create_weight_matrix(new_digits[0:i+1,:])
    #         res = res_letters(new_digits[0:i+1,:],weight_matrix,i,j)
    #         print("digits:", [*range(i + 1)],",mutation rate:", j,",accuracy= ", res)
    #         res_dig[counter] = res
    #         counter += 1
    #
    #
    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # ax.plot3D(x, y, res_dig, s=40, c='blue')
    # ax.set_xlabel('digits')
    # ax.set_ylabel('mutation rate')
    # ax.set_zlabel('predicted results')
    # plt.show()


    # Part 3:
    # # 1)
    # lst = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    # for i,j  in enumerate(lst):
    #     weight_matrix = create_weight_matrix(digits[0:j, :])
    #     res = res_letters(new_digits[0:i + 1,:], weight_matrix, i, 10)
    #     print("digits:", [*range(i + 1)], ".accuracy= ", res)
    # 2)
    #
    # x = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9])
    # y = np.array(
    #     [15, 10, 5, 15, 10, 5, 15, 10, 5, 15, 10, 5, 15, 10, 5, 15, 10, 5, 15, 10, 5, 15, 10, 5, 15, 10, 5, 15, 10,
    #      5])
    #
    # x2 = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    # y2 = np.array([15, 10, 5])
    #
    # res_dig = np.zeros(30)
    # lst = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    #
    # counter = 0
    # for i,k in enumerate(lst):
    #     for j in y2:
    #         weight_matrix = create_weight_matrix(new_digits[0:k, :])
    #         res = res_letters(new_digits[0:i + 1, :], weight_matrix, i, j)
    #         print("digits:", [*range(i + 1)], ",mutation rate:", j, ",accuracy= ", res)
    #         res_dig[counter] = res
    #         counter += 1
    #
    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # ax.plot3D(x, y, res_dig, 'gray')
    # ax.set_xlabel('digits')
    # ax.set_ylabel('mutation rate')
    # ax.set_zlabel('predicted results')
    # plt.show()
    #
    # lst = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    #

    #part 4:
    switch = 0
    list_results_0_1_accuracy = []
    list_results__1_1_accuracy = []
    list_results__1_1_gen = []
    list_results_0_1_gen = []
    gen_compare = 0
    for i in range (2):
        if switch == 0:
            switch = 1
            new_dig = new_digits
            new_dig = deepcopy(digits)
        else:
            new_dig = digits_update_zero_to_one(digits)
            switch = 2
        new_digits2 = np.empty([10, 100], dtype=int)
        new_digits2[0, :] = new_dig[0, :].copy()
        new_digits2[1, :] = new_dig[10, :].copy()
        new_digits2[2, :] = new_dig[20, :].copy()
        new_digits2[3, :] = new_dig[30, :].copy()
        new_digits2[4, :] = new_dig[40, :].copy()
        new_digits2[5, :] = new_dig[50, :].copy()
        new_digits2[6, :] = new_dig[60, :].copy()
        new_digits2[7, :] = new_dig[70, :].copy()
        new_digits2[8, :] = new_dig[80, :].copy()
        new_digits2[9, :] = new_dig[90, :].copy()
        iterations = 10
        for i in range(iterations):
            weight_matrix = create_weight_matrix(new_digits2[0:i+1,:])
            res,non_rel,non_rellevant,gen_compare = res_letters(new_digits2[0:i+1,:],weight_matrix,i,10)
            if switch == 1:
                list_results_0_1_accuracy.append(res)
                list_results_0_1_gen.append(gen_compare)
            else:
                list_results__1_1_accuracy.append(res)
                list_results__1_1_gen.append(gen_compare)
            print("digits2:", [*range(i+1)],".accuracy= ", res,"gens =",gen_compare )
    plt.plot([*range(iterations)],list_results_0_1_accuracy,label="0 1")
    plt.plot([*range(iterations)],list_results__1_1_accuracy,label="-1 1")
    # plt.plot([*range(iterations)],list_results_0_1_gen,label="gen 0 1")
    # plt.plot([*range(iterations)],list_results__1_1_gen,label="gen -1 1")
    plt.legend()

    plt.xlabel("Digits")
    plt.ylabel("Generations")
    plt.show()


