#!/usr/bin/env python
# -*- coding: utf-8 -*-

""""From Marcely: <https://github.com/mzboito/word_discovery>"""

from __future__ import unicode_literals, print_function, division
import argparse
from io import open
import sys
import os
import codecs
import glob
import evaluation
from util import plot_attention_from_file


# SEGMENTATION
# ------------

# NOTE: could maybe make a class Segmenter
def get_path(number, paths):
    for path in paths:
        if "." + str(number) + "." in path:
            return path
    return None

def read_matrix_file(path):
    return [line.strip("\n").split("\t") for line in open(path, mode='r', encoding='utf-8')]

# NOTE: probably possible to do better with an argmax routine
def get_max_prob_col(line, sentence_matrix):
    max_value = float(sentence_matrix[line][1]) #start from the first line after the characters
    col = 1
    for i in range(2, len(sentence_matrix[line])):
        if max_value < float(sentence_matrix[line][i]):
            col = i
            max_value = float(sentence_matrix[line][i])
    return col

def segment(file_path):
    matrix = read_matrix_file(file_path)

    final_string = ""
    last_col = -1
    for i in range(1, len(matrix)): #for each element
        col = get_max_prob_col(i, matrix)
        if last_col == -1: #first character
            final_string += matrix[i][0] #put the character in the beginning
        elif last_col == col: # if the current character and the last one are not separated
            final_string += matrix[i][0]
        else:
            final_string += " " + matrix[i][0]
        last_col = col
    # NOTE: probably could do a more elegant strip
    final_string = final_string.replace("  "," ")
    if final_string[-1] == " ":
        final_string = final_string[:-1]
    if final_string[0] == " ":
        final_string = final_string[1:]
    return final_string

def write_segmentation(final_string, output):
    with open(output, mode='a', encoding='utf-8') as outputFile:
        outputFile.write(final_string + "\n")


# SMOOTHING
# ---------

def write_matrix(matrix, path):
    with open(path,mode="w", encoding="utf-8") as o:
        for line in matrix:
            o.write("\t".join(line) + "\n")

def smooth(paths, output_path):
    for filem in paths:
        matrix = read_matrix_file(filem)
        #s_matrix = [[0.0 for col in range(len(matrix[0]))] for row in range(len(matrix))]
        #s_matrix[0] = matrix[0]
        for line in range(1,len(matrix)):
            #s_matrix[line][0] = matrix[line][0]
            for column in range(1,len(matrix[line])):
                if len(matrix[line]) == 2:
                    pass
                elif column == 1: #first line
                    matrix[line][column] = str((float(matrix[line][column]) + float(matrix[line][column+1]))/2)
                elif matrix[line][column] == matrix[line][-1]: #last line
                    matrix[line][column] = str((float(matrix[line][column]) + float(matrix[line][column-1]))/2)
                else:
                    matrix[line][column] = str((float(matrix[line][column-1]) + float(matrix[line][column]) + float(matrix[line][column+1]))/3)
        write_matrix(matrix, os.path.join(output_path, filem.split("/")[-1]))


def main(arguments):
    parser = argparse.ArgumentParser( description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input_path', help='Path for attention matrices', type=str)
    parser.add_argument('--gold', help='Path for gold standard', type=str)
    args = parser.parse_args(arguments)

    # smooth
    paths = glob.glob(os.path.join(args.input_path, "*.txt"))
    smoothed_path = os.path.join(os.path.dirname(args.input_path),
                                 'smoothed_' + os.path.basename(args.input_path) + '/')
    if not os.path.exists(smoothed_path):
        os.makedirs(smoothed_path)
    smooth(paths, smoothed_path)

    output_path = os.path.join(os.path.dirname(smoothed_path), os.path.basename(smoothed_path) + 'segmented.txt')

    # clean output file
    if os.path.exists(output_path):
        os.remove(output_path)
    smoothed_paths = glob.glob(os.path.join(smoothed_path, "*.txt"))
    for index in range(0, len(smoothed_paths)):
        file_path = get_path(index, smoothed_paths)
        final_str = segment(file_path).replace(" </s>","").replace("</s>","") #removing EOS
        # segment
        write_segmentation(final_str, output_path)
        # plot current attention file
        plot_attention_from_file(file_path)

    gold = args.gold
    x, wp, wr, wf, bp, br, bf, lp, lr, lf = evaluation.get_prf_metrics(gold, output_path)
    results = f"X {(x * 100):.2f} WP {(wp * 100):.2f} WR {( wr * 100):.2f} WF {(wf * 100):.2f}  BP {(bp * 100):.2f} BR {( br * 100):.2f} BF {(bf * 100):.2f} LP {(lp * 100):.2f} LR {( lr * 100):.2f} LF {(lf * 100):.2f}"
    with open(os.path.join(os.path.dirname(smoothed_path), os.path.basename(smoothed_path) + 'results.txt'), mode='w', encoding='utf-8') as r:
        r.write(results)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
