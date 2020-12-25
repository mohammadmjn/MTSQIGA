#!/usr/bin/env python3.6

from os import listdir
from os.path import join, isfile
import pandas


'''This script at first calculate average F-measure for each of .csv files and
write it in its corresponding .csv file. finally, it calculates total averge
and write it in a .txt file.'''


def main():
    # directory where .csv files are there and you want to calculate sum of F-maesure on them.
    main_path = input("Please enter path of csv files: ")
    each_file_average_f = []
    each_file_avg_recall = []
    each_file_avg_prec = []
    for direct in listdir(main_path):
        file = join(main_path, direct)
        if isfile(file) and direct.lower().endswith('.csv'):
            csvFile = pandas.read_csv(file)
            f_sorted = csvFile.sort_values('Avg_F-Score', ascending=False)
            new_csv = f_sorted.head(10)
            file_average_FScore = new_csv.loc[:, 'Avg_F-Score'].mean()
            file_average_recall = new_csv.loc[:, 'Avg_Recall'].mean()
            file_average_prec = new_csv.loc[:, 'Avg_Precision'].mean()
            each_file_average_f.append(file_average_FScore)
            each_file_avg_recall.append(file_average_recall)
            each_file_avg_prec.append(file_average_prec)
    total_FScore_average = sum(each_file_average_f) / \
        float(len(each_file_average_f))
    total_recall_average = sum(each_file_avg_recall) / \
        float(len(each_file_avg_recall))
    total_prec_average = sum(each_file_avg_prec) / \
        float(len(each_file_avg_prec))
    output = 'Average F-Score:   {}\n'.format(total_FScore_average) + \
             'Average Recall:    {}\n'.format(total_recall_average) + \
             'Average Precision: {}'.format(total_prec_average)
    with open('Average Scores.txt', mode='w') as out:
        out.write(output)


if __name__ == '__main__':
    main()
