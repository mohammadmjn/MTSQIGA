#!/usr/bin/env python3.6

from os import listdir, makedirs
from os.path import join, isdir, isfile, exists
from shutil import copy


'''multi-doc version: This script aims to make folders with "system" and "reference"
subfolders and then copy files from source_path to destin_path for locating summaries
in rouge format for evaluating with it.'''


def main():
    # main_path is the path where original dataset that is used for summarization is located.
    main_path = input('Please enter path of original dataset: ')
    # destin_path is the directory where you intended to store summaries for evaluation.
    destin_path = input("Please enter your destination path: ")
    # source_path is the directory where generated summaries are there now. In other word,
    # we want to copy summaries from source_path to destin_path and organize summaries.
    source_path = input("Please enter source path of summaries: ")
    for folder in listdir(main_path):
        if isdir(join(main_path, folder)):
            path = join(destin_path, folder)
            if not exists(path):
                makedirs(path)
            if not exists(join(path, 'system')):
                makedirs(join(path, 'system'))
            if not exists(join(path, 'reference')):
                makedirs(join(path, 'reference'))
            print('{}:	added System & Reference Folders'.format(folder))

            # This part copies system generated files from source_path to destin_path.
            summary_files = [file for file in listdir(source_path) if isfile(join(source_path, file))
                             and file.startswith('{}'.format(folder))]
            for f in summary_files:
                try:
                    copy(join(source_path, f),
                         r'{}/{}/system'.format(destin_path, folder))
                except FileNotFoundError:
                    print(r'--- {} not found in in "{}"'.format(f, path))
            print('{}:    Copied Files to System Folder'.format(folder))


if __name__ == "__main__":
    main()
