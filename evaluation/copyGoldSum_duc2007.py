#!/usr/bin/env python3.6

from os import walk, listdir, rename
from os.path import join, isdir, exists, isfile
from shutil import copy


'''DUC2007: This script aims to extract gold summaries and
write each summary to a seperate file. Then copy this files
to their corresponding folder in destination folder where
rouge evaluation will occur.'''


def main():
    # path to the original dataset where system summaries generated based on these dataset.
    dataset_original_path = input('Enter the path to the original dataset where ' +
                                  'system summaries generated based on these dataset:\n')
    folders_name = [direct for direct in listdir(
        dataset_original_path) if isdir(join(dataset_original_path, direct))]
    new_list = [direct[:-1] for direct in folders_name]
    # directory to the folders that contain gold summaries
    path = input('Enter your gold summary original path: ')
    # main_path is path to the destination folder contains multi-doc folders
    main_path = input('Enter the path to the destination folder where rouge '
                      + 'evaluation will occur:\n')
    for file in listdir(path):
        gold_summ_path = join(path, file)
        if isfile(gold_summ_path):
            docset = file.split(sep='.')[0]
            if docset in new_list:
                summarizer_id = file.split(sep='.')[-1]
                docset_id = folders_name[new_list.index(docset)]
                copy(gold_summ_path, '{}\\{}\\reference'.format(
                    main_path, docset_id))
                rename('{}\\{}\\reference\\{}'.format(main_path, docset_id, file),
                       '{}\\{}\\reference\\{}_{}.txt'.format(main_path, docset_id, docset_id, summarizer_id))
    print('{}:    Copied Gold Summary to reference Folder'.format(docset))


if __name__ == "__main__":
    main()
