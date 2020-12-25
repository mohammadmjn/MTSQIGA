#!/usr/bin/env python3

from os import listdir
from os.path import (join,
                     isdir,
                     isfile,
                     dirname,
                     abspath)
from re import sub
import subprocess
import nltk.data


'''This script aimed to evaluate generated summaries. In order to do this,
the reference summaries should be in plain text format (not like xml format).
At first, this script splits sentences and write each of them in a seperate
line (Rouge format). Then, it evaluates rouge value for each of folders and
generate a .csv result file'''


def normalizeRefSummary():
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    extra_abbreviation = {
        'a.m', 'p.m', 'ch', 'mr', 'mrs', 'prof', 'st', 'jan', 'conn', 'ariz',
        'feb', 'mar', 'apr', 'jun', 'jul', 'aug', 'sept', 'oct', 'nov', 'dec',
        'then-gov', 'sgt', 'w.h', 'gov', 'u.s', 'maj', 'gen', 'no', 'd', 'etc',
        'fig', 'dept', 'ave', 'law', 'd.a', 'i.e', 'u.n', 's.c', 'c.h', 'b.b',
        'e.g', 'c.o', 'b.c', 'm.n', 'j.d', 'm.p.h', 'n.y', 'e.d', 'n.w', 'i.d',
        'k.b', 'l.a', 'l.e', 'c.j', 'u.s.a', 'd.c', 'k.c', 'n.c', 'p.o', 'f.d.r',
        'r.i', 'c.k', 'b.s', 'n.j', 'v.m.d', 't.q', 'l.o', 'Ph.d', 'k.l', 'e.t',
        'o.c', 'rep', 'rey', 'sen', 'atty', 'col', 'corp', 'co', 'inc', 'ft',
        'ind', 'r', 'jr', 'd-md', 'd-fla', 'd-tex', 'd-wash', 'd-mich', 'd-calif',
        'd-n.y', 'd-wis', 'd-nev', 'd-ga', 'lt', 'dr', 's', 'mt', 'u', 's', 'blm',
        'r-Calif', 'r-ill', 'r-fla', 'rev', 'f', 'm', 'w', 'a', 'mg', 'sr', 'lbs',
        'ltd', 'vs', 'ga', 'cos', 'ore', 'va', 'md', 'pa', 'fla', 'ida', 'capt',
        'adm', 'assn', 'blvd', 'kent', 'supt', 'cmdr', 'Msgr', 'bros', 'mich',
        'dist', 'mass', 'reps', 'colo', 'asst', 'prop', 'sat'
    }
    tokenizer._params.abbrev_types.update(extra_abbreviation)
    # "main_path" is the destination directory where the Rouge test will occur there.
    main_path = input("Please enter the path where you want to run rouge test " +
                      "(destination test folder): ")
    for _direc in listdir(main_path):
        _path = join(main_path, _direc)
        for direc in listdir(_path):
            path = join(_path, direc)
            if direc == 'reference' and isdir(path):
                files = [file for file in listdir(
                    path) if isfile(join(path, file))]
                for file in files:
                    with open(join(path, file), mode='r') as i:
                        text = i.read()
                    summary = text.strip()
                    with open(join(path, file), mode='w') as summ:
                        summ.write(summary)
    return main_path


def main():
    option = int(input(
        'If you want to convert plain text summaries into a format ROUGE understands, enter \"1\" esle \"0\": '))
    if option == 1:
        main_path = normalizeRefSummary()
    else:
        main_path = input(
            'Please enter your intended path: ').replace('\\', '/')
    rouge_prop_abs_path = abspath(
        dirname(__file__) + '/../rouge2-1.2.1/rouge.properties').replace('\\', '/')
    rouge_jar_abs_path = abspath(
        dirname(__file__) + '/../rouge2-1.2.1/rouge2-1.2.1.jar').replace('\\', '/')
    for direc in listdir(main_path):
        path = join(main_path, direc)
        if isdir(path):
            with open(rouge_prop_abs_path) as rouge:
                rouge_properties = rouge.read()
            rouge_properties = sub(r'(project\.dir=).*?(\n)',
                                   r'\1{}/{}\2'.format(main_path, direc),
                                   rouge_properties)
            rouge_properties = sub(
                r'(outputFile=).*?(\.csv)', r'\1{}\2'.format(direc), rouge_properties)
            with open(rouge_prop_abs_path, mode='w') as rouge:
                rouge.write(rouge_properties)
            subprocess.call(['java', '-Drouge.prop={}'.format(rouge_prop_abs_path),
                             '-jar', rouge_jar_abs_path])


if __name__ == '__main__':
    main()
