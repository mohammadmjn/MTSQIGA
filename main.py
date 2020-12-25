#!/usr/bin/env python3.6

from os import listdir, mkdir
from os.path import (
    join,
    isdir,
    exists
)
from src.preprocess import Preprocessor
from src.quantum_ga import QuantumGA


def main():
    test_data_dir = input(
        'Please enter path to the directory contains source documents: ')
    destination_dir = input(
        'Please enter path where you want to store generated summaries: ')
    try:
        data_set = int(input(
            'Select dataset 1- DUC2005    2- DUC2007 (Enter 1 or 2): '))
        if data_set != 1 and data_set != 2:
            print('main: invalid selection!')
            return
    except ValueError:
        print('main: you should select 1 or 2!')
        return
    directory_name = ['popsize_4', 'popsize_10', 'transformed_popsize']
    for direc in listdir(test_data_dir):
        path = join(test_data_dir, direc)
        if isdir(path):
            preProc = Preprocessor(path)
            if data_set == 1:
                text, doc_title, task_name = preProc.normalizeText_duc05(True)
            elif data_set == 2:
                text, doc_title, task_name = preProc.normalizeText_duc07(True)
            else:
                print('main: dataset was not selected correctly by user!')
                return
            preProc.preprocessing_text(text)
            preproc_doc_title = preProc.preprocessing_titles(doc_title)
            quantumGA = QuantumGA(
                preProc.preprocSentences,
                preProc.word_of_sent,
                preProc.preprocTokens,
                preProc.distWordFreq,
                0.5, 0.5,
                userSummLen=250
            )
            quantumGA.tfisf_cosineSim_Calculate()
            quantumGA.cosineSimWithTitle(preproc_doc_title)
            print('---{}:'.format(task_name))
            # DUC05: min_old=356 & max_old=1807 *** DUC07: min_old=162 & max_old=1398
            # designed_sized will be in [50, 100]
            designed_sizes = [int(preProc.sentencesNum / 4), int(preProc.sentencesNum / 10),
                              int((((preProc.sentencesNum - 162) / 1236) * 50) + 50)]
            for i in range(len(designed_sizes)):
                pop_size = designed_sizes[i]
                new_path = join(destination_dir, str(directory_name[i]))
                if not exists(new_path):
                    mkdir(new_path)
                # With this loop 10 summaries are generated for each topic
                for index in range(10):
                    quantum_pop = quantumGA.initialPop(pop_size)
                    quantumGA.measurement(quantum_pop)
                    for q_indiv in quantum_pop:
                        quantumGA.evalFitness3(q_indiv)
                    best_indiv = quantumGA.bestIndividual(quantum_pop)
                    generation = 1
                    fitness_change = 0  # The number of consecutive generation that fitness has not changed
                    while fitness_change < 20 and generation < 500:
                        mating_pool = quantumGA.rouletteWheel(
                            quantum_pop, pop_size)
                        q_offsprings = quantumGA.twoPointCrossover(mating_pool)
                        quantumGA.cusMutation(q_offsprings)
                        for offspring in q_offsprings:
                            if not offspring.fitness.valid:
                                quantumGA.indivMeasure(offspring)
                                quantumGA.evalFitness3(offspring)
                        new_qOffsprings = quantumGA.rotationGate(
                            q_offsprings, best_indiv)
                        quantumGA.measurement(new_qOffsprings)
                        for q_indiv in new_qOffsprings:
                            quantumGA.evalFitness3(q_indiv)
                        new_qPop = quantumGA.bestReplacement(
                            quantum_pop, new_qOffsprings)
                        best_indiv = quantumGA.bestIndividual(new_qPop)
                        if quantumGA.terminCriterion1(quantum_pop, new_qPop):
                            fitness_change += 1
                        else:
                            fitness_change = 0
                        quantum_pop = new_qPop[:]
                        generation += 1
                    finalSummLen = 0
                    summary = ''
                    for i in range(len(best_indiv.binary)):
                        if best_indiv.binary[i] == 1:
                            finalSummLen += len(quantumGA.tokens[i])
                            summary += '{}\n'.format(preProc.splitedSent[i])
                    summary = summary.rstrip()
                    with open(join(new_path, '{}_syst{}.txt'.format(task_name, index + 1)), 'w') as finalSumm:
                        finalSumm.write(summary)
                    print("Length of summary in round {}: {}".format(
                        index+1, finalSummLen))


if __name__ == "__main__":
    main()
