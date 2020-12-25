#!/usr/bin/env python3.6

import random
import math
from operator import attrgetter, itemgetter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from deap import base, creator
from copy import deepcopy
import numpy as np


class QuantumGA:
    def __init__(self,
                 preprocSent,
                 word_of_sent,
                 stemmed_tokens,
                 freq_dist_words,
                 w1, w2,
                 userSummLen=100
                 ):
        # List of Preprocess Sentences
        self.preprocSentences = preprocSent
        # Length of each individual
        self.indiv_len = len(self.preprocSentences)
        self.tokens = word_of_sent  # List of preprocess tokens
        self.stemmedTokens = stemmed_tokens
        self.distWordFreq = freq_dist_words  # List of frequency of distinct words
        self.weight1 = w1
        self.weight2 = w2
        self.desired_len = userSummLen

    def initialPop(self, popSize):
        q_population = []
        creator.create("Fitness", base.Fitness, weights=(1.0,))
        creator.create("Individual", list,
                       fitness=creator.Fitness, binary=list)
        i = 0
        while i < popSize:
            tempIndiv = []
            for _ in range(self.indiv_len):
                amplitude = round(1 / math.sqrt(2), 5)
                tempIndiv.append(dict(alpha=amplitude, beta=amplitude))
            q_population.append(creator.Individual(tempIndiv))
            i += 1
        return q_population

    def measurement(self, q_pop):
        for q_indiv in q_pop:
            summary_len = 0
            rand_indices = random.sample(range(len(q_indiv)), len(q_indiv))
            bin_indiv = [None] * len(q_indiv)
            for j in rand_indices:
                if (summary_len+len(self.tokens[j])) < self.desired_len:
                    rand = random.random()
                    if rand <= q_indiv[j]['alpha']**2:
                        bin_indiv[j] = 0
                    else:
                        bin_indiv[j] = 1
                        summary_len += len(self.tokens[j])
                else:
                    bin_indiv[j] = 0
            while summary_len == 0:
                rand_indices = random.sample(range(len(q_indiv)), len(q_indiv))
                for i in rand_indices:
                    if (summary_len+len(self.tokens[i])) < self.desired_len:
                        rand = random.random()
                        if rand <= q_indiv[i]['alpha']**2:
                            bin_indiv[i] = 0
                        else:
                            bin_indiv[i] = 1
                            summary_len += len(self.tokens[i])
                    else:
                        bin_indiv[i] = 0
            q_indiv.binary = bin_indiv

    def indivMeasure(self, q_indiv):
        summary_len = 0
        rand_indices = random.sample(range(len(q_indiv)), len(q_indiv))
        bin_indiv = [None] * len(q_indiv)
        for i in rand_indices:
            if (summary_len+len(self.tokens[i])) < self.desired_len:
                rand = random.random()
                if rand <= q_indiv[i]['alpha']**2:
                    bin_indiv[i] = 0
                else:
                    bin_indiv[i] = 1
                    summary_len += len(self.tokens[i])
            else:
                bin_indiv[i] = 0
        while summary_len == 0:
            rand_indices = random.sample(range(len(q_indiv)), len(q_indiv))
            for i in rand_indices:
                if (summary_len+len(self.tokens[i])) < self.desired_len:
                    rand = random.random()
                    if rand <= q_indiv[i]['alpha']**2:
                        bin_indiv[i] = 0
                    else:
                        bin_indiv[i] = 1
                        summary_len += len(self.tokens[i])
                else:
                    bin_indiv[i] = 0
        q_indiv.binary = bin_indiv

    def tfisf_cosineSim_Calculate(self):
        # norm used for normalization of each element of tf-isf
        vectorizer = TfidfVectorizer(
            token_pattern=r"(?u)\b\w+\b", min_df=1, smooth_idf=False)
        docTerm_matrix = vectorizer.fit_transform(self.preprocSentences)
        self.tf_isf = docTerm_matrix.sum(axis=1).reshape((-1,)).tolist()[0]
        self.cosineSim_otherSent = linear_kernel(
            docTerm_matrix, docTerm_matrix)

    def cosineSimWithTitle(self, doc_title):
        self.cosineSim_with_title = np.zeros((1, len(self.preprocSentences)))
        if len(doc_title):
            for i in range(len(self.preprocSentences)):
                preprocSent_title = [self.preprocSentences[i]]
                preprocSent_title.extend(doc_title)
                # norm used for normalization of each element of tf-isf
                vectorizer = TfidfVectorizer(
                    token_pattern=r"(?u)\b\w+\b", min_df=1, smooth_idf=False)
                docTerm_matrix = vectorizer.fit_transform(preprocSent_title)
                cosineSim_each_sent = linear_kernel(
                    docTerm_matrix[:1], docTerm_matrix[1:])
                self.cosineSim_with_title[0][i] = sum(cosineSim_each_sent[0])

    def evalFitness1(self, q_indiv):
        '''This function evaluates individual according to words frequency and sentence position'''
        self.discoverSlope = -0.625
        _summary = []
        sumOfSummaryWordsFreq = 0
        sumOfMostFreqWords = 0
        numberOfSelectedSent = 0
        n = len(self.stemmedTokens)
        _distWordsFreqList = sorted(
            self.distWordFreq.items(), key=lambda x: x[1], reverse=True)
        for i in range(len(q_indiv.binary)):
            if q_indiv.binary[i] == 1:
                for word in self.stemmedTokens[i]:
                    if word not in _summary:
                        _summary.append(word)
                        sumOfSummaryWordsFreq += self.distWordFreq.get(
                            '{}'.format(word), 0)
                numberOfSelectedSent += 1
        for j in range(len(_summary)):
            sumOfMostFreqWords += _distWordsFreqList[j][1]
        beta = sumOfSummaryWordsFreq / sumOfMostFreqWords
        if self.discoverSlope == -1:
            pass
        elif self.discoverSlope == 0:
            gama = 1
        elif -1 < self.discoverSlope < 0:
            sum1, sum2 = 0, 0
            x = 1 + ((n - 1) / 2)
            for i in range(len(q_indiv.binary)):
                if q_indiv.binary[i] == 1:
                    sum1 += (self.discoverSlope * (i - x)) + x
            for j in range(numberOfSelectedSent):
                sum2 += (self.discoverSlope * (j - x)) + x
            gama = sum1 / sum2
        fitness = beta * gama
        q_indiv.fitness.values = (fitness, )

    def evalFitness2(self, q_indiv):
        '''This function evaluates each individual based on length of sentences,
        cosine similarity between each selected sentence with other sentences and
        cosine similarity between each selected sentence with title.'''
        sentencesLength_by_words = 0
        longestSentence = len(max(self.stemmedTokens, key=len))
        _inSenteceCosineSim = 0
        _titleCosineSim = 0
        summary = []
        for i in range(len(q_indiv.binary)):
            if q_indiv.binary[i] == 1:
                _inSenteceCosineSim += sum(
                    self.cosineSim_otherSent[i]) - self.cosineSim_otherSent[i, i]
                _titleCosineSim += self.cosineSim_with_title[0, i]
                try:
                    sentencesLength_by_words += (
                        len(self.stemmedTokens[i]) / longestSentence)
                except ZeroDivisionError:
                    sentencesLength_by_words += len(self.stemmedTokens[i])
                summary.append(self.preprocSentences[i])
        # norm used for normalization of each element of tf-isf
        new_vectorizer = TfidfVectorizer(
            token_pattern=r"(?u)\b\w+\b", min_df=1, smooth_idf=False)
        docTerm_matrix = new_vectorizer.fit_transform(summary)
        cosineSim_summaryMatrix = linear_kernel(docTerm_matrix, docTerm_matrix)
        cosineSim_summarySent = sum([sum(cosineSim_summaryMatrix[j])-cosineSim_summaryMatrix[j, j]
                                     for j in range(cosineSim_summaryMatrix.shape[0])])
        cosine_similarity = self.weight1 * \
            (_inSenteceCosineSim-cosineSim_summarySent) + \
            self.weight2*_titleCosineSim
        q_indiv.fitness.values = (
            (0.6 * cosine_similarity + 0.4 * sentencesLength_by_words), )

    def evalFitness3(self, q_indiv):
        '''Main objective function: this function evaluates each individual
        based on words frequency, sentence position, length of sentences,
        cosine similarity between each selected sentence with other sentences
        and cosine similarity between each selected sentence with title.'''
        self.discoverSlope = -0.625
        _summary = []
        sumOfSummaryWordsFreq = 0
        sumOfMostFreqWords = 0
        numberOfSelectedSent = 0
        n = len(self.stemmedTokens)
        _distWordsFreqList = sorted(
            self.distWordFreq.items(), key=lambda x: x[1], reverse=True)
        for i in range(len(q_indiv.binary)):
            if q_indiv.binary[i] == 1:
                for word in self.stemmedTokens[i]:
                    if word not in _summary:
                        _summary.append(word)
                        sumOfSummaryWordsFreq += self.distWordFreq.get(
                            '{}'.format(word), 0)
                numberOfSelectedSent += 1
        for j in range(len(_summary)):
            sumOfMostFreqWords += _distWordsFreqList[j][1]
        wordFreq_metric = sumOfSummaryWordsFreq / sumOfMostFreqWords
        if self.discoverSlope == -1:
            pass
        elif self.discoverSlope == 0:
            sent_position = 1
        elif -1 < self.discoverSlope < 0:
            sum1, sum2 = 0, 0
            x = 1 + ((n - 1) / 2)
            for i in range(len(q_indiv.binary)):
                if q_indiv.binary[i] == 1:
                    sum1 += (self.discoverSlope * (i - x)) + x
            for j in range(numberOfSelectedSent):
                sum2 += (self.discoverSlope * (j - x)) + x
            sent_position = sum1 / sum2
        sentencesLength_by_words = 0
        longestSentence = len(max(self.stemmedTokens, key=len))
        _inSenteceCosineSim = 0
        _titleCosineSim = 0
        summary = []
        for i in range(len(q_indiv.binary)):
            if q_indiv.binary[i] == 1:
                _inSenteceCosineSim += sum(
                    self.cosineSim_otherSent[i]) - self.cosineSim_otherSent[i, i]
                _titleCosineSim += self.cosineSim_with_title[0, i]
                try:
                    sentencesLength_by_words += (
                        len(self.stemmedTokens[i]) / longestSentence)
                except ZeroDivisionError:
                    sentencesLength_by_words += len(self.stemmedTokens[i])
                summary.append(self.preprocSentences[i])
        # norm used for normalization of each element of tf-isf
        new_vectorizer = TfidfVectorizer(
            token_pattern=r"(?u)\b\w+\b", min_df=1, smooth_idf=False)
        docTerm_matrix = new_vectorizer.fit_transform(summary)
        cosineSim_summaryMatrix = linear_kernel(docTerm_matrix, docTerm_matrix)
        cosineSim_summarySent = sum([sum(cosineSim_summaryMatrix[j])-cosineSim_summaryMatrix[j, j]
                                     for j in range(cosineSim_summaryMatrix.shape[0])])
        cosine_similarity = self.weight1 * \
            (_inSenteceCosineSim-cosineSim_summarySent) + \
            self.weight2*_titleCosineSim
        q_indiv.fitness.values = (
            (0.6 * cosine_similarity + 0.4 * sentencesLength_by_words) * (wordFreq_metric * sent_position), )

    # Select the 'k' best parents for mating using Roulette wheel sampling
    def rouletteWheel(self, quantum_pop, k):
        if k % 2 != 0:
            k += 1
        pool_size = int(k / 2)
        sorted_pop = sorted(quantum_pop, key=attrgetter("fitness"))
        sum_fits = sum(indiv.fitness.values[0] for indiv in quantum_pop)
        prob_indiv = []
        cumulative_indiv = []
        cumulativeProb = 0
        for index, indiv in enumerate(sorted_pop):
            prob_indiv.append(indiv.fitness.values[0]/sum_fits)
            cumulativeProb += prob_indiv[index]
            cumulative_indiv.append(cumulativeProb)
        cumulative_indiv[len(cumulative_indiv) - 1] = 1.0
        mating_pool = [None] * pool_size
        current_member = 0
        while current_member < pool_size:
            r1 = random.random()
            r2 = random.random()
            i, j = 0, 0
            while cumulative_indiv[i] < r1:
                i += 1
            tempIndividual1 = deepcopy(sorted_pop[i])
            while cumulative_indiv[j] < r2 or sorted_pop[j].binary == tempIndividual1.binary:
                j += 1
                if j >= len(cumulative_indiv):
                    r2 = random.random()
                    j = 0
            tempIndividual2 = deepcopy(sorted_pop[j])
            # mating pool is a list of tuples that each tuple contains two list of dictionaries (quantum individual)
            mating_pool[current_member] = (tempIndividual1, tempIndividual2)
            current_member += 1
            r1, r2 = 0, 0
        return mating_pool

    def twoPointCrossover(self, mating_pool, crossover_rate=0.75):
        offspring_pop = []
        for parent1, parent2 in mating_pool:
            if crossover_rate >= random.random():
                tempSize = min(len(parent1), len(parent2))
                cxpoint1 = random.randint(0, tempSize)
                cxpoint2 = random.randint(0, tempSize - 1)
                if cxpoint2 >= cxpoint1:
                    cxpoint2 += 1
                else:
                    cxpoint1, cxpoint2 = cxpoint2, cxpoint1
                parent1[cxpoint1: cxpoint2], parent2[cxpoint1:
                                                     cxpoint2] = parent2[cxpoint1:cxpoint2], parent1[cxpoint1:cxpoint2]
                del parent1.fitness.values
                del parent2.fitness.values
            offspring_pop.extend([parent1, parent2])
        return offspring_pop

    def cusMutation(self, offsprings, mutation_rate=0.005):
        for individual in offsprings:
            checker = False
            for q_gate in individual:
                if mutation_rate >= random.random():
                    q_gate['alpha'], q_gate['beta'] = q_gate['beta'], q_gate['alpha']
                    if not checker:
                        del individual.fitness.values
                        checker = True

    def rotationGate(self, q_population, best_indiv):
        rot = np.empty([2, 2])
        new_q_population = q_population[:]
        for i in range(len(q_population)):
            indiv_summary_len = self.checkSummaryLen(q_population[i])
            currIndiv_fitn, bestIndiv_fitn = q_population[i].fitness.values[0], best_indiv.fitness.values[0]
            modified_qbit_index = []
            if currIndiv_fitn < bestIndiv_fitn:
                for j in range(len(q_population[i])):
                    if q_population[i].binary[j] == 0 and best_indiv.binary[j] == 1 and indiv_summary_len < self.desired_len-20:
                        # Define the rotation angle: delta_theta
                        delta_theta = (0.005*math.pi + (0.05*math.pi-0.005*math.pi)*(abs(currIndiv_fitn-bestIndiv_fitn)/bestIndiv_fitn) +
                                       (abs(indiv_summary_len-self.desired_len)/indiv_summary_len)*0.25*math.pi)
                        rot[0, 0] = math.cos(delta_theta)
                        rot[0, 1] = -math.sin(delta_theta)
                        rot[1, 0] = math.sin(delta_theta)
                        rot[1, 1] = math.cos(delta_theta)
                        new_alpha = (rot[0, 0] * q_population[i][j]['alpha']
                                     ) + (rot[0, 1] * q_population[i][j]['beta'])
                        new_beta = (rot[1, 0] * q_population[i][j]['alpha']
                                    ) + (rot[1, 1] * q_population[i][j]['beta'])
                        new_q_population[i][j]['alpha'] = round(new_alpha, 2)
                        new_q_population[i][j]['beta'] = round(new_beta, 2)
                        indiv_summary_len += len(self.tokens[j])
                        modified_qbit_index.append(j)
                    elif q_population[i].binary[j] == 0 and best_indiv.binary[j] == 1 and indiv_summary_len > self.desired_len:
                        delta_theta = -(0.005*math.pi + (0.05*math.pi-0.005*math.pi)
                                        * (abs(currIndiv_fitn-bestIndiv_fitn)/bestIndiv_fitn))
                        rot[0, 0] = math.cos(delta_theta)
                        rot[0, 1] = -math.sin(delta_theta)
                        rot[1, 0] = math.sin(delta_theta)
                        rot[1, 1] = math.cos(delta_theta)
                        new_alpha = (rot[0, 0] * q_population[i][j]['alpha']
                                     ) + (rot[0, 1] * q_population[i][j]['beta'])
                        new_beta = (rot[1, 0] * q_population[i][j]['alpha']
                                    ) + (rot[1, 1] * q_population[i][j]['beta'])
                        new_q_population[i][j]['alpha'] = round(new_alpha, 2)
                        new_q_population[i][j]['beta'] = round(new_beta, 2)
                    elif q_population[i].binary[j] == 1 and best_indiv.binary[j] == 0 and indiv_summary_len > self.desired_len:
                        delta_theta = -(0.005*math.pi + (0.05*math.pi-0.005*math.pi)*(abs(currIndiv_fitn-bestIndiv_fitn)/bestIndiv_fitn) +
                                        (abs(indiv_summary_len-self.desired_len)/indiv_summary_len)*0.25*math.pi)
                        rot[0, 0] = math.cos(delta_theta)
                        rot[0, 1] = -math.sin(delta_theta)
                        rot[1, 0] = math.sin(delta_theta)
                        rot[1, 1] = math.cos(delta_theta)
                        new_alpha = (rot[0, 0] * q_population[i][j]['alpha']
                                     ) + (rot[0, 1] * q_population[i][j]['beta'])
                        new_beta = (rot[1, 0] * q_population[i][j]['alpha']
                                    ) + (rot[1, 1] * q_population[i][j]['beta'])
                        new_q_population[i][j]['alpha'] = round(new_alpha, 2)
                        new_q_population[i][j]['beta'] = round(new_beta, 2)
                        indiv_summary_len -= len(self.tokens[j])
                        modified_qbit_index.append(j)
                    elif q_population[i].binary[j] == 1 and best_indiv.binary[j] == 0 and indiv_summary_len < self.desired_len-20:
                        delta_theta = (0.005*math.pi + (0.05*math.pi-0.005*math.pi)
                                       * (abs(currIndiv_fitn-bestIndiv_fitn)/bestIndiv_fitn))
                        rot[0, 0] = math.cos(delta_theta)
                        rot[0, 1] = -math.sin(delta_theta)
                        rot[1, 0] = math.sin(delta_theta)
                        rot[1, 1] = math.cos(delta_theta)
                        new_alpha = (rot[0, 0] * q_population[i][j]['alpha']
                                     ) + (rot[0, 1] * q_population[i][j]['beta'])
                        new_beta = (rot[1, 0] * q_population[i][j]['alpha']
                                    ) + (rot[1, 1] * q_population[i][j]['beta'])
                        new_q_population[i][j]['alpha'] = round(new_alpha, 2)
                        new_q_population[i][j]['beta'] = round(new_beta, 2)
                del new_q_population[i].fitness.values
            if indiv_summary_len > self.desired_len:
                cosineSimWithTitle = self.cosineSim_with_title[0][:]
                indices = [j for j in range(len(
                    q_population[i])) if j not in modified_qbit_index and q_population[i].binary[j] == 1]
                cosineSimWithTitle = cosineSimWithTitle[indices].tolist()
                while indiv_summary_len > self.desired_len:
                    index = cosineSimWithTitle.index(min(cosineSimWithTitle))
                    selSent_index = indices[index]
                    delta_theta = -(0.005*math.pi + (0.05*math.pi-0.005*math.pi) *
                                    (abs(currIndiv_fitn-bestIndiv_fitn)/max(currIndiv_fitn, bestIndiv_fitn)) +
                                    (abs(indiv_summary_len-self.desired_len)/indiv_summary_len)*0.25*math.pi)
                    rot[0, 0] = math.cos(delta_theta)
                    rot[0, 1] = -math.sin(delta_theta)
                    rot[1, 0] = math.sin(delta_theta)
                    rot[1, 1] = math.cos(delta_theta)
                    new_alpha = (rot[0, 0] * q_population[i][selSent_index]['alpha']
                                 ) + (rot[0, 1] * q_population[i][selSent_index]['beta'])
                    new_beta = (rot[1, 0] * q_population[i][selSent_index]['alpha']
                                ) + (rot[1, 1] * q_population[i][selSent_index]['beta'])
                    new_q_population[i][selSent_index]['alpha'] = round(
                        new_alpha, 2)
                    new_q_population[i][selSent_index]['beta'] = round(
                        new_beta, 2)
                    indiv_summary_len -= len(self.tokens[selSent_index])
                    indices.remove(selSent_index)
                    del cosineSimWithTitle[index]
                del new_q_population[i].fitness.values
        return new_q_population

    def bestReplacement(self, q_parents, q_offsprings):
        pop_size = len(q_parents)
        sorted_parents = sorted(
            q_parents, key=attrgetter("fitness"), reverse=True)
        sorted_offspr = sorted(
            q_offsprings, key=attrgetter("fitness"), reverse=True)
        if pop_size % 2 == 0:
            new_population = sorted_parents[0:int(
                pop_size/2)] + sorted_offspr[0:int(pop_size/2)]
        else:
            new_population = sorted_parents[0:int(
                pop_size/2)] + sorted_offspr[0:int(pop_size/2+1)]
        return new_population

    def tournamReplec(self, q_parents, q_offsp):
        pop_size = len(q_parents)
        new_population = []
        current_member = 0
        while current_member < pop_size:
            k_random = random.sample(range(len(q_parents)), int(pop_size/2))
            temp_kParents, temp_kOffsp = itemgetter(
                *k_random)(q_parents[:]), itemgetter(*k_random)(q_offsp[:])
            best_parent = max(temp_kParents, key=attrgetter("fitness"))
            best_offsp = max(temp_kOffsp, key=attrgetter("fitness"))
            if best_parent != best_offsp:
                if best_parent not in new_population:
                    new_population.append(best_parent)
                    current_member += 1
                if best_offsp not in new_population:
                    new_population.append(best_offsp)
                    current_member += 1
            else:
                if best_parent not in new_population:
                    new_population.append(best_parent)
                    current_member += 1
        return new_population

    def bestIndividual(self, q_population):
        best_qIndiv = deepcopy(max(q_population, key=attrgetter("fitness")))
        return best_qIndiv

    # Difference between best individuals of current generation and best individuals of former generation
    def terminCriterion1(self, former_Qpop, next_Qpop):
        size = int(len(next_Qpop) / 2)
        if size != 0:
            sorted_currPop = sorted(
                next_Qpop, key=attrgetter("fitness"), reverse=True)
            sorted_formPop = sorted(
                former_Qpop, key=attrgetter("fitness"), reverse=True)
            mean_currPop = sum([indiv.fitness.values[0]
                                for indiv in sorted_currPop[0:size]]) / size
            mean_formPop = sum([indiv.fitness.values[0]
                                for indiv in sorted_formPop[0:size]]) / size
            if abs(mean_currPop - mean_formPop) < 0.05:
                return True
        return False

    # New termination criterion proposed in "Quantum-Inspired Evolutionary Algorithm With a New Termination Criterion" paper
    # This Termination Criterion checks the average convergence of population to "0" because it checks 'alpha' values
    def terminCriterion2(self, q_pop):
        pop_size = len(q_pop)
        individual_size = len(q_pop[0])
        delta = 0.95
        average_converge = 0
        for indiv in q_pop:
            indiv_converge = 0
            for q_bit in indiv:
                indiv_converge += abs(1 - 2 * (q_bit['alpha'] ** 2))
            indiv_converge /= individual_size
            average_converge += indiv_converge
        average_converge /= pop_size
        if average_converge > delta:
            return True
        return False

    def checkSummaryLen(self, indiv):
        indivSummLen = sum([len(self.tokens[i]) for i in range(
            len(indiv.binary)) if indiv.binary[i] == 1])
        return indivSummLen
