#!/usr/bin/env python3.6

import re
import string
import itertools
import spacy
import xml.etree.ElementTree as ET
from os import listdir, remove
from os.path import isfile, join, splitext, basename
from nltk.data import load
from nltk.tokenize import word_tokenize, TreebankWordTokenizer
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.stem import PorterStemmer
from truecase import get_true_case


class Preprocessor:
    def __init__(self, *args):
        if len(args) > 0:
            self.filesList = args
        else:
            self.filesList = (input(
                'You haven\'t entered any directory for document(s). Please enter a directory in Unix format:\n'), )
        self.sentencesNum = 0
        self.numOfWords = 0
        self._xmlPart = dict()

    def normalizeText_duc05(self, multi_doc):
        # Added list of not_hyphenated words that should correct in titles
        not_hyphenated = [
            'COLOMBIA-L.A.', 'MIAMI-TO-L.A.', 'Turkey-Syria', 'VW-Opel', 'GM-VW', 'O.C.-L.A.',
            'Ever-more-complex', 'ARGENTINE-BRITISH', '46-MILLION', 'UK-Argentine', 'PRO-MARIJUANA',
            'test-case', '2.46-MILLION', 'plasma-based', 'JOB-DEATH', 'GOLD-MINING', '1.3-MILLION',
            '150-MILLION-A-YEAR', 'Atom-smasher', 'LITHIUM-WATER', 'NERVE-GAS', 'drug-makers',
            'EAST-WEST', '2-MILLION-SQUARE-MILE', 'Hume-Adams', 'BEAR-POACHERS', 'Robot-selected',
            'self-rule', 'Ulster-style', '71-MILLION', '1.5-MILLION', '12-MILLION', '7-MILLION',
            'Iran-U.S.'
        ]
        type_of_titles = ['HEAD', 'HL', 'HEADLINE', 'H3']
        task_name = splitext(basename(self.filesList[0]))[0]
        if not multi_doc:
            while True:
                try:
                    tree = ET.parse(self.filesList[0])
                except (FileNotFoundError, PermissionError):
                    self.filesList = (input(
                        "The path doesn\'t contain file\'s name. Please enter complete path of a file containing file\'s name:\n"),)
                except ET.ParseError:
                    with open(self.filesList[0]) as xmlFile:
                        xml_content = xmlFile.read()
                    xml_content = re.sub(
                        r'(=)([0-9]+)(>)', r'\1"\2"\3', xml_content)
                    xml_content = re.sub(r'&', r'&amp;', xml_content)
                    with open('{}.xml'.format(task_name), 'w') as new_xmlFile:
                        new_xmlFile.write(xml_content)
                    tree = ET.parse('{}.xml'.format(task_name))
                    remove('{}.xml'.format(task_name))
                    break
                else:
                    break
            root = tree.getroot()
            for child in root:
                if child.tag not in self._xmlPart:
                    self._xmlPart[child.tag] = child.text
                else:
                    pass
                for subChild in child:
                    if subChild.text is not None:
                        self._xmlPart[child.tag] += subChild.text
                    else:
                        pass
                    if child.tag == 'TEXT' and subChild.tag == 'F':
                        self._xmlPart[child.tag] += subChild.tail
            text = self._xmlPart.get('TEXT')
            _title = [value for key, value in self._xmlPart.items()
                      if key in type_of_titles]
            doc_title = ''
            if _title:
                doc_title += _title[0]
                doc_title = re.sub(r'\n\n', '\n', doc_title)
                doc_title = re.sub(r'([^.])\n', r'\1 ', doc_title)
                doc_title = re.sub(r'@', r'', doc_title)
                doc_title = re.sub(r' +----+ +.*', r'', doc_title)
                doc_title = re.sub(r'[``|"|:]', '', doc_title)
                doc_title = re.sub(r' +', ' ', doc_title)
                doc_title.strip()
        else:
            if len(self.filesList) == 1:
                while True:
                    try:
                        files = [file for file in listdir(self.filesList[0]) if isfile(
                            join(self.filesList[0], file))]
                    except NotADirectoryError:
                        self.filesList = (
                            input("You should enter a folder directory:\n"),)
                    else:
                        break
                text, doc_title = '', ''
                for file in files:
                    try:
                        tree = ET.parse(join(self.filesList[0], file))
                    except ET.ParseError:
                        with open(join(self.filesList[0], file)) as xmlFile:
                            xml_content = xmlFile.read()
                        xml_content = re.sub(
                            r'(=)([0-9]+)(>)', r'\1"\2"\3', xml_content)
                        xml_content = re.sub(r'&', r'&amp;', xml_content)
                        with open('{}.xml'.format(task_name), 'w') as new_xmlFile:
                            new_xmlFile.write(xml_content)
                        tree = ET.parse('{}.xml'.format(task_name))
                        remove('{}.xml'.format(task_name))
                    root = tree.getroot()
                    xml_part = dict()
                    self.find_tags_recursively(root, xml_part)
                    temp_text = xml_part.get('TEXT')
                    _title = [value for key, value in xml_part.items()
                              if key in type_of_titles]
                    title_text = ''
                    if _title:
                        title_text += _title[0].lstrip()
                        title_text = re.sub(r'([^.])\n', r'\1 ', title_text)
                        title_text = re.sub('FT.*[0-9]+ +/ +', '', title_text)
                        title_text = re.sub(r'--+', r'', title_text)
                        title_text = re.sub(r' +- *', r' ', title_text)
                        title_text = re.sub(r'(``|\'\'|")', r'', title_text)
                        # For removing ' from beginning of a quotation
                        title_text = re.sub(
                            r'(\s)\'(\w+)', r'\1\2', title_text)
                        title_text = re.sub(
                            r' *\(.*\)', r'', title_text, re.DOTALL)
                        title_text = re.sub(r'\n', r' ', title_text)
                        title_text = title_text.strip()
                        if file.startswith('LA'):
                            title_text = get_true_case(title_text)
                        doc_title += '{}\n'.format(title_text)
                    if temp_text:
                        # For removing complete tables from text
                        temp_text = re.sub(
                            r'( +-{4,}\n)(.+\n)*( +-{4,}\n*)', r'', temp_text)
                        temp_text = re.sub(r'([^.])\n', r'\1 ', temp_text)
                        temp_text = temp_text.strip()
                        text += '{}\n'.format(temp_text)
                doc_title = re.sub(r'\n\n', r'\n', doc_title)
                doc_title = re.sub(r'\n ', r'\n', doc_title)
                hyphen_words = re.findall(
                    r'\s*(?:{})(?:\.|,|!|\?|/|\'|\s+)'.format('|'.join(not_hyphenated)), doc_title, re.I)
                if hyphen_words:
                    hyphen_dict = dict()
                    for word in hyphen_words:
                        hyphen_dict.update({word: word.replace('-', ' ')})
                    hyphen_dict = dict((re.escape(k), v)
                                       for k, v in hyphen_dict.items())
                    pattern = re.compile("|".join(hyphen_dict.keys()))
                    doc_title = pattern.sub(
                        lambda m: hyphen_dict[re.escape(m.group(0))], doc_title)
                doc_title = doc_title.replace('POP/ROCK', 'POP ROCK')
                doc_title = doc_title.replace('/LOCAL', 'LOCAL')
        text = text.strip()
        text = re.sub(r'\.( \"[A|a]nd)', r'\1', text)
        text = re.sub(r'(\.\.\.|\. \. \.)', r'', text)
        text = re.sub(r'--', r'', text)
        text = re.sub(r'([A-Z.][A-Z.])( +\n+ +)', r'\1 .\2', text)
        text = re.sub(r'(Inc\.)( +\n\n+ +)', r'\1 .\2', text)
        text = re.sub(r'\n\n', '\n', text)
        text = re.sub(r'.(\" [a-zA-Z0-9]* said.)', r',\1', text)
        text = re.sub(r'(``|\'\'|")', r'', text)
        # for removing ' from beginning of a quotation
        text = re.sub(r'(\s)\'(\w+)', r'\1\2', text)
        text = re.sub(r' \. \. \.', r'', text)
        text = re.sub(r'(.)\s+(\n)', r'\1\2', text)
        text = re.sub(r'([^.])\n', r'\1 ', text)
        text = re.sub(r'^.*\[Text\] ', r'', text)
        text = re.sub(r'\[.*\] ', r'', text)
        text = re.sub(r'(\n) (\S)', r'\1\2', text)
        text = re.sub(r'(AG|GM|VW|Volkswagen|and|he|his|Essex)/'
                      '(GM|Opel|General|or|she|her|London)', r'\1 \2', text)
        return text, doc_title, task_name

    def normalizeText_duc07(self, multi_doc):
        # Added list of not_hyphenated words that should correct in titles
        not_hyphenated = [
            'COLOMBIA-L.A.', 'MIAMI-TO-L.A.', 'Turkey-Syria', 'VW-Opel', 'GM-VW', 'O.C.-L.A.',
            'Ever-more-complex', 'ARGENTINE-BRITISH', '46-MILLION', 'UK-Argentine', 'PRO-MARIJUANA',
            'test-case', '2.46-MILLION', 'plasma-based', 'JOB-DEATH', 'GOLD-MINING', '1.3-MILLION',
            '150-MILLION-A-YEAR', 'Atom-smasher', 'LITHIUM-WATER', 'NERVE-GAS', 'drug-makers',
            'EAST-WEST', '2-MILLION-SQUARE-MILE', 'Hume-Adams', 'BEAR-POACHERS', 'Robot-selected',
            'self-rule', 'Ulster-style', '71-MILLION', '1.5-MILLION', '12-MILLION', '7-MILLION',
            'Iran-U.S.'
        ]
        type_of_titles = 'HEADLINE'
        task_name = splitext(basename(self.filesList[0]))[0]
        if not multi_doc:
            while True:
                try:
                    tree = ET.parse(self.filesList[0])
                except (FileNotFoundError, PermissionError):
                    self.filesList = (input(
                        "The path doesn\'t contain file\'s name. Please enter complete path of a file containing file\'s name:\n"),)
                except ET.ParseError:
                    with open(self.filesList[0]) as xmlFile:
                        xml_content = xmlFile.read()
                    xml_content = re.sub(
                        r'(=)([0-9]+)(>)', r'\1"\2"\3', xml_content)
                    xml_content = re.sub(r'&', r'&amp;', xml_content)
                    with open('{}.xml'.format(task_name), 'w') as new_xmlFile:
                        new_xmlFile.write(xml_content)
                    tree = ET.parse('{}.xml'.format(task_name))
                    remove('{}.xml'.format(task_name))
                    break
                else:
                    break
            root = tree.getroot()
            for child in root:
                if child.tag not in self._xmlPart:
                    self._xmlPart[child.tag] = child.text
                else:
                    pass
                for subChild in child:
                    if subChild.text is not None:
                        self._xmlPart[child.tag] += subChild.text
                    else:
                        pass
                    if child.tag == 'TEXT' and subChild.tag == 'F':
                        self._xmlPart[child.tag] += subChild.tail
            text = self._xmlPart.get('TEXT')
            _title = [value for key, value in self._xmlPart.items()
                      if key == type_of_titles]
            doc_title = ''
            if _title:
                doc_title += _title[0]
                doc_title = re.sub(r'\n\n', '\n', doc_title)
                doc_title = re.sub(r'([^.])\n', r'\1 ', doc_title)
                doc_title = re.sub(r'@', r'', doc_title)
                doc_title = re.sub(r' +----+ +.*', r'', doc_title)
                doc_title = re.sub(r'[``|"|:]', '', doc_title)
                doc_title = re.sub(r' +', ' ', doc_title)
                doc_title.strip()
        else:
            if len(self.filesList) == 1:
                while True:
                    try:
                        files = [file for file in listdir(self.filesList[0]) if isfile(
                            join(self.filesList[0], file))]
                    except NotADirectoryError:
                        self.filesList = (
                            input("You should enter a folder directory:\n"),)
                    else:
                        break
                text, doc_title = '', ''
                for file in files:
                    try:
                        tree = ET.parse(join(self.filesList[0], file))
                    except ET.ParseError:
                        with open(join(self.filesList[0], file)) as xmlFile:
                            xml_content = xmlFile.read()
                        xml_content = re.sub(
                            r'(=)([0-9]+)(>)', r'\1"\2"\3', xml_content)
                        xml_content = re.sub(r'&', r'&amp;', xml_content)
                        with open('{}.xml'.format(task_name), 'w') as new_xmlFile:
                            new_xmlFile.write(xml_content)
                        tree = ET.parse('{}.xml'.format(task_name))
                        remove('{}.xml'.format(task_name))
                    root = tree.getroot()
                    xml_part = dict()
                    self.find_tags_recursively(root, xml_part)
                    temp_text = xml_part.get('TEXT')
                    _title = [value for key, value in xml_part.items()
                              if key == type_of_titles]
                    title_text = ''
                    if _title and not re.match(r'\s*&HT;\s*', _title[0]):
                        title_text += _title[0].lstrip()
                        title_text = re.sub(r'([^.])\n', r'\1 ', title_text)
                        title_text = re.sub('FT.*[0-9]+ +/ +', '', title_text)
                        title_text = re.sub(r'--+', r'', title_text)
                        title_text = re.sub(r' +- *', r' ', title_text)
                        title_text = re.sub(r'(``|\'\'|")', r'', title_text)
                        # For removing ' from beginning of a quotation
                        title_text = re.sub(
                            r'(\s)\'(\w+)', r'\1\2', title_text)
                        title_text = re.sub(
                            r' *\(.*\)', r'', title_text, re.DOTALL)
                        title_text = re.sub(r'\n', r' ', title_text)
                        title_text = title_text.strip()
                        title_text = get_true_case(title_text)
                        doc_title += '{}\n'.format(title_text)
                    if temp_text:
                        # For removing complete tables from text
                        temp_text = re.sub(
                            r'( +-{4,}\n)(.+\n)*( +-{4,}\n*)', r'', temp_text)
                        temp_text = re.sub(r'([^.])\n', r'\1 ', temp_text)
                        temp_text = temp_text.strip()
                        text += '{}\n'.format(temp_text)
                doc_title = re.sub(r'\n\n', r'\n', doc_title)
                doc_title = re.sub(r'\n ', r'\n', doc_title)
                hyphen_words = re.findall(
                    r'\s*(?:{})(?:\.|,|!|\?|/|\'|\s+)'.format('|'.join(not_hyphenated)), doc_title, re.I)
                if hyphen_words:
                    hyphen_dict = dict()
                    for word in hyphen_words:
                        hyphen_dict.update({word: word.replace('-', ' ')})
                    hyphen_dict = dict((re.escape(k), v)
                                       for k, v in hyphen_dict.items())
                    pattern = re.compile("|".join(hyphen_dict.keys()))
                    doc_title = pattern.sub(
                        lambda m: hyphen_dict[re.escape(m.group(0))], doc_title)
                doc_title = doc_title.replace('POP/ROCK', 'POP ROCK')
                doc_title = doc_title.replace('/LOCAL', 'LOCAL')
        text = text.strip()
        text = re.sub(r'\.( \"[A|a]nd)', r'\1', text)
        text = re.sub(r'(\.\.\.|\. \. \.)', r'', text)
        text = re.sub(r'--', r'', text)
        text = re.sub(r'([A-Z.][A-Z.])( +\n+ +)', r'\1 .\2', text)
        text = re.sub(r'(Inc\.)( +\n\n+ +)', r'\1 .\2', text)
        text = re.sub(r'\n\n', '\n', text)
        text = re.sub(r'.(\" [a-zA-Z0-9]* said.)', r',\1', text)
        text = re.sub(r'(``|\'\'|")', r'', text)
        # for removing ' from beginning of a quotation
        text = re.sub(r'(\s)\'(\w+)', r'\1\2', text)
        text = re.sub(r' \. \. \.', r'', text)
        text = re.sub(r'(.)\s+(\n)', r'\1\2', text)
        text = re.sub(r'([^.])\n', r'\1 ', text)
        text = re.sub(r'^.*\[Text\] ', r'', text)
        text = re.sub(r'\[.*\] ', r'', text)
        text = re.sub(r'(\n) (\S)', r'\1\2', text)
        text = re.sub(r'(AG|GM|VW|Volkswagen|and|he|his|Essex)/'
                      '(GM|Opel|General|or|she|her|London)', r'\1 \2', text)
        return text, doc_title, task_name

    def preprocessing_text(self, text):
        tokenizer = load('tokenizers/punkt/english.pickle')
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
        tokenized_sents = tokenizer.tokenize(text)
        # text_tagged = self.part_of_speech_tagging(tokenized_sents) #Part-of-speech tagging of text sentences
        # working_sentence = [sent.lower() for sent in tokenized_sents] #****!!!!! lower after tokenization
        self.word_of_sent = []
        self.tokens = []
        punctuations = list(string.punctuation)
        punctuations.extend(["\'\'", "\"", "``", "--"])
        TreebankWordTokenizer.PUNCTUATION = [
            (re.compile(r'([:,])([^\d])'), r' \1 \2'),
            (re.compile(r'([:,])$'), r' \1 '),
            (re.compile(r'\.\.\.'), r' ... '),
            (re.compile(r'[;@#$%]'), r' \g<0> '),
            # Handles the final period.
            (re.compile(r'([^\.])(\.)([\]\)}>"\']*)\s*$'), r'\1 \2\3 '),
            (re.compile(r'[?!]'), r' \g<0> '),
            (re.compile(r"([^'])' "), r"\1 ' "),
        ]
        _index = []
        for i in range(len(tokenized_sents)):
            working_sentence = [word_tokenize(
                sentence) for sentence in tokenized_sents]
        for i in range(len(working_sentence)):
            _sentenceWords = [
                word.lower() for word in working_sentence[i] if word not in punctuations]
            if len(_sentenceWords) > 2:
                self.word_of_sent.append(_sentenceWords)
            else:
                _index.append(i)
        self.splitedSent = [tokenized_sents[i]
                            for i in range(len(tokenized_sents)) if i not in _index]
        self.sentencesNum = len(self.splitedSent)
        for i in range(len(self.word_of_sent)):
            self.tokens.append([word for word in self.word_of_sent[i]
                                if word not in stopwords.words('english-new')])
            self.numOfWords += len(self.tokens[i])
        stemmer = PorterStemmer()
        self.preprocTokens = []
        for j in range(len(self.tokens)):
            self.preprocTokens.append([stemmer.stem(words)
                                       for words in self.tokens[j]])
        _allTokens = itertools.chain.from_iterable(self.preprocTokens)
        self.distWordFreq = FreqDist(_allTokens)
        self.preprocSentences = []
        for i in range(len(self.preprocTokens)):
            self.preprocSentences.append(' '.join(self.preprocTokens[i]))

    def preprocessing_titles(self, doc_titles):
        titles_list = doc_titles.splitlines()
        # titles_tagged = self.part_of_speech_tagging(titles_list) #Part-of-speech tagging of text sentences
        word_of_title = []
        tokens = []
        punctuations = list(string.punctuation)
        punctuations.extend(["\'\'", "\"", "``", "--"])
        TreebankWordTokenizer.PUNCTUATION = [
            (re.compile(r'([:,])([^\d])'), r' \1 \2'),
            (re.compile(r'([:,])$'), r' \1 '),
            (re.compile(r'\.\.\.'), r' ... '),
            (re.compile(r'[;@#$%]'), r' \g<0> '),
            # Handles the final period.
            (re.compile(r'([^\.])(\.)([\]\)}>"\']*)\s*$'), r'\1 \2\3 '),
            (re.compile(r'[?!]'), r' \g<0> '),
            (re.compile(r"([^'])' "), r"\1 ' "),
        ]
        for i in range(len(titles_list)):
            _titleWords = [word.lower() for word in word_tokenize(
                titles_list[i]) if word not in punctuations]
            if len(_titleWords) > 0:
                word_of_title.append(_titleWords)
        for i in range(len(word_of_title)):
            tokens.append([word for word in word_of_title[i]
                           if word not in stopwords.words('english-new')])
        stemmer = PorterStemmer()
        preproc_tokens = []
        for j in range(len(tokens)):
            if len(tokens[j]) > 0:
                preproc_tokens.append([stemmer.stem(words)
                                       for words in tokens[j]])
        preproc_titles = []
        for i in range(len(preproc_tokens)):
            preproc_titles.append(' '.join(preproc_tokens[i]))
        return preproc_titles

    def part_of_speech_tagging(self, list_of_sentences):
        tagged_sentences = []
        nlp = spacy.load('en')
        for i in range(len(list_of_sentences)):
            tagged_sentences.append(list())
            doc = nlp(list_of_sentences[i])
            for token in doc:
                tagged_sentences[i].append((token.text, token.tag_))
        return tagged_sentences

    def find_tags_recursively(self, node, xml_part):
        for child in node:
            if child.tag not in xml_part:
                xml_part[child.tag] = child.text
            else:
                pass
            for subChild in child:
                if subChild.text is not None:
                    xml_part[child.tag] += '{} '.format(subChild.text.strip())
                else:
                    pass
                if child.tag == 'TEXT' and subChild.tag == 'F':
                    xml_part[child.tag] += subChild.tail
            self.find_tags_recursively(child, xml_part)
