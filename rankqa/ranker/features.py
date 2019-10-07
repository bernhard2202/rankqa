#!/usr/bin/env python3
""" Manually designed features as POS, NER, question types etc. """

import numpy as np


def get_jaccard_sim(str1, str2):
    a = set(str1)  # .split())
    b = set(str2)  # .split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


POS_DICT = {'NNP': 0, 'JJ': 1, 'NN': 2, 'IN': 3, ',': 4, 'CC': 5, 'DT': 6, 'VBG': 7, 'VB': 8, 'NNS': 9, 'POS': 10,
            'VBZ': 11, 'RB': 12, 'TO': 13, 'FW': 14, 'PRP$': 15, 'CD': 16, 'VBN': 17, 'NNPS': 18, 'JJR': 19, 'VBP': 20,
            ':': 21, 'VBD': 22, 'PRP': 23, '#': 24, 'JJS': 25, '$': 26, 'WRB': 27, '-LRB-': 28, '-RRB-': 29, '.': 30,
            '``': 31, "''": 32, 'PDT': 33, 'MD': 34, 'WP': 35, 'RP': 36, 'WDT': 37, 'EX': 38, 'UH': 39, 'SYM': 40,
            'LS': 41, 'RBS': 42, 'RBR': 43, 'WP$': 44}


def get_pos_features(sample):
    vec = np.zeros(len(POS_DICT))
    for pos in sample['span_pos']:
        vec[POS_DICT[pos]] = 1
    return vec


NER_DICT = {'location': 0, 'person': 1, 'organization': 2, 'money': 3, 'percent': 4, 'date': 5, 'time': 6, 'o': 7,
            'set': 8, 'duration': 9, 'number': 10, 'ordinal': 11, 'misc': 12}


def get_ner_features(sample):
    vec = np.zeros(len(NER_DICT))
    for ner in sample['span_ner']:
        vec[NER_DICT[ner.lower()]] = 1
    return vec


Q_TYPE = {'what was': 0, 'what is': 1, 'what': 2, 'in what': 3, 'in which': 4, 'in': 5,
          'when': 6, 'where': 7, 'who': 8, 'why': 9, 'which': 10, 'is': 11, 'other': 12}


def get_question_type_features(sample):
    vec = np.zeros(len(Q_TYPE))
    qwords = sample['question'].split(' ')
    other = True
    if qwords[0].lower() in Q_TYPE:
        vec[Q_TYPE[qwords[0].lower()]] = 1
        other = False
    if ' '.join(list(map(lambda x: x.lower(), qwords[0:2]))) in Q_TYPE:
        vec[Q_TYPE[' '.join(list(map(lambda x: x.lower(), qwords[0:2])))]] = 1
        other = False
    if other:
        vec[Q_TYPE['other']] = 1
    return vec
