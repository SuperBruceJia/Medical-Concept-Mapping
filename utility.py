#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import jieba
import numpy as np
import pandas as pd
import re
import codecs


def read_csv(path_txt: str) -> np.array:
    """
    Read the CSV file and convert it into numpy array
    :param path_txt: The path of the CSV file
    :return csv: CSV info as numpy array
    """
    csv = pd.read_csv(path_txt, encoding='utf-8', keep_default_na=False, header=None)
    csv = np.array(csv)
    return csv


def cos_similarity(vec1: list, vec2: list) -> list:
    """
    Calculate Cosine Similarity between a sample and X samples
    :param vec1: One-dimensional vector [1 x Y]
                 1: 1 sample
                 Y: Y dimensional vector
    :param vec2: Two-dimensional Matrix [X x Y]
                 X: X different samples
                 Y: Y dimensional vector
    :return score: Scores of similarity w.r.t. X samples
    """
    shape = np.shape(vec2)[0]
    vec1 = np.tile(vec1, [shape, 1])
    dot = np.sum(vec1 * vec2, axis=1)
    normal = (np.linalg.norm(vec1, axis=1) * np.linalg.norm(vec2, axis=1))
    score = (dot / normal).tolist()
    return score


def load_word_vector(path: str, word_dim: int) -> dict:
    """
    Load standard-trained word vectors
    """
    print("Start to load standard-trained word embeddings!!")
    wordvec = {}
    for i, line in enumerate(codecs.open(path + "word_vectors.vec", 'r', encoding='utf-8')):
        line = line.rstrip().split()
        if len(line) == word_dim + 1:
            wordvec[line[0]] = np.array([float(x) for x in line[1:]]).astype(np.float32)
    return wordvec


def kmp(m_str: str, s_str: str) -> int:
    """
    KMP Algorithm (a String Matching Algorithm)
    Reference: https://blog.csdn.net/achen0511/article/details/105983444/
    :param m_str: Main string
    :param s_str: Pattern String
    :return: Location of the matched string, if didn't exist, return -1
    """
    next_ls = [-1] * len(s_str)
    m = 1
    s = 0

    while m < len(s_str) - 1:
        if s_str[m] == s_str[s] or s == -1:
            m += 1
            s += 1
            next_ls[m] = s
        else:
            s = next_ls[s]

    i = j = 0
    while i < len(m_str) and j < len(s_str):
        if m_str[i] == s_str[j] or j == -1:
            i += 1
            j += 1
        else:
            j = next_ls[j]

    if j == len(s_str):
        return i - j
    return -1


def remove_punctuation(term: str) -> str:
    """
    Remove the punctuations from the string
    :param term: the term list or a string
    :return term: the term whose punctuations are removed
    """
    punctuation = ['(', ')', '[', ']', '，', '。', '！', ',', '.', '!', '_', '\n', '-', '/', '',
                   '?', '？', ':', '：', '{', '}', '「', '」', '@', '#', '$', '%', '^', '&', '*', '[]',
                   '+', '=', '"', '~', '`', '|', '<', '>', '……', '￥', '〔', '〕', '“', '”', '—', '\\',
                   "'", ';', '、', '↓', '≤', '≥', '①', '②', '─', '【', '】', '°', '·', '…', '﹒', '､',
                   '『', '』', '｜']
    for j in punctuation:
        if j in term:
            term = term.replace(j, '')
    return term


def find_English_term(term: str) -> tuple:
    """
    Find and remove English from the term
    :param term: the term list or a string
    :return term: the term whose English item(s) are removed
    :return Eng_in_term: the removed English item(s)
    """
    Eng_in_term = []

    # Find all the English from the term
    result = re.findall(r'[a-zA-Z0-9]+', term)

    # Remove the English from the term
    for i in result:
        term = re.sub(pattern=i, repl='', string=term)

    # Lower case the English term
    for j in range(len(result)):
        result[j] = result[j].replace(result[j], result[j].lower())

    # Find the term and the English containing in the term
    Eng_in_term.append(result)
    return term, Eng_in_term