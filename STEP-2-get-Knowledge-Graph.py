#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import json
import numpy as np
import pandas as pd
import copy


def read_json(path: str) -> dict:
    """
    Read JSON data
    :param path: The path of the JSON file
    :return json_data: the JSON data as dictionary type
    """
    with open(path, 'r', encoding='utf-8')as fp:
        json_data = json.load(fp)
    return json_data


def read_csv(path_txt: str) -> np.array:
    """
    Read the CSV file and convert it into numpy array
    :param path_txt: The path of the CSV file
    :return csv: The CSV info as numpy array
    """
    csv = pd.read_csv(path_txt, encoding='utf-8', keep_default_na=False)
    csv = np.array(csv)
    return csv


def prefer_names(json: dict) -> list:
    """
    Get the preferred names, synonym names, and synonym numbers
    :param json: input JSON dictionary (Knowledge Graph)
    :return pre_names: preferred names list
    :return pointer: [synonym name -> preferred name] list
    :return synonym_length: synonym lengths
    """
    concept = json.get("concept")
    pre_names = []
    pointer = []
    length = []
    for i in concept.keys():
        ID = concept.get(i)
        pre_name = ID.get("preferred_name")
        pre_names.append(pre_name)
        pointer.append(pre_name)

        synonym_names = ID.get("synonym")
        if len(synonym_names) == 1 and len(synonym_names[0]) == 0:
            length.append(len(synonym_names))
        else:
            length.append(len(synonym_names) + 1)

        if len(synonym_names) >= 1 and len(synonym_names[0]) != 0:
            for j in synonym_names:
                pre_names.append(j)
                pointer.append(pre_name)
                length.append(len(synonym_names) + 1)
    return pre_names, pointer, length


def Q2B(uchar):
    """
    Convert the full-width to half-width
    """
    inside_code = ord(uchar)
    if inside_code == 0x3000:
        inside_code = 0x0020
    else:
        inside_code -= 0xfee0
    if inside_code < 0x0020 or inside_code > 0x7e:
        return uchar
    return chr(inside_code)


def stringQ2B(ustring):
    """
    Convert full-width to half-width for a sentence or string
    """
    return "".join([Q2B(uchar) for uchar in ustring])


def remove_punctuation(term: list) -> list:
    """
    Remove the punctuations from the string
    :param term: the input term
    :return term: the term removed punctuations
    """
    punctuation = ['(', ')', '[', ']', '，', '。', '！', ',', '.', '!', '_', ' ', '\n', '-', '/', '',
                   '?', '？', ':', '：', '{', '}', '「', '」', '@', '#', '$', '%', '^', '&', '*', '[]',
                   '+', '=', '"', '~', '`', '|', '<', '>', '……', '￥', '〔', '〕', '“', '”', '—', '\\',
                   "'", ';', '、', '↓', '≤', '≥', '①', '②', '─', '【', '】', '°', '·', '…', '﹒', '､',
                   '『', '』', '｜']

    for i in range(len(term)):
        for j in punctuation:
            if j in term[i]:
                term[i] = term[i].replace(j, '')
    return term


def find_English_term(term: list) -> tuple:
    """
    Find and remove English and numbers from the term
    :param term: the input term
    :return term: the term removed English and numbers
    """
    Eng_in_term = []
    for i in range(len(term)):
        string = term[i]
        result = re.findall(r'[a-zA-Z0-9]+', string)

        # Find the term and the English containing in the term
        Eng_in_term.append(result)

        # Remove the English from the term
        for j in result:
            term[i] = re.sub(pattern=j, repl='', string=term[i])
    return term, Eng_in_term


def kmp(m_str, s_str) -> int:
    """
    The string matching algorithms
    :param m_str: main string
    :param s_str: pattern string
    :return: matching location or -1 if there is no matching
    """
    next_ls = [-1] * len(s_str)
    m = 1
    s = 0

    next_ls[0] = -1
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


def get_subword(pre_word: list, subword: list) -> list:
    pre_subs = []
    for word in pre_word:
        temp_sub = []
        if len(word) != 0:
            for sin_sub in subword:
                loc = kmp(m_str=word, s_str=sin_sub)
                if loc != -1 and len(sin_sub) != 0:
                    temp_sub.append(sin_sub)
                    word = word.replace(sin_sub, '')
        else:
            temp_sub.append('')

        if len(word) != 0:
            other = word.split()
            temp_sub.append(other)
        print(temp_sub)
        pre_subs.append(temp_sub)
    return pre_subs


def eng_with_sub(Eng: list, subwords: list) -> list:
    Eng_with_sub = []
    for i in range(len(subwords)):
        if len(Eng[i]) != 0:
            sub = ' '.join(subwords[i])
            temp = Eng[i][0].lower() + ' ' + sub
        else:
            if [] in subwords[i]:
                subwords[i].remove([])

            for j in range(len(subwords[i])):
                if type(subwords[i][j]) is list:
                    subwords[i][j] = subwords[i][j][0]
            temp = ' '.join(subwords[i])

        Eng_with_sub.append(temp)
    return Eng_with_sub


def save_csv(pointer: list, pre_names: list):
    # pointer = np.expand_dims(pointer, axis=1)  # Preferred names
    # synonym_length = np.expand_dims(synonym_length, axis=1)  # Number of synonyms
    # pre_names = np.expand_dims(pre_names, axis=1)  # preferred names and synonym names
    # Eng_with_sub = np.expand_dims(Eng_with_sub, axis=1)  # English + subwords w.r.t. preferred names
    #
    # preferred_words = np.concatenate([pointer, synonym_length, pre_names, Eng_with_sub], axis=1)
    # preferred_words = pd.DataFrame(preferred_words)
    # preferred_words.to_csv('pre_words_dict-200.csv', sep=',', index=False, header=None)
    pointer = np.expand_dims(pointer, axis=1)  # Preferred names
    # synonym_length = np.expand_dims(synonym_length, axis=1)  # Number of synonyms
    pre_names = np.expand_dims(pre_names, axis=1)  # preferred names and synonym names
    # Eng_with_sub = np.expand_dims(Eng_with_sub, axis=1)  # English + subwords w.r.t. preferred names

    # preferred_words = np.concatenate([pointer, synonym_length, pre_names, Eng_with_sub], axis=1)
    preferred_words = np.concatenate([pointer, pre_names], axis=1)
    preferred_words = pd.DataFrame(preferred_words)
    preferred_words.to_csv('pre_words_dict-200.csv', sep=',', index=False, header=None)


# def subword_and_synonym(subword: list, synonym: list):
#     subword = np.expand_dims(subword, axis=1)
#     synonym = np.expand_dims(synonym, axis=1)
#     save = np.concatenate([subword, synonym], axis=0)
#
#     rows = np.shape(save)[0]
#     data = []
#     for i in range(rows):
#         data.append([len(save[i][0]), save[i][0]])
#
#     sort_data = [value for index, value in sorted(enumerate(data), key=lambda data: data[1], reverse=True)]
#     sort_data = np.array(sort_data)[:, 1]
#     save_data = pd.DataFrame(sort_data)
#     save_data.to_csv('New-subword-list-200.csv', sep=',', index=False, header=None)


# The main function
if __name__ == '__main__':
    # Read the Knowledge Graph
    data_path = 'data/'
    json_data = read_json(data_path + 'onto_resource_0.0.3.json')

    # Get the subword list
    sub_and_freq = read_csv('subwords_freq_200.csv')
    sub_list = sub_and_freq[1:, 1]

    # Get the preferred names
    ori_pre, pointer, synonym_length = prefer_names(json=json_data)
    pre_names = copy.deepcopy(ori_pre)

    # Remove the English and numbers from the preferred names
    # pre_names, Eng_in_term = find_English_term(term=pre_names)

    # Remove the punctuations from the preferred names
    # pre_names = remove_punctuation(term=pre_names)

    # Get the subwords of the preferred names
    # pre_subs = get_subword(pre_word=pre_names, subword=sub_list)

    # Combine the English (lower case) with the subwords
    # Eng_with_sub = eng_with_sub(Eng=Eng_in_term, subwords=pre_subs)

    # Save the preferred names and subwords
    for i in range(len(ori_pre)):
        ori_pre[i] = ori_pre[i].replace(ori_pre[i], ori_pre[i].lower())
    temp_pre_names = remove_punctuation(term=ori_pre)

    # temp_pre_names, _ = find_English_term(term=temp_pre_names)

    save_csv(pointer=pointer, pre_names=temp_pre_names)

    # subword_and_synonym(subword=sub_list, synonym=temp_pre_names)
