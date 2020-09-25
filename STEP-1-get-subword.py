#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import necessary packages
import re
import json
import collections
import numpy as np
import pandas as pd
from collections import Counter


def read_txt(path_txt: str) -> np.array:
    """
    Read the TXT file and convert it into numpy array
    :param path_txt: The path of the TXT file
    :return txt: The TXT info as numpy array
    """
    txt = pd.read_table(path_txt, encoding='utf-8', keep_default_na=False)
    txt = np.array(txt)
    return txt


def read_csv(path_txt: str) -> np.array:
    """
    Read the CSV file and convert it into numpy array
    :param path_txt: The path of the CSV file
    :return txt: The CSV info as numpy array
    """
    csv = pd.read_csv(path_txt, encoding='utf-8', keep_default_na=False)
    csv = np.array(csv)
    return csv


def read_xls(path_txt: str) -> np.array:
    """
    Read the XLS file and convert it into numpy array
    :param path_txt: The path of the XLS file
    :return txt: The XLS info as the numpy array
    """
    xls = pd.read_excel(path_txt, keep_default_na=False)
    xls = np.array(xls)
    return xls


def read_xls_sheet(path_txt: str, sheet_name: str) -> np.array:
    """
    Read the xls' sheet file and convert it into numpy array
    :param sheet_name: The sheet name of the xls file
    :param path_txt: The path of the xls file
    :return txt: The xls sheet info as numpy array
    """
    xls = pd.read_excel(path_txt, header=None, sheet_name=sheet_name, keep_default_na=False)
    xls = np.array(xls)
    return xls


def read_json(path: str) -> dict:
    """
    Read JSON data
    :param path: The path of the JSON file
    :return json_data: The JSON data as dictionary type
    """
    with open(path, 'r', encoding='utf-8')as fp:
        json_data = json.load(fp)
        # print('This is the JSON data in the file：', json_data)
        # print('Type of the read JSON data: ', type(json_data))
    return json_data


def KG_words(json: dict) -> list:
    """
    Get the preferred names and synonym names from the Knowledge Graph
    :param json: The JSON dictionary
    :return words: The words containing preferred names and synonym names
    """
    concept = json.get("concept")
    words = []
    for i in concept.keys():
        mapping = concept.get(i)
        preferred_name = mapping.get("preferred_name")
        synonym_name = mapping.get("synonym")

        words.append(preferred_name)
        if len(synonym_name) != 0 and len(synonym_name[0]) != 0:
            for j in synonym_name:
                words.append(j)
    return words


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
    Remove the punctuations from a term list
    :param term: The term list
    :return term: The term list that removed the punctuations
    """
    punctuation = ['(', ')', '[', ']', '，', '。', '！', ',', '.', '!', '_', ' ', '\n', '-', '/', '',
                   '?', '？', ':', '：', '{', '}', '「', '」', '@', '#', '$', '%', '^', '&', '*', '[]',
                   '+', '=', '"', '~', '`', '|', '<', '>', '……', '￥', '〔', '〕', '“', '”', '—', '\\',
                   "'", ';', '、', '↓', '≤', '≥', '①', '②', '─', '【', '】', '°', '·', '…', '﹒', '､',
                   '『', '』', '｜', '_']

    for i in range(len(term)):
        term[i] = stringQ2B(term[i])
        for j in punctuation:
            if j in term[i]:
                term[i] = term[i].replace(j, '')
    return term


def find_English_term(term: list) -> tuple:
    """
    Find the English and numbers from a term list
    and remove the English and numbers from the term
    :param term: the term list
    :return term: the term removed the English and numbers
    :return Eng_terms: the removed English
    """
    temp_terms = []
    Eng_terms = []
    for i in range(len(term)):
        string = term[i]
        result = re.findall(r'[a-zA-Z0-9]+', string)
        for j in result:
            temp_terms.append(j)
            term[i] = re.sub(pattern=j, repl='', string=term[i])
    temp_terms = set(temp_terms)
    for k in temp_terms:
        Eng_terms.append(k)
    return term, Eng_terms


def count_terms(terms: list) -> dict:
    """
    Count the number of terms
    :param terms: term list
    :return dict_term: The dictionary containing terms and their numbers
    """
    entity_dict = dict(Counter(terms))
    print('There are %s entities in total.\n' % entity_dict.__len__())
    # print({key: value for key, value in entity_dict.items()})
    return entity_dict


def make_vocab(dict_term: dict) -> dict:
    """
    Make vocabulary (Word -> Characters)
    For example, {'123': 10} ------> {'1 2 3': 10}
    :param dict_term: The dictionary containing words, and their numbers
    :return temp_dict: The dictionary containing Characters and their numbers
    """
    temp_dict = dict_term
    for dict_key in list(dict_term.keys()):
        temp_key = ''
        for i in range(len(dict_key)):
            temp_key += dict_key[i]
            if i < len(dict_key) - 1:
                temp_key += ' '
        temp_dict[temp_key] = temp_dict.pop(dict_key)
    return temp_dict


def get_pairs(vocab: dict) -> dict:
    """
    Get the pairs and their frequency of the vocabulary
    :param vocab: The input vocabulary
    :return pairs: Pairs and their frequency
    """
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i + 1]] += freq
    return pairs


def merge_pair(pair: tuple, voc_in: dict) -> dict:
    """
    Merge the most frequent pair
    :param pair: one character pair, e.g., the most frequent character pair
    :param voc_in: Input vocabulary
    :return voc_out: Output vocabulary after merge
    """
    voc_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in voc_in:
        w_out = p.sub(''.join(pair), word)
        voc_out[w_out] = voc_in[word]
    return voc_out


def get_merged_subwords(vocab: dict) -> list:
    """
    Get the merged subwords list
    :param vocab: The input vocabulary dictionary
    :return sub_list: Subwords list
    """
    sub_list = []
    for sub_key in vocab.keys():
        sub_key = sub_key.split()
        for i in sub_key:
            sub_list.append(i)
    return sub_list


def get_subword_frequency(vocab: dict) -> list:
    """
    Get the subwords frequency list
    :param vocab: The input vocabulary dictionary
    :return sub_freqs: Subwords frequency list
    """
    sub_freqs = []
    for freq in vocab.values():
        sub_freqs.append(int(freq))
    return sub_freqs


def save_csv(csv_path: str, subword: np.array, subword_freq: np.array):
    """
    Save the data into CSV file
    """
    # rows = np.shape(subword)[0]
    # data = []
    # for i in range(rows):
    #     data.append([int(subword_freq[i]), subword[i]])
    #
    # sort_data = [value for index, value in sorted(enumerate(data), key=lambda data: data[1], reverse=True)]
    # save_data = pd.DataFrame(sort_data)
    # save_data.to_csv(csv_path, sep=',', index=False, header=['Subword', 'Frequency'])
    rows = np.shape(subword)[0]
    data = []
    for i in range(rows):
        data.append([len(subword[i]), subword[i]])

    sort_data = [value for index, value in sorted(enumerate(data), key=lambda data: data[1], reverse=True)]
    save_data = pd.DataFrame(sort_data)
    save_data.to_csv(csv_path, sep=',', index=False, header=['subword', 'length'])


# The main function
if __name__ == '__main__':
    # Read the Dataset from different files
    data_path = 'data/'
    path_1 = data_path + 'ICD-O-3形态学编码.csv'
    path_2 = data_path + '北京市住院病案首页手术操作名称与代码标准v5.0.xls'
    path_3 = data_path + '北京市住院病案首页手术操作名称与代码标准V6.01版本.xls'
    path_4 = data_path + '北京版RC022-ICD-9手术编码.xls'
    path_5 = data_path + '广东省ICD-9-CM-3手术与操作代码(2016版).xls'
    path_6 = data_path + '广东省ICD-9-CM-3手术与操作代码(2017版).xls'
    path_7 = data_path + '手术操作编码ICD-9-CM-3(2017维护版).xls'
    path_8 = data_path + '上海2018年手术操作与代码标准库.xlsx'
    path_9 = data_path + '北京版手术操作名称v6.0.xlsx'
    path_10 = data_path + '四川省ICD手术编码.xlsx'
    path_11 = data_path + '国标1.0-手术操作分类代码国家临床版1.0.xlsx'
    path_12 = data_path + '国际疾病分类第十一次修订本（ICD-11）中文版.xlsx'
    path_13 = data_path + '山东省-临床版国际疾病分类ICD-9-CM-3(V6.01版).xlsx'
    path_14 = data_path + '山东省医疗机构手术操作分类代码及级别目录(2018).xlsx'
    path_15 = data_path + '手术操作分类与代码 全国2017版.xlsx'
    path_16 = data_path + '手术操作分类代码国家临床版1.1.xlsx'
    path_17 = data_path + '手术操作分类代码国家临床版2.0.xlsx'
    path_18 = data_path + 'OMAHA术语集_术语_完整版_20180720.txt'
    path_19 = data_path + '成人（造影、负荷、心超）.xls'

    txt_1 = read_csv(path_txt=path_1)
    txt_2 = read_xls(path_txt=path_2)
    txt_3 = read_xls(path_txt=path_3)
    txt_4 = read_xls(path_txt=path_4)
    txt_5 = read_xls(path_txt=path_5)
    txt_6 = read_xls(path_txt=path_6)
    txt_7 = read_xls(path_txt=path_7)
    txt_8 = read_xls(path_txt=path_8)
    txt_9 = read_xls(path_txt=path_9)
    txt_10 = read_xls(path_txt=path_10)
    txt_11 = read_xls(path_txt=path_11)
    txt_12 = read_xls(path_txt=path_12)
    txt_13 = read_xls(path_txt=path_13)
    txt_14 = read_xls(path_txt=path_14)
    txt_15 = read_xls(path_txt=path_15)
    txt_16 = read_xls(path_txt=path_16)

    txt_17 = read_xls_sheet(path_txt=path_17, sheet_name='2.0')
    txt_18 = read_xls_sheet(path_txt=path_17, sheet_name='信息中心（20170125）')
    txt_19 = read_xls_sheet(path_txt=path_17, sheet_name='补丁1')
    txt_20 = read_xls_sheet(path_txt=path_17, sheet_name='补丁2')
    txt_21 = read_xls_sheet(path_txt=path_17, sheet_name='停用及修订')

    txt_22 = read_txt(path_txt=path_18)
    txt_23 = read_xls(path_txt=path_19)
    json_data = read_json(data_path + 'onto_resource_0.0.3.json')

    term_1 = txt_1[:, 1]
    term_2 = txt_2[:, 1]
    term_3 = txt_3[:, 1]
    term_4 = txt_4[:, 1]
    term_5 = txt_5[:, 1]
    term_6 = txt_6[:, 1]
    term_7 = txt_7[:, 1]
    term_8 = txt_8[:, 1]
    term_9 = txt_9[:, 0]
    term_10 = txt_10[:, 2]
    term_11 = txt_11[:, 2]
    term_12 = txt_12[:, 1]
    term_13 = txt_13[:, 2]
    term_14 = txt_14[:, 2]
    term_15 = txt_15[:, 2]
    term_16 = txt_16[:, 1]
    term_17 = txt_17[:, 2]
    term_18 = txt_18[:, 2]
    term_19 = txt_19[:, 2]
    term_20 = txt_20[:, 2]
    term_21 = txt_21[:, 1]
    term_22 = txt_22[:, 4]
    term_23 = txt_23[:, 3]

    KGwords = KG_words(json=json_data)

    term = np.concatenate([term_1, term_2, term_3, term_4, term_5, term_6,
                           term_7, term_8, term_9, term_10, term_11, term_12,
                           term_13, term_14, term_15, term_16, term_17, term_18,
                           term_19, term_20, term_21, term_22, term_23, KGwords])

    ####################################################################################################################################
    # Remove English from the term
    term, Eng_term = find_English_term(term=term)

    ####################################################################################################################################
    # Remove the punctuations from the terms and see the number of terms
    term = remove_punctuation(term=term)
    print('There are %s terms in total, including those repeated.\n' % np.shape(term)[0])

    # Save all the (repeated) terms
    all_term = pd.DataFrame(term)
    all_term.to_csv('all_term_repeated.csv', sep=',', index=False)

    ####################################################################################################################################
    # Remove the repeated ones
    temp_term = []
    term = set(term)
    for i in term:
        temp_term.append(i)

    # Save all the (unrepeated) terms
    unrepeated = pd.DataFrame(temp_term)
    unrepeated.to_csv('all_term_unrepeated.csv', sep=',', index=False)

    # Get the number of every term
    # print('The number of (unrepeated) terms is ')
    dict_term = count_terms(terms=temp_term)

    ####################################################################################################################################
    # Make dictionary vocabulary (Word -> Characters)
    vocab = make_vocab(dict_term=dict_term)

    # Set the number of operations
    max_freq = 10000
    merged_vocab = vocab
    merge_index = 0

    pair_chars = []
    while max_freq > 500:
        # Get pairs for the vocabulary
        pairs = get_pairs(vocab=merged_vocab)

        # Get the maximum frequency character pair
        max_freq_pair = max(pairs, key=pairs.get)

        # Update current max frequency
        max_freq = pairs.get(max_freq_pair)

        merge_index += 1
        print(merge_index, max_freq_pair, pairs.get(max_freq_pair))
        pair_chars.append(max_freq_pair)

        # Merge the most frequent two characters (bigram)
        merged_vocab = merge_pair(pair=max_freq_pair, voc_in=merged_vocab)

    ####################################################################################################################################
    # Save the paired characters
    pair_chars = pd.DataFrame(pair_chars)
    pair_chars.to_csv('Combined_Characters_' + str(max_freq) + '.csv', sep=',', index=False, header=None)

    # Get the merged subwords list (with repeated subwords)
    merged_vocab_list = get_merged_subwords(vocab=merged_vocab)

    # Get unrepeated subwords and their numbers
    dict_subword = count_terms(terms=merged_vocab_list)

    # Get the merged subwords list (without repeated subwords)
    subwords = get_merged_subwords(vocab=dict_subword)

    # Get the subwords frequency w.r.t. each word
    subword_freqs = get_subword_frequency(vocab=dict_subword)

    # Save the subwords to CSV File
    csv_path = 'subwords_freq_' + str(max_freq) + '.csv'
    save_csv(csv_path=csv_path, subword=subwords, subword_freq=subword_freqs)
    print('The ', csv_path, ' has been saved successfully!')
