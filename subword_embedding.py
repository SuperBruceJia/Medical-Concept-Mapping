#!/usr/bin/env python
# -*- coding: utf-8 -*-

import jieba
from utility import *


class Subword_Embedding:
    """
    Get Subwords via FMM and BMM algorithm
    Get Word Embedding of sub-word
    """
    def __init__(self, sub_list, pre_trained, standard_synonym):
        self.sub_list = sub_list
        self.pre_trained = pre_trained
        self.standard_synonym = standard_synonym

    def FMM(self, term: str):
        """
        Forward Maximum Matching (FMM)
        Get Embeddings
        """
        start = 0
        while start != self.len_term:
            index = start + self.max_len
            if index > self.len_term:
                index = self.len_term
            for i in range(self.max_len):
                if (term[start:index] in self.sub_list) or (len(term[start:index]) == 1):
                    self.standard_subs.append(term[start:index])
                    start = index
                    break
                index += -1

    def BMM(self, term: str):
        """
        Backward Maximum Matching (BMM)
        Get Embeddings
        """
        start = self.len_term
        while start != 0:
            index = start - self.max_len
            if index < 0:
                index = 0
            for i in range(self.max_len):
                if (term[index:start] in self.sub_list) or (len(term[index:start]) == 1):
                    self.standard_subs.append(term[index:start])
                    start = index
                    break
                index += 1

    def get_subword(self, term: str, is_print: bool) -> list:
        """
        Get Sub-word(s) of the term through
            Forward Maximum Matching (FMM)
            Backward Maximum Matching (BMM)
        Reference: https://zhuanlan.zhihu.com/p/103392455
        :param term: The Input Term
        """
        self.standard_subs = []
        self.len_term = len(term)
        self.max_len = len(self.sub_list[0])

        if self.len_term != 0:
            self.FMM(term=term)
            self.BMM(term=term)

            # Remove repeated sub-words
            self.standard_subs = list(set(self.standard_subs))
        else:
            self.standard_subs.append('')

        if is_print is True:
            print('Sub-word(s) are ', self.standard_subs)
        return self.standard_subs

    def jieba_subword(self, term: str, negative: int) -> tuple:
        """
        Jieba tokenizatin Embedding and Subword Embedding
        """
        neg_word = ['非', '不', '无', '否', '假']
        jieba_token = jieba.lcut(term, HMM=True)
        subword_token = self.get_subword(term=term, is_print=False)
        tokens = jieba_token + subword_token

        for token in tokens:
            if token in neg_word:
                negative = 1
                term.replace(token, '')
                continue
            else:
                temp_vec = self.pre_trained.get(token)
                if temp_vec is not None and token != '':
                    temp_vec = temp_vec.tolist()
                    if temp_vec not in self.grams:
                        self.grams.append(temp_vec)
        return term, negative

    # def jieba_subword(self, term: str) -> tuple:
    #     """
    #     Jieba tokenizatin Embedding and Subword Embedding
    #     """
    #     jieba_token = jieba.lcut(term, HMM=True)
    #     subword_token = self.get_subword(term=term, is_print=False)
    #     tokens = jieba_token + subword_token
    #
    #     for token in tokens:
    #         temp_vec = self.pre_trained.get(token)
    #         if temp_vec is not None and token != '':
    #             temp_vec = temp_vec.tolist()
    #             if temp_vec not in self.grams:
    #                 self.grams.append(temp_vec)
    #     return term

    def n_gram(self, term: list):
        """
        Get N-gram Embeddings from the term
        """
        index = 0
        for i in range(len(term)):
            temp_grams = [term[index:t + 1] for t in range(len(term))]

            for temp_gram in temp_grams:
                temp_vec = self.pre_trained.get(temp_gram)
                if temp_vec is not None and temp_gram != '':
                    temp_vec = temp_vec.tolist()
                    if temp_vec not in self.grams:
                        self.grams.append(temp_vec)
            index += 1

    def get_embedding(self, term: str) -> np.array:
        """
        Get a word's embedding
        """
        vec = self.pre_trained.get(term)

        if vec is not None:
            return vec.tolist()
        else:
            # Some standard-defined rules
            self.grams = []
            negative = 0
            term, negative = self.jieba_subword(term=term, negative=negative)  # Jieba and sub-word Embedding
            # term = self.jieba_subword(term=term)  # Jieba and sub-word Embedding
            self.n_gram(term=term)  # N-gram Embedding

            # Return outvec
            if negative == 1:
                outvec = None if self.grams == [] else (np.mean(self.grams, axis=0) * -1).tolist()
            else:
                outvec = None if self.grams == [] else np.mean(self.grams, axis=0).tolist()
            # outvec = None if self.grams == [] else np.mean(self.grams, axis=0).tolist()
            return outvec

    def load_standard_vector(self) -> tuple:
        """
        Load word vector for term(s)]
        Notice: We use the sub-words to get the word Embedding
                instead of the original word!
        """
        output_vec = []
        output_term = []

        # Iterate the Standard and Synonym Terms
        for i in self.standard_synonym:
            temp_out = []

            # Remove Punctuations from the synonym term
            i = remove_punctuation(term=i)

            # Get sub-words from the synonym term
            subs = self.get_subword(term=i, is_print=False)  # Get Sub-words of this term

            # Iterate the terms
            for sub in subs:
                outvec = self.get_embedding(term=sub)

                if outvec is not None and outvec != []:
                    temp_out.append(outvec)

            if temp_out != []:
                temp_out = np.mean(temp_out, axis=0).tolist()
                output_vec.append(temp_out)
                output_term.append(i)
        return output_vec, output_term
