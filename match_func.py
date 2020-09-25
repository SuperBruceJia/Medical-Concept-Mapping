#!/usr/bin/env python
# -*- coding: utf-8 -*-

import heapq
from subword_embedding import Subword_Embedding
from utility import *


class Match:
    def __init__(self, input_str, knowledge, standard_terms, standard_synonym, sub_list, pre_trained, synonym_vec, synonym_term):
        """
        :param input_str: Input String
        :param knowledge: Knowledge Graph
        :param standard_terms: standard terms (Standard Terms)
        :param standard_synonym: standard terms and their Synonyms
        :param sub_list: Sub-words Frequency List
        :param pre_trained: standard-trained word vectors
        :param synonym_vec: Word vectors of (standard terms and their Synonyms)
        :param synonym_term: standard terms and their Synonyms **that can be restandardsented by a vector**
        """
        self.input_str = input_str
        self.knowledge = knowledge
        self.standard_terms = standard_terms
        self.standard_synonym = standard_synonym
        self.sub_list = sub_list
        self.pre_trained = pre_trained
        self.synonym_vec = synonym_vec
        self.synonym_term = synonym_term
        self.subword_embed_calss = Subword_Embedding(sub_list=self.sub_list, pre_trained=self.pre_trained, standard_synonym=self.standard_synonym)

    def eng_with_sub(self, eng: list, subword: list) -> list:
        """
        Combine English terms with sub-words
        """
        subwords = subword + eng[0]
        while [] in subwords:
            subwords.remove([])
        out = " ".join('%s' % id for id in subwords).split()
        return out

    def non_match_word(self, matched_loc: list):
        """
        Get non-matched (False) sub-words
        """
        sort_com = [value for index, value in sorted(enumerate(matched_loc), key=lambda matched_loc: matched_loc[1])]
        subs = []
        end = 0
        for i in sort_com:
            start = i[0]
            false_word = self.input_str[end:start]
            end = i[1]
            subs.append(false_word)
        subs.append(self.input_str[end:])
        while '' in subs:
            subs.remove('')
        return subs

    def match_score(self, main_str: list, pattern_str: list):
        """
        Use string mapping algorithm (KMP algorithm) to obtain the matched scores
        :param main_str: the sub-word of the target word, e.g., '我爱中国' -> '我 爱 中国'
        :param pattern_str: the sub-words of the sub-words list
        :return total_score: the total scores list
        """
        total_score = []
        for mains in main_str:
            score = 0
            # One target sub-word sentence
            for pattern in pattern_str:
                # Use string mapping algorithm to get the scores
                loc = kmp(m_str=mains, s_str=pattern)
                if loc != -1:
                    score += 1
            total_score.append(score)
        return total_score

    def find_max_score(self, scores: list) -> list:
        """
        Find the maximum score from the total scores
        :param scores: total scores
        :return index: maximum score index
        """
        index = []
        max_score = max(scores)
        for i in range(len(scores)):
            if scores[i] == max_score:
                index.append(i)
        return index

    def top_k_result(self, score, k=1):
        """
        Get the top K scores
        """
        self.candidates = []
        self.candidate_sub = []
        self.top_k = heapq.nlargest(n=k, iterable=score)

        # Iterate the Top k scores, print each score and standard term
        for top_i in self.top_k:
            max_index = score.index(top_i)

            # Find the synonym term (might include standard terms)
            match_standard = self.synonym_term[max_index]

            # Find the standard term
            p_index = self.standard_synonym.tolist().index(match_standard)
            synonym_term = self.knowledge[p_index, 1]
            standard_term = self.knowledge[p_index, 0]

            # These lines of codes are mainly for Sub-words frequency
            temp_pre_name = remove_punctuation(term=synonym_term)
            temp_pre_name, _ = find_English_term(term=temp_pre_name)
            synonym_term_sub = self.subword_embed_calss.get_subword(term=temp_pre_name, is_print=False)
            synonym_term_sub = ' '.join(synonym_term_sub)
            self.candidates.append(standard_term)
            self.candidate_sub.append(synonym_term_sub)

            print('Top 10 Mapping ::: ', self.input_str, '----->', synonym_term, '----->', standard_term, ' (Similarity: ', top_i, ')')
        return standard_term

    def subword_frequency(self, input_sub):
        # Find all the *sub-words frequency scores*
        total_score = self.match_score(main_str=self.candidate_sub, pattern_str=input_sub)
        print('-' * 100)
        print('Total matched frequency: ', total_score)

        # Get the maximum frequency scores
        maxscore = self.find_max_score(scores=total_score)
        print('Max matched Frequency: ', maxscore)

        # Output the Results of *maximum frequency scores*
        final_map = []
        final_score = []
        for k in maxscore:
            final_map.append(self.candidates[k])
            final_score.append(self.top_k[k])
        final_map = list(set(final_map))
        print('Maximum Sub-words\' Frequency Mapping Results: ', final_map)

        # Output the Results of *max similarity* in the *maximum frequency score(s)*
        top_map = max(final_score)
        top_index = self.top_k.index(top_map)
        print('Model Final Mapping ::: ', self.input_str, '----->', self.candidates[top_index])
        final_output = self.candidates[top_index]
        return final_output

    def find_standard_term(self, score: list, is_final=False):
        """
        Find the standard term
        if is_final is True:
            find the top 5 maximum scores standard terms with regard to **
        else:
            find the maximum score standard term with regard to *Non-matched sub-word Embedding*
        """
        # If it's the Final Mapping
        if is_final is True:
            standard_term = self.top_k_result(score=score, k=5)
            input_sub = self.subword_embed_calss.get_subword(term=self.input_str, is_print=False)
            input_sub = ' '.join(input_sub).split()
            final_output = self.subword_frequency(input_sub=input_sub)
            return final_output

        # If it's the non-matched sub-words Mapping
        else:
            standard_term = self.top_k_result(score=score, k=1)
            return standard_term

    def subword_mapping(self, non_match: list) -> tuple:
        """
        Calculate Cosine Similarity and find the most similar [term <-> standard term] (the shortest cosine distance)
        """
        out_standard = []
        # out_standard_vec = []
        for i in non_match:
            # Each non-matched word vector
            vec = self.subword_embed_calss.get_embedding(term=i)

            # All the vectors of the Knowledge base
            if vec is not None:
                # Calculate Cosine Distance
                score = cos_similarity(vec1=vec, vec2=self.synonym_vec)

                # [Sub-words] standard term Mapping
                standard_term = self.find_standard_term(score=score, is_final=False)

                out_standard.append(standard_term)
            else:
                print('There was no word vector for ', i)
        return out_standard

    def final_mapping(self, all_standard: list) -> str:
        """
        Final mapping
        """
        all_vec = []
        for i in all_standard:
            # Each non-matched word vector
            temp_vec = self.subword_embed_calss.get_embedding(term=i)
            if temp_vec is not None and temp_vec not in all_vec:
                all_vec.append(temp_vec)

        # Get the final output Embedding
        vec = np.mean(all_vec, axis=0)

        # Calculate Cosine Distance
        score = cos_similarity(vec1=vec, vec2=self.synonym_vec)

        # [Final] standard term Mapping
        final_output = self.find_standard_term(score=score, is_final=True)
        return final_output
