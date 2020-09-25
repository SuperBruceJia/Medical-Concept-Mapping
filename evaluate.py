#!/usr/bin/env python
# -*- coding: utf-8 -*-

from utility import *
from subword_embedding import Subword_Embedding
from match_func import Match


# The main function
if __name__ == '__main__':
    ########################--LOAD standard-DEFINED DICT--##################################################
    # Get the standard terms, synonym terms, and terms' sub-words
    file_threshold = '200'
    knowledge = read_csv(path_txt='pre_words_dict-' + file_threshold + '.csv')
    standard_terms, standard_synonym = knowledge[:, 0], knowledge[:, 1]
    print('There are', np.shape(standard_synonym)[0], 'standard and synonym terms!')

    # Get the sub-words list
    subword_list = read_csv(path_txt='subwords_freq_' + file_threshold + '.csv')
    subword_list = subword_list[:, 1]

    # Load standard-trained vectors and get word Embeddings of standard and synonym words
    pre_trained = load_word_vector(path='data/', word_dim=128)
    subword_embed = Subword_Embedding(sub_list=subword_list, pre_trained=pre_trained, standard_synonym=standard_synonym)
    synonym_vec, synonym_term = subword_embed.load_standard_vector()
    print('There are', np.shape(synonym_term)[0], 'standard and synonym terms that own word vectors!')

    print('Start to evaluate.....................')
    score = 0
    for k in range(np.shape(standard_synonym)[0]):
        input_str = standard_synonym[k]
        print('The input string is ', input_str)

        # Load Match Class
        match_class = Match(input_str=input_str,
                            knowledge=knowledge,
                            standard_terms=standard_terms,
                            standard_synonym=standard_synonym,
                            sub_list=subword_list,
                            pre_trained=pre_trained,
                            synonym_vec=synonym_vec,
                            synonym_term=synonym_term)

        #################--FIND STANDARD TERM--####################################################
        # Get all the mapping w.r.t. standard terms
        temp_str = remove_punctuation(term=input_str)
        temp_str = temp_str.replace(temp_str, temp_str.lower())  # Use lowercase if there is English

        # Find and remove English from term
        re_eng, eng_subword = find_English_term(term=temp_str)

        #####################--GET SUB-WORDS--#########################################################
        input_subword = subword_embed.get_subword(term=re_eng, is_print=False)

        # Combine the sub-words with the removed English term(s)
        subwords = match_class.eng_with_sub(eng=eng_subword, subword=input_subword)
        print('All the sub-words are', subwords)

        matched = []
        matched_loc = []
        for i in subwords:
            # This sub-word is in the standard terms
            if i in standard_terms and len(i) > 1:
                print(i, '----->', i)
                matched.append(i)

                try:
                    start_loc = temp_str.index(i)
                    end_loc = start_loc + len(i)
                    matched_loc.append([start_loc, end_loc])
                except ValueError:
                    print('{} not found in the search space.'.format(i))
                    continue

            # This sub-word is in the synonym terms
            elif i in standard_synonym and len(i) > 1:
                s_index = standard_synonym.tolist().index(i)
                print(i, '----->', knowledge[s_index, 0])
                matched.append(knowledge[s_index, 0])

                try:
                    start_loc = temp_str.index(i)
                    end_loc = start_loc + len(i)
                    matched_loc.append([start_loc, end_loc])
                except ValueError:
                    print('{} not found in the search space.'.format(i))
                    continue

            # other non-matched sub-word
            else:
                print(i, '----->', False)

        #################--FIND THE NON-MATCHED term--###################################################
        # Get the Non-matched sub-words
        non_match = match_class.non_match_word(matched_loc=matched_loc)
        input_jieba = jieba.lcut(re_eng, HMM=True)

        # If there was no non-matched sub-words
        if non_match == []:
            # One Matched Standard Term
            if len(matched) == 1:
                print('Final Mapping ::: ', input_str, '----->', matched[0])
                final_output = matched[0]
            # Multiple Matched Standard Terms
            else:
                out_standard = list(set(matched + subwords + input_jieba))
                final_output = match_class.final_mapping(all_standard=out_standard)
        # If there were non-matched sub-words
        else:
            print('The None-matched sub-words are ', non_match, '\n', '-' * 100)

            # Sub-words mapped to standard term
            # out_standard = match_class.subword_mapping(non_match=non_match)

            # out_standard: The Standard term mapped by the Non-matched sub-word
            # matched: Matched Standard Term of the sub-word
            # non_match: The Non-matched sub-word
            # subwords: The sub-words of the input string
            # input_str: The input string
            input_jieba = jieba.lcut(re_eng, HMM=True)
            out_standard = list(set(matched + non_match + subwords + input_jieba))
            print('All the sub-words\' mapped standard terms: ', out_standard)

            # [Final] standard term Mapping
            final_output = match_class.final_mapping(all_standard=out_standard)

        # Calculate Model Accuracy
        if final_output == standard_terms[k]:
            print('Yes')
            score += 1
        else:
            print('No')

    print('Cheers! ', score, 'terms got right!')
    data_num = np.shape(standard_synonym)[0]
    acc = score / data_num
    print('Model Accuracy is ', acc)
