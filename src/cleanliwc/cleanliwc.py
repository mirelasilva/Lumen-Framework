import os
import string
import json
from nltk.stem import PorterStemmer


def obtain_chosen_liwc_word(in_clean_liwc_dic={}):
    '''
    Get desired liwc word list

    Input
    - in_clean_liwc_dic
        - liwc_location
        - id_feature_dic
        - save_dic_flag
        - liwc_feature_word_dic_location
    Output
    - 1
    '''
    # obtain parameter
    liwc_location = in_clean_liwc_dic.get('liwc_location')
    id_feature_dic = in_clean_liwc_dic.get('id_feature_dic')
    save_dic_flag = in_clean_liwc_dic.get('save_dic_flag', 0)
    liwc_feature_word_dic_location = in_clean_liwc_dic.get('liwc_feature_word_dic_location')

    # obtain default parameter

    if liwc_location is None:
        liwc_folder = os.path.abspath(os.path.join(os.pardir, 'data', 'liwc', 'original_liwc'))
        liwc_file = 'LIWC2015_English.dic'
        liwc_location = os.path.join(liwc_folder, liwc_file)

    if id_feature_dic is None:
        id_feature_dic = {
            '31': 'posemo',
            '32': 'negemo',
            '33': 'anx',
            '34': 'anger',
            '35': 'sad',
            '84': 'reward',
            '85': 'risk',
            '103': 'time',
            '113': 'money',
        }

    # main
    with open(liwc_location, 'r') as f:
        tmp_all_lines = f.readlines()

    ps = PorterStemmer()
    id_list = list(id_feature_dic.keys())

    feature_word_dic = {}
    for tmp_id in id_list:
        tmp_feature = id_feature_dic[tmp_id]
        feature_word_dic[tmp_feature] = []

    for i in tmp_all_lines[88:]:

        i = i.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))

        tmp_str_list = i.strip().split('\t')

        for tmp_id in id_list:
            if tmp_id in tmp_str_list:
                tmp_word = tmp_str_list[0].replace(' ', '')
                tmp_word_stem = ps.stem(tmp_word)
                tmp_feature = id_feature_dic[tmp_id]
                if tmp_word_stem not in feature_word_dic[tmp_feature]:
                    feature_word_dic[tmp_feature] += [tmp_word_stem]

    # save dictionary
    if save_dic_flag:
        if liwc_feature_word_dic_location is None:
            liwc_feature_word_dic_folder = os.path.abspath(os.path.join(os.pardir, 'data', 'liwc', 'processed_liwc'))
            liwc_feature_word_dic_file = 'liwc_feature_word_dic.json'
            liwc_feature_word_dic_location = os.path.join(liwc_feature_word_dic_folder, liwc_feature_word_dic_file)

        print(liwc_feature_word_dic_location)
        with open(liwc_feature_word_dic_location, 'w') as fp:
            json.dump(feature_word_dic, fp)

    return 1
