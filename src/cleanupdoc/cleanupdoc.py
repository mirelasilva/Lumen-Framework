import numpy as np
from nltk.tokenize import word_tokenize
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pandas as pd
import os


def get_raw_weapon_matrix(in_weapon_dic):
    '''
    Transform weapon code from decimal to binary, and save in a array.

    Input
    - in_weapon_dic
        - raw_weapon_series: a Series of binary weapon list
        - total_weapon_num: total number of weapons
                            default total_weapon_num=13
    Output
    - out_raw_weapon_dic:
        - bi_weapon_array: number of doc * number of weapons
        - weapon_name_list: weapon name
    '''

    # get input dic
    raw_weapon_series = in_weapon_dic.get('raw_weapon_series')
    total_weapon_num = in_weapon_dic.get('total_weapon_num', 13)

    # main function
    bi_weapon_array = np.zeros([len(raw_weapon_series), total_weapon_num])

    for i, tmp_c1_weapon in enumerate(raw_weapon_series):
        tmp_bi_weapons_rev = bin(tmp_c1_weapon)[2:].zfill(total_weapon_num)
        tmp_bi_weapons = tmp_bi_weapons_rev[::-1]

        for j in range(total_weapon_num):
            bi_weapon_array[i][j] = int(tmp_bi_weapons[j])

    # original weapon name list
    weapon_name_list = ['authority',
                        'contrast',
                        'commitment',
                        'curiosity',
                        'liking',
                        'reciprocation',
                        'scarcity',
                        'social',
                        'other',
                        'time',
                        'omission',
                        'reward',
                        'loss']

    # output dic
    out_raw_weapon_dic = {}
    out_raw_weapon_dic['bi_weapon_array'] = bi_weapon_array
    out_raw_weapon_dic['weapon_name_list'] = weapon_name_list

    return out_raw_weapon_dic


def cleanup_raw_doc(in_cleanup_dic):
    '''
     Tokenize and clean up the document list, prepare for downstream algorithms.
     Tasks: 1. remove stopword; 2. stemming

    Input:
    - in_cleanup_dic:
        - raw_email_text_series: series of str
        - stopword_flag: binary, denote if remove stopword,
                            default stopword_flag=1, remove stopword
        - eng_stopword: users can choose their own stopword list
                        default eng_stopword comes from NLTK package
        - print_eng_stopword_flag: if print stopword list for inspection
                                    default print_eng_stopword_flag=0, do not print
        - stemming_flag: binary, denote if conduct stemming,
                            default stemming_flag=1, stemming all words

    Output:
    - out_cleanup_doc_dic
        - email_list_nostop_stem
    '''

    # get input
    raw_email_text_series = in_cleanup_dic.get('raw_email_text_series')

    stopword_flag = in_cleanup_dic.get('stopword_flag', 1)
    eng_stopword = in_cleanup_dic.get('eng_stopword', None)
    print_eng_stopword_flag = in_cleanup_dic.get('print_eng_stopword_flag', 0)
    stemming_flag = in_cleanup_dic.get('stemming_flag', 1)

    # pre-prosessing
    # # stopword
    if eng_stopword is None:
        eng_stopword = stopwords.words('english')
    if stopword_flag == 0:
        eng_stopword = []

    # # if print stopword
    if print_eng_stopword_flag == 1:
        print(eng_stopword)

    # # stemming
    ps = PorterStemmer()

    # main
    email_list = []
    for i in range(len(raw_email_text_series)):
        tmp_doc = raw_email_text_series[i]
        tmp_doc_lower = tmp_doc.lower().replace('\\n', ' ').replace('\\xa0', ' ')
        tmp_doc_lower_nopun = tmp_doc_lower.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
        tmp_word_token_list = word_tokenize(tmp_doc_lower_nopun)
        if stemming_flag == 1:
            tmp_word_list = [ps.stem(j) for j in tmp_word_token_list if len(j) > 2 if j not in eng_stopword if j.isalpha()]
        if stemming_flag == 0:
            tmp_word_list = [j for j in tmp_word_token_list if len(j) > 2 if j not in eng_stopword if j.isalpha()]

        email_list += [' '.join(tmp_word_list)]

    # set output
    out_cleanup_doc_dic = {}
    out_cleanup_doc_dic['email_list'] = email_list

    return out_cleanup_doc_dic


def setup_all_clean_dataframe(in_all_clean_data_dic):
    '''
    Save useful data into a dataframe

    Input
    - in_all_clean_data_dic
        - save_df_flag: binary denote if save the dataframe into a csv file
                        default save_df_flag=0, do not save

        - bi_weapon_array: np array
        - weapon_name_list: all weapon names
        - email_list_stem: email list only stem
        - email_list_stem: email list remove stopword & stem
        - drop_columns_list: list of to-be-dropped persuation principles
                            default drop_columns_list - source code
        - save_df_csv_path: the location of the saved csv file
                            default drop_columns_list - source code
    Output
    - 1
    '''
    # check save_df_flag first
    save_df_flag = in_all_clean_data_dic.get('save_df_flag', 0)
    if save_df_flag == 0:
        print('save_df_flag == 0: have NOT saved data!!!')
        return 1

    # get input variables
    bi_weapon_array = in_all_clean_data_dic.get('bi_weapon_array')
    weapon_name_list = in_all_clean_data_dic.get('weapon_name_list')
    email_list_nostop_stem = in_all_clean_data_dic.get('email_list_nostop_stem')
    email_list = in_all_clean_data_dic.get('email_list')

    drop_columns_list = in_all_clean_data_dic.get('drop_columns_list', None)

    save_df_csv_path = in_all_clean_data_dic.get('save_df_csv_path', None)

    # get default values
    if drop_columns_list is None:
        drop_columns_list = ['contrast', 'curiosity', 'scarcity', 'other', 'time', 'omission', 'reciprocation']

    if save_df_csv_path is None:
        clean_data_dir = os.path.abspath(os.path.join(os.pardir, 'data', 'clean_data'))
        clean_data_file = 'clean_label_doc_with_label_6p2.csv'
        clean_data_path = os.path.join(clean_data_dir, clean_data_file)

    # main

    df_all_clean_data = pd.DataFrame(bi_weapon_array, columns=weapon_name_list)  # create new data frame

    df_all_clean_data['scarcity_time'] = ((df_all_clean_data.scarcity + df_all_clean_data.time) / 2 > 0.1) + 0  # combine scarcity & time into scarcity

    df_all_clean_data = df_all_clean_data.drop(columns=drop_columns_list)  # delete un-necessary columns

    sum_column_list = ['authority', 'commitment', 'liking', 'social', 'reward', 'loss', 'scarcity_time']
    df_all_clean_data['all_label'] = df_all_clean_data[sum_column_list].sum(axis=1)  # create a new columns: number of persuasion in one doc

    df_all_clean_data['email_nostop_stem'] = email_list_nostop_stem  # add email-remove stop-stem

    df_all_clean_data['email'] = email_list  # add email simple clean

    # save final dataframe

    df_all_clean_data.to_csv(clean_data_path, index=0)

    return 1
