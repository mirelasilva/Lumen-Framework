from nltk.stem import PorterStemmer
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer


def obtain_sia_feature_from_email_updateDf(email_6p2_df):
    '''
    Aim: sentiment analysis for each email
    sia == Sentiment Intensity Analyzer
    '''

    # get sia for each email
    sia = SentimentIntensityAnalyzer()

    result_dict = {}
    result_dict['pos'] = []
    result_dict['compound'] = []
    result_dict['neu'] = []
    result_dict['neg'] = []

    for tmp_doc in email_6p2_df.email:
        tmp_dic_sentiment = sia.polarity_scores(tmp_doc)
        result_dict['pos'] += [tmp_dic_sentiment['pos']]
        result_dict['compound'] += [tmp_dic_sentiment['compound']]
        result_dict['neu'] += [tmp_dic_sentiment['neu']]
        result_dict['neg'] += [tmp_dic_sentiment['neg']]

    # update pandas.df
    out_email_6p2_df = email_6p2_df.copy()
    for i in result_dict.keys():
        out_email_6p2_df[i + '_sia'] = result_dict[i]

    return out_email_6p2_df


def obtain_liwc_feature_from_email_updateDf(email_6p2_df, liwc_feature_word_dic):
    '''
    Input:
    - email_6p2_df: dataframe for email content & weapon matrix
    - liwc_feature_word_dic: dictionary for liwc features; class id - feature word list

    Output:
    - liwc_feature_count_list_dict:
        keys: each liwc features
        values: list of number of liwc word in each email, len(list) = number of emails
    '''

    # obtain_liwc_feature_from_email
    ps = PorterStemmer()

    liwc_feature_list = list(liwc_feature_word_dic.keys())

    liwc_feature_count_list_dict = {}
    for i in liwc_feature_list:
        liwc_feature_count_list_dict[i] = []

    for tmp_string in email_6p2_df.email:

        tmp_liwc_feature_count_dict = {}
        for i in liwc_feature_list:
            tmp_liwc_feature_count_dict[i] = 0

        for tmp_word in tmp_string.split():
            tmp_word_stem = ps.stem(tmp_word)

            for tmp_liwc_feature in liwc_feature_list:
                if tmp_word_stem in liwc_feature_word_dic[tmp_liwc_feature]:
                    tmp_liwc_feature_count_dict[tmp_liwc_feature] += 1

        for i in liwc_feature_list:
            liwc_feature_count_list_dict[i] += [tmp_liwc_feature_count_dict[i]]

    # updateDf
    out_email_6p2_df = email_6p2_df.copy()

    out_email_6p2_df['email_doc_length'] = [len(tmp_string.split()) for tmp_string in out_email_6p2_df.email]

    for i in liwc_feature_list:
        out_email_6p2_df[i + '_liwc'] = liwc_feature_count_list_dict[i]

    out_email_6p2_df['liwc_all_count'] = pd.DataFrame(liwc_feature_count_list_dict).sum(axis=1).values

    return out_email_6p2_df, liwc_feature_count_list_dict
