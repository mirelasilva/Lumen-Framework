# Add path in the script
# get the current location of the script, not work in the jupyter notebook
import os
import sys
current_file_location = os.path.dirname(os.path.realpath(__file__))
src_dir = os.path.abspath(os.path.join(current_file_location, os.pardir, 'src'))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)


import numpy as np
from sklearn.model_selection import KFold
import gensim as gs
from gensim.models.ldamodel import LdaModel
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from evaluation.evaluation import doc_class_evaluation_fscore
from obtainLDA.obtainLDA import get_ptd_from_lda


def run_lda_liwc_sia_rf(doc_list, liwc_sia_array, bi_weapon_array, n_splits=5, shuffle=True, num_topics=50, n_estimators=200, random_state=2):

    kf = KFold(n_splits=n_splits, shuffle=shuffle)

    doc_array = np.array(doc_list)

    df_result_f1_score_save = pd.DataFrame()

    for train_index, test_index in kf.split(doc_array):
        # print("TRAIN:", len(train_index), "TEST:", len(test_index))

        liwc_sia_array_train = liwc_sia_array[train_index]
        liwc_sia_array_test = liwc_sia_array[test_index]

        doc_array_train = doc_array[train_index]
        doc_array_test = doc_array[test_index]

        dictionary = 0
        dictionary = gs.corpora.Dictionary(doc_array_train)
        print('len of dictionary.keys: ', len(dictionary.keys()))

        corpus_array_train = [dictionary.doc2bow(text) for text in doc_array_train]
        corpus_array_test = [dictionary.doc2bow(text) for text in doc_array_test]

        bi_weapon_array_train = bi_weapon_array[train_index]
        bi_weapon_array_test = bi_weapon_array[test_index]

        # train lda on train
        lda = 0
        lda = LdaModel(corpus_array_train, num_topics=num_topics, alpha='asymmetric', eval_every=3)

        # get ptd for train and test
        ptd_train = get_ptd_from_lda(lda, corpus_array_train)
        ptd_test = get_ptd_from_lda(lda, corpus_array_test)

        ptd_liwc_sia_train = np.concatenate((ptd_train, liwc_sia_array_train), axis=1)
        ptd_liwc_sia_test = np.concatenate((ptd_test, liwc_sia_array_test), axis=1)

        # train RF on train
        rfc = 0
        rfc = RandomForestClassifier(n_estimators=n_estimators, random_state=2)
        rfc.fit(ptd_liwc_sia_train, bi_weapon_array_train)

        # predict on test
        bi_weapon_array_predict_test = rfc.predict(ptd_liwc_sia_test)

        # evaluation
        tmp_f1_score_dic = doc_class_evaluation_fscore(bi_weapon_array_predict_test, bi_weapon_array_test)

        # save result
        df_result_f1_score_save = df_result_f1_score_save.append(tmp_f1_score_dic, ignore_index=True)

    return df_result_f1_score_save
