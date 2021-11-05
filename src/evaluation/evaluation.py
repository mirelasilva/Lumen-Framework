import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from collections import defaultdict
import pandas as pd


def get_label_via_training(result_ptd, y_train_input):

    result_class = np.zeros(result_ptd.shape)

    true_pencent_ary = y_train_input.sum(axis=0) / len(y_train_input)

    for tmp_class_id in range(result_ptd.shape[1]):

        tmp_list = list(result_ptd[:, tmp_class_id])

        tmp_list.sort()

        tmp_true_test_num = int(true_pencent_ary[tmp_class_id] * len(tmp_list))

        tmp_cutoff = tmp_list[-tmp_true_test_num]

        result_class[:, tmp_class_id] = (result_ptd[:, tmp_class_id] >= tmp_cutoff) + 0

    return result_class


def doc_class_evaluation_fscore(result_test, y_test):
    f1_score_macro = f1_score(y_test, result_test, average="macro")
    f1_score_micro = f1_score(y_test, result_test, average="micro")
    acc_score = accuracy_score(y_test.ravel(), result_test.ravel())

    out_dic = {}
    out_dic['f1_score_macro'] = f1_score_macro
    out_dic['f1_score_micro'] = f1_score_micro
    out_dic['acc_score'] = acc_score

    return out_dic


def baseline_doc_class_evaluation_fscore(y_test):

    out_dic = defaultdict(list)

    # baseline 1: all email belong to the same class
    random_result = np.zeros(y_test.shape)
    random_result[:, 2] = 1

    f1_score_macro = f1_score(y_test, random_result, average="macro")
    f1_score_micro = f1_score(y_test, random_result, average="micro")
    acc_score = accuracy_score(y_test.ravel(), random_result.ravel())

    out_dic['f1_score_macro'] += [f1_score_macro]
    out_dic['f1_score_micro'] += [f1_score_micro]
    out_dic['acc_score'] += [acc_score]
    out_dic['baseline case'] += ['all email belong to the same class']

    # baseline 2: randomly generate label
    random_result = np.random.randint(2, size=y_test.shape)

    f1_score_macro = f1_score(y_test, random_result, average="macro")
    f1_score_micro = f1_score(y_test, random_result, average="micro")
    acc_score = accuracy_score(y_test.ravel(), random_result.ravel())

    out_dic['f1_score_macro'] += [f1_score_macro]
    out_dic['f1_score_micro'] += [f1_score_micro]
    out_dic['acc_score'] += [acc_score]
    out_dic['baseline case'] += ['randomly generate label']

    # baseline 4: all 1, then randomly choose one column and set as 0
    random_result = np.ones(y_test.shape)
    random_result[:, 2] = 0

    f1_score_macro = f1_score(y_test, random_result, average="macro")
    f1_score_micro = f1_score(y_test, random_result, average="micro")
    acc_score = accuracy_score(y_test.ravel(), random_result.ravel())

    out_dic['f1_score_macro'] += [f1_score_macro]
    out_dic['f1_score_micro'] += [f1_score_micro]
    out_dic['acc_score'] += [acc_score]
    out_dic['baseline case'] += ['all 1, then randomly choose one column and set as 0']

    # baseline 4: set all as 1
    random_result = np.ones(y_test.shape)

    f1_score_macro = f1_score(y_test, random_result, average="macro")
    f1_score_micro = f1_score(y_test, random_result, average="micro")
    acc_score = accuracy_score(y_test.ravel(), random_result.ravel())

    out_dic['f1_score_macro'] += [f1_score_macro]
    out_dic['f1_score_micro'] += [f1_score_micro]
    out_dic['acc_score'] += [acc_score]
    out_dic['baseline case'] += ['set all as 1']

    out_df = pd.DataFrame(out_dic)

    return out_df
