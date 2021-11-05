import numpy as np


def get_ptd_from_lda(trained_lda, corpus_array):
    num_topics = trained_lda.get_topics().shape[0]
    ptd_array = np.zeros([len(corpus_array), num_topics])

    for tmp_i, tmp_doc in enumerate(corpus_array):
        tmp_topic_dis_list = trained_lda.get_document_topics(tmp_doc, minimum_probability=0)
        ptd_array[tmp_i, :] = np.array(tmp_topic_dis_list)[:, 1]

    return ptd_array
