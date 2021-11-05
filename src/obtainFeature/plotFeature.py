import pandas as pd
from scipy.stats import ttest_ind


def obtain_csv_doc_with_label_6p2_liwc_sia(save_email_6p2_location):

    raw_data_df = pd.read_csv(save_email_6p2_location)

    raw_data_df = raw_data_df.rename(columns={"reward": "gain", "scarcity_time": "scarcity"})

    # re-scale liwc features
    liwc_feature_list = ['anx_liwc', 'anger_liwc', 'sad_liwc', 'reward_liwc', 'risk_liwc', 'time_liwc', 'money_liwc']

    for i in liwc_feature_list:
        raw_data_df[i] = raw_data_df[i] / raw_data_df.email_doc_length * 100

    return raw_data_df


def get_p_value_single_vs_multiple(y, df_plot_all):
    array_a = df_plot_all[df_plot_all['Email type'] == 'Single'][y].values
    array_b = df_plot_all[df_plot_all['Email type'] == 'Multiple'][y].values

    pvalue_float = ttest_ind(array_a, array_b).pvalue

    pvalue_str = "{:.2e}".format(pvalue_float)

    return pvalue_float, pvalue_str
