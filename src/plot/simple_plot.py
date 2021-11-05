import matplotlib.pyplot as plt
import os


def short_principle_xlabel_set(principle_set_str_list):
    '''
    Create a short version of xlabel, which is a pair or triplet of persuasion principles

    Input
    - principle_set_str_list: the original pair or triplet of persuasion principles

    Output
    - principle_set_shortstr_list: short version
    '''

    principle_set_shortstr_list = []

    for tmp_principle_set_str in principle_set_str_list:
        tmp_principle_set_list = tmp_principle_set_str.split('_')

        tmp_principle_set_list_short = [i[:4] for i in tmp_principle_set_list]

        tmp_principle_set_str_short = '\n'.join(tmp_principle_set_list_short)

        principle_set_shortstr_list += [tmp_principle_set_str_short]

    return principle_set_shortstr_list


def get_bartop_infor_list(count_list, float_percent_list):

    bartop_infor_list = []
    for i in range(len(float_percent_list)):
        tmp_count = count_list[i]
        tmp_perc = float_percent_list[i]

        tmp_str = str(tmp_count) + '\n' + str(int(tmp_perc * 100)) + '%'
        bartop_infor_list += [tmp_str]

    return bartop_infor_list


def plot_barplot(in_plot_para_dic):
    '''
    Plot a simple bar plot

    Input
    - in_plot_para_dic
        - x_list
        - y_list
        - x_label
        - y_label
        - figsize
        - save_fig_path
        - save_fig_name
        - save_fig_flag: if save the fig
                         default save_fig_flag=0, do not save the fig
    Output
    - 1
    '''

    # get input parameters
    x_list = in_plot_para_dic.get('x_list', range(5))
    y_list = in_plot_para_dic.get('y_list', range(5))

    bartop_infor_list = in_plot_para_dic.get('bartop_infor_list')

    y_limit = in_plot_para_dic.get('y_limit', None)

    x_ticks = in_plot_para_dic.get('x_ticks')
    x_ticklabels = in_plot_para_dic.get('x_ticklabels')
    x_ticklabels_angle = in_plot_para_dic.get('x_ticklabels_angle', 0)

    x_label = in_plot_para_dic.get('x_label', 'x_label')
    y_label = in_plot_para_dic.get('y_label', 'y_label')

    figsize = in_plot_para_dic.get('figsize', (8, 6))

    save_fig_path = in_plot_para_dic.get('save_fig_path', os.path.abspath(os.path.join(os.pardir, 'fig',)))
    save_fig_name = in_plot_para_dic.get('save_fig_name', 'tmp00_delete_test_fig.pdf')
    save_fig_flag = in_plot_para_dic.get('save_fig_flag', 0)

    # main
    fig, ax = plt.subplots(figsize=figsize)

    ax.bar(x_list, y_list)

    if bartop_infor_list is not None:
        for i, v in enumerate(bartop_infor_list):
            ax.text(x_list[i] - .3, y_list[i] + 10, str(v))

    if y_limit is not None:
        ax.set_ylim(y_limit)

    if x_ticks is not None:
        ax.set_xticks(x_ticks)
    if x_ticklabels is not None:
        ax.set_xticklabels(x_ticklabels, rotation=x_ticklabels_angle)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    if save_fig_flag:
        save_fig_location = os.path.join(save_fig_path, save_fig_name)
        fig.savefig(save_fig_location, bbox_inches='tight', transparent=1, dpi=1000)

    return 1
