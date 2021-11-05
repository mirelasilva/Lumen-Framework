from itertools import permutations


def get_permutation_set(principle_list, perm_num=1):
    '''
    Get the permutation set

    Input
    - principle_list: list of principle list
    - perm_num: int, number of chosen elements
                default perm_num=1
    Output
    - perm_set_list: all possible sorted combination of permutation
    '''
    perm_repeated = ['_'.join(sorted(i)) for i in permutations(principle_list, perm_num)]

    perm_set_list = list(set(perm_repeated))

    return perm_set_list


def get_principle_combination_num(principle_set_list, df_all_weapons):
    '''
    Given a set of principles, find the number of emails containing the set

    Input:
    - principle_set_list: list of principle set, linked by '_'
    - df_all_weapons: dataframe for emails * weapons

    Output
    - principles_num_dic: dictionary, key-principle set, value-number of emails containing the set
    '''
    principle_num_dic = {}
    principle_num_dic['principle_set'] = []
    principle_num_dic['email_count'] = []
    for tmp_principle_link in principle_set_list:
        tmp_principle_list = tmp_principle_link.split('_')
        tmp_principle_num = (df_all_weapons[tmp_principle_list].sum(axis=1) == len(tmp_principle_list)).sum()
        principle_num_dic['principle_set'] += [tmp_principle_link]
        principle_num_dic['email_count'] += [tmp_principle_num]

    return principle_num_dic
