import numpy as np


def get_rank_matrix(result_matix, label_matrix):
    query_num = result_matix.shape[0]

    ranks = np.zeros(query_num)
    ranks2 = np.zeros(query_num)
    for i in range(query_num):

        label = label_matrix[i].tolist()
        true_pos = label.index(1)
        true = result_matix[i, true_pos]
        l = result_matix[i, :].tolist()
        ids = list(range(len(l)))
        pairs = list(zip(ids, l))
        l.sort(reverse=True)
        pairs.sort(key=lambda x: x[1], reverse=True)
        rank1 = l.index(true) + 1
        pair_true = (true_pos, true)
        rank2 = pairs.index(pair_true) + 1
        ranks[i] = rank1
        ranks2[i] = rank2
    return ranks


def get_result_by_ranks(ranks, rec_k_list):
    result = np.zeros(len(rec_k_list) + 1)
    mean_rank = np.mean(ranks)
    result[0] = mean_rank
    for index, k in enumerate(rec_k_list):
        result[index + 1] = sum(ranks < k + 1) / float(len(ranks))
    return result








