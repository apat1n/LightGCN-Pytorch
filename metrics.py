import numpy as np


def user_hitrate(rank, ground_truth, k=20):
    """
    :param rank: shape [n_recommended_items]
    :param ground_truth: shape [n_relevant_items]
    :param k: number of top recommended items
    :return: single hitrate
    """
    return len(set(rank[:k]).intersection(set(ground_truth)))


def hitrate(rank, ground_truth, k=20):
    """
    :param rank: shape [n_users, n_recommended_items]
    :param ground_truth: shape [n_users, n_relevant_items]
    :param k: number of top recommended items
    :return: shape [n_users]
    """
    return np.array([
        user_hitrate(user_rank, user_ground_truth, k)
        for user_rank, user_ground_truth in zip(rank, ground_truth)
    ])


def user_precision(rank, ground_truth, k=20):
    """
    :param rank: shape [n_recommended_items]
    :param ground_truth: shape [n_relevant_items]
    :param k: number of top recommended items
    :return: single precision
    """
    return user_hitrate(rank, ground_truth, k) / len(rank[:k])


def precision(rank, ground_truth, k=20):
    """
    :param rank: shape [n_users, n_recommended_items]
    :param ground_truth: shape [n_users, n_relevant_items]
    :param k: number of top recommended items
    :return: shape [n_users]
    """
    return np.array([
        user_precision(user_rank, user_ground_truth, k)
        for user_rank, user_ground_truth in zip(rank, ground_truth)
    ])


def user_recall(rank, ground_truth, k=20):
    """
    :param rank: shape [n_recommended_items]
    :param ground_truth: shape [n_relevant_items]
    :param k: number of top recommended items
    :return: single recall
    """
    return user_hitrate(rank, ground_truth, k) / len(ground_truth)


def recall(rank, ground_truth, k=20):
    """
    :param rank: shape [n_users, n_recommended_items]
    :param ground_truth: shape [n_users, n_relevant_items]
    :param k: number of top recommended items
    :return: shape [n_users]
    """
    return np.array([
        user_recall(user_rank, user_ground_truth, k)
        for user_rank, user_ground_truth in zip(rank, ground_truth)
    ])


def user_ap(rank, ground_truth, k=20):
    """
    :param rank: shape [n_recommended_items]
    :param ground_truth: shape [n_relevant_items]
    :param k: number of top recommended items
    :return: single ap
    """
    return np.sum([
        user_precision(rank, ground_truth, idx + 1)
        for idx, item in enumerate(rank[:k]) if item in ground_truth
    ]) / len(rank[:k])


def ap(rank, ground_truth, k=20):
    """
    :param rank: shape [n_users, n_recommended_items]
    :param ground_truth: shape [n_users, n_relevant_items]
    :param k: number of top recommended items
    :return: shape [n_users]
    """
    return np.array([
        user_ap(user_rank, user_ground_truth, k)
        for user_rank, user_ground_truth in zip(rank, ground_truth)
    ])


def map(rank, ground_truth, k=20):
    """
    :param rank: shape [n_users, n_recommended_items]
    :param ground_truth: shape [n_users, n_relevant_items]
    :param k: number of top recommended items
    :return: single map
    """
    return np.mean([ap(rank, ground_truth, k)])


def user_ndcg(rank, ground_truth, k=20):
    """
    :param rank: shape [n_recommended_items]
    :param ground_truth: shape [n_relevant_items]
    :param k: number of top recommended items
    :return: single ndcg
    """
    dcg = 0
    idcg = 0
    for idx, item in enumerate(rank[:k]):
        dcg += 1.0 / np.log2(idx + 2) if item in ground_truth else 0.0
        idcg += 1.0 / np.log2(idx + 2)
    return dcg / idcg


def ndcg(rank, ground_truth, k=20):
    """
    :param rank: shape [n_users, n_recommended_items]
    :param ground_truth: shape [n_users, n_relevant_items]
    :param k: number of top recommended items
    :return: shape [n_users]
    """
    return np.array([
        user_ndcg(user_rank, user_ground_truth, k)
        for user_rank, user_ground_truth in zip(rank, ground_truth)
    ])


def user_mrr(rank, ground_truth, k=20):
    """
    :param rank: shape [n_recommended_items]
    :param ground_truth: shape [n_relevant_items]
    :param k: number of top recommended items
    :return: single mrr
    """
    for idx, item in enumerate(rank[:k]):
        if item in ground_truth:
            return 1 / (idx + 1)
    return 0


def mrr(rank, ground_truth, k=20):
    """
    :param rank: shape [n_users, n_recommended_items]
    :param ground_truth: shape [n_users, n_relevant_items]
    :param k: number of top recommended items
    :return: shape [n_users]
    """
    return np.array([
        user_mrr(user_rank, user_ground_truth, k)
        for user_rank, user_ground_truth in zip(rank, ground_truth)
    ])


metric_dict = {
    'Hitrate': hitrate,
    'Precision': precision,
    'Recall': recall,
    'MAP': map,
    'NDCG': ndcg,
    'MRR': mrr}
