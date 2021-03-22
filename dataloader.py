import scipy
import torch
import numpy as np
import pandas as pd
from config import config
import scipy.sparse as sp


class BasicDataset:
    def __init__(self):
        print("init dataset")

    @property
    def n_users(self) -> int:
        raise NotImplementedError

    @property
    def m_items(self) -> int:
        raise NotImplementedError

    def get_sparse_graph(self) -> scipy.sparse.csr_matrix:
        """
        build a graph in torch.sparse.IntTensor.
        Details in NGCF's matrix form
        A =
            |0,   R|
            |R^T, 0|
        """
        raise NotImplementedError

    def get_all_users(self):
        raise NotImplementedError

    def get_user_positives(self, user):
        raise NotImplementedError

    def get_user_negatives(self, user, k=10):
        raise NotImplementedError


class GowallaDataset(BasicDataset):
    def __init__(self, path, train=True):
        super().__init__()

        dataset = pd.read_csv(path, names=['userId', 'timestamp', 'long', 'lat', 'loc_id'])
        if train:
            # GroupBy item
            temp = dataset.groupby('loc_id').agg({'userId': 'count'}).reset_index()
            # Treshold item ratings
            items = temp.loc[temp.userId > config['NUM_RAT_FOR_ITEM']].loc_id
            dataset = dataset.loc[dataset.loc_id.isin(items)].copy()
            # GroupBy user
            temp = dataset.groupby('userId').agg({'loc_id': 'count'}).reset_index()
            # Threshold user ratings
            users = temp.loc[temp.loc_id > config['NUM_RAT_FOR_USER']].userId
            dataset = dataset.loc[dataset.userId.isin(users)].copy()

        dataset['feed'] = 1
        users = dataset['userId']
        self.unique_users = users.unique()
        items = dataset['loc_id']
        feed = dataset['feed']
        self.user_positive_items = dataset.groupby('userId')['loc_id'].apply(list).to_dict()
        del dataset

        # suppose user and item ids are begins from 1
        n_nodes = self.n_users + self.m_items

        # build scipy sparse matrix
        user_np = np.array(users.values, dtype=np.int32)
        item_np = np.array(items.values, dtype=np.int32)
        ratings = np.array(feed.values, dtype=np.int32)

        tmp_adj = sp.csr_matrix((ratings, (user_np, item_np + self.n_users)),
                                shape=(n_nodes, n_nodes))
        adj_mat = tmp_adj + tmp_adj.T

        # normalize matrix
        # TODO: not only rowsum, also colsum
        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)

        norm_adj_tmp = d_mat_inv.dot(adj_mat)
        normalized_adj_matrix = norm_adj_tmp.dot(d_mat_inv)

        # convert to torch sparse matrix
        adj_mat_coo = normalized_adj_matrix.tocoo()

        values = adj_mat_coo.data
        indices = np.vstack((adj_mat_coo.row, adj_mat_coo.col))

        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = adj_mat_coo.shape

        self.adj_matrix = torch.sparse_coo_tensor(i, v, torch.Size(shape))

    def get_all_users(self):
        return self.unique_users

    def get_user_positives(self, user):
        if user not in self.user_positive_items:
            return []
        return self.user_positive_items[user]

    def get_user_negatives(self, user, k=10):
        neg = []
        positives = set(self.get_user_positives(user))
        while len(neg) < k:
            candidate = np.random.randint(1, self.n_users)
            if candidate not in positives:
                neg.append(candidate)
        return neg

    @property
    def n_users(self):
        return 107092

    @property
    def m_items(self):
        return 1280969

    def get_sparse_graph(self):
        return self.adj_matrix
