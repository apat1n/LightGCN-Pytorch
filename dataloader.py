import torch
import numpy as np
import pandas as pd
import scipy.sparse as sp
from config import config


class GowallaDataset:
    def __init__(self, train):
        print('init ' + ('train' if train else 'test') + ' dataset')

    def get_all_users(self):
        raise NotImplemented

    def get_user_positives(self, user):
        raise NotImplemented

    def get_user_negatives(self, user, k):
        raise NotImplemented

    @property
    def n_users(self):
        # TODO: parse user_list.txt
        return 107092

    @property
    def m_items(self):
        # TODO: parse item_list.txt
        return 1280969


class GowallaTopNDataset(GowallaDataset):
    def __init__(self, path, train=True):
        super().__init__(train)
        self.df = pd.read_csv(path, names=['userId', 'timestamp', 'long', 'lat', 'loc_id'])

        self.unique_users = self.df['userId'].unique()
        self.user_positive_items = self.df.groupby('userId')['loc_id'].apply(list).to_dict()

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
            candidate = np.random.randint(1, self.m_items)
            if candidate not in positives:
                neg.append(candidate)
        return neg


class GowallaLightGCNDataset(GowallaDataset):
    def __init__(self, path, train=True):
        super().__init__(train)
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
            candidate = np.random.randint(1, self.m_items)
            if candidate not in positives:
                neg.append(candidate)
        return neg

    def get_sparse_graph(self):
        """
        build a graph in torch.sparse.IntTensor.
        Details in NGCF's matrix form
        A =
            |0,   R|
            |R^T, 0|
        """
        return self.adj_matrix


class GowallaALSDataset(GowallaDataset):
    def __init__(self, path, train=True):
        super().__init__(train)
        self.path = path
        self.train = train
        self.df = pd.read_csv(path, names=['userId', 'timestamp', 'long', ' lat', 'loc_id'])

    def get_dataset(self):
        if self.train:
            users = self.df['userId'].values
            items = self.df['loc_id'].values
            ratings = np.ones(len(users))

            user_item_data = sp.csr_matrix((ratings, (users, items)),
                                           shape=(self.n_users, self.m_items))
            item_user_data = user_item_data.T.tocsr()
            return self.df, user_item_data, item_user_data
        else:
            return self.df
