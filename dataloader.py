import torch
import numpy as np
import pandas as pd
import scipy.sparse as sp
from torch.utils.data import Dataset


class GowallaDataset(Dataset):
    def __init__(self, train, path='dataset'):
        print('init ' + ('train' if train else 'test') + ' dataset')
        self.n_users_ = int(open(f'{path}/user_list.txt').readlines()[-1][:-1].split(' ')[1]) + 1
        self.m_items_ = int(open(f'{path}/item_list.txt').readlines()[-1][:-1].split(' ')[1]) + 1

    def get_all_users(self):
        raise NotImplemented

    def get_user_positives(self, user):
        raise NotImplemented

    def get_user_negatives(self, user, k):
        raise NotImplemented

    @property
    def n_users(self):
        return self.n_users_

    @property
    def m_items(self):
        return self.m_items_

    def __len__(self):
        return self.n_users_

    def __getitem__(self, idx):
        raise NotImplemented


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
    def __init__(self, path, train=True, n_negatives: int = 10):
        super().__init__(train)
        self.n_negatives = n_negatives

        # dataset = pd.read_csv(path, names=['userId', 'timestamp', 'long', 'lat', 'loc_id'])
        dataset = pd.read_csv(path, names=['userId', 'loc_id'])

        dataset['feed'] = 1
        users = dataset['userId']
        items = dataset['loc_id']
        feed = dataset['feed']
        self.unique_users = users.unique()
        self.user_positive_items = dataset.groupby('userId')['loc_id'].apply(list).to_dict()
        del dataset

        n_nodes = self.n_users + self.m_items

        # build scipy sparse matrix
        user_np = np.array(users.values, dtype=np.int32)
        item_np = np.array(items.values, dtype=np.int32)
        ratings = np.array(feed.values, dtype=np.int32)

        tmp_adj = sp.csr_matrix((ratings, (user_np, item_np + self.n_users)),
                                shape=(n_nodes, n_nodes))
        adj_mat = tmp_adj + tmp_adj.T

        # normalize matrix
        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)

        # normalize by user counts
        norm_adj_tmp = d_mat_inv.dot(adj_mat)
        # normalize by item counts
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

    def __len__(self):
        return len(self.unique_users)

    def __getitem__(self, idx):
        """
        returns user, pos_items, neg_items

        :param idx: index of user from unique_users
        :return:
        """
        user = self.unique_users[idx]
        pos = np.random.choice(self.get_user_positives(user), self.n_negatives)
        neg = self.get_user_negatives(user, self.n_negatives)
        return user, pos, neg


class GowallaALSDataset(GowallaDataset):
    def __init__(self, path, train=True):
        super().__init__(train)
        self.path = path
        self.train = train
        self.df = pd.read_csv(path, names=['userId', 'timestamp', 'long', ' lat', 'loc_id'])

    def get_dataset(self, n_users=None, m_items=None):
        if self.train:
            users = self.df['userId'].values
            items = self.df['loc_id'].values
            ratings = np.ones(len(users))

            n_users = self.n_users if n_users is None else n_users
            m_items = self.m_items if m_items is None else m_items
            user_item_data = sp.csr_matrix((ratings, (users, items)),
                                           shape=(n_users, m_items))
            item_user_data = user_item_data.T.tocsr()
            return self.df, user_item_data, item_user_data
        else:
            return self.df
