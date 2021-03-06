import faiss
import torch
import utils
import metrics
import torch.nn as nn
from tqdm import tqdm
from loguru import logger
from collections import defaultdict
from torch.utils.data import DataLoader
from config import config, tensorboard_writer
from dataloader import GowallaLightGCNDataset, GowallaTopNDataset


class TopNModel:
    def __init__(self, top_n):
        self.top_n = top_n
        self.top_items = []

    def fit(self, dataset: GowallaTopNDataset):
        item_counts = dataset.df.groupby('loc_id')['userId'].count().reset_index(name='count')
        self.top_items = item_counts.sort_values('count', ascending=False).head(self.top_n)[
            'loc_id'].values

    def recommend(self, users: list, k: int = 20):
        return [self.top_items[:k] for _ in users]

    def eval(self, test_dataset: GowallaTopNDataset):
        users = []
        ground_truth = []

        for user in test_dataset.get_all_users():
            user_positive_items = test_dataset.get_user_positives(user)
            if user_positive_items:
                users.append(user)
                ground_truth.append(user_positive_items)

        preds = self.recommend(users)
        max_length = max(map(len, metrics.metric_dict.keys())) + max(
            map(lambda x: len(str(x)), config['METRICS_REPORT']))
        for metric_name, metric_func in metrics.metric_dict.items():
            for k in config['METRICS_REPORT']:
                metric_name_total = f'{metric_name}@{k}'
                metric_value = metric_func(preds, ground_truth, k).mean()
                logger.info(f'{metric_name_total: >{max_length + 1}} = {metric_value}')


class TopNPersonalized:
    def __init__(self, top_n):
        self.top_n = top_n
        self.top_items = []
        self.top_user_items = defaultdict(lambda: self.top_items)

    def fit(self, dataset: GowallaTopNDataset):
        item_counts = dataset.df.groupby('loc_id')['userId'].count().reset_index(name='count')
        self.top_items = item_counts.sort_values('count', ascending=False) \
            .head(self.top_n)['loc_id'].values
        for user_id, df in dataset.df.groupby('userId'):
            item_counts = df.groupby('loc_id')['timestamp'].count().reset_index(name='count')
            self.top_user_items[user_id] = item_counts.sort_values('count', ascending=False) \
                .head(self.top_n)['loc_id'].values

    def recommend(self, users: list, k: int = 20):
        return [self.top_user_items[user][:k] for user in users]

    def eval(self, test_dataset: GowallaTopNDataset):
        users = []
        ground_truth = []

        for user in test_dataset.get_all_users():
            user_positive_items = test_dataset.get_user_positives(user)
            if user_positive_items:
                users.append(user)
                ground_truth.append(user_positive_items)

        preds = self.recommend(users)
        max_length = max(map(len, metrics.metric_dict.keys())) + max(
            map(lambda x: len(str(x)), config['METRICS_REPORT']))
        for metric_name, metric_func in metrics.metric_dict.items():
            for k in config['METRICS_REPORT']:
                metric_name_total = f'{metric_name}@{k}'
                metric_value = metric_func(preds, ground_truth, k).mean()
                logger.info(f'{metric_name_total: >{max_length + 1}} = {metric_value}')


class TopNNearestModel:
    def __init__(self, top_n, calc_nearest):
        self.top_n = top_n
        self.top_items = []
        self.calc_nearest = calc_nearest
        self.user_last_item = defaultdict(lambda: -1)
        self.top_nearest_items = defaultdict(lambda: self.top_items)

    def fit(self, dataset: GowallaTopNDataset):
        item_counts = dataset.df.groupby('loc_id')['userId'].count().reset_index(name='count')
        self.top_items = item_counts.sort_values('count', ascending=False) \
            .head(self.top_n)['loc_id'].values
        for user, item in dataset.df.sort_values('timestamp') \
                .drop_duplicates('userId', keep='last').set_index('userId')['loc_id'] \
                .to_dict().items():
            self.user_last_item[user] = item
        for user, last_item in tqdm(self.user_last_item.items()):
            self.top_nearest_items[user] = self.calc_nearest(last_item, self.top_n)

    def recommend(self, users: list, k: int = 20):
        return [self.top_nearest_items[self.user_last_item[user]][:k] for user in users]

    def eval(self, test_dataset: GowallaTopNDataset):
        users = []
        ground_truth = []

        for user in test_dataset.get_all_users():
            user_positive_items = test_dataset.get_user_positives(user)
            if user_positive_items:
                users.append(user)
                ground_truth.append(user_positive_items)

        preds = self.recommend(users)
        max_length = max(map(len, metrics.metric_dict.keys())) + max(
            map(lambda x: len(str(x)), config['METRICS_REPORT']))
        for metric_name, metric_func in metrics.metric_dict.items():
            for k in config['METRICS_REPORT']:
                metric_name_total = f'{metric_name}@{k}'
                metric_value = metric_func(preds, ground_truth, k).mean()
                logger.info(f'{metric_name_total: >{max_length + 1}} = {metric_value}')


class LightGCN(nn.Module):
    def __init__(self, dataset: GowallaLightGCNDataset):
        """
        :param dataset: dataset derived from BasicDataset
        """
        super(LightGCN, self).__init__()
        self.dataset: GowallaLightGCNDataset = dataset
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.latent_dim = config['LATENT_DIM']
        self.n_layers = config['N_LAYERS']
        self.__init_weight()

    def __init_weight(self):
        """
        Initialize embeddings with normal distribution
        :return:
        """
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)

        if 'pretrain' not in config or not config['PRETRAIN']:
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)
        else:
            self.embedding_user.weight.data.copy_(torch.from_numpy(config['USER_EMB_FILE']))
            self.embedding_item.weight.data.copy_(torch.from_numpy(config['ITEM_EMB_FILE']))
            print('use pretrained data')

        self.Graph = self.dataset.get_sparse_graph()
        print('LightGCN is ready to go')

    def computer(self) -> tuple:
        """
        Propagate high-hop embeddings for lightGCN
        :return: user embeddings, item embeddings
        """
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])

        layer_embeddings = [all_emb]
        for _ in range(self.n_layers):
            all_emb = torch.sparse.mm(self.Graph, all_emb)
            layer_embeddings.append(all_emb)
        layer_embeddings = torch.stack(layer_embeddings, dim=1)

        final_embeddings = layer_embeddings.mean(dim=1)  # output is mean of all layers
        users, items = torch.split(final_embeddings, [self.num_users, self.num_items])
        return users, items

    def get_users_rating(self, users: torch.tensor) -> torch.tensor:
        """
        Compute item ratings for users
        :param users: user ids for which compute ratings
        :return:
        """
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = torch.matmul(users_emb, items_emb.t())
        return rating

    def get_embedding(self, users: torch.tensor, pos_items: torch.tensor,
                      neg_items: torch.tensor) -> tuple:
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def bpr_loss(self, users: torch.tensor, pos: torch.tensor, neg: torch.tensor) -> tuple:
        """
        Calculate BPR loss as - sum ln(sigma(pos_scores - neg_scores)) + L2 norm
        :param users: users for which calculate loss
        :param pos: positive items
        :param neg: negative items
        :return: loss, reg_loss
        """
        (users_emb, pos_emb, neg_emb,
         userEmb0, posEmb0, negEmb0) = self.get_embedding(users.long(), pos.long(), neg.long())
        reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) +
                              posEmb0.norm(2).pow(2) +
                              negEmb0.norm(2).pow(2)) / float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        loss = - (pos_scores - neg_scores).sigmoid().log().mean()
        return loss, reg_loss

    def forward(self, users: torch.tensor, items: torch.tensor):
        # compute embedding
        all_users, all_items = self.computer()

        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_prod = torch.mul(users_emb, items_emb)
        return torch.sum(inner_prod, dim=1).sigmoid()

    def fit(self, n_epochs: int = 10, dataset: GowallaLightGCNDataset = None):
        """
        Fitting model with BPR loss.
        :param n_epochs: number of epochs to fit model
        :param dataset: dataset for training
        :return:
        """
        optimizer = torch.optim.Adam(self.parameters())
        pbar = tqdm(range(n_epochs))
        dataloader = DataLoader(
            self.dataset, batch_size=config['BATCH_SIZE'],
            shuffle=True, collate_fn=utils.collate_function)
        for epoch in pbar:
            for users, pos, neg in dataloader:
                optimizer.zero_grad()
                loss, reg_loss = self.bpr_loss(users, pos, neg)
                total_loss = loss + config['BPR_REG_ALPHA'] * reg_loss

                total_loss.backward()
                optimizer.step()

                if tensorboard_writer:
                    tensorboard_writer.add_scalar('Train/bpr_loss', loss.item())
                    tensorboard_writer.add_scalar('Train/bpr_reg_loss', reg_loss.item())
                    tensorboard_writer.add_scalar('Train/bpr_total_loss', total_loss.item())
                pbar.set_postfix({'bpr_loss': total_loss.item()})

            if dataset and config['EVAL_EPOCHS'] > 0 and epoch % config['EVAL_EPOCHS'] == 0:
                self.eval(dataset)

    @torch.no_grad()
    def recommend(self, users: torch.tensor, k: int = 20):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()].numpy()
        items_emb = all_items.numpy()

        index = faiss.IndexFlatIP(self.latent_dim)
        index.add(items_emb)
        return index.search(users_emb, k)[1]

    @torch.no_grad()
    def eval(self, test_dataset: GowallaLightGCNDataset):
        users = []
        ground_truth = []

        for user in test_dataset.get_all_users():
            user_positive_items = test_dataset.get_user_positives(user)
            if user_positive_items:
                users.append(user)
                ground_truth.append(user_positive_items)

        preds = self.recommend(torch.tensor(users), max(config['METRICS_REPORT']))
        max_length = max(map(len, metrics.metric_dict.keys())) + max(
            map(lambda x: len(str(x)), config['METRICS_REPORT']))
        for metric_name, metric_func in metrics.metric_dict.items():
            for k in config['METRICS_REPORT']:
                metric_name_total = f'{metric_name}@{k}'
                metric_value = metric_func(preds, ground_truth, k).mean()
                logger.info(f'{metric_name_total: >{max_length + 1}} = {metric_value}')
                if tensorboard_writer:
                    tensorboard_writer.add_scalar(f'Eval/{metric_name_total}', metric_value)
