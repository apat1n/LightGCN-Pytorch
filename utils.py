import sys
import torch
import metrics
import haversine
from loguru import logger
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter


def print_progressbar(current, total, width=80):
    progress_message = "Downloading: %d%% [%d / %d] bytes" % (current / total * 100, current, total)
    # Don't use print() as it will print in new line every time.
    sys.stdout.write("\r" + progress_message)
    sys.stdout.flush()


class TensorboardWriter(SummaryWriter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_iter = defaultdict(lambda: 0)

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None):
        if not global_step:
            global_step = self.n_iter[tag]
            self.n_iter[tag] += 1
        super().add_scalar(tag, scalar_value, global_step, walltime)


def eval_als_model(model, user_item_data, gowalla_test):
    from config import config

    def inner(iteration, elapsed):
        preds = []
        ground_truth = []
        n_recommend = max(config['METRICS_REPORT'])
        test_interactions = gowalla_test.groupby('userId')['loc_id'].apply(list).to_dict()
        for userId in gowalla_test['userId'].unique():
            preds.append(
                list(map(lambda x: x[0], model.recommend(userId, user_item_data, n_recommend))))
            ground_truth.append(test_interactions[userId])

        logger.info(f'{iteration} iteration:')
        max_length = max(map(len, metrics.metric_dict.keys())) + max(
            map(lambda x: len(str(x)), config['METRICS_REPORT']))
        for metric_name, metric_func in metrics.metric_dict.items():
            for k in config['METRICS_REPORT']:
                metric_name_total = f'{metric_name}@{k}'
                metric_value = metric_func(preds, ground_truth, k).mean()
                logger.info(f'{metric_name_total: >{max_length + 1}} = {metric_value}')

    return inner


def calc_nearest(df):
    df = df.set_index('loc_id')
    item_lat = df['lat'].to_dict()
    item_long = df['long'].to_dict()
    locations = {item: (item_long[item], item_lat[item]) for item in item_lat}

    def inner(item_id, k=20):
        loc = locations[item_id]
        distances = [
            (item, haversine.haversine(loc, location)) for item, location in locations.items()]
        return list(map(lambda x: x[0], sorted(distances, key=lambda x: x[1])[:k]))
    return inner


def collate_function(batch):
    users = []
    pos_items = []
    neg_items = []
    for user, pos, neg in batch:
        users.extend([user for _ in pos])
        pos_items.extend(pos)
        neg_items.extend(neg)
    return list(map(torch.tensor, [users, pos_items, neg_items]))
