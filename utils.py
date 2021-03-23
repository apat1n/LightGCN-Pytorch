import sys
import metrics
from loguru import logger
from config import config


def print_progressbar(current, total, width=80):
    progress_message = "Downloading: %d%% [%d / %d] bytes" % (current / total * 100, current, total)
    # Don't use print() as it will print in new line every time.
    sys.stdout.write("\r" + progress_message)
    sys.stdout.flush()


def eval_als_model(model, user_item_data, gowalla_test):
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
