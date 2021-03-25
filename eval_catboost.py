import metrics
import numpy as np
import pandas as pd
from collections import defaultdict
from catboost import CatBoostClassifier

if __name__ == '__main__':
    candidates_dataset = pd.read_csv('catboost_eval_dataset.csv', names=['userId', 'itemId'])

    gowalla_train = pd.read_csv('dataset/gowalla.train',
                                names=['userId', 'timestamp', 'long', 'lat', 'itemId'])
    gowalla_test = pd.read_csv('dataset/gowalla.test',
                               names=['userId', 'timestamp', 'long', 'lat', 'itemId'])
    gowalla_train = pd.concat([gowalla_train, gowalla_test])
    del gowalla_test

    gowalla_val = pd.read_csv('dataset/gowalla.val',
                              names=['userId', 'timestamp', 'long', 'lat', 'itemId'])
    gowalla_dataset = pd.concat([gowalla_train, gowalla_val])

    # count geopositions of items
    tmp = gowalla_dataset.drop_duplicates('itemId').set_index('itemId')
    item_lat = tmp['lat'].to_dict()
    item_long = tmp['long'].to_dict()
    del tmp

    user_num_interactions = defaultdict(
        lambda: 0, gowalla_train.groupby('userId')['itemId'].count().to_dict())
    item_num_interactions = defaultdict(
        lambda: 0, gowalla_train.groupby('itemId')['userId'].count().to_dict())
    user_item_interactions = defaultdict(
        lambda: 0, gowalla_train.groupby(['userId', 'itemId'])['timestamp'].count().to_dict())
    candidates_user_item_interactions = list(
        map(lambda x: user_item_interactions[(x[0], x[1])],
            candidates_dataset.loc[:, ['userId', 'itemId']].values))

    gowalla_friendships = pd.read_csv('dataset/gowalla.friends', header=None,
                                      names=['user1', 'user2'])
    user_num_friends = gowalla_friendships.groupby('user1')['user2'].count().to_dict()

    candidates_dataset['num_friends'] = candidates_dataset['userId'].map(user_num_friends)
    candidates_dataset['user_num_interactions'] = \
        candidates_dataset['userId'].map(user_num_interactions)
    candidates_dataset['item_num_interactions'] = \
        candidates_dataset['userId'].map(item_num_interactions)
    candidates_dataset['user_item_interactions'] = candidates_user_item_interactions
    candidates_dataset['long'] = candidates_dataset['itemId'].map(item_long)
    candidates_dataset['lat'] = candidates_dataset['itemId'].map(item_lat)

    # train_pool = Pool(
    #     train_df.drop(['userId', 'itemId', 'target'], axis=1),
    #     train_df['target']
    # )
    # eval_pool = Pool(
    #     eval_df.drop(['userId', 'itemId', 'target'], axis=1),
    #     eval_df['target']
    # )
    #
    model = CatBoostClassifier(
        iterations=5000, loss_function='Logloss', eval_metric='AUC', verbose=10)
    model.load_model('catboost.cbm')
    candidates_dataset['target_pred'] = model.predict_proba(
        candidates_dataset.drop(['userId', 'itemId'], axis=1))[:, 1]

    hitrates = []
    for user, df in candidates_dataset.groupby('userId'):
        k = 20
        preds = df.sort_values('target_pred').head(k)['itemId'].values
        ground_truth = gowalla_val[gowalla_val['userId'] == user]['itemId'].values
        if len(ground_truth) > 0:
            # print(preds, ground_truth)
            # print(user, metrics.user_hitrate(preds, ground_truth))
            hitrates.append(metrics.user_hitrate(preds, ground_truth))
    print(np.mean(hitrates))
