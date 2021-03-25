import pandas as pd
from collections import defaultdict
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    candidates_dataset = pd.read_csv('catboost_train_dataset.csv', names=['userId', 'itemId'])

    gowalla_train = pd.read_csv('dataset/gowalla.train',
                                names=['userId', 'timestamp', 'long', 'lat', 'itemId'])
    gowalla_test = pd.read_csv('dataset/gowalla.test',
                               names=['userId', 'timestamp', 'long', 'lat', 'itemId'])
    gowalla_val = pd.read_csv('dataset/gowalla.val',
                              names=['userId', 'timestamp', 'long', 'lat', 'itemId'])
    gowalla_dataset = pd.concat([gowalla_train, gowalla_test, gowalla_val])

    # add test candidates as positives for better training
    candidates_dataset = pd.concat([candidates_dataset, gowalla_test.loc[:, ['userId', 'itemId']]])

    test_user_item_pairs = set(
        map(lambda x: (x[0], x[1]), gowalla_test.loc[:, ['userId', 'itemId']].values))

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

    candidates_target_values = list(
        map(lambda x: int((x[0], x[1]) in test_user_item_pairs),
            candidates_dataset.loc[:, ['userId', 'itemId']].values))

    candidates_dataset['num_friends'] = candidates_dataset['userId'].map(user_num_friends)
    candidates_dataset['user_num_interactions'] = \
        candidates_dataset['userId'].map(user_num_interactions)
    candidates_dataset['item_num_interactions'] = \
        candidates_dataset['userId'].map(item_num_interactions)
    candidates_dataset['user_item_interactions'] = candidates_user_item_interactions
    candidates_dataset['long'] = candidates_dataset['itemId'].map(item_long)
    candidates_dataset['lat'] = candidates_dataset['itemId'].map(item_lat)
    candidates_dataset['target'] = candidates_target_values

    print(candidates_dataset['target'].value_counts())

    train_df, eval_df = train_test_split(candidates_dataset, test_size=0.2)

    train_pool = Pool(
        train_df.drop(['userId', 'itemId', 'target'], axis=1),
        train_df['target']
    )
    eval_pool = Pool(
        eval_df.drop(['userId', 'itemId', 'target'], axis=1),
        eval_df['target']
    )

    model = CatBoostClassifier(
        iterations=5000, loss_function='Logloss', eval_metric='AUC', verbose=10)
    model.fit(train_pool, eval_set=eval_pool)
    model.save_model('catboost.cbm')

    train_df['preds'] = model.predict(train_pool)
    print(len(train_df[(train_df['preds'] == 0) & (train_df['target'] == 1)]))
    print(len(train_df[train_df['target'] == 1]))
