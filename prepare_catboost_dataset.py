import utils
import torch
import implicit
import pandas as pd
from model import LightGCN, TopNModel, TopNPersonalized, TopNNearestModel
from dataloader import GowallaLightGCNDataset, GowallaTopNDataset, GowallaALSDataset


def get_als_recommendations(path):
    gowalla_train, user_item_data, item_user_data = GowallaALSDataset(path) \
        .get_dataset(5739, 56261)  # n_users and m_items from list.txt

    model = implicit.als.AlternatingLeastSquares(iterations=5, factors=64)
    model.fit(item_user_data)

    users = []
    items_pred = []
    for user in range(5738 + 1):
        preds = list(map(lambda x: x[0], model.recommend(user, user_item_data, 20)))
        users.extend([user for _ in preds])
        items_pred.extend(preds)
    return pd.DataFrame({'userId': users, 'itemId': items_pred}) \
        .drop_duplicates(['userId', 'itemId'])


def get_topn_recommendations(path):
    train_dataset = GowallaTopNDataset(path)

    model = TopNPersonalized(15)
    model.fit(train_dataset)

    users = []
    items_pred = []
    for user in range(5738 + 1):
        preds = list(model.recommend([user])[0])
        users.extend([user for _ in preds])
        items_pred.extend(preds)
    return pd.DataFrame({'userId': users, 'itemId': items_pred}) \
        .drop_duplicates(['userId', 'itemId'])


def get_top_nearest_recommendations(train_path, locations_path):
    train_dataset = GowallaTopNDataset(train_path)
    df = pd.read_csv(locations_path, names=['loc_id', 'long', 'lat'])
    calc_nearest = utils.calc_nearest(df)
    model = TopNNearestModel(15, calc_nearest)
    model.fit(train_dataset)

    users = []
    items_pred = []
    for user in range(5738 + 1):
        preds = list(model.recommend([user])[0])
        users.extend([user for _ in preds])
        items_pred.extend(preds)
    return pd.DataFrame({'userId': users, 'itemId': items_pred}) \
        .drop_duplicates(['userId', 'itemId'])


def get_lightgcn_recommendations(path):
    train_dataset = GowallaLightGCNDataset(path)
    model = LightGCN(train_dataset)
    model.fit(100)

    users = []
    items_pred = []
    preds = list(model.recommend(torch.tensor([user for user in range(5738 + 1)])))
    for user in range(5738 + 1):
        users.extend([user for _ in preds[user]])
        items_pred.extend(preds[user])
    return pd.DataFrame({'userId': users, 'itemId': items_pred}) \
        .drop_duplicates(['userId', 'itemId'])


def get_recommendations(train_path, locations_path):
    als_recommendations = get_als_recommendations(train_path)
    topn_recommendations = get_topn_recommendations(train_path)
    # top_nearest_recommendations = get_top_nearest_recommendations(train_path, locations_path)
    lightgcn_recommendations = get_lightgcn_recommendations(train_path)
    return pd.concat([als_recommendations, topn_recommendations, lightgcn_recommendations]) \
        .drop_duplicates(['userId', 'itemId'])


if __name__ == '__main__':
    # make candidates for catboost training
    get_recommendations('dataset/gowalla.train', 'dataset/gowalla.locations') \
        .to_csv('catboost_train_dataset.csv', index=False, header=False)

    # make candidates for catboost eval
    get_recommendations('dataset/gowalla.traintest', 'dataset/gowalla.locations') \
        .to_csv('catboost_eval_dataset.csv', index=False, header=False)

    get_als_recommendations('dataset/gowalla.traintest') \
        .to_csv('als_candidates.csv', index=False, header=False)
    get_topn_recommendations('dataset/gowalla.traintest') \
        .to_csv('topn_candidates.csv', index=False, header=False)
    get_lightgcn_recommendations('dataset/gowalla.traintest') \
        .to_csv('lightgcn_candidates.csv', index=False, header=False)
