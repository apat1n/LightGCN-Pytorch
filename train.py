import utils
import implicit
import pandas as pd
from config import config
from model import LightGCN, TopNModel, TopNPersonalized, TopNNearestModel
from dataloader import GowallaLightGCNDataset, GowallaTopNDataset, GowallaALSDataset

if __name__ == '__main__':
    dataset = config['DATASET']
    if config['MODEL'] == 'LightGCN':
        train_dataset = GowallaLightGCNDataset(f'dataset/{dataset}.train')
        test_dataset = GowallaLightGCNDataset(f'dataset/{dataset}.test', train=False)

        model = LightGCN(train_dataset)
        model.fit(config['TRAIN_EPOCHS'], test_dataset)
    elif config['MODEL'] == 'TopNModel':
        train_dataset = GowallaTopNDataset(f'dataset/{dataset}.train')
        test_dataset = GowallaTopNDataset(f'dataset/{dataset}.test', train=False)

        model = TopNModel(config['TOP_N'])
        model.fit(train_dataset)
        model.eval(test_dataset)
    elif config['MODEL'] == 'TopNPersonalized':
        train_dataset = GowallaTopNDataset(f'dataset/{dataset}.train')
        test_dataset = GowallaTopNDataset(f'dataset/{dataset}.test', train=False)

        model = TopNPersonalized(config['TOP_N'])
        model.fit(train_dataset)
        model.eval(test_dataset)
    elif config['MODEL'] == 'TopNNearestModel':
        train_dataset = GowallaTopNDataset(f'dataset/{dataset}.train')
        test_dataset = GowallaTopNDataset(f'dataset/{dataset}.test', train=False)

        df = pd.concat([train_dataset.df, test_dataset.df])
        calc_nearest = utils.calc_nearest(df)
        model = TopNNearestModel(config['TOP_N'], calc_nearest)
        model.fit(train_dataset)
        model.eval(test_dataset)
    elif config['MODEL'] == 'iALS':
        gowalla_train, user_item_data, item_user_data = GowallaALSDataset(
            f'dataset/{dataset}.train').get_dataset()
        gowalla_test = GowallaALSDataset(f'dataset/{dataset}.test', train=False).get_dataset()
        model = implicit.als.AlternatingLeastSquares(
            iterations=config['ALS_N_ITERATIONS'], factors=config['ALS_N_FACTORS'])
        model.fit_callback = utils.eval_als_model(model, user_item_data, gowalla_test)
        model.fit(item_user_data)
