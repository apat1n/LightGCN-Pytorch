import utils
import implicit
from config import config
from model import LightGCN, TopNModel, TopNPersonalized, TopNNearestModel
from dataloader import GowallaLightGCNDataset, GowallaTopNDataset, GowallaALSDataset

if __name__ == '__main__':
    if config['MODEL'] == 'LightGCN':
        train_dataset = GowallaLightGCNDataset('dataset/gowalla.train')
        test_dataset = GowallaLightGCNDataset('dataset/gowalla.test', train=False)

        model = LightGCN(train_dataset)
        model.fit(config['TRAIN_EPOCHS'], test_dataset)
    elif config['MODEL'] == 'TopNModel':
        train_dataset = GowallaTopNDataset('dataset/gowalla.train')
        test_dataset = GowallaTopNDataset('dataset/gowalla.test', train=False)

        model = TopNModel(config['TOP_N'])
        model.fit(train_dataset)
        model.eval(test_dataset)
    elif config['MODEL'] == 'TopNPersonalized':
        train_dataset = GowallaTopNDataset('dataset/gowalla.train')
        test_dataset = GowallaTopNDataset('dataset/gowalla.test', train=False)

        model = TopNPersonalized(config['TOP_N'])
        model.fit(train_dataset)
        model.eval(test_dataset)
    elif config['MODEL'] == 'TopNNearestModel':
        train_dataset = GowallaTopNDataset('dataset/gowalla.train')
        test_dataset = GowallaTopNDataset('dataset/gowalla.test', train=False)

        calc_nearest = utils.calc_nearest(train_dataset, test_dataset)
        model = TopNNearestModel(config['TOP_N'], calc_nearest)
        model.fit(train_dataset)
        model.eval(test_dataset)
    elif config['MODEL'] == 'iALS':
        gowalla_train, user_item_data, item_user_data = GowallaALSDataset(
            'dataset/gowalla.train').get_dataset()
        gowalla_test = GowallaALSDataset('dataset/gowalla.test', train=False).get_dataset()
        model = implicit.als.AlternatingLeastSquares(
            iterations=config['ALS_N_ITERATIONS'], factors=config['ALS_N_FACTORS'])
        model.fit_callback = utils.eval_als_model(model, user_item_data, gowalla_test)
        model.fit(item_user_data)
