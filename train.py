import utils
import implicit
from loguru import logger
from config import config
from time import gmtime, strftime
from model import LightGCN, TopNModel
from dataloader import GowallaLightGCNDataset, GowallaTopNDataset, GowallaALSDataset

if __name__ == '__main__':
    current_time = strftime("%Y-%m-%d_%H:%M:%S", gmtime())
    logger.add(f'train_{current_time}.log')

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
    elif config['MODEL'] == 'iALS':
        gowalla_train, user_item_data, item_user_data = GowallaALSDataset(
            'dataset/gowalla.train').get_dataset()
        gowalla_test = GowallaALSDataset('dataset/gowalla.test', train=False).get_dataset()
        model = implicit.als.AlternatingLeastSquares(
            iterations=config['ALS_N_ITERATIONS'], factors=config['ALS_N_FACTORS'])
        model.fit_callback = utils.eval_als_model(model, user_item_data, gowalla_test)
        model.fit(item_user_data)
