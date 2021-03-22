from loguru import logger
from config import config
from model import LightGCN, TopNModel
from time import gmtime, strftime
from dataloader import GowallaLightGCNDataset, GowallaTopNDataset

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
