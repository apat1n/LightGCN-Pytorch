from loguru import logger
from config import config
from model import LightGCN
from time import gmtime, strftime
from dataloader import GowallaLightGCNDataset

if __name__ == '__main__':
    current_time = strftime("%Y-%m-%d_%H:%M:%S", gmtime())
    logger.add(f'train_{current_time}.log')

    train_dataset = GowallaLightGCNDataset('dataset/gowalla.train')
    test_dataset = GowallaLightGCNDataset('dataset/gowalla.test', train=False)

    model = LightGCN(train_dataset)
    model.fit(config['TRAIN_EPOCHS'], test_dataset)
