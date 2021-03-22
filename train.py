from loguru import logger
from model import LightGCN
from time import gmtime, strftime
from dataloader import GowallaDataset

if __name__ == '__main__':
    current_time = strftime("%Y-%m-%d_%H:%M:%S", gmtime())
    logger.add(f'train_{current_time}.log')

    train_dataset = GowallaDataset('dataset/gowalla.train')
    test_dataset = GowallaDataset('dataset/gowalla.test', train=False)

    model = LightGCN(train_dataset)
    model.fit(100, test_dataset)
