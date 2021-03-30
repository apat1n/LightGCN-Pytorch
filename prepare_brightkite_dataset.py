import wget
import pandas as pd
from pathlib import Path
from config import config
from utils import print_progressbar

if __name__ == '__main__':
    dataset_dir = Path('dataset')
    if not dataset_dir.exists():
        dataset_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = dataset_dir / 'loc-brightkite_totalCheckins.txt.gz'
    if not dataset_path.exists():
        wget.download('https://snap.stanford.edu/data/loc-brightkite_edges.txt.gz',
                      out=str(dataset_path), bar=print_progressbar)
        wget.download('https://snap.stanford.edu/data/loc-brightkite_totalCheckins.txt.gz',
                      out=str(dataset_path), bar=print_progressbar)
    brightkite_dataset = pd.read_csv(
        dataset_path, sep='\t', names=['userId', 'timestamp', 'long', 'lat', 'loc_id'])
    brightkite_dataset['timestamp'] = pd.to_datetime(brightkite_dataset['timestamp']).dt.tz_localize(None)

    split_date = pd.to_datetime(config['SPLIT_DATE'])
    start_date = brightkite_dataset['timestamp'].min() \
        if 'TRAIN_DAYS' not in config \
        else pd.to_datetime(split_date - pd.DateOffset(days=config['TRAIN_DAYS']))
    end_test_date = split_date + pd.DateOffset(days=config['TEST_DAYS'])
    end_date = pd.to_datetime(
        end_test_date + pd.DateOffset(days=config['VAL_DAYS']) if 'VAL_DAYS' in config else 0)

    timestamp_filter = (brightkite_dataset['timestamp'] >= start_date) & (
                brightkite_dataset['timestamp'] <= end_date)
    brightkite_dataset = brightkite_dataset[timestamp_filter]
    brightkite_dataset.sort_values('timestamp', inplace=True)

    new_user_ids = {k: v for v, k in enumerate(brightkite_dataset['userId'].unique())}
    new_item_ids = {k: v for v, k in enumerate(brightkite_dataset['loc_id'].unique())}

    brightkite_dataset['userId'] = brightkite_dataset['userId'].map(new_user_ids)
    brightkite_dataset['loc_id'] = brightkite_dataset['loc_id'].map(new_item_ids)

    with open(dataset_dir / 'user_list.txt', 'w') as f:
        f.write('org_id remap_id\n')
        for org_id, remap_id in new_user_ids.items():
            f.write(f'{org_id} {remap_id}\n')

    print('user_list.txt saved')

    with open(dataset_dir / 'item_list.txt', 'w') as f:
        f.write('org_id remap_id\n')
        for org_id, remap_id in new_item_ids.items():
            f.write(f'{org_id} {remap_id}\n')

    print('item_list.txt saved')

    train_filter = (brightkite_dataset['timestamp'] >= start_date) & (
                brightkite_dataset['timestamp'] <= split_date)
    brightkite_train = brightkite_dataset[train_filter]

    test_filter = (brightkite_dataset['timestamp'] > split_date) & (
            brightkite_dataset['timestamp'] <= end_test_date)
    brightkite_test = brightkite_dataset[test_filter]

    if 'VAL_DAYS' in config:
        val_filter = (brightkite_dataset['timestamp'] > end_test_date) & (
            brightkite_dataset['timestamp'] <= end_date)
        brightkite_val = brightkite_dataset[val_filter]
        pd.concat([brightkite_train, brightkite_test]).to_csv(
            dataset_dir / 'brightkite.traintest', index=None, header=None)
        brightkite_val.to_csv(dataset_dir / 'brightkite.val', index=None, header=None)

    brightkite_train.to_csv(dataset_dir / 'brightkite.train', index=None, header=None)
    brightkite_test.to_csv(dataset_dir / 'brightkite.test', index=None, header=None)
    brightkite_dataset.loc[:, ['loc_id', 'long', 'lat']] \
        .to_csv(dataset_dir / 'brightkite.locations', index=None, header=None)

    print('dataset splits saved')

    unique_users = set(brightkite_dataset['userId'].unique())
    brightkite_friendships = pd.read_csv(
        'dataset/loc-brightkite_edges.txt.gz', sep='\t', names=['user1', 'user2'])
    brightkite_friendships[(brightkite_friendships['user1'].isin(unique_users)) &
                        (brightkite_friendships['user2'].isin(unique_users))] \
        .to_csv(dataset_dir / 'brightkite.friends', index=None, header=None)
