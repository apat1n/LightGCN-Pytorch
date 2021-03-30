import wget
import pandas as pd
from pathlib import Path
from config import config
from utils import print_progressbar

if __name__ == '__main__':
    dataset_dir = Path('dataset')
    if not dataset_dir.exists():
        dataset_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = dataset_dir / 'loc-gowalla_totalCheckins.txt.gz'
    if not dataset_path.exists():
        wget.download('https://snap.stanford.edu/data/loc-gowalla_edges.txt.gz',
                      out=str(dataset_path), bar=print_progressbar)
        wget.download('https://snap.stanford.edu/data/loc-gowalla_totalCheckins.txt.gz',
                      out=str(dataset_path), bar=print_progressbar)
    gowalla_dataset = pd.read_csv(
        dataset_path, sep='\t', names=['userId', 'timestamp', 'long', 'lat', 'loc_id'])
    gowalla_dataset['timestamp'] = pd.to_datetime(gowalla_dataset['timestamp']).dt.tz_localize(None)

    split_date = pd.to_datetime(config['SPLIT_DATE'])
    start_date = gowalla_dataset['timestamp'].min() \
        if 'TRAIN_DAYS' not in config \
        else pd.to_datetime(split_date - pd.DateOffset(days=config['TRAIN_DAYS']))
    end_test_date = split_date + pd.DateOffset(days=config['TEST_DAYS'])
    end_date = pd.to_datetime(
        end_test_date + pd.DateOffset(days=config['VAL_DAYS']) if 'VAL_DAYS' in config else 0)

    timestamp_filter = (gowalla_dataset['timestamp'] >= start_date) & (
                gowalla_dataset['timestamp'] <= end_date)
    gowalla_dataset = gowalla_dataset[timestamp_filter]
    gowalla_dataset.sort_values('timestamp', inplace=True)

    new_user_ids = {k: v for v, k in enumerate(gowalla_dataset['userId'].unique())}
    new_item_ids = {k: v for v, k in enumerate(gowalla_dataset['loc_id'].unique())}

    gowalla_dataset['userId'] = gowalla_dataset['userId'].map(new_user_ids)
    gowalla_dataset['loc_id'] = gowalla_dataset['loc_id'].map(new_item_ids)

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

    train_filter = (gowalla_dataset['timestamp'] >= start_date) & (
                gowalla_dataset['timestamp'] <= split_date)
    gowalla_train = gowalla_dataset[train_filter]

    test_filter = (gowalla_dataset['timestamp'] > split_date) & (
            gowalla_dataset['timestamp'] <= end_test_date)
    gowalla_test = gowalla_dataset[test_filter]

    if 'VAL_DAYS' in config:
        val_filter = (gowalla_dataset['timestamp'] > end_test_date) & (
            gowalla_dataset['timestamp'] <= end_date)
        gowalla_val = gowalla_dataset[val_filter]
        pd.concat([gowalla_train, gowalla_test]).to_csv(
            dataset_dir / 'gowalla.traintest', index=None, header=None)
        gowalla_val.to_csv(dataset_dir / 'gowalla.val', index=None, header=None)

    gowalla_train.to_csv(dataset_dir / 'gowalla.train', index=None, header=None)
    gowalla_test.to_csv(dataset_dir / 'gowalla.test', index=None, header=None)
    gowalla_dataset.loc[:, ['loc_id', 'long', 'lat']] \
        .to_csv(dataset_dir / 'gowalla.locations', index=None, header=None)

    print('dataset splits saved')

    unique_users = set(gowalla_dataset['userId'].unique())
    gowalla_friendships = pd.read_csv(
        'dataset/loc-gowalla_edges.txt.gz', sep='\t', names=['user1', 'user2'])
    gowalla_friendships[(gowalla_friendships['user1'].isin(unique_users)) &
                        (gowalla_friendships['user2'].isin(unique_users))] \
        .to_csv(dataset_dir / 'gowalla.friends', index=None, header=None)

    print('dataset friendships saved')
