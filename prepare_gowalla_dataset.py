import wget
import pandas as pd
from pathlib import Path
from utils import print_progressbar

if __name__ == '__main__':
    dataset_dir = Path('dataset')
    if not dataset_dir.exists():
        dataset_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = dataset_dir / 'loc-gowalla_totalCheckins.txt.gz'
    if not dataset_path.exists():
        wget.download('https://snap.stanford.edu/data/loc-gowalla_totalCheckins.txt.gz',
                      out=dataset_path, bar=print_progressbar)

    gowalla_dataset = pd.read_csv(
        dataset_path, sep='\t', names=['userId', 'timestamp', 'long', ' lat', 'loc_id'])
    gowalla_dataset['timestamp'] = pd.to_datetime(gowalla_dataset['timestamp'])
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

    n_train = int(len(gowalla_dataset) * 0.8)
    n_test = len(gowalla_dataset) - n_train

    gowalla_train = gowalla_dataset.head(n_train)
    gowalla_test = gowalla_dataset.tail(n_test)

    gowalla_train.to_csv(dataset_dir / 'gowalla.train', index=None, header=None)
    gowalla_test.to_csv(dataset_dir / 'gowalla.test', index=None, header=None)

    print('dataset splits saved')
