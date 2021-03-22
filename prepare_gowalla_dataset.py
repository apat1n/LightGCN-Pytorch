import os
import pandas as pd
from glob import glob

if __name__ == '__main__':
    path = 'dataset'
    if not os.path.exists(path):
        os.makedirs(path)

    # TODO: automatically download dataset
    gowalla_dataset = pd.read_csv(
        glob(os.path.join(path, '*gowalla*'))[0], sep='\t',
        names=['userId', 'timestamp', 'long', ' lat', 'loc_id'])
    gowalla_dataset['timestamp'] = pd.to_datetime(gowalla_dataset['timestamp'])
    gowalla_dataset.sort_values('timestamp', inplace=True)

    new_user_ids = {k: v for v, k in enumerate(gowalla_dataset['userId'].unique())}
    new_item_ids = {k: v for v, k in enumerate(gowalla_dataset['loc_id'].unique())}

    gowalla_dataset['userId'] = gowalla_dataset['userId'].map(new_user_ids)
    gowalla_dataset['loc_id'] = gowalla_dataset['loc_id'].map(new_item_ids)

    with open(os.path.join(path, 'user_list.txt'), 'w') as f:
        f.write('org_id remap_id\n')
        for org_id, remap_id in new_user_ids.items():
            f.write(f'{org_id} {remap_id}\n')

    print('user_list.txt saved')

    with open(os.path.join(path, 'item_list.txt'), 'w') as f:
        f.write('org_id remap_id\n')
        for org_id, remap_id in new_item_ids.items():
            f.write(f'{org_id} {remap_id}\n')

    print('item_list.txt saved')

    n_train = int(len(gowalla_dataset) * 0.8)
    n_test = len(gowalla_dataset) - n_train

    gowalla_train = gowalla_dataset.head(n_train)
    gowalla_test = gowalla_dataset.tail(n_test)

    gowalla_train.to_csv(os.path.join(path, 'gowalla.train'), index=None, header=None)
    gowalla_test.to_csv(os.path.join(path, 'gowalla.test'), index=None, header=None)

    print('dataset splits saved')
