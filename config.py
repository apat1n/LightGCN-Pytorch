import yaml
from pprint import pprint

with open('config.yaml') as f:
    config = yaml.safe_load(f)

print('config loaded:')
pprint(config)
