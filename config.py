import os
import yaml
from loguru import logger
from time import gmtime, strftime
from utils import TensorboardWriter


if not os.path.isdir('logs'):
    os.mkdir('logs')
current_time = strftime("%Y-%m-%d_%H:%M:%S", gmtime())
logger.add(f'logs/train_{current_time}.log')

# problem on macOS
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

with open('config.yaml') as f:
    config = yaml.safe_load(f)

tensorboard_writer = None
if 'USE_TENSORBOARD' in config and config['USE_TENSORBOARD']:
    tensorboard_writer = TensorboardWriter(f'runs/{current_time}')

logger.info(f'config loaded: {config}')
