import os
import yaml
from loguru import logger
from time import gmtime, strftime
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter


current_time = strftime("%Y-%m-%d_%H:%M:%S", gmtime())
logger.add(f'train_{current_time}.log')


class TensorboardWriter(SummaryWriter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_iter = defaultdict(lambda: 0)

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None):
        if not global_step:
            global_step = self.n_iter[tag]
            self.n_iter[tag] += 1
        super().add_scalar(tag, scalar_value, global_step, walltime)


# problem on macOS
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

with open('config.yaml') as f:
    config = yaml.safe_load(f)

tensorboard_writer = None
if 'USE_TENSORBOARD' in config and config['USE_TENSORBOARD']:
    tensorboard_writer = TensorboardWriter()

logger.info(f'config loaded: {config}')
