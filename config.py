import os
import yaml
from loguru import logger
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter


class TensorboardWriter(SummaryWriter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_iter = defaultdict(lambda: 0)

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None):
        super().add_scalar(tag, scalar_value, self.n_iter[tag], walltime)
        self.n_iter[tag] += 1


# problem on macOS
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

with open('config.yaml') as f:
    config = yaml.safe_load(f)

tensorboard_writer = None
if 'USE_TENSORBOARD' in config and config['USE_TENSORBOARD']:
    tensorboard_writer = TensorboardWriter()

logger.info(f'config loaded: {config}')
