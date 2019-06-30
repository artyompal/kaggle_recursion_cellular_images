''' Reads config file and merges settings with default ones. '''

import multiprocessing
import os
import re
import yaml

import torch

from typing import Any
from easydict import EasyDict as edict

from debug import dprint

IN_KERNEL = os.environ.get('KAGGLE_WORKING_DIR') is not None
INPUT_PATH = '../input/imet-2019-fgvc6/' if IN_KERNEL else '../input/'

def _get_default_config(filename: str, fold: int) -> edict:
    cfg = edict()
    cfg.in_kernel = False
    cfg.version = os.path.splitext(os.path.basename(filename))[0]
    cfg.experiment_dir = f'../models/{cfg.version}/fold_{fold}/' \
                         if not IN_KERNEL else '.'
    cfg.num_workers = min(12, multiprocessing.cpu_count())

    cfg.model = edict()
    cfg.model.arch = 'resnet50'
    cfg.model.image_size = 0
    cfg.model.input_size = 0
    cfg.model.num_classes = None
    cfg.model.num_folds = 5
    cfg.model.bottleneck_fc = None
    cfg.model.dropout = 0

    cfg.data = edict()
    cfg.data.train_dir = INPUT_PATH + 'train/'
    cfg.data.test_dir = INPUT_PATH + 'test/'

    cfg.data.rect_crop = edict()
    cfg.data.rect_crop.enable = False
    cfg.data.min_ratio = 0.08
    cfg.data.max_ratio = 1.0
    cfg.data.scale_both_dims = False

    cfg.train = edict()
    cfg.train.csv = ''
    cfg.train.batch_size = 32 * torch.cuda.device_count()
    cfg.train.num_epochs = 10 ** 9
    cfg.train.shuffle = True
    cfg.train.images_per_class = None
    cfg.train.max_steps_per_epoch = 10 ** 9
    cfg.train.log_freq = 100
    cfg.train.min_lr = 3e-7
    cfg.train.use_balancing_sampler = False
    cfg.train.enable_warmup = False
    cfg.train.head_only_warmup = False
    cfg.train.accum_batches_num = 1
    cfg.train.lr_decay_coeff = 0
    cfg.train.lr_decay_milestones = []

    cfg.train.mixup = edict()
    cfg.train.mixup.enable = False
    cfg.train.mixup.beta_a = 0.5

    cfg.train.warmup = edict()
    cfg.train.warmup.steps = None
    cfg.train.warmup.max_lr = None

    cfg.train.lr_finder = edict()
    cfg.train.lr_finder.num_steps = 10 ** 9     # one epoch max
    cfg.train.lr_finder.beta = 0.98
    cfg.train.lr_finder.init_value = 1e-8
    cfg.train.lr_finder.final_value = 10

    cfg.val = edict()
    cfg.val.images_per_class = None

    cfg.test = edict()
    cfg.test.csv = ''
    cfg.test.batch_size = 64 * torch.cuda.device_count()
    cfg.test.num_ttas = 1
    cfg.test.tta_combine_func = 'mean'

    cfg.optimizer = edict()
    cfg.optimizer.name = 'adam'
    cfg.optimizer.params = edict()

    cfg.scheduler = edict()
    cfg.scheduler.name = ''
    cfg.scheduler.params = edict()

    cfg.scheduler2 = edict()
    cfg.scheduler2.name = ''
    cfg.scheduler2.params = edict()

    cfg.cosine = edict()
    cfg.cosine.period = 1
    cfg.cosine.period_inc = 1
    cfg.cosine.max_period = 1000
    cfg.cosine.min_metric_val = 0.6

    cfg.loss = edict()
    cfg.loss.name = 'none'
    cfg.loss.params = edict()

    cfg.augmentations = edict()
    cfg.augmentations.global_prob = 1.0

    cfg.augmentations.hflip = False
    cfg.augmentations.vflip = False
    cfg.augmentations.rotate90 = False
    cfg.augmentations.affine = 'none'

    cfg.augmentations.rect_crop = edict()
    cfg.augmentations.rect_crop.enable = False
    cfg.augmentations.rect_crop.rect_min_area = 0.1
    cfg.augmentations.rect_crop.rect_min_ratio = 0.75

    cfg.augmentations.noise = 0
    cfg.augmentations.blur = 0
    cfg.augmentations.distortion = 0
    cfg.augmentations.color = 0

    cfg.augmentations.erase = edict()
    cfg.augmentations.erase.prob = 0
    cfg.augmentations.erase.min_area = 0.02
    cfg.augmentations.erase.max_area = 0.4
    cfg.augmentations.erase.min_ratio = 0.3
    cfg.augmentations.erase.max_ratio = 3.33

    return cfg

def _merge_config(src: edict, dst: edict) -> edict:
    if not isinstance(src, edict):
        return

    for k, v in src.items():
        if isinstance(v, edict):
            _merge_config(src[k], dst[k])
        else:
            dst[k] = v

def load_config(config_path: str, fold: int) -> edict:
    loader = yaml.SafeLoader
    loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(u'''^(?:
         [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.'))

    with open(config_path) as f:
        yaml_config = edict(yaml.load(f, Loader=loader))

    config = _get_default_config(config_path, fold)
    _merge_config(yaml_config, config)

    return config
