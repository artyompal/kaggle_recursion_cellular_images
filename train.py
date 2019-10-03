#!/usr/bin/python3.6
''' Trains a model or infers predictions. '''

import argparse
import hashlib
import math
import os
import pprint
import random
import re
import sys
import time
import yaml

from typing import *
from collections import defaultdict, Counter
from glob import glob

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from easydict import EasyDict as edict

import albumentations as albu

from data_loader import ImageDataset, AllVsAllDataset
from utils import create_logger, AverageMeter
from debug import dprint

from parse_config import load_config
from losses import get_loss
from schedulers import get_scheduler, is_scheduler_continuous, get_warmup_scheduler
from optimizers import get_optimizer, get_lr, set_lr
from metrics import accuracy
from random_rect_crop import RandomRectCrop
from random_erase import RandomErase
from model import create_model, freeze_layers, unfreeze_layers
from torch.optim.lr_scheduler import ReduceLROnPlateau
from schedulers import CosineLRWithRestarts

IN_KERNEL = os.environ.get('KAGGLE_WORKING_DIR') is not None
INPUT_PATH = '../input/' if IN_KERNEL else 'data/'

if not IN_KERNEL:
    import torchsummary
    from hyperopt import hp, tpe, fmin


def find_input_file(path: str) -> str:
    return os.path.join(INPUT_PATH, os.path.basename(path))

def make_folds(df: pd.DataFrame, policy: str) -> pd.DataFrame:
    logger.info(f'make_folds: policy is {policy}')

    skf = StratifiedKFold(config.general.num_folds, shuffle=True, random_state=7)
    folds = -np.ones(df.shape[0], dtype=int)

    if policy == 'strat_by_target':
        for i, (train_idx, val_idx) in enumerate(skf.split(df, df.sirna)):
            dprint(train_idx.shape)
            dprint(val_idx.shape)
            folds[val_idx] = i
    elif policy == 'strat_by_target_per_type':
        types = ['HEPG2', 'HUVEC']
        df['cell_type'] = df.experiment.str.split('-').apply(lambda x: x[0])

        part_df = df.loc[df.cell_type == types[0]]
        print('type', types[0], 'dataframe', df.shape)

        for i, (train_idx, val_idx) in enumerate(skf.split(part_df, part_df.sirna)):
            dprint(train_idx.shape)
            dprint(val_idx.shape)
            folds[part_df.index[val_idx]] = i

        part_df = df.loc[df.cell_type == types[1]]
        print('type', types[1], 'dataframe', df.shape)

        for i, (train_idx, val_idx) in enumerate(skf.split(part_df, part_df.sirna)):
            dprint(train_idx.shape)
            dprint(val_idx.shape)
            folds[part_df.index[val_idx]] = i

        part_df = df.loc[(df.cell_type != types[0]) & (df.cell_type != types[1])]
        print('type other dataframe', df.shape)

        for i, (train_idx, val_idx) in enumerate(skf.split(part_df, part_df.sirna)):
            dprint(train_idx.shape)
            dprint(val_idx.shape)
            folds[part_df.index[val_idx]] = i
    elif policy == 'split_by_exp':
        experiments = sorted(df.experiment.unique())
        fold_by_exp = {exp: i % 3 for i, exp in enumerate(experiments)}

        if config.general.num_folds == 3:
            folds = df.experiment.apply(lambda exp: fold_by_exp[exp]).values
        elif config.general.num_folds == 6:
            folds = df.experiment.apply(lambda exp: fold_by_exp[exp]).values
            folds += (df.plate > 2) * 3
        else:
            assert False

        for f in range(config.general.num_folds):
            logger.info(f'fold {f}: {sum(folds == f)} rows')
    else:
        assert False

    assert all(folds != -1)
    return folds

def train_val_split(df: pd.DataFrame, fold: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    policy = config.general.validation_policy
    folds_file = f'folds_{policy}_{config.general.num_folds}.npy'

    if not os.path.exists(folds_file):
        folds = make_folds(df, policy)
        np.save(folds_file, folds)
    else:
        folds = np.load(folds_file)

    assert folds.shape[0] == df.shape[0]
    return df.loc[folds != fold], df.loc[folds == fold]

def load_data(fold: int) -> Any:
    torch.multiprocessing.set_sharing_strategy('file_system') # type: ignore
    cudnn.benchmark = True # type: ignore

    logger.info('config:')
    logger.info(pprint.pformat(config))

    full_df = pd.read_csv(find_input_file(config.train.csv))
    train_controls = pd.read_csv(find_input_file('train_controls.csv'))
    test_df = pd.read_csv(find_input_file(config.test.csv))
    test_controls = pd.read_csv(find_input_file('test_controls.csv'))

    print('full_df', full_df.shape)
    train_df, val_df = train_val_split(full_df, fold)
    print('train_df', train_df.shape)
    print('val_df', val_df.shape)
    print('train_controls', train_controls.shape)
    print('test_controls', test_controls.shape)

    if config.model.add_controls_to_train:
        train_df = pd.concat([train_df, train_controls, test_controls], sort=False)
        print('train_df total', train_df.shape)

    augs: List[Union[albu.BasicTransform, albu.OneOf]] = []

    if config.augmentations.hflip:
        augs.append(albu.HorizontalFlip(.5))
    if config.augmentations.vflip:
        augs.append(albu.VerticalFlip(.5))
    if config.augmentations.rotate90:
        augs.append(albu.RandomRotate90())
    if config.augmentations.rotate:
        augs.append(albu.Rotate())

    if config.augmentations.affine == 'soft':
        augs.append(albu.ShiftScaleRotate(shift_limit=0.075, scale_limit=0.15, rotate_limit=10, p=.75))
    elif config.augmentations.affine == 'medium':
        augs.append(albu.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2))
    elif config.augmentations.affine == 'hard':
        augs.append(albu.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.50, rotate_limit=45, p=.75))

    if config.augmentations.rect_crop.enable:
        augs.append(RandomRectCrop(min_area=config.augmentations.rect_crop.min_area,
                                   max_area=config.augmentations.rect_crop.max_area,
                                   min_ratio=config.augmentations.rect_crop.min_ratio,
                                   max_ratio=config.augmentations.rect_crop.max_ratio,
                                   image_size=config.model.image_size,
                                   input_size=config.model.input_size))

    if config.augmentations.noise >= 0.1:
        augs.append(albu.OneOf([
            albu.IAAAdditiveGaussianNoise(),
            albu.GaussNoise(),
        ], p=config.augmentations.noise))

    if config.augmentations.blur >= 0.1:
        augs.append(albu.OneOf([
            albu.MotionBlur(p=.2),
            albu.MedianBlur(blur_limit=3, p=0.1),
            albu.Blur(blur_limit=3, p=0.1),
        ], p=config.augmentations.blur))

    if config.augmentations.distortion >= 0.1:
        augs.append(albu.OneOf([
            albu.OpticalDistortion(p=0.3),
            albu.GridDistortion(p=.1),
            albu.IAAPiecewiseAffine(p=0.3),
        ], p=config.augmentations.distortion))

    if config.augmentations.color >= 0.1:
        augs.append(albu.OneOf([
            albu.CLAHE(clip_limit=2),
            albu.IAASharpen(),
            albu.IAAEmboss(),
            albu.RandomBrightnessContrast(),
        ], p=config.augmentations.color))

    if config.augmentations.erase.prob >= 0.1:
        augs.append(RandomErase(min_area=config.augmentations.erase.min_area,
                                max_area=config.augmentations.erase.max_area,
                                min_ratio=config.augmentations.erase.min_ratio,
                                max_ratio=config.augmentations.erase.max_ratio,
                                input_size=config.model.input_size,
                                p=config.augmentations.erase.prob))

    transform_train = albu.Compose([
        albu.PadIfNeeded(config.model.input_size, config.model.input_size),
        albu.RandomCrop(height=config.model.input_size, width=config.model.input_size),
        albu.Compose(augs, p=config.augmentations.global_prob),
        ])

    if config.test.num_ttas > 1:
        transform_test = albu.Compose([
            albu.PadIfNeeded(config.model.input_size, config.model.input_size),
            albu.RandomCrop(height=config.model.input_size, width=config.model.input_size),
            # horizontal flip is done by the data loader
        ])
    else:
        transform_test = albu.Compose([
            albu.PadIfNeeded(config.model.input_size, config.model.input_size),
            albu.CenterCrop(height=config.model.input_size, width=config.model.input_size),
            # albu.RandomCrop(height=config.model.input_size, width=config.model.input_size)    ,
            # albu.HorizontalFlip(.5)
        ])


    train_dataset = ImageDataset(train_df, train_controls,
                                 mode='train', config=config,
                                 num_sites=config.model.num_sites,
                                 augmentor=transform_train,
                                 debug_save=args.save_images)

    num_ttas_for_val = config.test.num_ttas if args.predict_oof else 1

    train_subset = train_df.loc[train_df.sirna < config.general.num_supported_sirna]
    train_subset = train_subset.groupby('sirna').apply(
        lambda row: row.sample(config.test.num_train_samples_per_class))
    train_feature_dataset = ImageDataset(train_subset, train_controls,
                                         mode='val', config=config,
                                         num_ttas=num_ttas_for_val,
                                         augmentor=transform_test)

    val_subset = val_df.groupby('sirna').apply(
        lambda row: row.sample(config.test.num_val_samples_per_class))
    val_feature_dataset = ImageDataset(val_subset, train_controls,
                                       mode='val', config=config,
                                       num_ttas=num_ttas_for_val,
                                       augmentor=transform_test)

    test_dataset = ImageDataset(test_df, test_controls,
                                mode='test', config=config,
                                num_ttas=config.test.num_ttas,
                                augmentor=transform_test)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.train.batch_size, shuffle=True,
        num_workers=config.general.num_workers, drop_last=True)

    train_feature_loader = torch.utils.data.DataLoader(
        train_feature_dataset, batch_size=config.test.batch_size, shuffle=False,
        num_workers=config.general.num_workers)

    val_feature_loader = torch.utils.data.DataLoader(
        val_feature_dataset, batch_size=config.test.batch_size, shuffle=False,
        num_workers=config.general.num_workers)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config.test.batch_size, shuffle=False,
        num_workers=config.general.num_workers)

    return train_loader, train_feature_loader, val_feature_loader, test_loader

def lr_finder(train_loader: Any, model: Any, criterion: Any, optimizer: Any) -> None:
    ''' Finds the optimal LR range and sets up first optimizer parameters. '''
    logger.info('lr_finder called')

    batch_time = AverageMeter()
    num_steps = min(len(train_loader), config.train.lr_finder.num_steps)
    logger.info(f'total batches: {num_steps}')
    end = time.time()
    lr_str = ''
    model.train()

    init_value = config.train.lr_finder.init_value
    final_value = config.train.lr_finder.final_value
    beta = config.train.lr_finder.beta

    mult = (final_value / init_value) ** (1 / (num_steps - 1))
    lr = init_value

    avg_loss = best_loss = 0.0
    losses = np.zeros(num_steps)
    logs = np.zeros(num_steps)

    for i, (input_, target) in enumerate(train_loader):
        if i >= num_steps:
            break

        set_lr(optimizer, lr)

        output = model(input_.cuda())
        loss = criterion(output, target.cuda().float())
        loss_val = loss.data.item()

        predict = output.detach() > 0.5
        score = accuracy(predict, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lr_str = f'\tlr {lr:.02e}'

        # compute the smoothed loss
        avg_loss = beta * avg_loss + (1 - beta) * loss_val
        smoothed_loss = avg_loss / (1 - beta ** (i + 1))

        # stop if the loss is exploding
        if i > 0 and smoothed_loss > 4 * best_loss:
            break

        # record the best loss
        if smoothed_loss < best_loss or i == 0:
            best_loss = smoothed_loss

        # store the values
        losses[i] = smoothed_loss
        logs[i] = math.log10(lr)

        # update the lr for the next step
        lr *= mult

        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.train.log_freq == 0:
            logger.info(f'lr_finder [{i}/{num_steps}]\t'
                        f'time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        f'loss {loss:.4f} ({smoothed_loss:.4f})\t'
                        f'acc {score:.4f} {lr_str}')

    np.savez(os.path.join(config.general.experiment_dir, f'lr_finder_{config.version}'),
             logs=logs, losses=losses)

    d1 = np.zeros_like(losses); d1[1:] = losses[1:] - losses[:-1]
    first, last = np.argmin(d1), np.argmin(losses)

    MAGIC_COEFF = 4

    highest_lr = 10 ** logs[last]
    best_high_lr = highest_lr / MAGIC_COEFF
    best_low_lr = 10 ** logs[first]
    logger.info(f'best_low_lr={best_low_lr} best_high_lr={best_high_lr} '
                f'highest_lr={highest_lr}')

    def find_nearest(array: np.array, value: float) -> int:
        return (np.abs(array - value)).argmin()

    last = find_nearest(logs, math.log10(best_high_lr))
    logger.info(f'first={first} last={last}')

    import matplotlib.pyplot as plt
    plt.plot(logs, losses, '-D', markevery=[first, last])
    plt.savefig(os.path.join(config.general.experiment_dir, 'lr_finder_plot.png'))

def mixup(x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    ''' Performs mixup: https://arxiv.org/pdf/1710.09412.pdf '''
    coeff = np.random.beta(config.train.mixup.beta_a, config.train.mixup.beta_a)
    indices = np.roll(np.arange(x.shape[0]), np.random.randint(1, x.shape[0]))
    indices = torch.tensor(indices).cuda()

    x = x * coeff + x[indices] * (1 - coeff)
    y = y * coeff + y[indices] * (1 - coeff)
    return x, y

def train_epoch(train_loader: Any, model: Any, criterion: Any, optimizer: Any,
                epoch: int, lr_scheduler: Any) -> float:
    logger.info(f'epoch: {epoch}')
    logger.info(f'learning rate: {get_lr(optimizer)}')

    batch_time = AverageMeter()
    losses = AverageMeter()
    avg_score = AverageMeter()

    model.train()
    optimizer.zero_grad()

    num_steps = min(len(train_loader), config.train.max_steps_per_epoch)
    num_steps -= num_steps % config.train.accum_batches_num

    logger.info(f'total batches: {num_steps}')
    end = time.time()
    lr_str = ''

    for i, (input_, target) in enumerate(train_loader):
        if i >= num_steps:
            break

        input_ = input_.cuda()

        if config.train.mixup.enable:
            input_, target = mixup(input_, target)

        output = model(input_)
        loss = criterion(output, target.float().cuda())

        predict = output.detach() > 0.5
        avg_score.update(accuracy(predict, target))

        losses.update(loss.data.item(), input_.size(0))
        loss.backward()

        if (i + 1) % config.train.accum_batches_num == 0:
            optimizer.step()
            optimizer.zero_grad()

        if is_scheduler_continuous(lr_scheduler):
            lr_scheduler.step()
            lr_str = f'\tlr {get_lr(optimizer):.02e}'

        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.train.log_freq == 0:
            logger.info(f'{epoch} [{i}/{num_steps}]\t'
                        f'time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        f'loss {losses.val:.4f} ({losses.avg:.4f})\t'
                        f'acc {avg_score.val:.4f} ({avg_score.avg:.4f})'
                        + lr_str)
        # FIXME: this prints random accuracy value from the last batch

    logger.info(f' * average acc on train {avg_score.avg:.4f}')
    return avg_score.avg

def inference(data_loader: Any, model: Any, func: Any, activation: Any = None) -> np.array:
    ''' Returns predictions array. '''
    model.eval()
    predicts_list, targets_list = [], []

    # if config.loss.name == 'binary_cross_entropy':
    #     activation: nn.Module = nn.Sigmoid()
    # elif config.loss.name == 'cross_entropy':
    #     activation = nn.Softmax(dim=1)
    # else:
    #     assert None

    with torch.no_grad():
        for input_ in tqdm(data_loader, disable=IN_KERNEL):
            if data_loader.dataset.num_ttas != 1:
                bs, ncrops, c, h, w = input_.size()
                input_ = input_.view(-1, c, h, w)
                output = getattr(model, func)(input_.cuda())

                if config.test.tta_combine_func == 'max':
                    output = output.view(bs, ncrops, -1).max(1)[0]
                elif config.test.tta_combine_func == 'mean':
                    output = output.view(bs, ncrops, -1).mean(1)
                else:
                    assert False
            else:
                output = getattr(model, func)(input_.cuda())

            if activation is not None:
                output = activation(output)

            predicts_list.append(output.detach().cpu().numpy())

    predicts = np.concatenate(predicts_list)
    return predicts

def siamese_inference(train_feature_loader: Any, test_feature_loader: Any,
                      model: Any) -> np.array:
    ''' Returns predictions array. '''
    logger.info('calculating features for the train')
    train_features = inference(train_feature_loader, model, 'features')
    logger.info('calculating features for validation/test')
    test_features = inference(test_feature_loader, model, 'features')

    dataset = AllVsAllDataset(torch.tensor(train_features),
                              torch.tensor(test_features), config)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=config.test.batch_size, shuffle=False,
        num_workers=config.general.num_workers)
    logger.info('calculating distances')
    predicts = inference(data_loader, model, 'classifier', nn.Sigmoid())

    # TODO: use both sites for the test stage, one site for the validation stage
    # if config.model.num_sites == 2:
    #     sz = predicts.shape[0]
    #     predicts = np.mean(np.dstack([predicts[:sz], predicts[sz:]]), axis=-1)

    predicts = predicts.reshape(len(test_feature_loader.dataset),
                                len(train_feature_loader.dataset))
    dprint(predicts.shape)
    predicts = np.argmax(predicts, axis=1)
    dprint(predicts.shape)

    # convert train sample indices to sirna indices here
    predicts = train_feature_loader.dataset.df.sirna.values[predicts]
    dprint(predicts.shape)

    return predicts

def validate(train_feature_loader: Any, val_feature_loader: Any, model: Any,
             epoch: int) -> float:
    ''' Infers predictions and calculates validation score. '''
    logger.info('validate()')
    predicts = siamese_inference(train_feature_loader, val_feature_loader, model)

    targets = val_feature_loader.dataset.df.sirna.values
    score = accuracy(predicts, targets)

    logger.info(f' * epoch {epoch} acc on validation {score:.4f}')
    return score

def gen_train_prediction(data_loader: Any, model: Any, epoch: int,
                         model_path: str) -> np.ndarray:
    predicts, _ = siamese_inference(data_loader, model)
    filename = os.path.splitext(os.path.basename(model_path))[0]
    np.save(f'level1_train_{filename}.npy', predicts)

def gen_test_prediction(data_loader: Any, model: Any, model_path: str) -> np.ndarray:
    predicts, _ = siamese_inference(data_loader, model)
    filename = f'level1_test_{os.path.splitext(os.path.basename(model_path))[0]}'
    np.save(filename, predicts)

def run(hyperparams: Optional[Dict[str, str]] = None) -> float:
    np.random.seed(0)
    logger.info('=' * 50)

    if hyperparams:
        hash = hashlib.sha224(str(hyperparams).encode()).hexdigest()[:8]
        model_dir = os.path.join(config.general.experiment_dir, f'{hash}')

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        str_params = str(hyperparams)
        logger.info(f'hyperparameters: {hyperparams}')
        config.augmentations.update(hyperparams)
    else:
        model_dir = config.general.experiment_dir

    train_loader, train_feature_loader, val_feature_loader, test_loader = load_data(args.fold)
    epoch_size = min(len(train_loader), config.train.max_steps_per_epoch)

    logger.info(f'creating a model {config.model.arch}')
    model = create_model(config, pretrained=args.weights is None).cuda()
    criterion = get_loss(config)

    if args.summary:
        torchsummary.summary(model, (config.model.num_channels * 2,
                                     config.model.input_size,
                                     config.model.input_size))

    if args.lr_finder:
        optimizer = get_optimizer(config, model.parameters())
        lr_finder(train_loader, model, criterion, optimizer)
        sys.exit()

    if args.weights is None and config.train.head_only_warmup:
        logger.info('-' * 50)
        logger.info(f'doing warmup for {config.train.warmup.epochs} epochs')
        logger.info(f'max lr will be {config.optimizer.params.lr}')

        optimizer = get_optimizer(config, model.parameters())
        warmup_scheduler = get_warmup_scheduler(config, optimizer, epoch_size)
        freeze_layers(model)

        for epoch in range(config.train.warmup.epochs):
            train_epoch(train_loader, model, criterion, optimizer, epoch,
                        warmup_scheduler)

        unfreeze_layers(model)

    if args.weights is None and config.train.enable_warmup:
        logger.info('-' * 50)
        logger.info(f'doing warmup for {config.train.warmup.epochs} epochs')
        logger.info(f'max lr will be {config.optimizer.params.lr}')

        optimizer = get_optimizer(config, model.parameters())
        warmup_scheduler = get_warmup_scheduler(config, optimizer, epoch_size)

        for epoch in range(config.train.warmup.epochs):
            train_epoch(train_loader, model, criterion, optimizer, epoch,
                        warmup_scheduler)

    optimizer = get_optimizer(config, model.parameters())

    if args.weights is None:
        last_epoch = -1
    else:
        last_checkpoint = torch.load(args.weights)
        model_arch = last_checkpoint['arch'].replace('se_', 'se')

        if model_arch != config.model.arch:
            dprint(model_arch)
            dprint(config.model.arch)
            assert model_arch == config.model.arch

        model.load_state_dict(last_checkpoint['state_dict'])
        if 'optimizer' in last_checkpoint.keys():
            optimizer.load_state_dict(last_checkpoint['optimizer'])
        logger.info(f'checkpoint loaded: {args.weights}')

        last_epoch = last_checkpoint['epoch'] if 'epoch' in last_checkpoint.keys() else 99
        logger.info(f'loaded the model from epoch {last_epoch}')

        if args.lr != 0:
            set_lr(optimizer, float(args.lr))
        elif 'lr' in config.optimizer.params:
            set_lr(optimizer, config.optimizer.params.lr)
        elif 'base_lr' in config.scheduler.params:
            set_lr(optimizer, config.scheduler.params.base_lr)

    lr_scheduler = get_scheduler(config, optimizer, epoch_size=epoch_size)

    if args.predict_oof or args.predict_test:
        print('inference mode')
        assert args.weights is not None

        if args.predict_oof:
            gen_train_prediction(val_loader, model, last_epoch, args.weights)
        else:
            gen_test_prediction(test_loader, model, args.weights)

        sys.exit()

    logger.info(f'training will start from epoch {last_epoch + 1}')

    best_score = 0.0
    best_epoch = 0

    last_lr = get_lr(optimizer)
    best_model_path = args.weights

    for epoch in range(last_epoch + 1, config.train.num_epochs):
        logger.info('-' * 50)

        if not is_scheduler_continuous(lr_scheduler):
            # if we have just reduced LR, reload the best saved model
            lr = get_lr(optimizer)

            if lr < last_lr - 1e-10 and best_model_path is not None:
                logger.info(f'learning rate dropped: {lr}, reloading')
                last_checkpoint = torch.load(best_model_path)

                assert(last_checkpoint['arch']==config.model.arch)
                model.load_state_dict(last_checkpoint['state_dict'])
                optimizer.load_state_dict(last_checkpoint['optimizer'])
                logger.info(f'checkpoint loaded: {best_model_path}')
                set_lr(optimizer, lr)
                last_lr = lr

        if isinstance(lr_scheduler, CosineLRWithRestarts):
            restart = lr_scheduler.epoch_step()
            if restart:
                logger.info('cosine annealing restarted, resetting the best metric')
                best_score = min(config.train.restart_metric_val, best_score)

        train_epoch(train_loader, model, criterion, optimizer, epoch, lr_scheduler)
        score = validate(train_feature_loader, val_feature_loader, model, epoch)

        if type(lr_scheduler) == ReduceLROnPlateau:
            lr_scheduler.step(metrics=score)
        elif not is_scheduler_continuous(lr_scheduler):
            lr_scheduler.step()

        is_best = score > best_score
        best_score = max(score, best_score)
        if is_best:
            best_epoch = epoch

        if is_best:
            best_model_path = os.path.join(model_dir,
                f'{config.version}_f{args.fold}_e{epoch:02d}_{score:.04f}.pth')

            data_to_save = {
                'epoch': epoch,
                'arch': config.model.arch,
                'state_dict': model.state_dict(),
                'score': score,
                'optimizer': optimizer.state_dict(),
                'config': config
            }

            torch.save(data_to_save, best_model_path)
            logger.info(f'a snapshot was saved to {best_model_path}')

    logger.info(f'best score: {best_score:.04f}')
    return -best_score

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='model configuration file (YAML)', type=str)
    parser.add_argument('--lr_finder', help='invoke LR finder and exit', action='store_true')
    parser.add_argument('--weights', help='model to resume training', type=str)
    parser.add_argument('--fold', help='fold number', type=int, default=0)
    parser.add_argument('--predict_oof', help='make predictions for the train set and return', action='store_true')
    parser.add_argument('--predict_test', help='make predictions for the test set and return', action='store_true')
    parser.add_argument('--summary', help='show model summary', action='store_true')
    parser.add_argument('--lr', help='override learning rate', type=float, default=0)
    parser.add_argument('--num_epochs', help='override number of epochs', type=int, default=0)
    parser.add_argument('--num_ttas', help='override number of TTAs', type=int, default=0)
    parser.add_argument('--save_images', help='debug save images', action='store_true')
    # parser.add_argument('--cosine', help='enable cosine annealing', type=bool, default=True)
    args = parser.parse_args()

    if not args.config:
        if not args.weights:
            print('you must specify either --config or --weights')
            sys.exit()

        # f'{config.version}_f{args.fold}_e{epoch:02d}_{score:.04f}.pth')
        m = re.match(r'(.*)_f(\d)_e(\d+)_([.0-9]+)\.pth', os.path.basename(args.weights))
        if not m:
            print('could not parse model name', os.path.basename(args.weights))
            assert False

        args.config = f'config/{m.group(1)}.yml'
        args.fold = int(m.group(2))

        print(f'detected config={args.config} fold={args.fold}')

    config = load_config(args.config, args.fold)

    if args.num_epochs:
        config.train.num_epochs = args.num_epochs

    if args.num_ttas:
        config.test.num_ttas = args.num_ttas

    if args.save_images:
        config.general.num_workers = 0

    if not os.path.exists(config.general.experiment_dir):
        os.makedirs(config.general.experiment_dir)

    log_filename = 'log_predict.txt' if args.predict_oof or args.predict_test \
                    else 'log_training.txt'
    logger = create_logger(os.path.join(config.general.experiment_dir, log_filename))

    if not config.hyperopt.enable:
        run()
    else:
        # use hyperopt to find hyperparameters
        hyperopt_space = {}

        for key, value in config.hyperopt.augmentations.items():
            type, params = value['type'], value['args']

            if type == 'choice':
                hyperopt_space[key] = hp.choice(key, params)
            else:
                hyperopt_space[key] = hp.__dict__[type](key, *params)

        best = fmin(fn=run, space=hyperopt_space, algo=tpe.suggest,
                    max_evals=config.hyperopt.max_evals)
