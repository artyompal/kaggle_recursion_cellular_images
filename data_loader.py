''' Data loaders for training & validation. '''

import math
import os
import pickle
import random

from collections import defaultdict
from glob import glob
from typing import *

import numpy as np
import pandas as pd
import torch
import torch.utils.data
import torchvision.transforms as transforms
import albumentations as albu

from PIL import Image
from tqdm import tqdm
from debug import dprint
from scipy.stats import describe


class ImageDataset(torch.utils.data.Dataset): # type: ignore
    def __init__(self, dataframe: pd.DataFrame, controls_df: pd.DataFrame,
                 mode: str, config: Any, num_ttas: int = 1,
                 augmentor: Any = None, debug_save: bool = False) -> None:
        print(f'creating data_loader for {config.version} in mode={mode}')
        assert mode in ['train', 'val', 'test']

        self.df = dataframe
        self.mode = mode

        self.version = config.version
        self.path = config.train.path if mode != 'test' else config.test.path
        self.num_classes = config.model.num_classes
        self.image_size = config.model.image_size
        self.num_ttas = num_ttas
        self.num_channels = config.model.num_channels
        self.debug_save = debug_save
        self.use_one_hot = config.loss.name != 'cross_entropy'
        self.num_sites = config.model.num_sites

        # if we use augmentations, create group transform
        self.augmentor = None

        if augmentor is not None:
            targets = {'image1': 'image'}
            self.augmentor = albu.Compose(augmentor, p=1,
                                          additional_targets=targets)

        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.02645905, 0.05782904, 0.0412261,
                                       0.04099516, 0.02156723, 0.03849208],
                                  std=[0.03776616, 0.05301339, 0.03087561,
                                       0.03875584, 0.02616441, 0.03077043])
        ])

    def _load_image(self, path: str) -> np.array:
        ''' Loads image into np.array with optional resize. '''
        image = Image.open(path)

        if self.image_size != 0:
            image = image.resize((self.image_size, self.image_size), Image.LANCZOS)

        return np.array(image)

    def _load_images(self, index: int) -> np.array:
        # df_index, site = index // 2, index % 2
        site, df_index = index // self.df.shape[0], index % self.df.shape[0]
        exp, plate, well = self.df.iloc[df_index, 1], self.df.iloc[df_index, 2], \
                           self.df.iloc[df_index, 3],
        layers = []

        for channel in range(self.num_channels):
            filename = f'{exp}/Plate{plate}/{well}_s{site+1}_w{channel+1}.png'
            layers.append(self._load_image(os.path.join(self.path, filename)))

        # neg_ctl_well = self.neg_control[f'{exp}_{plate}']
        #
        # for channel in range(self.num_channels):
        #     filename = f'{exp}/Plate{plate}/{neg_ctl_well}_s{site+1}_w{channel+1}.png'
        #     layers.append(self._load_image(os.path.join(self.path, filename)))

        image = np.dstack(layers)
        return image

    def _transform_images(self, image: np.array, index: int) -> torch.Tensor:
        ''' Applies augmentations, if any. '''
        if self.augmentor is not None:
            if self.num_channels == 3:
                image = self.augmentor(image=image)['image']
            elif self.num_channels == 6:
                image0, image1 = image[:, :, :3], image[:, :, 3:]
                results = self.augmentor(image=image0, image1=image1)
                image0, image1 = results['image'], results['image1']
                image = np.dstack([image0, image1])
            else:
                raise RuntimeError('unsupported number of channels')

        if self.debug_save:
            os.makedirs(f'debug_images_{self.version}/', exist_ok=True)

            orig_img = image[:, :, :self.num_channels]
            sirna_img = image[:, :, self.num_channels:]

            orig_img = Image.fromarray(orig_img)
            sirna_img = Image.fromarray(sirna_img)
            orig_img.save(f'debug_images_{self.version}/{index}_orig.png')
            sirna_img.save(f'debug_images_{self.version}/{index}_sirna.png')

        return self.transforms(image)

    def __getitem__(self, index: int) -> Any:
        ''' Returns: tuple (sample, target) '''
        image = self._load_images(index)

        if self.num_ttas == 1:
            image = self._transform_images(image, index)
        else:
            crops = [self._transform_images(image, index) for _ in range(self.num_ttas)]

            for i in range(len(crops)):
                if i % 2 != 0:
                    crops[i] = torch.flip(crops[i], dims=[-1])

            image = torch.stack(crops)

        if self.mode != 'test':
            target = int(self.df.sirna.values[index % self.df.shape[0]])

            if self.use_one_hot:
                targets = np.zeros(self.num_classes, dtype=np.float32)
                targets[target] = 1
                target = targets

            return image, target
        else:
            return image

    def __len__(self) -> int:
        ''' We have two sets of images per well. '''
        return self.df.shape[0] * self.num_sites
