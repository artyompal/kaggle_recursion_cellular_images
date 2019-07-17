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
from rxrx1.rxrx.io import convert_tensor_to_rgb


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
        self.siamese_input = config.model.type == 'SiameseModel'


        # load negative control information
        self.neg_control: Dict[str, str] = {}
        controls_df = controls_df.loc[controls_df.well_type == 'negative_control']

        for row in controls_df.itertuples():
            exp_plate = f'{row.experiment}_{row.plate}'
            self.neg_control[exp_plate] = row.well

        # check if there's negative control for every sample
        for row in self.df.itertuples():
            exp_plate = f'{row.experiment}_{row.plate}'
            assert self.neg_control[exp_plate] != ''


        # if we use augmentations, create group transform
        self.augmentor = None

        if augmentor is not None:
            targets = {'image1': 'image', 'image2': 'image', 'image3': 'image'}
            self.augmentor = albu.Compose(augmentor, p=1,
                                          additional_targets=targets)

        rep = 2 if config.model.type == 'SiameseModel' else 1
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.02645905, 0.05782904, 0.0412261,
                                       0.04099516, 0.02156723, 0.03849208] * rep,
                                  std=[0.03776616, 0.05301339, 0.03087561,
                                       0.03875584, 0.02616441, 0.03077043] * rep)
        ])

    def _load_image(self, path: str) -> np.array:
        ''' Loads image into np.array with optional resize. '''
        image = Image.open(path)

        if self.image_size != 0:
            image = image.resize((self.image_size, self.image_size), Image.LANCZOS)

        return np.array(image)

    def _load_images(self, index: int) -> np.array:
        site, df_index = index // self.df.shape[0], index % self.df.shape[0]
        exp, plate, well = self.df.iloc[df_index, 1], self.df.iloc[df_index, 2], \
                           self.df.iloc[df_index, 3]
        layers = []

        for channel in range(self.num_channels):
            filename = f'{exp}/Plate{plate}/{well}_s{site+1}_w{channel+1}.png'
            layers.append(self._load_image(os.path.join(self.path, filename)))

        if self.siamese_input:
            neg_ctl_well = self.neg_control[f'{exp}_{plate}']

            for channel in range(self.num_channels):
                filename = f'{exp}/Plate{plate}/{neg_ctl_well}_s{site+1}_w{channel+1}.png'
                layers.append(self._load_image(os.path.join(self.path, filename)))

        image = np.dstack(layers)
        return image

    def _transform_images(self, image: np.array, index: int) -> torch.Tensor:
        if self.debug_save:
            print('image', index)
            os.makedirs(f'debug_images_{self.version}/', exist_ok=True)

            if self.siamese_input:
                sirna_img = image[:, :, :6]
                ctl_img = image[:, :, 6:]

                sirna_img = convert_tensor_to_rgb(sirna_img)
                sirna_img = sirna_img.astype(np.uint8)
                sirna_img = Image.fromarray(sirna_img)

                ctl_img = convert_tensor_to_rgb(ctl_img)
                ctl_img = ctl_img.astype(np.uint8)
                ctl_img = Image.fromarray(ctl_img)

                sirna_img.save(f'debug_images_{self.version}/{index}_before_sirna.png')
                ctl_img.save(f'debug_images_{self.version}/{index}_before_control.png')

        ''' Applies augmentations, if any. '''
        if self.augmentor is not None:
            if self.num_channels == 3:
                image = self.augmentor(image=image)['image']
            elif self.num_channels == 6:
                if not self.siamese_input:
                    image0, image1 = image[:, :, :3], image[:, :, 3:]
                    results = self.augmentor(image=image0, image1=image1)
                    image0, image1 = results['image'], results['image1']
                    image = np.dstack([image0, image1])
                else:
                    image0, image1, image2, image3 = image[:, :, :3], image[:, :, 3:6], \
                                                     image[:, :, 6:9], image[:, :, 9:]
                    results = self.augmentor(image=image0, image1=image1,
                                             image2=image2, image3=image3)
                    image0, image1, image2, image3 = results['image'], results['image1'], \
                                                     results['image2'], results['image3']

                    image = np.dstack([image0, image1, image2, image3])
            else:
                raise RuntimeError('unsupported number of channels')

        if self.debug_save:
            os.makedirs(f'debug_images_{self.version}/', exist_ok=True)

            if self.siamese_input:
                sirna_img = image[:, :, :6]
                ctl_img = image[:, :, 6:]

                sirna_img = convert_tensor_to_rgb(sirna_img)
                sirna_img = sirna_img.astype(np.uint8)
                sirna_img = Image.fromarray(sirna_img)

                ctl_img = convert_tensor_to_rgb(ctl_img)
                ctl_img = ctl_img.astype(np.uint8)
                ctl_img = Image.fromarray(ctl_img)

                sirna_img.save(f'debug_images_{self.version}/{index}_sirna.png')
                ctl_img.save(f'debug_images_{self.version}/{index}_control.png')

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
