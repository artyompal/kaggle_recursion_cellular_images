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


SAVE_DEBUG_IMAGES = False


class ImageDataset(torch.utils.data.Dataset): # type: ignore
    def __init__(self, dataframe: pd.DataFrame, controls_df: pd.DataFrame,
                 mode: str, config: Any, num_ttas: int = 1, augmentor: Any = None) -> None:
        print(f'creating data_loader for {config.version} in mode={mode}')
        assert mode in ['train', 'val', 'test']

        self.df = dataframe
        self.mode = mode

        self.version = config.version
        self.path = config.data.train_dir if mode != 'test' else config.data.test_dir
        self.num_classes = config.model.num_classes
        self.image_size = config.model.image_size
        self.num_ttas = num_ttas
        self.num_channels = config.model.num_channels


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
            targets = {'image1': 'image'}
            self.augmentor = albu.Compose(augmentor, p=1,
                                          additional_targets=targets)

        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * 6, std=[0.5] * 6)
        ])

        # if 'ception' in config.model.arch:
        #     self.transforms = transforms.Compose([
        #         transforms.ToTensor(),
        #         transforms.Normalize(mean=[0.5, 0.5, 0.5],
        #                               std=[0.5, 0.5, 0.5])
        #     ])
        # else:
        #     self.transforms = transforms.Compose([
        #         transforms.ToTensor(),
        #         transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                               std=[0.229, 0.224, 0.225]),
        #     ])

    def _load_image(self, path: str) -> np.array:
        ''' Loads image into np.array with optional resize. '''
        image = Image.open(path)

        if self.image_size != 0:
            image = image.resize((self.image_size, self.image_size), Image.LANCZOS)

        return np.array(image)

    def _load_images(self, index: int) -> np.array:
        ''' Loads two N-channel images and stacks them. '''
        df_index, site = index // 2, index % 2
        exp, plate, well = self.df.iloc[df_index, 1], self.df.iloc[df_index, 2], \
                           self.df.iloc[df_index, 3],
        layers = []

        for channel in range(self.num_channels):
            filename = f'{exp}/Plate{plate}/{well}_s{site+1}_w{channel+1}.png'
            layers.append(self._load_image(os.path.join(self.path, filename)))

        neg_ctl_well = self.neg_control[f'{exp}_{plate}']

        for channel in range(self.num_channels):
            filename = f'{exp}/Plate{plate}/{neg_ctl_well}_s{site+1}_w{channel+1}.png'
            layers.append(self._load_image(os.path.join(self.path, filename)))

        image = np.dstack(layers)
        return image

    def _transform_images(self, image: np.array, index: int) -> torch.Tensor:
        ''' Applies augmentations, if any. '''
        if self.augmentor is not None:
            image0, image1 = image[:, :, :self.num_channels], image[:, :, self.num_channels:]
            results = self.augmentor(image=image0, image1=image1)
            image0, image1 = results['image'], results['image1']
            image = np.dstack([image0, image1])

        if SAVE_DEBUG_IMAGES:
            os.makedirs(f'../debug_images_{self.version}/', exist_ok=True)
            Image.fromarray(image).save(f'../debug_images_{self.version}/{index}.png')

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
            targets = int(self.df.sirna.values[index // 2])
            return image, targets
        else:
            return image

    def __len__(self) -> int:
        ''' We have two sets of images per well. '''
        return self.df.shape[0] * 2
