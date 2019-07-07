''' Data loaders for training & validation. '''

import os
import random
import albumentations as albu
import numpy as np

from PIL import Image


class RandomRectCrop(albu.DualTransform):
    def __init__(self, min_area, max_area, min_ratio, max_ratio, image_size, input_size,
                 always_apply=True, p=1.0) -> None:
        super().__init__(always_apply, p)
        assert min_area >= 0 and min_area <= max_area and max_area <= 1.0
        assert min_ratio > 0 and min_ratio <= max_ratio

        self.min_area = min_area
        self.max_area = max_area
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.image_size = image_size
        self.input_size = input_size

    def apply(self, img, y=0, x=0, h=0, w=0, **params):
        array = array[y : y + h, x : x + w]

        img = Image.fromarray(array)
        img = img.resize((self.input_size, self.input_size), Image.LANCZOS)
        return np.array(img)

    def get_params(self):
        area = random.uniform(self.min_area, self.max_area)
        ratio = random.uniform(self.min_ratio, self.max_ratio)

        h = min(int(((area / ratio) ** 0.5) * self.image_size), self.image_size)
        w = min(int(((area * ratio) ** 0.5) * self.image_size), self.image_size)

        return {'y': int(random.uniform(0, self.image_size - h)),
                'x': int(random.uniform(0, self.image_size - w)),
                'h': h,
                'w': w}

    def apply_to_bbox(self, bbox, **params):
        return bbox

    def apply_to_keypoint(self, keypoint, **params):
        return keypoint
