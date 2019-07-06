#!/usr/bin/python3.6
''' Crops and downsamples image sets. Saves result into jpeg. '''

import multiprocessing
import os
import sys

import numpy as np

from glob import glob
from tqdm import tqdm
from PIL import Image

from rxrx1.rxrx.io import convert_tensor_to_rgb
from debug import dprint


def convert_image(path: str) -> int:
    ''' Loads image file, crops it and resizes into the proper resolution. '''
    try:
        img = Image.open(path)

        if img.size != (image_size, image_size):
            img = img.resize((image_size, image_size), resample=Image.LANCZOS)

        layers = [np.array(img)]

        for plane in range(2, 7):
            mod_path = path.replace('_w1.', f'_w{plane}.')
            img = Image.open(mod_path)

            if img.size != (image_size, image_size):
                img = img.resize((image_size, image_size), resample=Image.LANCZOS)

            layers.append(np.array(img))

        img = np.dstack(layers)

        img = convert_tensor_to_rgb(img)
        img = img.astype(np.uint8)

        assert path.startswith(source_dir)
        dest = os.path.join(dest_dir, path[len(source_dir) + 1:])
        dest = dest.replace('_w1.', '_rgb.')

        os.makedirs(os.path.dirname(dest), exist_ok=True)
        Image.fromarray(img).save(dest)
        return 0
    except IOError:
        print(f'{path}: read/write error')
        return 1

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('Usage: %s dest_dir source_dir size' % sys.argv[0])
        sys.exit(0)

    dest_dir, source_dir, image_size = sys.argv[1], sys.argv[2], int(sys.argv[3])
    file_list = glob(os.path.join(source_dir, '**/*_w1.png'), recursive=True)

    pool = multiprocessing.Pool()
    num_errors = sum(tqdm(pool.imap_unordered(convert_image, file_list),
                        total=len(file_list)))
    print('total number of errors:', num_errors)
