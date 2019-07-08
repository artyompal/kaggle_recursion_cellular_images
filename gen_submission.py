#!/usr/bin/python3.6

import os
import re
import sys
import yaml

from glob import glob
from collections import OrderedDict
from typing import List

import numpy as np
import pandas as pd

from tqdm import tqdm
from metrics import F_score
from debug import dprint


IN_KERNEL = os.environ.get('KAGGLE_WORKING_DIR') is not None
INPUT_PATH = '../input/imet-2019-fgvc6/' if IN_KERNEL else 'data/'
NUM_CLASSES = 1103

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f'usage: {sys.argv[0]} predict.npy')
        sys.exit()

    predict = np.load(sys.argv[1])
    dprint(predict.shape)

    # sz = predict.shape[0] // 2
    # predict = np.mean(np.dstack([predict[:sz], predict[sz:]]), axis=2)
    labels = np.argmax(predict, axis=1)

    sub = pd.read_csv(INPUT_PATH + 'sample_submission.csv')
    assert sub.shape[0] == predict.shape[0]

    dprint(len(labels))
    print('labels')
    print(np.array(labels))

    sub['sirna'] = labels
    filename = os.path.splitext(os.path.basename(sys.argv[1]))[0]
    if filename.startswith('level1_test_'):
        filename = filename[12:]
    sub.to_csv(filename + '.csv', index=False)
