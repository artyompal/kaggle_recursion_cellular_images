#!/usr/bin/python3.6

import os
import re
import sys
import yaml

from glob import glob
from collections import OrderedDict
from typing import List
from collections import Counter

import numpy as np
import pandas as pd

from tqdm import tqdm
from metrics import F_score
from debug import dprint


IN_KERNEL = os.environ.get('KAGGLE_WORKING_DIR') is not None
INPUT_PATH = '../input/imet-2019-fgvc6/' if IN_KERNEL else 'data/'

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f'usage: {sys.argv[0]} predict.npy')
        sys.exit()

    predict = np.load(sys.argv[1])
    dprint(predict.shape)

    # Greedy algorithm: iteratively take the most confident prediction,
    # as long as its class is not full yet.

    num_rows, num_classes = predict.shape
    max_predicts = int(np.ceil(num_rows / num_classes))
    min_predicts = max_predicts - 1
    counts = np.zeros(num_classes, dtype=int)
    results = -np.ones(num_rows)

    for i in tqdm(range(num_rows)):
        most_likely_idx = np.argmax(predict.flatten())
        most_likely_sample = most_likely_idx // num_classes
        most_likely_cls = most_likely_idx % num_classes

        predict[most_likely_sample, :] = 0
        counts[most_likely_cls] += 1
        results[most_likely_sample] = most_likely_cls

        if counts[most_likely_cls] == max_predicts:
            predict[:, most_likely_cls] = 0
        elif counts[most_likely_cls] == max_predicts - 1:
            # stop if there are just enough predictions left for other classes
            avail_predicts = num_rows - i - 1
            samples_needed = np.sum(min_predicts - counts[counts < min_predicts])
            # assert samples_needed <= avail_predicts

            if samples_needed > avail_predicts:
                dprint(samples_needed)
                dprint(avail_predicts)

                assert False

            if samples_needed == avail_predicts:
                predict[:, most_likely_cls] = 0

    assert all(results != -1)
    dprint(counts)

    sub = pd.read_csv(INPUT_PATH + 'sample_submission.csv')
    assert sub.shape[0] == predict.shape[0]

    dprint(len(results))
    print('results')
    print(np.array(results))

    sub['sirna'] = results
    filename = os.path.splitext(os.path.basename(sys.argv[1]))[0]
    if filename.startswith('level1_test_'):
        filename = filename[12:]
    sub.to_csv(filename + '.csv', index=False)
