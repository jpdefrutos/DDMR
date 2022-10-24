import os
import argparse
import random
import warnings

import math
from shutil import copyfile, move
from tqdm import tqdm
import concurrent.futures
import numpy as np


def copy_file_fnc(s_d):
    s, d = s_d
    file_name = os.path.split(s)[-1]
    copyfile(s, os.path.join(d, file_name))
    return int(os.path.exists(d))


def move_file_fnc(s_d):
    s, d = s_d
    file_name = os.path.split(s)[-1]
    move(s, os.path.join(d, file_name))
    return int(os.path.exists(d))


def split(train_perc: float=0.7,
          validation_perc: float=0.15,
          test_perc: float=0.15,
          data_dir: str='',
          file_format: str='h5',
          random_split: bool=True,
          move_files: bool=False):
    assert train_perc + validation_perc + test_perc == 1.0, 'Train+Validation+Test != 1 (100%)'
    assert train_perc > 0 and test_perc > 0, 'Train and test percentages must be greater than zero'

    file_set = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(file_format)]
    random.shuffle(file_set) if random_split else file_set.sort()

    num_files = len(file_set)
    num_validation = math.floor(num_files * validation_perc)
    num_test = math.floor(num_files * test_perc)
    num_train = num_files - num_test - num_validation

    dataset_root, dataset_name = os.path.split(data_dir)
    dst_train = os.path.join(dataset_root, 'SPLIT_' + dataset_name, 'train_set')
    dst_validation = os.path.join(dataset_root, 'SPLIT_' + dataset_name, 'validation_set')
    dst_test = os.path.join(dataset_root, 'SPLIT_' + dataset_name, 'test_set')

    print('OUTPUT INFORMATION\n=============')
    print('Train:\t\t{}'.format(num_train))
    print('Validation:\t{}'.format(num_validation))
    print('Test:\t\t{}'.format(num_test))
    print('Num. samples\t{}'.format(num_files))
    print('Path:\t\t', os.path.join(dataset_root, 'SPLIT_' + dataset_name))

    dest = [dst_train] * num_train + [dst_validation] * num_validation + [dst_test] * num_test

    os.makedirs(dst_train, exist_ok=True)
    os.makedirs(dst_validation, exist_ok=True)
    os.makedirs(dst_test, exist_ok=True)

    progress_bar = tqdm(zip(file_set, dest), desc='Copying files', total=num_files)
    operation = move_file_fnc if move_files else copy_file_fnc
    desc = 'Moving files' if move_files else 'Copying files'
    with concurrent.futures.ProcessPoolExecutor(max_workers=10) as ex:
        results = list(tqdm(ex.map(operation, zip(file_set, dest)), desc=desc, total=num_files))

    num_copies = np.sum(results)
    if num_copies == num_files:
        print('Done successfully')
    else:
        warnings.warn('Missing files: {}'.format(num_files - num_copies))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', '-t', type=float, default=.70, help='Train percentage. Default: 0.70')
    parser.add_argument('--validation', '-v', type=float, default=0.15, help='Validation percentage. Default: 0.15')
    parser.add_argument('--test', '-s', type=float, default=0.15, help='Test percentage. Default: 0.15')
    parser.add_argument('-d', '--dir', type=str, help='Directory where the data is')
    parser.add_argument('-f', '--format', type=str, help='Format of the data files. Default: h5', default='h5')
    parser.add_argument('-r', '--random', help='Randomly split the dataset or not. Default: True', action='store_true', default=True)
    parser.add_argument('-m', '--movefiles', help='Move files. Otherwise copy. Default: False', action='store_true', default=False)

    args = parser.parse_args()

    split(args.train, args.validation, args.test, args.dir, args.format, args.random, args.movefiles)


