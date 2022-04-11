import os
import argparse
import random
import warnings

import math
from shutil import copyfile
from tqdm import tqdm
import concurrent.futures
import numpy as np


def copy_file(s_d):
    s, d = s_d
    file_name = os.path.split(s)[-1]
    copyfile(s, os.path.join(d, file_name))
    return int(os.path.exists(d))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', '-t', type=float, default=.70, help='Train percentage. Default: 0.70')
    parser.add_argument('--validation', '-v', type=float, default=0.15, help='Validation percentage. Default: 0.15')
    parser.add_argument('--test', '-s', type=float, default=0.15, help='Test percentage. Default: 0.15')
    parser.add_argument('-d', '--dir', type=str, help='Directory where the data is')
    parser.add_argument('-f', '--format', type=str, help='Format of the data files. Default: h5', default='h5')
    parser.add_argument('-r', '--random', type=bool, help='Randomly split the dataset or not. Default: True', default=True)

    args = parser.parse_args()

    assert args.train + args.validation + args.test == 1.0, 'Train+Validation+Test != 1 (100%)'

    file_set = [os.path.join(args.dir, f) for f in os.listdir(args.dir) if f.endswith(args.format)]
    random.shuffle(file_set) if args.random else file_set.sort()

    num_files = len(file_set)
    num_validation = math.floor(num_files * args.validation)
    num_test = math.floor(num_files * args.test)
    num_train = num_files - num_test - num_validation

    dataset_root, dataset_name = os.path.split(args.dir)
    dst_train = os.path.join(dataset_root, 'SPLIT_'+dataset_name, 'train_set')
    dst_validation = os.path.join(dataset_root, 'SPLIT_'+dataset_name, 'validation_set')
    dst_test = os.path.join(dataset_root, 'SPLIT_'+dataset_name, 'test_set')

    print('OUTPUT INFORMATION\n=============')
    print('Train:\t\t{}'.format(num_train))
    print('Validation:\t{}'.format(num_validation))
    print('Test:\t\t{}'.format(num_test))
    print('Num. samples\t{}'.format(num_files))
    print('Path:\t\t', os.path.join(dataset_root, 'SPLIT_'+dataset_name))

    dest = [dst_train] * num_train + [dst_validation] * num_validation + [dst_test] * num_test

    os.makedirs(dst_train, exist_ok=True)
    os.makedirs(dst_validation, exist_ok=True)
    os.makedirs(dst_test, exist_ok=True)

    progress_bar = tqdm(zip(file_set, dest), desc='Copying files', total=num_files)
    with concurrent.futures.ProcessPoolExecutor(max_workers=10) as ex:
        results = list(tqdm(ex.map(copy_file, zip(file_set, dest)), desc='Copying files', total=num_files))

    num_copies = np.sum(results)
    if num_copies == num_files:
        print('Done successfully')
    else:
        warnings.warn('Missing files: {}'.format(num_files - num_copies))

