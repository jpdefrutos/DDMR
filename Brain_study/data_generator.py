import warnings
import time
import numpy as np
from tensorflow import keras
import os
import h5py
import random
from PIL import Image
import nibabel as nib
from nilearn.image import resample_img
from skimage.exposure import equalize_adapthist
from scipy.ndimage import zoom
import tensorflow as tf

import DeepDeformationMapRegistration.utils.constants as C
from DeepDeformationMapRegistration.utils.operators import min_max_norm
from DeepDeformationMapRegistration.utils.misc import segmentation_cardinal_to_ohe
from DeepDeformationMapRegistration.utils.thin_plate_splines import ThinPlateSplines
from voxelmorph.tf.layers import SpatialTransformer
from Brain_study.format_dataset import SEGMENTATION_NR2LBL_LUT, SEGMENTATION_LBL2NR_LUT

from tensorflow.python.keras.preprocessing.image import Iterator
from tensorflow.python.keras.utils import Sequence
import sys

from collections import defaultdict

from Brain_study.format_dataset import SEGMENTATION_LOC

#import concurrent.futures
#import multiprocessing as mp
import time

class BatchGenerator:
    def __init__(self,
                 directory,
                 batch_size,
                 shuffle=True,
                 split=0.7,
                 combine_segmentations=True,
                 labels=['all'],
                 directory_val=None,
                 return_isotropic_shape=False):
        self.file_directory = directory
        self.batch_size = batch_size
        self.combine_segmentations = combine_segmentations
        self.labels = labels
        self.shuffle = shuffle
        self.split = split
        self.return_isotropic_shape=return_isotropic_shape

        if directory_val is None:
            self.file_list = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(('h5', 'hd5'))]
            random.shuffle(self.file_list) if self.shuffle else self.file_list.sort()
            self.num_samples = len(self.file_list)
            training_samples = self.file_list[:int(self.num_samples * self.split)]

            self.train_iter = BatchIterator(training_samples, batch_size, shuffle, combine_segmentations, labels, return_isotropic_shape=return_isotropic_shape)
            if self.split < 1.:
                validation_samples = list(set(self.file_list) - set(training_samples))
                self.validation_iter = BatchIterator(validation_samples, batch_size, shuffle, combine_segmentations, ['all'],
                                                     validation=True, return_isotropic_shape=return_isotropic_shape)
            else:
                self.validation_iter = None
        else:
            training_samples = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(('h5', 'hd5'))]
            random.shuffle(training_samples) if self.shuffle else training_samples.sort()

            validation_samples = [os.path.join(directory_val, f) for f in os.listdir(directory_val) if f.endswith(('h5', 'hd5'))]
            random.shuffle(validation_samples) if self.shuffle else validation_samples.sort()

            self.num_samples = len(training_samples) + len(validation_samples)
            self.file_list = training_samples + validation_samples

            self.train_iter = BatchIterator(training_samples, batch_size, shuffle, combine_segmentations, labels)
            self.validation_iter = BatchIterator(validation_samples, batch_size, shuffle, combine_segmentations, labels,
                                                 validation=True)

    def get_train_generator(self):
        return self.train_iter

    def get_validation_generator(self):
        if self.validation_iter is not None:
            return self.validation_iter
        else:
            raise ValueError('No validation iterator. Split must be < 1.0')

    def get_file_list(self):
        return self.file_list

    def get_data_shape(self):
        return self.train_iter.get_data_shape()


ALL_LABELS = {2., 3., 4., 6., 8., 9., 11., 12., 14., 16., 20., 23., 29., 33., 39., 53., 67., 76., 102., 203., 210.,
              211., 218., 219., 232., 233., 254., 255.}
ALL_LABELS_LOC = {label: loc for label, loc in zip(ALL_LABELS, range(0, len(ALL_LABELS)))}


class BatchIterator(Sequence):
    def __init__(self, file_list, batch_size, shuffle, combine_segmentations=True, labels=['all'],
                                            zero_grads=[64, 64, 64, 3], validation=False, sequential_labels=True,
                 return_isotropic_shape=False, **kwargs):
        # super(BatchIterator, self).__init__(n=len(file_list),
        #                                     batch_size=batch_size,
        #                                     shuffle=shuffle,
        #                                     seed=None,
        #                                     **kwargs)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.file_list = file_list
        self.combine_segmentations = combine_segmentations
        self.labels = labels
        self.zero_grads = np.zeros(zero_grads)
        self.idx_list = np.arange(0, len(self.file_list))
        self.validation = validation
        self.sequential_labels = sequential_labels
        self.return_isotropic_shape = return_isotropic_shape
        self._initialize()
        self.shuffle_samples()

    def _initialize(self):
        if (isinstance(self.labels[0], str) and self.labels[0].lower() != 'none'):
            if self.labels[0] != 'all':
                # Labels are tag names. Convert to numeric and check if the expected labels are in sequence or not
                self.labels = [SEGMENTATION_LBL2NR_LUT[lbl] for lbl in self.labels]
                if not self.sequential_labels:
                    self.labels = [SEGMENTATION_LOC[lbl] for lbl in self.labels]
                    self.labels_dict = lambda x: SEGMENTATION_LOC[x] if x in self.labels else 0
                else:
                    self.labels_dict = lambda x: ALL_LABELS_LOC[x] if x in self.labels else 0
            else:
                # Use all labels
                if self.sequential_labels:
                    self.labels = list(set(SEGMENTATION_LOC.values()))
                    self.labels_dict = lambda x: SEGMENTATION_LOC[x] if x else 0
                else:
                    self.labels = list(ALL_LABELS)
                    self.labels_dict = lambda x: ALL_LABELS_LOC[x] if x in self.labels else 0
        elif hasattr(self.labels[0], 'lower') and self.labels[0].lower() == 'none':
            # self.labels = list()
            self.labels_dict = dict()
        else:
            assert np.all([isinstance(lbl, (int, float)) for lbl in self.labels]), "Labels must be a str, int or float"
            # Nothing to do, the self.labels contains a list of numbers

        self.num_steps = len(self.file_list) // self.batch_size + (1 if len(self.file_list) % self.batch_size else 0)
        #self.executor = concurrent.futures.ProcessPoolExecutor(max_workers=self.batch_size)
        #self.mp_pool = mp.Pool(self.batch_size)

        with h5py.File(self.file_list[0], 'r') as f:
            self.image_shape = list(f['image'][:].shape)
            self.segm_shape = self.image_shape.copy()
            self.segm_shape[-1] = len(self.labels) if not self.combine_segmentations else 1

        self.batch_shape = self.image_shape.copy()
        self.batch_shape[-1] = self.image_shape[-1] + self.segm_shape[-1]

    def shuffle_samples(self):
        np.random.shuffle(self.idx_list)

    def __len__(self):
        return self.num_steps

    def _filter_segmentations(self, segm, segm_labels):
        if self.combine_segmentations:
            # TODO
            warnings.warn('Cannot select labels when combine_segmentations options is active')
        if self.labels[0] != 'all':
            if set(self.labels).issubset(set(segm_labels)):
                # If labels in self.labels are in segm
                idx = [ALL_LABELS_LOC[l] for l in self.labels]
                segm = segm[..., idx]
            else:
                # Else we have to collect those labels that are contained and complete with zeros
                idx = [ALL_LABELS_LOC[l] for l in list(set(self.labels).intersection(set(segm_labels)))]
                aux = segm.copy()
                segm = np.zeros(self.segm_shape)
                segm[..., :len(idx)] = aux[..., idx]
                # TODO: leave the zero-ed segmentations before or after the selected labels based on the order
        return segm

    def _load_sample(self, file_path):
        with h5py.File(file_path, 'r') as f:
            img = f['image'][:]
            segm = f['segmentation'][:]
            isot_shape = f['isotropic_shape'][:]

        if not self.combine_segmentations:
            if self.sequential_labels:
                # TODO: I am assuming I want all the labels
                segm = np.squeeze(np.eye(len(self.labels))[segm])
            else:
                lbls_list = list(ALL_LABELS) if self.labels[0] == 'all' else self.labels
                segm = segmentation_cardinal_to_ohe(segm, lbls_list)  # Filtering is done here
            #     aux = np.zeros(self.segm_shape)
            #     aux[..., :segm.shape[-1]] = segm     # Ensure the same shape in case there are missing labels in aux
            #     segm = aux
            # TODO: selection label segm = aux[..., self.labels]  but:
            #       what if aux does not have a label in self.labels??

        img = np.asarray(img, dtype=np.float32)
        segm = np.asarray(segm, dtype=np.float32)
        if not isinstance(self.labels[0], str) or self.labels[0].lower() != 'none' or self.validation:  # I expect to ask for the segmentations during val
            # segm = self._filter_segmentations(segm, segm_labels)

            if self.validation:
                ret_val = np.concatenate([img, segm], axis=-1), (img, segm, self.zero_grads), isot_shape
            else:
                ret_val = np.concatenate([img, segm], axis=-1), (img, self.zero_grads), isot_shape
        else:
            ret_val = img, (img, self.zero_grads), isot_shape
        return ret_val

    def __getitem__(self, idx):
        in_batch = list()
        isotropic_shape = list()
        # out_batch = list()

        batch_idxs = self.idx_list[idx * self.batch_size:(idx + 1) * self.batch_size]
        file_list = [self.file_list[i] for i in batch_idxs]
        # if self.batch_size > 1:
        #     # Multiprocessing to speed up laoding
        #
        #     for ret in self.executor.map(self._load_sample, file_list):
        #         b, i = ret
        #         in_batch.append(b)
        #         # out_batch.append(i)
        # else:
            # No need for multithreading, we are loading a single file
        # in_batch = np.zeros([self.batch_size] + self.batch_shape, dtype=np.float32)
        for batch_idx, f in enumerate(file_list):
            b, i, isot_shape = self._load_sample(f)
            # in_batch[batch_idx, :, :, :, :] = b
            if self.return_isotropic_shape:
                isotropic_shape.append(isot_shape)
            in_batch.append(b)
            # out_batch.append(i)

        in_batch = np.asarray(in_batch, dtype=np.float32)
        ret_val = (in_batch, in_batch)
        if self.return_isotropic_shape:
            isotropic_shape = np.asarray(isotropic_shape, dtype=np.int)
            ret_val += (isotropic_shape,)
        # out_batch = np.asarray(out_batch)
        return ret_val

    def __iter__(self):
        """Create a generator that iterate over the Sequence."""
        for item in (self[i] for i in range(len(self))):
            yield item

    def get_data_shape(self):
        return self.batch_shape, self.image_shape, self.segm_shape

    def on_epoch_end(self):
        self.shuffle_samples()

    def get_segmentation_labels(self):
        if self.combine_segmentations:
            labels = [1]
        else:
            labels = self.labels
        return labels






'''
def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size


class BatchIterator(Iterator):
    def __init__(self, generator, file_list, input_shape, output_shape, batch_size, shuffle, all_files_in_batch):
        self.file_list = file_list
        self.generator = generator
        self.input_shape = input_shape
        self.nr_of_inputs = len(input_shape)
        self.output_shape = output_shape
        self.nr_of_outputs = len(output_shape)
        self.all_files_in_batch = all_files_in_batch
        self.preload_to_memory = False
        self.file_cache = {}
        self.max_cache_size = 10*1024
        self.verbose = False
        if self.preload_to_memory:
            for filename, file_index in self.file_list:
                file = h5py.File(filename, 'r')
                inputs = {}
                for name, data in file['input'].items():
                    inputs[name] = np.copy(data)
                self.file_cache[filename] = {'input': inputs, 'output': np.copy(file['output'])}
                file.close()
                if get_size(self.file_cache) / (1024*1024) >= self.max_cache_size:
                    print('File cache has reached limit of', self.max_cache_size, 'MBs')
                    break
        epoch_size = len(file_list)
        if all_files_in_batch:
            epoch_size = len(file_list) * 10
        super(BatchIterator, self).__init__(epoch_size, batch_size, shuffle, None)

    def _get_sample(self, index):
        filename, file_index = self.file_list[index]
        if filename in self.file_cache:
            file = self.file_cache[filename]
        else:
            file = h5py.File(filename, 'r')
        inputs = []
        outputs = []
        for name, data in file['input'].items():
            inputs.append(data[file_index, :])
        for name, data in file['output'].items():
            if len(data.shape) > 1:
                outputs.append(data[file_index, :])
            else:
                outputs.append(data[file_index])
        #outputs.append(file['output'][file_index, :]) # TODO fix
        if filename not in self.file_cache:
            file.close()
        return inputs, outputs

    def _get_random_sample_in_file(self, file_index):
        filename = self.file_list[file_index]
        file = h5py.File(filename, 'r')
        x = file['output/0']
        sample = np.random.randint(0, x.shape[0])
        #print('Sampling image', sample, 'from file', filename)
        inputs = []
        outputs = []
        for name, data in file['input'].items():
            inputs.append(data[sample, :])
        for name, data in file['output'].items():
            outputs.append(data[file_index, :])
        #outputs.append(file['output'][sample, :]) # TODO FIX output
        file.close()
        return inputs, outputs

    def next(self):

        with self.lock:
            index_array = next(self.index_generator)

        #print(len(index_array))
        return self._get_batches_of_transformed_samples(index_array)

    def _get_batches_of_transformed_samples(self, index_array):
        start_batch = time.time()
        batches_x = []
        batches_y = []
        for input_index in range(self.nr_of_inputs):
            batches_x.append(np.zeros(tuple([len(index_array)] + list(self.input_shape[input_index]))))
        for output_index in range(self.nr_of_outputs):
            batches_y.append(np.zeros(tuple([len(index_array)] + list(self.output_shape[output_index]))))

        timings_sampling = np.zeros((len(index_array,)))
        timings_transform = np.zeros((len(index_array,)))
        for batch_index, sample_index in enumerate(index_array):
            # Have to copy here in order to not modify original data
            start = time.time()
            if self.all_files_in_batch:
                input, output = self._get_random_sample_in_file(batch_index)
            else:
                input, output = self._get_sample(sample_index)
            timings_sampling[batch_index] = time.time() - start
            start = time.time()
            input, output = self.generator.transform(input, output)
            timings_transform[batch_index] = time.time() - start

            #print('inputs', self.nr_of_inputs, len(input))
            for input_index in range(self.nr_of_inputs):
                batches_x[input_index][batch_index] = input[input_index]
            for output_index in range(self.nr_of_outputs):
                batches_y[output_index][batch_index] = output[output_index]

        elapsed = time.time() - start_batch
        if self.verbose:
            print('Time to prepare batch:', round(elapsed,3), 'seconds')
            print('Sampling mean:', round(timings_sampling.mean(), 3), 'seconds')
            print('Transform mean:', round(timings_transform.mean(), 3), 'seconds')

        return batches_x, batches_y


CLASSIFICATION = 'classification'
SEGMENTATION = 'segmentation'


class BatchGenerator():
    def __init__(self, filelist, all_files_in_batch=False):
        self.methods = []
        self.args = []
        self.crop_width_to = None
        self.image_list = []
        self.input_shape = []
        self.output_shape = []
        self.all_files_in_batch = all_files_in_batch
        self.transforms = []

        if all_files_in_batch:
            file = h5py.File(filelist[0], 'r')
            for name, data in file['input'].items():
                self.input_shape.append(data.shape[1:])
            for name, data in file['output'].items():
                self.output_shape.append(data.shape[1:])
            # TODO fix
            #self.output_shape.append(file['output'].shape[1:])
            file.close()
            self.image_list = filelist
            return

        # Go through filelist
        first = True
        for filename in filelist:
            samples = None
            # Open file to see how many samples it has
            file = h5py.File(filename, 'r')
            for name, data in file['input'].items():
                if first:
                    self.input_shape.append(data.shape[1:])
                samples = data.shape[0]
            # TODO fix
            for name, data in file['output'].items():
                if first:
                    self.output_shape.append(data.shape[1:])
                if samples != data.shape[0]:
                    raise ValueError()
            #self.output_shape.append(file['output'].shape[1:])
            if len(self.output_shape) == 1:
                self.problem_type = CLASSIFICATION
            else:
                self.problem_type = SEGMENTATION

            file.close()
            if samples is None:
                raise ValueError()
            # Append a tuple to image_list for each image consisting of filename and index
            print(filename, samples)
            for i in range(samples):
                self.image_list.append((filename, i))
            first = False

        print('Image generator with', len(self.image_list), ' image samples created')

    def flow(self, batch_size, shuffle=True):

        return BatchIterator(self, self.image_list, self.input_shape, self.output_shape, batch_size, shuffle, self.all_files_in_batch)

    def transform(self, inputs, outputs):
        #input = input.astype(np.float32) # TODO
        #output = output.astype(np.float32)
        for input_indices, output_indices, transform in self.transforms:
            transform.randomize()
            inputs, outputs = transform.transform_all(inputs, outputs, input_indices, output_indices)
        return inputs, outputs

    def add_transform(self, input_indices: Union[int, List[int], None], output_indices: Union[int, List[int], None], transform: Transform):
        if type(input_indices) is int:
            input_indices = [input_indices]
        if type(output_indices) is int:
            output_indices = [output_indices]

        self.transforms.append((
            input_indices,
            output_indices,
            transform
        ))

    def get_size(self):
        if self.all_files_in_batch:
            return 10*len(self.image_list)
        else:
            return len(self.image_list)

'''
