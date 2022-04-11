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
from DeepDeformationMapRegistration.utils.thin_plate_splines import ThinPlateSplines
from voxelmorph.tf.layers import SpatialTransformer


class DataGeneratorManager(keras.utils.Sequence):
    def __init__(self, dataset_path, batch_size=32, shuffle=True,
                 num_samples=None, validation_split=None, validation_samples=None, clip_range=[0., 1.],
                 input_labels=[C.H5_MOV_IMG, C.H5_FIX_IMG], output_labels=[C.H5_FIX_IMG, 'zero_gradient']):
        # Get the list of files
        self.__list_files = self.__get_dataset_files(dataset_path)
        self.__list_files.sort()
        self.__dataset_path = dataset_path
        self.__shuffle = shuffle
        self.__total_samples = len(self.__list_files)
        self.__validation_split = validation_split
        self.__clip_range = clip_range
        self.__batch_size = batch_size

        self.__validation_samples = validation_samples

        self.__input_labels = input_labels
        self.__output_labels = output_labels

        if num_samples is not None:
            self.__num_samples = self.__total_samples if num_samples > self.__total_samples else num_samples
        else:
            self.__num_samples = self.__total_samples

        self.__internal_idxs = np.arange(self.__num_samples)

        # Split it accordingly
        if validation_split is None:
            self.__validation_num_samples = None
            self.__validation_idxs = list()
            if self.__shuffle:
                random.shuffle(self.__internal_idxs)
            self.__training_idxs = self.__internal_idxs

            self.__validation_generator = None
        else:
            self.__validation_num_samples = int(np.ceil(self.__num_samples * validation_split))
            if self.__shuffle:
                self.__validation_idxs = np.random.choice(self.__internal_idxs, self.__validation_num_samples)
            else:
                self.__validation_idxs = self.__internal_idxs[0: self.__validation_num_samples]
            self.__training_idxs = np.asarray([idx for idx in self.__internal_idxs if idx not in self.__validation_idxs])
            # Build them DataGenerators
            self.__validation_generator = DataGenerator(self, 'validation')

        self.__train_generator = DataGenerator(self, 'train')
        self.reshuffle_indices()

    @property
    def dataset_path(self):
        return self.__dataset_path

    @property
    def dataset_list_files(self):
        return self.__list_files

    @property
    def train_idxs(self):
        return self.__training_idxs

    @property
    def validation_idxs(self):
        return self.__validation_idxs

    @property
    def batch_size(self):
        return self.__batch_size

    @property
    def clip_rage(self):
        return self.__clip_range

    @property
    def shuffle(self):
        return self.__shuffle

    @property
    def input_labels(self):
        return self.__input_labels

    @property
    def output_labels(self):
        return self.__output_labels

    def get_generator_idxs(self, generator_type):
        if generator_type == 'train':
            return self.train_idxs
        elif generator_type == 'validation':
            return self.validation_idxs
        else:
            raise ValueError('Invalid generator type: ', generator_type)

    @staticmethod
    def __get_dataset_files(search_path):
        """
        Get the path to the dataset files
        :param  search_path: dir path to search for the hd5 files
        :return:
        """
        file_list = list()
        for root, dirs, files in os.walk(search_path):
            file_list.sort()
            for data_file in files:
                file_name, extension = os.path.splitext(data_file)
                if extension.lower() == '.hd5' or '.h5':
                    file_list.append(os.path.join(root, data_file))

        if not file_list:
            raise ValueError('No files found to train in ', search_path)

        print('Found {} files in {}'.format(len(file_list), search_path))
        return file_list

    def reshuffle_indices(self):
        if self.__validation_num_samples is None:
            if self.__shuffle:
                random.shuffle(self.__internal_idxs)
            self.__training_idxs = self.__internal_idxs
        else:
            if self.__shuffle:
                self.__validation_idxs = np.random.choice(self.__internal_idxs, self.__validation_num_samples)
            else:
                self.__validation_idxs = self.__internal_idxs[0: self.__validation_num_samples]
            self.__training_idxs = np.asarray([idx for idx in self.__internal_idxs if idx not in self.__validation_idxs])

            # Update the indices
            self.__validation_generator.update_samples(self.__validation_idxs)

        self.__train_generator.update_samples(self.__training_idxs)

    def get_generator(self, type='train'):
        if type.lower() == 'train':
            return self.__train_generator
        elif type.lower() == 'validation':
            if self.__validation_generator is not None:
                return self.__validation_generator
            else:
                raise Warning('No validation generator available. Set a non-zero validation_split to build one.')
        else:
            raise ValueError('Unknown dataset type "{}". Expected "train" or "validation"'.format(type))


class DataGenerator(DataGeneratorManager):
    def __init__(self, GeneratorManager: DataGeneratorManager, dataset_type='train'):
        self.__complete_list_files = GeneratorManager.dataset_list_files
        self.__list_files = [self.__complete_list_files[idx] for idx in GeneratorManager.get_generator_idxs(dataset_type)]
        self.__batch_size = GeneratorManager.batch_size
        self.__total_samples = len(self.__list_files)
        self.__clip_range = GeneratorManager.clip_rage
        self.__manager = GeneratorManager
        self.__shuffle = GeneratorManager.shuffle

        self.__num_samples = len(self.__list_files)
        self.__internal_idxs = np.arange(self.__num_samples)
        # These indices are internal to the generator, they are not the same as the dataset_idxs!!

        self.__dataset_type = dataset_type

        self.__last_batch = 0
        self.__batches_per_epoch = int(np.floor(len(self.__internal_idxs) / self.__batch_size))

        self.__input_labels = GeneratorManager.input_labels
        self.__output_labels = GeneratorManager.output_labels

    @staticmethod
    def __get_dataset_files(search_path):
        """
        Get the path to the dataset files
        :param  search_path: dir path to search for the hd5 files
        :return:
        """
        file_list = list()
        for root, dirs, files in os.walk(search_path):
            for data_file in files:
                file_name, extension = os.path.splitext(data_file)
                if extension.lower() == '.hd5':
                    file_list.append(os.path.join(root, data_file))

        if not file_list:
            raise ValueError('No files found to train in ', search_path)

        print('Found {} files in {}'.format(len(file_list), search_path))
        return file_list

    def update_samples(self, new_sample_idxs):
        self.__list_files = [self.__complete_list_files[idx] for idx in new_sample_idxs]
        self.__num_samples = len(self.__list_files)
        self.__internal_idxs = np.arange(self.__num_samples)

    def on_epoch_end(self):
        """
        To be executed at the end of each epoch. Reshuffle the assigned samples
        :return:
        """
        if self.__shuffle:
            random.shuffle(self.__internal_idxs)
        self.__last_batch = 0

    def __len__(self):
        """
        Number of batches per epoch
        :return:
        """
        return self.__batches_per_epoch

    @staticmethod
    def __build_list(data_dict, labels):
        ret_list = list()
        for label in labels:
            if label in data_dict.keys():
                if label in [C.DG_LBL_FIX_IMG, C.DG_LBL_MOV_IMG]:
                    ret_list.append(min_max_norm(data_dict[label]).astype(np.float32))
                elif label in [C.DG_LBL_FIX_PARENCHYMA, C.DG_LBL_FIX_VESSELS, C.DG_LBL_FIX_TUMOR,
                               C.DG_LBL_MOV_PARENCHYMA, C.DG_LBL_MOV_VESSELS, C.DG_LBL_MOV_TUMOR]:
                    aux = data_dict[label]
                    aux[aux > 0.] = 1.
                    ret_list.append(aux)
            elif label == C.DG_LBL_ZERO_GRADS:
                ret_list.append(np.zeros([data_dict['BATCH_SIZE'], *C.DISP_MAP_SHAPE]))
        return ret_list

    def __getitem1(self, index):
        idxs = self.__internal_idxs[index * self.__batch_size:(index + 1) * self.__batch_size]

        data_dict = self.__load_data(idxs)

        # https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit
        # A generator or keras.utils.Sequence returning (inputs, targets) or (inputs, targets, sample_weights)
        # The second element must match the outputs of the model, in this case (image, displacement map)
        inputs = self.__build_list(data_dict, self.__input_labels)
        outputs = self.__build_list(data_dict, self.__output_labels)

        return (inputs, outputs)

    def __getitem__(self, index):
        """
        Generate one batch of data
        :param index: epoch index
        :return:
        """
        return self.__getitem2(index)

    def next_batch(self):
        if self.__last_batch > self.__batches_per_epoch:
            raise ValueError('No more batches for this epoch')
        batch = self.__getitem__(self.__last_batch)
        self.__last_batch += 1
        return batch

    def __try_load(self, data_file, label, append_array=None):
        if label in self.__input_labels or label in self.__output_labels:
            # To avoid extra overhead
            try:
                retVal = data_file[label][:][np.newaxis, ...]
            except KeyError:
                # That particular label is not found in the file. But this should be known by the user by now
                retVal = None

            if append_array is not None and retVal is not None:
                return np.append(append_array, retVal, axis=0)
            elif append_array is None:
                return retVal
            else:
                return retVal  # None
        else:
            return None

    def __load_data(self, idx_list):
        """
        Build the batch with the samples in idx_list
        :param idx_list:
        :return:
        """
        if isinstance(idx_list, (list, np.ndarray)):
            fix_img = np.empty((0, ) + C.IMG_SHAPE, np.float32)
            mov_img = np.empty((0, ) + C.IMG_SHAPE, np.float32)

            fix_parench = np.empty((0, ) + C.IMG_SHAPE, np.float32)
            mov_parench = np.empty((0, ) + C.IMG_SHAPE, np.float32)

            fix_vessels = np.empty((0, ) + C.IMG_SHAPE, np.float32)
            mov_vessels = np.empty((0, ) + C.IMG_SHAPE, np.float32)

            fix_tumors = np.empty((0, ) + C.IMG_SHAPE, np.float32)
            mov_tumors = np.empty((0, ) + C.IMG_SHAPE, np.float32)

            disp_map = np.empty((0, ) + C.DISP_MAP_SHAPE, np.float32)

            fix_centroid = np.empty((0, 3))
            mov_centroid = np.empty((0, 3))

            for idx in idx_list:
                data_file = h5py.File(self.__list_files[idx], 'r')

                fix_img = self.__try_load(data_file, C.H5_FIX_IMG, fix_img)
                mov_img = self.__try_load(data_file, C.H5_MOV_IMG, mov_img)

                fix_parench = self.__try_load(data_file, C.H5_FIX_PARENCHYMA_MASK, fix_parench)
                mov_parench = self.__try_load(data_file, C.H5_MOV_PARENCHYMA_MASK, mov_parench)

                fix_vessels = self.__try_load(data_file, C.H5_FIX_VESSELS_MASK, fix_vessels)
                mov_vessels = self.__try_load(data_file, C.H5_MOV_VESSELS_MASK, mov_vessels)

                fix_tumors = self.__try_load(data_file, C.H5_FIX_TUMORS_MASK, fix_tumors)
                mov_tumors = self.__try_load(data_file, C.H5_MOV_TUMORS_MASK, mov_tumors)

                disp_map = self.__try_load(data_file, C.H5_GT_DISP, disp_map)

                fix_centroid = self.__try_load(data_file, C.H5_FIX_CENTROID, fix_centroid)
                mov_centroid = self.__try_load(data_file, C.H5_MOV_CENTROID, mov_centroid)

                data_file.close()
            batch_size = len(idx_list)
        else:
            data_file = h5py.File(self.__list_files[idx_list], 'r')

            fix_img = self.__try_load(data_file, C.H5_FIX_IMG)
            mov_img = self.__try_load(data_file, C.H5_MOV_IMG)

            fix_parench = self.__try_load(data_file, C.H5_FIX_PARENCHYMA_MASK)
            mov_parench = self.__try_load(data_file, C.H5_MOV_PARENCHYMA_MASK)

            fix_vessels = self.__try_load(data_file, C.H5_FIX_VESSELS_MASK)
            mov_vessels = self.__try_load(data_file, C.H5_MOV_VESSELS_MASK)

            fix_tumors = self.__try_load(data_file, C.H5_FIX_TUMORS_MASK)
            mov_tumors = self.__try_load(data_file, C.H5_MOV_TUMORS_MASK)

            disp_map = self.__try_load(data_file, C.H5_GT_DISP)

            fix_centroid = self.__try_load(data_file, C.H5_FIX_CENTROID)
            mov_centroid = self.__try_load(data_file, C.H5_MOV_CENTROID)

            data_file.close()
            batch_size = 1

        data_dict = {C.H5_FIX_IMG: fix_img,
                     C.H5_FIX_TUMORS_MASK: fix_tumors,
                     C.H5_FIX_VESSELS_MASK: fix_vessels,
                     C.H5_FIX_PARENCHYMA_MASK: fix_parench,
                     C.H5_MOV_IMG: mov_img,
                     C.H5_MOV_TUMORS_MASK: mov_tumors,
                     C.H5_MOV_VESSELS_MASK: mov_vessels,
                     C.H5_MOV_PARENCHYMA_MASK: mov_parench,
                     C.H5_GT_DISP: disp_map,
                     C.H5_FIX_CENTROID: fix_centroid,
                     C.H5_MOV_CENTROID: mov_centroid,
                     'BATCH_SIZE': batch_size
                     }

        return data_dict

    @staticmethod
    def __get_data_shape(file_path, label):
        f = h5py.File(file_path, 'r')
        shape = f[label][:].shape
        f.close()
        return shape

    def __load_data_by_label(self, label, idx_list):
        if isinstance(idx_list, (list, np.ndarray)):
            data_shape = self.__get_data_shape(self.__list_files[idx_list[0]], label)
            container = np.empty((0, *data_shape), np.float32)
            # if label == C.H5_GT_DISP:
            #     container = np.empty((0, ) + C.DISP_MAP_SHAPE, np.float32)
            # elif label == C.H5_MOV_CENTROID or label == C.H5_FIX_CENTROID:
            #     container = np.empty((0, 3), np.float32)
            # else:
            #     container = np.empty((0, ) + C.IMG_SHAPE, np.float32)

            for idx in idx_list:
                data_file = h5py.File(self.__list_files[idx], 'r')
                container = self.__try_load(data_file, label, container)
                data_file.close()
        else:
            data_file = h5py.File(self.__list_files[idx_list], 'r')
            container = self.__try_load(data_file, label)
            data_file.close()

        return container

    def __build_list2(self, label_list, file_idxs):
        ret_list = list()
        for label in label_list:
            if label is C.DG_LBL_ZERO_GRADS:
                aux = np.zeros([len(file_idxs), *C.DISP_MAP_SHAPE])
            else:
                aux = self.__load_data_by_label(label, file_idxs)

                if label in [C.DG_LBL_MOV_IMG, C.DG_LBL_FIX_IMG]:
                    aux = min_max_norm(aux).astype(np.float32)
            ret_list.append(aux)
        return ret_list

    def __getitem2(self, index):
        f_indices = self.__internal_idxs[index * self.__batch_size:(index + 1) * self.__batch_size]

        return self.__build_list2(self.__input_labels, f_indices), self.__build_list2(self.__output_labels, f_indices)


    def get_samples(self, num_samples, random=False):
        if random:
            idxs = np.random.randint(0, self.__num_samples, num_samples)
        else:
            idxs = np.arange(0, num_samples)
        data_dict = self.__load_data(idxs)
        # return X, y
        return self.__build_list(data_dict, self.__input_labels), self.__build_list(data_dict, self.__output_labels)

    def get_input_shape(self):
        input_batch, _ = self.__getitem__(0)
        data_dict = self.__load_data(0)

        ret_val = data_dict[self.__input_labels[0]].shape
        ret_val = (None, ) + ret_val[1:]
        return ret_val  # const.BATCH_SHAPE_SEGM

    def who_are_you(self):
        return self.__dataset_type

    def print_datafiles(self):
        return self.__list_files


class DataGeneratorManager2D:
    FIX_IMG_H5 = 'input/1'
    MOV_IMG_H5 = 'input/0'

    def __init__(self, h5_file_list, batch_size=32, data_split=0.7, img_size=None,
                 fix_img_tag=FIX_IMG_H5, mov_img_tag=MOV_IMG_H5, multi_loss=False):
        self.__file_list = h5_file_list #h5py.File(h5_file, 'r')
        self.__batch_size = batch_size
        self.__data_split = data_split

        self.__initialize()

        self.__train_generator = DataGenerator2D(self.__train_file_list,
                                                 batch_size=self.__batch_size,
                                                 img_size=img_size,
                                                 fix_img_tag=fix_img_tag,
                                                 mov_img_tag=mov_img_tag,
                                                 multi_loss=multi_loss)
        self.__val_generator = DataGenerator2D(self.__val_file_list,
                                               batch_size=self.__batch_size,
                                               img_size=img_size,
                                               fix_img_tag=fix_img_tag,
                                               mov_img_tag=mov_img_tag,
                                               multi_loss=multi_loss)

    def __initialize(self):
        num_samples = len(self.__file_list)
        random.shuffle(self.__file_list)

        data_split = int(np.floor(num_samples * self.__data_split))
        self.__val_file_list = self.__file_list[0:data_split]
        self.__train_file_list = self.__file_list[data_split:]

    @property
    def train_generator(self):
        return self.__train_generator

    @property
    def validation_generator(self):
        return self.__val_generator


class DataGenerator2D(keras.utils.Sequence):
    FIX_IMG_H5 = 'input/1'
    MOV_IMG_H5 = 'input/0'

    def __init__(self, file_list: list, batch_size=32, img_size=None, fix_img_tag=FIX_IMG_H5, mov_img_tag=MOV_IMG_H5, multi_loss=False):
        self.__file_list = file_list  # h5py.File(h5_file, 'r')
        self.__file_list.sort()
        self.__batch_size = batch_size
        self.__idx_list = np.arange(0, len(self.__file_list))
        self.__multi_loss = multi_loss

        self.__tags = {'fix_img': fix_img_tag,
                       'mov_img': mov_img_tag}

        self.__batches_seen = 0
        self.__batches_per_epoch = 0

        self.__img_size = img_size

        self.__initialize()

    def __len__(self):
        return self.__batches_per_epoch

    def __initialize(self):
        random.shuffle(self.__idx_list)

        if self.__img_size is None:
            f = h5py.File(self.__file_list[0], 'r')
            self.input_shape = f[self.__tags['fix_img']].shape  # Already defined in super()
            f.close()
        else:
            self.input_shape = self.__img_size

        if self.__multi_loss:
            self.input_shape = (self.input_shape, (*self.input_shape[:-1], 2))

        self.__batches_per_epoch = int(np.ceil(len(self.__file_list) / self.__batch_size))

    def __load_and_preprocess(self, fh, tag):
        img = fh[tag][:]

        if (self.__img_size is not None) and (img[..., 0].shape != self.__img_size):
            im = Image.fromarray(img[..., 0])  # Can't handle the 1 channel
            img = np.array(im.resize(self.__img_size[:-1], Image.LANCZOS)).astype(np.float32)
            img = img[..., np.newaxis]

        if img.max() > 1. or img.min() < 0.:
            try:
                img = min_max_norm(img).astype(np.float32)
            except ValueError:
                print(fh, tag, img.shape)
                er_str = 'ERROR:\t[file]:\t{}\t[tag]:\t{}\t[img.shape]:\t{}\t'.format(fh, tag, img.shape)
                raise ValueError(er_str)
        return img.astype(np.float32)

    def __getitem__(self, idx):
        idxs = self.__idx_list[idx * self.__batch_size:(idx + 1) * self.__batch_size]

        fix_imgs, mov_imgs = self.__load_samples(idxs)

        zero_grad = np.zeros((*fix_imgs.shape[:-1], 2))

        inputs = [mov_imgs, fix_imgs]
        outputs = [fix_imgs, zero_grad]

        if self.__multi_loss:
            return [mov_imgs, fix_imgs, zero_grad],
        else:
            return (inputs, outputs)

    def __load_samples(self, idx_list):
        if self.__multi_loss:
            img_shape = (0, *self.input_shape[0])
        else:
            img_shape = (0, *self.input_shape)

        fix_imgs = np.empty(img_shape)
        mov_imgs = np.empty(img_shape)
        for i in idx_list:
            f = h5py.File(self.__file_list[i], 'r')
            fix_imgs = np.append(fix_imgs, [self.__load_and_preprocess(f, self.__tags['fix_img'])], axis=0)
            mov_imgs = np.append(mov_imgs, [self.__load_and_preprocess(f, self.__tags['mov_img'])], axis=0)
            f.close()

        return fix_imgs, mov_imgs

    def on_epoch_end(self):
        np.random.shuffle(self.__idx_list)

    def get_single_sample(self):
        idx = random.randint(0, len(self.__idx_list))
        fix, mov = self.__load_samples([idx])
        return mov, fix


FILE_EXT = {'nifti': '.nii.gz',
            'h5': '.h5'}
CTRL_GRID = C.CoordinatesGrid()
CTRL_GRID.set_coords_grid([128]*3, [C.TPS_NUM_CTRL_PTS_PER_AXIS]*3, batches=False, norm=False, img_type=tf.float32)

FINE_GRID = C.CoordinatesGrid()
FINE_GRID.set_coords_grid([128]*3, [128]*3, batches=FINE_GRID, norm=False)

class DataGeneratorAugment(DataGeneratorManager):
    def __init__(self, GeneratorManager: DataGeneratorManager, file_type='nifti', dataset_type='train'):
        self.__complete_list_files = GeneratorManager.dataset_list_files
        self.__list_files = [self.__complete_list_files[idx] for idx in GeneratorManager.get_generator_idxs(dataset_type)]
        self.__batch_size = GeneratorManager.batch_size
        self.__augm_per_sample = 10
        self.__samples_per_batch = np.ceil(self.__batch_size / (self.__augm_per_sample + 1))  # B = S + S*A
        self.__total_samples = len(self.__list_files)
        self.__clip_range = GeneratorManager.clip_rage
        self.__manager = GeneratorManager
        self.__shuffle = GeneratorManager.shuffle
        self.__file_extension = FILE_EXT[file_type]

        self.__num_samples = len(self.__list_files)
        self.__internal_idxs = np.arange(self.__num_samples)
        # These indices are internal to the generator, they are not the same as the dataset_idxs!!

        self.__dataset_type = dataset_type

        self.__last_batch = 0
        self.__batches_per_epoch = int(np.floor(len(self.__internal_idxs) / self.__batch_size))

        self.__input_labels = GeneratorManager.input_labels
        self.__output_labels = GeneratorManager.output_labels


    def __get_dataset_files(self, search_path):
        """
        Get the path to the dataset files
        :param  search_path: dir path to search for the hd5 files
        :return:
        """
        file_list = list()
        for root, dirs, files in os.walk(search_path):
            for data_file in files:
                file_name, extension = os.path.splitext(data_file)
                if extension.lower() == self.__file_extension:
                    file_list.append(os.path.join(root, data_file))

        if not file_list:
            raise ValueError('No files found to train in ', search_path)

        print('Found {} files in {}'.format(len(file_list), search_path))
        return file_list

    def update_samples(self, new_sample_idxs):
        self.__list_files = [self.__complete_list_files[idx] for idx in new_sample_idxs]
        self.__num_samples = len(self.__list_files)
        self.__internal_idxs = np.arange(self.__num_samples)

    def on_epoch_end(self):
        """
        To be executed at the end of each epoch. Reshuffle the assigned samples
        :return:
        """
        if self.__shuffle:
            random.shuffle(self.__internal_idxs)
        self.__last_batch = 0

    def __len__(self):
        """
        Number of batches per epoch
        :return:
        """
        return self.__batches_per_epoch

    def __getitem__(self, index):
        """
        Generate one batch of data
        :param index: epoch index
        :return:
        """
        return self.__getitem(index)

    def next_batch(self):
        if self.__last_batch > self.__batches_per_epoch:
            raise ValueError('No more batches for this epoch')
        batch = self.__getitem__(self.__last_batch)
        self.__last_batch += 1
        return batch

    def __try_load(self, data_file, label, append_array=None):
        if label in self.__input_labels or label in self.__output_labels:
            # To avoid extra overhead
            try:
                retVal = data_file[label][:][np.newaxis, ...]
            except KeyError:
                # That particular label is not found in the file. But this should be known by the user by now
                retVal = None

            if append_array is not None and retVal is not None:
                return np.append(append_array, retVal, axis=0)
            elif append_array is None:
                return retVal
            else:
                return retVal  # None
        else:
            return None

    @staticmethod
    def __get_data_shape(file_path, label):
        f = h5py.File(file_path, 'r')
        shape = f[label][:].shape
        f.close()
        return shape

    def __load_data_by_label(self, label, idx_list):
        if isinstance(idx_list, (list, np.ndarray)):
            data_shape = self.__get_data_shape(self.__list_files[idx_list[0]], label)
            container = np.empty((0, *data_shape), np.float32)
            # if label == C.H5_GT_DISP:
            #     container = np.empty((0, ) + C.DISP_MAP_SHAPE, np.float32)
            # elif label == C.H5_MOV_CENTROID or label == C.H5_FIX_CENTROID:
            #     container = np.empty((0, 3), np.float32)
            # else:
            #     container = np.empty((0, ) + C.IMG_SHAPE, np.float32)

            for idx in idx_list:
                data_file = h5py.File(self.__list_files[idx], 'r')
                container = self.__try_load(data_file, label, container)
                data_file.close()
        else:
            data_file = h5py.File(self.__list_files[idx_list], 'r')
            container = self.__try_load(data_file, label)
            data_file.close()

        return container

    def __build_list(self, label_list, file_idxs):
        ret_list = list()
        for label in label_list:
            if label is C.DG_LBL_ZERO_GRADS:
                aux = np.zeros([len(file_idxs), *C.DISP_MAP_SHAPE])
            else:
                aux = self.__load_data_by_label(label, file_idxs)

                if label in [C.DG_LBL_MOV_IMG, C.DG_LBL_FIX_IMG]:
                    aux = min_max_norm(aux).astype(np.float32)
            ret_list.append(aux)
        return ret_list

    def __getitem(self, index):
        f_indices = self.__internal_idxs[index * self.__samples_per_batch:(index + 1) * self.__samples_per_batch]
        # https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit
        # A generator or keras.utils.Sequence returning (inputs, targets) or (inputs, targets, sample_weights)
        # The second element must match the outputs of the model, in this case (image, displacement map)
        if 'h5' in self.__file_extension:
            return self.__build_list(self.__input_labels, f_indices), self.__build_list(self.__output_labels, f_indices)
        else:
            f_list = [self.__list_files[i] for i in f_indices]
            return self.__augment(f_list, 'fixed', C.H5_FIX_IMG), self.__augment(f_list, 'moving', C.H5_FIX_IMG)


    def __intensity_preprocessing(self, img_data):
        # Histogram normalization
        processed_img = equalize_adapthist(img_data, clip_limit=0.03)
        processed_img = min_max_norm(processed_img)

        return processed_img


    def __resize_img(self, img, output_shape):
        if isinstance(output_shape, int):
            output_shape = [output_shape] * len(img.shape)
        # Resize
        zoom_vals = np.asarray(output_shape) / np.asarray(img.shape)
        return zoom(img, zoom_vals)


    def __build_augmented_batch(self, f_list, mode):
        for f_path in f_list:
            h5_file = h5py.File(f_path, 'r')
            img_nib = nib.load(h5_file[C.H5_FIX_IMG][:])
            img_nib = resample_img(img_nib, np.eye(3))
            try:
                seg_nib = nib.load(h5_file[C.H5_FIX_SEGMENTATIONS][:])
                seg_nib = resample_img(seg_nib, np.eye(3))
            except FileNotFoundError:
                seg_nib = None

            img_nib = self.__intensity_preprocessing(img_nib)
            img_nib = self.__resize_img(img_nib, 128)








    def get_samples(self, num_samples, random=False):
        return

    def get_input_shape(self):
        input_batch, _ = self.__getitem__(0)
        data_dict = self.__load_data(0)

        ret_val = data_dict[self.__input_labels[0]].shape
        ret_val = (None, ) + ret_val[1:]
        return ret_val  # const.BATCH_SHAPE_SEGM

    def who_are_you(self):
        return self.__dataset_type

    def print_datafiles(self):
        return self.__list_files


def tf_graph_deform():
    # Place holders
    fix_img = tf.placeholder(tf.float32, [128]*3, 'fix_img')
    fix_segmentations = tf.placeholder_with_default(np.zeros([128]*3), shape=[128]*3, name='fix_segmentations')
    max_deformation = tf.placeholder(tf.float32, shape=(), name='max_deformation')
    max_displacement = tf.placeholder(tf.float32, shape=(), name='max_displacement')
    max_rotation = tf.placeholder(tf.float32, shape=(), name='max_rotation')
    num_moved_points = tf.placeholder_with_default(50, shape=(), name='num_moved_points')
    only_image = tf.placeholder_with_default(True, shape=(), name='only_image')

    search_voxels = tf.cond(only_image,
                            lambda: fix_img,
                            lambda: fix_segmentations)

    # Apply TPS deformation
    # Get points in the segmentation or image, and add it to the control grid and target grid
    # Indices of the points in the seaerch image with intensity greater than 0  (It would be bad if we only move the bg)
    idx_points_in_label = tf.where(tf.greater(search_voxels, 0.0))

    # Randomly select one of the points
    random_idx = tf.random.uniform((num_moved_points,), minval=0, maxval=tf.shape(idx_points_in_label)[0], dtype=tf.int32)

    disp_location = tf.gather_nd(idx_points_in_label, random_idx)  # And get the coordinates
    disp_location = tf.cast(disp_location, tf.float32)
    # Get the coordinates of the control point displaces
    rand_disp = tf.random.uniform((num_moved_points, 3), minval=-1, maxval=1, dtype=tf.float32) * max_deformation
    warped_location = disp_location + rand_disp

    # Add the selected locations to the control grid and the warped locations to the target grid
    control_grid = tf.concat([CTRL_GRID.grid_flat(), disp_location], axis=0)
    trg_grid = tf.concat([CTRL_GRID.grid_flat(), warped_location], axis=0)

    # Add global affine transformation
    trg_grid, aff = transform_points(trg_grid, max_displacement=max_displacement, max_rotation=max_rotation)

    tps = ThinPlateSplines(control_grid, trg_grid)
    def_grid = tps.interpolate(FINE_GRID.grid_flat())

    disp_map = FINE_GRID.grid_flat() - def_grid
    disp_map = tf.reshape(disp_map, (*FINE_GRID.shape, -1))
    # disp_map = interpn(disp_map, FULL_FINE_GRID.grid)

    # add the batch and channel dimensions
    fix_img = tf.expand_dims(tf.expand_dims(fix_img, -1), 0)
    fix_segmentations = tf.expand_dims(tf.expand_dims(fix_img, -1), 0)
    disp_map = tf.cast(tf.expand_dims(disp_map, 0), tf.float32)

    mov_img = SpatialTransformer(interp_method='linear', indexing='ij', single_transform=False)([fix_img, disp_map])
    mov_segmentations = SpatialTransformer(interp_method='linear', indexing='ij', single_transform=False)([fix_segmentations, disp_map])

    return tf.squeeze(mov_img),\
           tf.squeeze(mov_segmentations),\
           tf.squeeze(disp_map),\
           disp_location,\
           rand_disp,\
           aff #, w, trg_grid, def_grid


def transform_points(points: tf.Tensor, max_displacement, max_rotation):
    axis = tf.random.uniform((), 0, 3)

    alpha = tf.cond(tf.less_equal(axis, 0.),
                    lambda: tf.random.uniform((1,), -max_rotation, max_rotation),
                    lambda: tf.zeros((1,), tf.float32))
    beta = tf.cond(tf.less_equal(axis, 1.),
                   lambda: tf.random.uniform((1,), -max_rotation, max_rotation),
                   lambda: tf.zeros((1,), tf.float32))
    gamma = tf.cond(tf.less_equal(axis, 2.),
                    lambda: tf.random.uniform((1,), -max_rotation, max_rotation),
                    lambda: tf.zeros((1,), tf.float32))

    ti = tf.random.uniform((), minval=-1, maxval=1, dtype=tf.float32) * max_displacement
    tj = tf.random.uniform((), minval=-1, maxval=1, dtype=tf.float32) * max_displacement
    tk = tf.random.uniform((), minval=-1, maxval=1, dtype=tf.float32) * max_displacement

    M = build_affine_trf(tf.convert_to_tensor(FINE_GRID.shape, tf.float32), alpha, beta, gamma, ti, tj, tk)
    if points.shape.as_list()[-1] == 3:
        points = tf.transpose(points)
    new_pts = tf.matmul(M[:3, :3], points)
    new_pts = tf.expand_dims(M[:3, -1], -1) + new_pts
    return tf.transpose(new_pts), M  # Remove the last row of ones


def build_affine_trf(img_size, alpha, beta, gamma, ti, tj, tk):
    img_centre = tf.expand_dims(tf.divide(img_size, 2.), -1)

    # Rotation matrix around the image centre
    # R* = T(p) R(ang) T(-p)
    # tf.cos and tf.sin expect radians
    zero = tf.zeros((1,))
    one = tf.ones((1,))

    T = tf.convert_to_tensor([[one, zero, zero, ti],
                              [zero, one, zero, tj],
                              [zero, zero, one, tk],
                              [zero, zero, zero, one]], tf.float32)
    T = tf.squeeze(T)

    R = tf.convert_to_tensor([[tf.math.cos(gamma) * tf.math.cos(beta),
                               tf.math.cos(gamma) * tf.math.sin(beta) * tf.math.sin(alpha) - tf.math.sin(gamma) * tf.math.cos(alpha),
                               tf.math.cos(gamma) * tf.math.sin(beta) * tf.math.cos(alpha) + tf.math.sin(gamma) * tf.math.sin(alpha),
                               zero],
                              [tf.math.sin(gamma) * tf.math.cos(beta),
                               tf.math.sin(gamma) * tf.math.sin(beta) * tf.math.sin(gamma) + tf.math.cos(gamma) * tf.math.cos(alpha),
                               tf.math.sin(gamma) * tf.math.sin(beta) * tf.math.cos(gamma) - tf.math.cos(gamma) * tf.math.sin(gamma),
                               zero],
                              [-tf.math.sin(beta),
                               tf.math.cos(beta) * tf.math.sin(alpha),
                               tf.math.cos(beta) * tf.math.cos(alpha),
                               zero],
                              [zero, zero, zero, one]], tf.float32)

    R = tf.squeeze(R)

    Tc = tf.convert_to_tensor([[one, zero, zero, img_centre[0]],
                               [zero, one, zero, img_centre[1]],
                               [zero, zero, one, img_centre[2]],
                               [zero, zero, zero, one]], tf.float32)
    Tc = tf.squeeze(Tc)
    Tc_ = tf.convert_to_tensor([[one, zero, zero, -img_centre[0]],
                                [zero, one, zero, -img_centre[1]],
                                [zero, zero, one, -img_centre[2]],
                                [zero, zero, zero, one]], tf.float32)
    Tc_ = tf.squeeze(Tc_)

    return tf.matmul(T, tf.matmul(Tc, tf.matmul(R, Tc_)))
