import sys, os
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)  # PYTHON > 3.3 does not allow relative referencing

PYCHARM_EXEC = os.getenv('PYCHARM_EXEC') == 'True'

import numpy as np
from tensorflow import keras
import os
import h5py
import random
from PIL import Image

import DeepDeformationMapRegistration.utils.constants as C
from DeepDeformationMapRegistration.utils.operators import min_max_norm


class DataGeneratorManager(keras.utils.Sequence):
    def __init__(self, dataset_path, batch_size=32, shuffle=True, num_samples=None, validation_split=None, validation_samples=None,
                 clip_range=[0., 1.], voxelmorph=False, segmentations=False,
                 seg_labels: dict = {'bg': 0, 'vessels': 1, 'tumour': 2, 'parenchyma': 3}):
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

        self.__voxelmorph = voxelmorph
        self.__segmentations = segmentations
        self.__seg_labels = seg_labels

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
                if extension.lower() == '.hd5':
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

    @property
    def is_voxelmorph(self):
        return self.__voxelmorph

    @property
    def give_segmentations(self):
        return self.__segmentations

    @property
    def seg_labels(self):
        return self.__seg_labels


class DataGenerator(DataGeneratorManager):
    def __init__(self, GeneratorManager: DataGeneratorManager, dataset_type='train'):
        self.__complete_list_files = GeneratorManager.dataset_list_files
        self.__list_files = [self.__complete_list_files[idx] for idx in GeneratorManager.get_generator_idxs(dataset_type)]
        self.__batch_size = GeneratorManager.batch_size
        self.__total_samples = len(self.__list_files)
        self.__clip_range = GeneratorManager.clip_rage
        self.__manager = GeneratorManager
        self.__shuffle = GeneratorManager.shuffle

        self.__seg_labels = GeneratorManager.seg_labels

        self.__num_samples = len(self.__list_files)
        self.__internal_idxs = np.arange(self.__num_samples)
        # These indices are internal to the generator, they are not the same as the dataset_idxs!!

        self.__dataset_type = dataset_type

        self.__last_batch = 0
        self.__batches_per_epoch = int(np.floor(len(self.__internal_idxs) / self.__batch_size))

        self.__voxelmorph = GeneratorManager.is_voxelmorph
        self.__segmentations = GeneratorManager.is_voxelmorph and GeneratorManager.give_segmentations

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

    def __getitem__(self, index):
        """
        Generate one batch of data
        :param index: epoch index
        :return:
        """
        idxs = self.__internal_idxs[index * self.__batch_size:(index + 1) * self.__batch_size]

        fix_img, mov_img, fix_vessels, mov_vessels, fix_tumour, mov_tumour, disp_map = self.__load_data(idxs)

        try:
            fix_img = min_max_norm(fix_img).astype(np.float32)
            mov_img = min_max_norm(mov_img).astype(np.float32)
        except ValueError:
            print(idxs, fix_img.shape, mov_img.shape)
            er_str = 'ERROR:\t[file]:\t{}\t[idx]:\t{}\t[fix_img.shape]:\t{}\t[mov_img.shape]:\t{}\t'.format(self.__list_files[idxs], idxs, fix_img.shape, mov_img.shape)
            raise ValueError(er_str)

        fix_vessels[fix_vessels > 0.] = self.__seg_labels['vessels']
        mov_vessels[mov_vessels > 0.] = self.__seg_labels['vessels']

        # fix_tumour[fix_tumour > 0.] = self.__seg_labels['tumour']
        # mov_tumour[mov_tumour > 0.] = self.__seg_labels['tumour']

        # https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit
        # A generator or keras.utils.Sequence returning (inputs, targets) or (inputs, targets, sample_weights)
        # The second element must match the outputs of the model, in this case (image, displacement map)
        if self.__voxelmorph:
            zero_grad = np.zeros([fix_img.shape[0], *C.DISP_MAP_SHAPE])
            if self.__segmentations:
                inputs = [mov_vessels, fix_vessels, mov_img, fix_img, zero_grad]
                outputs = []  #[fix_img, zero_grad]
            else:
                inputs = [mov_img, fix_img]
                outputs = [fix_img, zero_grad]
            return (inputs, outputs)
        else:
            return (fix_img, mov_img, fix_vessels, mov_vessels), # (None, fix_seg, fix_seg, fix_img)

    def next_batch(self):
        if self.__last_batch > self.__batches_per_epoch:
            raise ValueError('No more batches for this epoch')
        batch = self.__getitem__(self.__last_batch)
        self.__last_batch += 1
        return batch

    def __load_data(self, idx_list):
        """
        Build the batch with the samples in idx_list
        :param idx_list:
        :return:
        """
        if isinstance(idx_list, (list, np.ndarray)):
            fix_img = np.empty((0, ) + C.IMG_SHAPE)
            mov_img = np.empty((0, ) + C.IMG_SHAPE)
            disp_map = np.empty((0, ) + C.DISP_MAP_SHAPE)

            # fix_segm = np.empty((0, ) + const.IMG_SHAPE)
            # mov_segm = np.empty((0, ) + const.IMG_SHAPE)

            fix_vessels = np.empty((0, ) + C.IMG_SHAPE)
            mov_vessels = np.empty((0, ) + C.IMG_SHAPE)
            fix_tumors = np.empty((0, ) + C.IMG_SHAPE)
            mov_tumors = np.empty((0, ) + C.IMG_SHAPE)
            for idx in idx_list:
                data_file = h5py.File(self.__list_files[idx], 'r')

                fix_img = np.append(fix_img, [data_file[C.H5_FIX_IMG][:]], axis=0)
                mov_img = np.append(mov_img, [data_file[C.H5_MOV_IMG][:]], axis=0)

                # fix_segm = np.append(fix_segm, [data_file[const.H5_FIX_PARENCHYMA_MASK][:]], axis=0)
                # mov_segm = np.append(mov_segm, [data_file[const.H5_MOV_PARENCHYMA_MASK][:]], axis=0)

                disp_map = np.append(disp_map, [data_file[C.H5_GT_DISP][:]], axis=0)

                fix_vessels = np.append(fix_vessels, [data_file[C.H5_FIX_VESSELS_MASK][:]], axis=0)
                mov_vessels = np.append(mov_vessels, [data_file[C.H5_MOV_VESSELS_MASK][:]], axis=0)
                fix_tumors = np.append(fix_tumors, [data_file[C.H5_FIX_TUMORS_MASK][:]], axis=0)
                mov_tumors = np.append(mov_tumors, [data_file[C.H5_MOV_TUMORS_MASK][:]], axis=0)

                data_file.close()
        else:
            data_file = h5py.File(self.__list_files[idx_list], 'r')

            fix_img = np.expand_dims(data_file[C.H5_FIX_IMG][:], 0)
            mov_img = np.expand_dims(data_file[C.H5_MOV_IMG][:], 0)

            # fix_segm = np.expand_dims(data_file[const.H5_FIX_PARENCHYMA_MASK][:], 0)
            # mov_segm = np.expand_dims(data_file[const.H5_MOV_PARENCHYMA_MASK][:], 0)

            fix_vessels = np.expand_dims(data_file[C.H5_FIX_VESSELS_MASK][:], axis=0)
            mov_vessels = np.expand_dims(data_file[C.H5_MOV_VESSELS_MASK][:], axis=0)
            fix_tumors = np.expand_dims(data_file[C.H5_FIX_TUMORS_MASK][:], axis=0)
            mov_tumors = np.expand_dims(data_file[C.H5_MOV_TUMORS_MASK][:], axis=0)

            disp_map = np.expand_dims(data_file[C.H5_GT_DISP][:], 0)

            data_file.close()

        return fix_img, mov_img, fix_vessels, mov_vessels, fix_tumors, mov_tumors, disp_map

    def get_single_sample(self):
        fix_img, mov_img, fix_segm, mov_segm, _ = self.__load_data(0)
        # return X, y
        return np.expand_dims(np.concatenate([mov_img, fix_img, mov_segm, mov_segm], axis=-1), axis=0)

    def get_random_sample(self, num_samples):
        idxs = np.random.randint(0, self.__num_samples, num_samples)
        fix_img, mov_img, fix_segm, mov_segm, disp_map = self.__load_data(idxs)

        return (fix_img, mov_img, fix_segm, mov_segm, disp_map), [self.__list_files[f] for f in idxs]

    def get_input_shape(self):
        input_batch, _ = self.__getitem__(0)
        if self.__voxelmorph:
            ret_val = list(input_batch[0].shape)
            ret_val[-1] = 2
            ret_val = (None, ) + tuple(ret_val[1:])
        else:
            ret_val = input_batch.shape
            ret_val = (None, ) + ret_val[1:]
        return ret_val # const.BATCH_SHAPE_SEGM

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


