import os, sys

import shutil

import matplotlib.pyplot as plt

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)  # PYTHON > 3.3 does not allow relative referencing

import tensorflow as tf
# tf.enable_eager_execution(config=config)

import numpy as np
import h5py

import DeepDeformationMapRegistration.utils.constants as C
from DeepDeformationMapRegistration.utils.nifti_utils import save_nifti
from DeepDeformationMapRegistration.layers import AugmentationLayer
from DeepDeformationMapRegistration.utils.visualization import save_disp_map_img, plot_predictions
from DeepDeformationMapRegistration.utils.misc import get_segmentations_centroids, DisplacementMapInterpolator

from tqdm import tqdm

from Brain_study.data_generator import BatchGenerator

from skimage.measure import regionprops
from scipy.interpolate import griddata

import argparse


DATASET = '/mnt/EncryptedData1/Users/javier/ext_datasets/IXI_dataset/T1/training'
MODEL_FILE = '/mnt/EncryptedData1/Users/javier/train_output/Brain_study/ERASE/MS_SSIM/BASELINE_L_ssim__MET_mse_ncc_ssim_162756-29062021/checkpoints/best_model.h5'
DATA_ROOT_DIR = '/mnt/EncryptedData1/Users/javier/train_output/Brain_study/ERASE/MS_SSIM/BASELINE_L_ssim__MET_mse_ncc_ssim_162756-29062021/'

POINTS = None
MISSING_CENTROID = np.asarray([[np.nan]*3])


def get_mov_centroids(fix_seg, disp_map, nb_labels=28, exclude_background_lbl=False, brain_study=True, dm_interp=None):
    if exclude_background_lbl:
        fix_centroids, _ = get_segmentations_centroids(fix_seg[0, ..., 1:], ohe=True, expected_lbls=range(1, nb_labels), brain_study=brain_study)
    else:
        fix_centroids, _ = get_segmentations_centroids(fix_seg[0, ...], ohe=True, expected_lbls=range(1, nb_labels), brain_study=brain_study)
    if dm_interp is None:
        disp = griddata(POINTS, disp_map.reshape([-1, 3]), fix_centroids, method='linear')
    else:
        disp = dm_interp(disp_map, fix_centroids, backwards=False)
    return fix_centroids, fix_centroids + disp, disp


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', type=str, help='Directory where to store the files', default='')
    parser.add_argument('--reldir', type=str, help='Relative path to dataset, in where to store the files', default='')
    parser.add_argument('--gpu', type=int, help='GPU', default=0)
    parser.add_argument('--dataset', type=str, help='Dataset to build the test set', default='')
    parser.add_argument('--erase', type=bool, help='Erase the content of the output folder', default=False)
    parser.add_argument('--output_shape', help='If an int, a cubic shape is presumed. Otherwise provide it as a space separated sequence', nargs='+', default=128)
    args = parser.parse_args()

    assert args.dataset != '', "Missing original dataset dataset"
    if args.dir == '' and args.reldir != '':
        OUTPUT_FOLDER_DIR = os.path.join(args.dataset, 'test_dataset')
    elif args.dir != '' and args.reldir == '':
        OUTPUT_FOLDER_DIR = args.dir
    else:
        raise ValueError("Either provide 'dir' or 'reldir'")

    if args.erase:
        shutil.rmtree(OUTPUT_FOLDER_DIR, ignore_errors=True)
    os.makedirs(OUTPUT_FOLDER_DIR, exist_ok=True)
    print('DESTINATION FOLDER: ', OUTPUT_FOLDER_DIR)

    DATASET = args.dataset

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)  # Check availability before running using 'nvidia-smi'

    data_generator = BatchGenerator(DATASET, 1, False, 1.0, False, ['all'], return_isotropic_shape=True)

    img_generator = data_generator.get_train_generator()
    nb_labels = len(img_generator.get_segmentation_labels())
    image_input_shape = img_generator.get_data_shape()[-1][:-1]

    if isinstance(args.output_shape, int):
        image_output_shape = [args.output_shape] * 3
    elif isinstance(args.output_shape, list):
        assert len(args.output_shape) == 3, 'Invalid output shape, expected three values and got {}'.format(len(args.output_shape))
        image_output_shape = [int(s) for s in args.output_shape]
    else:
        raise ValueError('Invalid output_shape. Must be an int or a space-separated sequence of ints')
    print('Scaling to: ', image_output_shape)
    # Build model

    xx = np.linspace(0, image_output_shape[0], image_output_shape[0], endpoint=False)
    yy = np.linspace(0, image_output_shape[1], image_output_shape[2], endpoint=False)
    zz = np.linspace(0, image_output_shape[2], image_output_shape[1], endpoint=False)

    xx, yy, zz = np.meshgrid(xx, yy, zz)

    POINTS = np.stack([xx.flatten(), yy.flatten(), zz.flatten()], axis=0).T

    input_augm = tf.keras.Input(shape=img_generator.get_data_shape()[0], name='input_augm')
    augm_layer = AugmentationLayer(max_displacement=C.MAX_AUG_DISP,  # Max 30 mm in isotropic space
                                   max_deformation=C.MAX_AUG_DEF,  # Max 6 mm in isotropic space
                                   max_rotation=C.MAX_AUG_ANGLE,  # Max 10 deg in isotropic space
                                   num_control_points=C.NUM_CONTROL_PTS_AUG,
                                   num_augmentations=C.NUM_AUGMENTATIONS,
                                   gamma_augmentation=C.GAMMA_AUGMENTATION,
                                   brightness_augmentation=C.BRIGHTNESS_AUGMENTATION,
                                   in_img_shape=image_input_shape,
                                   out_img_shape=image_output_shape,
                                   only_image=False,
                                   only_resize=False,
                                   trainable=False,
                                   return_displacement_map=True)
    augm_model = tf.keras.Model(inputs=input_augm, outputs=augm_layer(input_augm))

    fix_img_ph = tf.placeholder(dtype=tf.float32, shape=[1,] + image_input_shape + [1+nb_labels,], name='fix_image')

    augmentation_pipeline = augm_model(fix_img_ph)

    config = tf.compat.v1.ConfigProto()  # device_count={'GPU':0})
    config.gpu_options.allow_growth = True
    config.log_device_placement = False  ## to log device placement (on which device the operation ran)

    dm_interp = DisplacementMapInterpolator(image_output_shape, 'griddata', step=8)

    sess = tf.Session(config=config)
    tf.keras.backend.set_session(sess)
    with sess.as_default():
        sess.run(tf.global_variables_initializer())
        progress_bar = tqdm(enumerate(img_generator, 1), desc='Generating samples', total=len(img_generator))
        for step, (in_batch, _, isotropic_shape) in progress_bar:
            # fix_img, mov_img, fix_seg, mov_seg, disp_map = augm_model.predict(in_batch)
            fix_img, mov_img, fix_seg, mov_seg, disp_map = sess.run(augmentation_pipeline,
                                                                    feed_dict={'fix_image:0': in_batch})

            fix_centroids, mov_centroids, disp_centroids = get_mov_centroids(fix_seg, disp_map, nb_labels, dm_interp=dm_interp)

            out_file = os.path.join(OUTPUT_FOLDER_DIR, 'test_sample_{:04d}.h5'.format(step))
            out_file_dm = os.path.join(OUTPUT_FOLDER_DIR, 'test_sample_dm_{:04d}.h5'.format(step))
            img_shape = fix_img.shape
            segm_shape = fix_seg.shape
            disp_shape = disp_map.shape
            centroids_shape = fix_centroids.shape
            with h5py.File(out_file, 'w') as f:
                f.create_dataset('fix_image', shape=img_shape[1:], dtype=np.float32, data=fix_img[0, ...])
                f.create_dataset('mov_image', shape=img_shape[1:], dtype=np.float32, data=mov_img[0, ...])
                f.create_dataset('fix_segmentations', shape=segm_shape[1:], dtype=np.uint8, data=fix_seg[0, ...])
                f.create_dataset('mov_segmentations', shape=segm_shape[1:], dtype=np.uint8, data=mov_seg[0, ...])
                f.create_dataset('fix_centroids', shape=centroids_shape, dtype=np.float32, data=fix_centroids)
                f.create_dataset('mov_centroids', shape=centroids_shape, dtype=np.float32, data=mov_centroids)
                f.create_dataset('isotropic_shape', data=np.squeeze(isotropic_shape))
            with h5py.File(out_file_dm, 'w') as f:
                f.create_dataset('disp_map', shape=disp_shape[1:], dtype=np.float32, data=disp_map)
                f.create_dataset('disp_centroids', shape=centroids_shape, dtype=np.float32, data=disp_centroids)

    print('Done')
