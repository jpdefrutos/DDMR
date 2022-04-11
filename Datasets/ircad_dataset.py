import os, sys

os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "1"  # Check availability before running using 'nvidia-smi'
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
import multiprocessing as mp

mp.set_start_method('spawn')

import tensorflow as tf

# tf.enable_eager_execution()

import numpy as np
import nibabel as nib
from skimage.transform import resize
from skimage.filters import median
from scipy.ndimage import binary_dilation, generate_binary_structure
from nilearn.image import math_img
import h5py
from tqdm import tqdm
import re

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)  # PYTHON > 3.3 does not allow relative referencing
from DeepDeformationMapRegistration.utils.cmd_args_parser import parse_arguments
import DeepDeformationMapRegistration.utils.constants as const
from tools.thinPlateSplines_tf import ThinPlateSplines
from keras_model.ext.neuron.layers import SpatialTransformer
from tools.voxelMorph import interpn
from generate_dataset.utils import plot_central_slices, plot_def_map, single_img_gif, two_img_gif, plot_slices, \
    crop_images, plot_displacement_map, bbox_3D
from generate_dataset import utils
from tools.misc import try_mkdir
from generate_dataset.utils import unzip_file, delete_temp

DATASTE_RAW_FILES = '/mnt/EncryptedData1/Users/javier/vessel_registration/3Dirca/nifti'
LITS_SEGMENTATION_FILE = 'segmentations'
LITS_CT_FILE = 'volume'

IMG_SIZE_LARGE = const.IMG_SHAPE[:-1]
IMG_SIZE_LARGE_x2 = [2 * x for x in const.IMG_SHAPE[:-1]]
FINE_GRID_SHAPE = tuple(x // 1 for x in IMG_SIZE_LARGE_x2)  # tuple(np.asarray(IMG_SIZE_LARGE) // 10)
CTRL_GRID = const.CoordinatesGrid()
CTRL_GRID.set_coords_grid(IMG_SIZE_LARGE_x2, [const.TPS_NUM_CTRL_PTS_PER_AXIS, const.TPS_NUM_CTRL_PTS_PER_AXIS,
                                              const.TPS_NUM_CTRL_PTS_PER_AXIS], batches=False, norm=False,
                          img_type=tf.float32)

FULL_FINE_GRID = const.CoordinatesGrid()
FULL_FINE_GRID.set_coords_grid(IMG_SIZE_LARGE_x2, FINE_GRID_SHAPE, batches=False, norm=False)

OFFSET_NAME_NUM = 0

TH_BIN = 0.50

DILATION_STRUCT = generate_binary_structure(3, 1)

LARGE_PT_DIM = CTRL_GRID.shape_grid_flat + np.asarray([9, 0])
SINGLE_PT_DIM = CTRL_GRID.shape_grid_flat + np.asarray([1, 0])
USE_LARGE_PT = False
ADD_AFFINE_TRF = False

config = tf.compat.v1.ConfigProto()  # device_count={'GPU':0})
config.gpu_options.allow_growth = True
config.log_device_placement = False  ## to log device placement (on which device the operation ran)



def tf_graph_translation():
    # Place holders
    fix_img = tf.placeholder(tf.float32, IMG_SIZE_LARGE_x2, 'fix_img')
    fix_tumors = tf.placeholder(tf.float32, IMG_SIZE_LARGE_x2, 'fix_tumors')
    fix_parenchyma = tf.placeholder(tf.float32, IMG_SIZE_LARGE_x2, 'fix_parenchyma')

    # Apply Affine translation
    w = tf.constant(np.random.uniform(-1, 1, 3) * const.MAX_DISP_DM_PERC * IMG_SIZE_LARGE_x2[0], dtype=tf.float32)
    pad = tf.cast(tf.abs(w) + 1., tf.int32)
    padding = tf.stack([pad, pad], 1)
## PURE TRANSLATION
    # Shift the target grid 'w' units
    #control_grid = tf.identity(CTRL_GRID.grid_flat())
    #trg_grid = tf.add(control_grid, w)

    #tps = ThinPlateSplines(control_grid, trg_grid)
    #def_grid = tps.interpolate(FULL_FINE_GRID.grid_flat())
## PURE TRANSLATION

    def_grid = tf.add(FULL_FINE_GRID.grid_flat(), w)
    disp_map = def_grid - FULL_FINE_GRID.grid_flat()
    disp_map = tf.reshape(disp_map, (*FINE_GRID_SHAPE, -1))
    # disp_map = interpn(disp_map, FULL_FINE_GRID.grid)

    # add the batch and channel dimensions
    fix_img = tf.pad(fix_img, padding, "CONSTANT", constant_values=0.)
    fix_tumors = tf.pad(fix_tumors, padding, "CONSTANT", constant_values=0.)
    fix_parenchyma = tf.pad(fix_parenchyma, padding, "CONSTANT", constant_values=0.)

    sampl_grid = tf.add(def_grid, tf.cast(pad, def_grid.dtype)) # Because of the padding, the sampling points are now translated 'pad' units
    fix_img = tf.expand_dims(fix_img, -1)
    fix_tumors = tf.expand_dims(fix_tumors, -1)
    fix_parenchyma = tf.expand_dims(fix_parenchyma, -1)

    mov_img = interpn(fix_img, sampl_grid, interp_method='linear')
    mov_img = tf.squeeze(tf.reshape(mov_img, IMG_SIZE_LARGE_x2))

    mov_tumors = interpn(fix_tumors, sampl_grid, interp_method='linear')
    mov_tumors = tf.squeeze(tf.reshape(mov_tumors, IMG_SIZE_LARGE_x2))

    mov_parenchyma = interpn(fix_parenchyma, sampl_grid, interp_method='linear')
    mov_parenchyma = tf.squeeze(tf.reshape(mov_parenchyma, IMG_SIZE_LARGE_x2))

    disp_map = tf.cast(disp_map, tf.float32)
    return mov_img, mov_parenchyma, mov_tumors, disp_map, w  # , w, trg_grid, def_grid


def build_affine_trf(img_size, alpha, beta, gamma, ti, tj, tk):
    img_centre = tf.expand_dims(tf.divide(img_size, 2.), -1)

    # Rotation matrix around the image centre
    # R* = T(p) R(ang) T(-p)
    # tf.cos and tf.sin expect radians
    zero = tf.zeros((1,))
    one = tf.ones((1,))
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

    T = tf.convert_to_tensor([[one, zero, zero, ti],
                              [zero, one, zero, tj],
                              [zero, zero, one, tk],
                              [zero, zero, zero, one]], tf.float32)
    T = tf.squeeze(T)

    return tf.matmul(T, tf.matmul(Tc, tf.matmul(R, Tc_)))


def transform_points(points: tf.Tensor):
    alpha = tf.random.uniform((1,), -const.MAX_ANGLE_RAD, const.MAX_ANGLE_RAD)
    beta = tf.random.uniform((1,), -const.MAX_ANGLE_RAD, const.MAX_ANGLE_RAD)
    gamma = tf.random.uniform((1,), -const.MAX_ANGLE_RAD, const.MAX_ANGLE_RAD)

    ti = tf.constant(np.random.uniform(-1, 1, 1) * const.MAX_DISP_DM / 2, dtype=tf.float32)
    tj = tf.constant(np.random.uniform(-1, 1, 1) * const.MAX_DISP_DM / 2, dtype=tf.float32)
    tk = tf.constant(np.random.uniform(-1, 1, 1) * const.MAX_DISP_DM / 2, dtype=tf.float32)

    M = build_affine_trf(tf.convert_to_tensor(IMG_SIZE_LARGE_x2, tf.float32), alpha, beta, gamma, ti, tj, tk)
    if points.shape.as_list()[-1] == 3:
        points = tf.transpose(points)
    new_pts = tf.matmul(M[:3, :3], points)
    new_pts = tf.expand_dims(M[:3, -1], -1) + new_pts
    return tf.transpose(new_pts), M  # Remove the last row of ones


def tf_graph_deform():
    # Place holders
    fix_img = tf.placeholder(tf.float32, IMG_SIZE_LARGE_x2, 'fix_img')
    fix_tumors = tf.placeholder(tf.float32, IMG_SIZE_LARGE_x2, 'fix_tumors')
    fix_vessels = tf.placeholder(tf.float32, IMG_SIZE_LARGE_x2, 'fix_vessels')
    fix_parenchyma = tf.placeholder(tf.float32, IMG_SIZE_LARGE_x2, 'fix_parenchyma')
    large_point = tf.placeholder_with_default(input=tf.constant(False, tf.bool), shape=(), name='large_point')
    add_affine = tf.placeholder_with_default(input=tf.constant(False, tf.bool), shape=(), name='add_affine')

    search_voxels = tf.cond(tf.equal(tf.reduce_sum(fix_tumors), 0.0),
                            lambda: fix_parenchyma,
                            lambda: fix_tumors)

    # Apply TPS deformation
    # 1. get a point in the label img and add it to the control grid and target grid
    idx_points_in_label = tf.where(tf.greater(search_voxels, 0.0))  # Indices of the points in the label image with intensity greater than 0

    random_idx = tf.random.uniform([], minval=0, maxval=tf.shape(idx_points_in_label)[0],
                                  dtype=tf.int32)  # Randomly select one of the points
    disp_location = tf.gather_nd(idx_points_in_label, tf.expand_dims(random_idx, 0))  # And get the coordinates
    disp_location = tf.cast(disp_location, tf.float32)
    # Get the coordinates of the control point displaces
    rand_disp = tf.constant(np.random.uniform(-1, 1, 3) * const.MAX_DISP_DM, dtype=tf.float32)
    warped_location = disp_location + rand_disp

    def get_box_neighbours(location, radius=3):
        n1 = tf.add(rand_disp, tf.constant(np.asarray([radius, radius, radius]), location.dtype))
        n2 = tf.add(rand_disp, tf.constant(np.asarray([-radius, radius, radius]), location.dtype))
        n3 = tf.add(rand_disp, tf.constant(np.asarray([radius, -radius, radius]), location.dtype))
        n4 = tf.add(rand_disp, tf.constant(np.asarray([-radius, -radius, radius]), location.dtype))
        n5 = tf.add(rand_disp, tf.constant(np.asarray([radius, radius, -radius]), location.dtype))
        n6 = tf.add(rand_disp, tf.constant(np.asarray([-radius, radius, -radius]), location.dtype))
        n7 = tf.add(rand_disp, tf.constant(np.asarray([radius, -radius, -radius]), location.dtype))
        n8 = tf.add(rand_disp, tf.constant(np.asarray([-radius, -radius, -radius]), location.dtype))

        return tf.stack([location, n1, n2, n3, n4, n5, n6, n7, n8], 0)

    disp_location, warped_location = tf.cond(large_point,
                               lambda: (get_box_neighbours(disp_location, 3), get_box_neighbours(warped_location, 3)),
                               lambda: (tf.expand_dims(rand_disp, 0), tf.expand_dims(warped_location, 0)))

    # 2. Add the new point to the control grid and the target grid
    control_grid = tf.concat([CTRL_GRID.grid_flat(), disp_location], axis=0)
    trg_grid = tf.concat([CTRL_GRID.grid_flat(), warped_location], axis=0)

    trg_grid, aff = tf.cond(add_affine,
                            lambda: transform_points(trg_grid),
                            lambda: (trg_grid, tf.eye(4, 4)))

    # I need to know the shape before running TPS
    control_grid.set_shape([73, 3] if USE_LARGE_PT else [65, 3])
    trg_grid.set_shape([73, 3] if USE_LARGE_PT else [65, 3])

    tps = ThinPlateSplines(control_grid, trg_grid)
    def_grid = tps.interpolate(FULL_FINE_GRID.grid_flat())

    disp_map = def_grid - FULL_FINE_GRID.grid_flat()
    disp_map = tf.reshape(disp_map, (*FINE_GRID_SHAPE, -1))
    # disp_map = interpn(disp_map, FULL_FINE_GRID.grid)

    # add the batch and channel dimensions
    fix_img = tf.expand_dims(tf.expand_dims(fix_img, -1), 0)
    fix_tumors = tf.expand_dims(tf.expand_dims(fix_tumors, -1), 0)
    fix_vessels = tf.expand_dims(tf.expand_dims(fix_vessels, -1), 0)
    fix_parenchyma = tf.expand_dims(tf.expand_dims(fix_parenchyma, -1), 0)
    disp_map = tf.cast(tf.expand_dims(disp_map, 0), tf.float32)

    mov_tumors = SpatialTransformer(interp_method='linear', indexing='ij', single_transform=False)([fix_tumors, disp_map])
    mov_vessels = SpatialTransformer(interp_method='linear', indexing='ij', single_transform=False)([fix_vessels, disp_map])
    mov_parenchyma = SpatialTransformer(interp_method='linear', indexing='ij', single_transform=False)([fix_parenchyma, disp_map])
    mov_img = SpatialTransformer(interp_method='linear', indexing='ij', single_transform=False)([fix_img, disp_map])

    return tf.squeeze(mov_img),\
           tf.squeeze(mov_parenchyma),\
           tf.squeeze(mov_tumors),\
           tf.squeeze(mov_vessels),\
           tf.squeeze(disp_map),\
           disp_location,\
           rand_disp,\
           aff #, w, trg_grid, def_grid


if __name__ == '__main__':
    parse_arguments(sys.argv[1:])
    volume_list = [os.path.join(DATASTE_RAW_FILES, f) for f in os.listdir(DATASTE_RAW_FILES) if f.startswith(LITS_CT_FILE)]
    volume_list.sort()
    segmentation_list = [os.path.join(DATASTE_RAW_FILES, f) for f in os.listdir(DATASTE_RAW_FILES) if
                         f.startswith(LITS_SEGMENTATION_FILE)]
    segmentation_list.sort()

    file_path_pairs = [[v, s] for v, s in zip(volume_list, segmentation_list)]

    print('Generating HD5 files at {} ...', format(const.DESTINATION_FOLDER))
    # with Pool(10) as p, tf.Session(config=config) as sess:
    #     tqdm(p.map(generate_training_sample, file_path_pairs))
    intensity_window_w = 350
    intensity_window_l = 40
    intensity_clipping_range = intensity_window_l + np.asarray([-intensity_window_w // 2, intensity_window_w // 2],
                                                               np.int)  # Slicer range for abdominal CT

    try_mkdir(const.DESTINATION_FOLDER)

    print('PART 1: Deformation')
    # Then do the fancy stuff
    init = tf.initialize_all_variables()
    get_mov_img = tf_graph_deform()
    sess = tf.Session(config=config)
    with sess.as_default():
        sess.run(init)
        sess.graph.finalize()
        for img_path, labels_path in tqdm(file_path_pairs):
            if img_path is not None and labels_path is not None:
                #img_path = unzip_file(img_path)
                #labels_path = unzip_file(labels_path)

                fix_img = nib.load(img_path)  # By convention, nibabel world axes are always in RAS+ orientation
                img_header = fix_img.header
                fix_labels = nib.load(labels_path)
                fix_img = np.asarray(fix_img.dataobj)
                fix_labels = np.asarray(fix_labels.dataobj)
                if fix_labels.shape[-1] < 4:
                    print('[INF] ' + img_path + ' has no tumor segmentations')
                    continue
                # fix_artery = fix_labels[..., 0]
                fix_vessels = fix_labels[..., 1]
                fix_parenchyma = fix_labels[..., 2]
                fix_tumors = fix_labels[..., 3]

                # Clip intensity values
                fix_img = utils.intesity_clipping(fix_img, intensity_clipping_range, augment=True)

                # Reshape
                fix_img = resize(fix_img, IMG_SIZE_LARGE_x2)
                fix_parenchyma = resize(fix_parenchyma, IMG_SIZE_LARGE_x2)
                fix_tumors = resize(fix_tumors, IMG_SIZE_LARGE_x2)
                fix_vessels = resize(fix_vessels, IMG_SIZE_LARGE_x2)

                fix_parenchyma = median(fix_parenchyma, np.ones((5, 5, 5)))

                # Compute deformation
                mov_img, mov_parenchyma, mov_tumors, mov_vessels, disp_map, disp_loc, disp_vec, aff = sess.run(get_mov_img,
                                                                                   feed_dict={
                                                                                       'fix_img:0': fix_img,
                                                                                       'fix_tumors:0': fix_tumors,
                                                                                       'fix_vessels:0': fix_vessels,
                                                                                       'fix_parenchyma:0': fix_parenchyma,
                                                                                       'large_point:0': USE_LARGE_PT,
                                                                                       'add_affine:0': ADD_AFFINE_TRF})
                # Cleaning
                mov_img = utils.intesity_clipping(mov_img, intensity_clipping_range)

                if USE_LARGE_PT:
                    disp_loc = disp_loc[0, ...]

                # Define the bbox around the union of the parenchyma of both volumes, so none falls outside
                bbox_mask = np.sign(mov_parenchyma + fix_parenchyma)
                bbox_mask = binary_dilation(bbox_mask, DILATION_STRUCT)
                bbox_mask = binary_dilation(bbox_mask, DILATION_STRUCT).astype(np.float32)

                # The point of application is referred to the whole image coordinate, not to the local BB
                min_i, _, min_j, _, min_k, _ = bbox_3D(bbox_mask)
                disp_loc = (disp_loc - np.asarray([min_i, min_j, min_k])) / 2
                # Crop the image to only contain the liver
                # The origin moved according to the mask information. And the images will be resized in a factor of 2!!

                fix_img, _ = crop_images(fix_img, bbox_mask, IMG_SIZE_LARGE)
                fix_tumors, _ = crop_images(fix_tumors, bbox_mask, IMG_SIZE_LARGE)
                fix_vessels, _ = crop_images(fix_vessels, bbox_mask, IMG_SIZE_LARGE)
                disp_map, _ = crop_images(disp_map, bbox_mask, IMG_SIZE_LARGE)
                fix_parenchyma, _ = crop_images(fix_parenchyma, bbox_mask, IMG_SIZE_LARGE)

                # We will later crop even further, so we don't want to downsample too much
                # Crop the image to only contain the liver
                mov_img, _ = crop_images(mov_img, bbox_mask, IMG_SIZE_LARGE)
                mov_tumors, _ = crop_images(mov_tumors, bbox_mask, IMG_SIZE_LARGE)
                mov_vessels, _ = crop_images(mov_vessels, bbox_mask, IMG_SIZE_LARGE)
                mov_parenchyma, _ = crop_images(mov_parenchyma, bbox_mask, IMG_SIZE_LARGE)

                # Just to be sure we have binary masks
                fix_tumors[fix_tumors > TH_BIN] = 1.0
                fix_tumors[fix_tumors < 1.0] = 0.0

                fix_vessels[fix_vessels > TH_BIN] = 1.0
                fix_vessels[fix_vessels < 1.0] = 0.0

                fix_parenchyma[fix_parenchyma > TH_BIN] = 1.0
                fix_parenchyma[fix_parenchyma < 1.0] = 0.0

                mov_tumors[mov_tumors > TH_BIN] = 1.0
                mov_tumors[mov_tumors < 1.0] = 0.0

                mov_vessels[mov_vessels > TH_BIN] = 1.0
                mov_vessels[mov_vessels < 1.0] = 0.0

                mov_parenchyma[mov_parenchyma > TH_BIN] = 1.0
                mov_parenchyma[mov_parenchyma < 1.0] = 0.0

                # Save everything
                fix_img = np.expand_dims(fix_img, -1)
                fix_tumors = np.expand_dims(fix_tumors, -1)
                fix_vessels = np.expand_dims(fix_vessels, -1)
                fix_parenchyma = np.expand_dims(fix_parenchyma, -1)
                fix_segmentations = np.stack([fix_parenchyma, fix_vessels, fix_tumors], -1)

                mov_img = np.expand_dims(mov_img, -1)
                mov_tumors = np.expand_dims(mov_tumors, -1)
                mov_vessels = np.expand_dims(mov_vessels, -1)
                mov_parenchyma = np.expand_dims(mov_parenchyma, -1)

                # Save everything
                file_name = os.path.split(img_path)[-1].split('.')[0]
                vol_num = int(re.split('-|_', file_name)[-1])
                hd5_filename = 'volume-{:04d}'.format(vol_num + OFFSET_NAME_NUM)
                hd5_filename = os.path.join(const.DESTINATION_FOLDER, hd5_filename + '.hd5')
                hd5_file = h5py.File(hd5_filename, 'w')

                hd5_file.create_dataset(const.H5_FIX_IMG, data=fix_img, dtype='float32')
                hd5_file.create_dataset(const.H5_FIX_PARENCHYMA_MASK, data=fix_parenchyma, dtype='float32')
                hd5_file.create_dataset(const.H5_FIX_VESSELS_MASK, data=fix_vessels, dtype='float32')
                hd5_file.create_dataset(const.H5_FIX_TUMORS_MASK, data=fix_tumors, dtype='float32')
                hd5_file.create_dataset(const.H5_FIX_SEGMENTATIONS, data=fix_segmentations, dtype='float32')

                hd5_file.create_dataset(const.H5_PARAMS_INTENSITY_RANGE, (2,), data=intensity_clipping_range,
                                        dtype='float32')

                hd5_file.create_dataset(const.H5_MOV_IMG, const.IMG_SHAPE, data=mov_img, dtype='float32')
                hd5_file.create_dataset(const.H5_MOV_PARENCHYMA_MASK, const.IMG_SHAPE, data=mov_parenchyma,
                                        dtype='float32')
                hd5_file.create_dataset(const.H5_MOV_VESSELS_MASK, const.IMG_SHAPE, data=mov_vessels, dtype='float32')
                hd5_file.create_dataset(const.H5_MOV_TUMORS_MASK, const.IMG_SHAPE, data=mov_tumors, dtype='float32')
                hd5_file.create_dataset(const.H5_MOV_SEGMENTATIONS, data=fix_segmentations, dtype='float32')

                hd5_file.create_dataset(const.H5_GT_DISP, const.DISP_MAP_SHAPE, data=disp_map, dtype='float32')
                hd5_file.create_dataset(const.H5_GT_DISP_VECT_LOC, data=disp_loc, dtype='float32')
                hd5_file.create_dataset(const.H5_GT_DISP_VECT, data=disp_vec, dtype='float32')
                hd5_file.create_dataset(const.H5_GT_AFFINE_M, data=aff, dtype='float32')

                hd5_file.create_dataset('params/voxel_size', data=img_header.get_zooms()[:3])
                hd5_file.create_dataset('params/original_shape', data=img_header.get_data_shape())
                hd5_file.create_dataset('params/bbox_origin', data=[min_i, min_j, min_k])
                hd5_file.create_dataset('params/first_reshape', data=IMG_SIZE_LARGE_x2)

                # delete_temp(img_path)
                # delete_temp(labels_path)

                hd5_file.close()
        sess.close()
    print('...Done generating HD5 files')
