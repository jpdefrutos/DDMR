import os, sys

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)  # PYTHON > 3.3 does not allow relative referencing

# import tensorflow as tf
# tf.enable_eager_execution()
# import neurite.py.utils as neurite_utils

from skimage.morphology import skeletonize_3d, ball
from skimage.morphology import binary_closing, binary_opening
from skimage.filters import median
from skimage.measure import regionprops, label
from skimage.transform import warp

from scipy.ndimage import zoom
from scipy.interpolate import LinearNDInterpolator, Rbf

import h5py
import numpy as np
from tqdm import tqdm
import re
import nibabel as nib
from nilearn.image import resample_img

from Centerline.graph_utils import graph_to_ndarray, deform_graph, get_bifurcation_nodes, subsample_graph, \
    apply_displacement
from Centerline.skeleton_to_graph import get_graph_from_skeleton
from Centerline.visualization_utils import plot_skeleton, compare_graphs

from DeepDeformationMapRegistration.utils.operators import min_max_norm
from DeepDeformationMapRegistration.utils import constants as C

import cupy
from cupyx.scipy.ndimage import zoom as zoom_gpu
from cupyx.scipy.ndimage import map_coordinates

DATASET_LOCATION = '/mnt/EncryptedData1/Users/javier/vessel_registration/3Dirca/dataset/EVAL'
DATASET_NAMES = ['Affine', 'None', 'Translation']
DATASET_FILENAME = 'volume'
IMGS_FOLDER = '/home/jpdefrutos/workspace/DeepDeformationMapRegistration/Centerline/centerlines'

DATASTE_RAW_FILES = '/mnt/EncryptedData1/Users/javier/vessel_registration/3Dirca/nifti3'
LITS_SEGMENTATION_FILE = 'segmentation'
LITS_CT_FILE = 'volume'


def warp_volume(volume, disp_map, indexing='ij'):
    assert indexing is 'ij' or 'xy', 'Invalid indexing option. Only "ij" or "xy"'
    grid_i = np.linspace(0, disp_map.shape[0], disp_map.shape[0], endpoint=False)
    grid_j = np.linspace(0, disp_map.shape[1], disp_map.shape[1], endpoint=False)
    grid_k = np.linspace(0, disp_map.shape[2], disp_map.shape[2], endpoint=False)
    grid_i, grid_j, grid_k = np.meshgrid(grid_i, grid_j, grid_k, indexing=indexing)
    grid_i = (grid_i.flatten() + disp_map[..., 0].flatten())[..., np.newaxis]
    grid_j = (grid_j.flatten() + disp_map[..., 1].flatten())[..., np.newaxis]
    grid_k = (grid_k.flatten() + disp_map[..., 2].flatten())[..., np.newaxis]
    coords = np.hstack([grid_i, grid_j, grid_k]).reshape([*disp_map.shape[:-1], -1])
    coords = coords.transpose((-1, 0, 1, 2))
    # The returned volume has indexing xy
    return warp(volume, coords)


def keep_largest_segmentation(img):
    label_img = label(img)
    rp = regionprops(label_img)  # Regions labeled with 0 (bg) are ignored
    biggest_area = (0, 0)
    for l in range(0, label_img.max()):
        if rp[l].area > biggest_area[1]:
            biggest_area = (l + 1, rp[l].area)
    img[label_img != biggest_area[0]] = 0.
    return img


def preprocess_image(img, keep_largest=False):
    ret = binary_closing(img, ball(1))
    ret = binary_opening(ret, ball(1))
    #ret = median(ret, ball(1), mode='constant')
    if keep_largest:
        ret = keep_largest_segmentation(ret)
    return ret.astype(np.float)


def build_displacement_map_interpolator(disp_map, backwards=False, indexing='ij'):
    grid_i = np.linspace(0, disp_map.shape[0], disp_map.shape[0], endpoint=False)
    grid_j = np.linspace(0, disp_map.shape[1], disp_map.shape[1], endpoint=False)
    grid_k = np.linspace(0, disp_map.shape[2], disp_map.shape[2], endpoint=False)
    grid_i, grid_j, grid_k = np.meshgrid(grid_i, grid_j, grid_k, indexing=indexing)
    grid_i = grid_i.flatten()
    grid_j = grid_j.flatten()
    grid_k = grid_k.flatten()
    # To generate the moving image, we used backwards mapping were the input was the fix image
    # Now we are doing direct mapping from the fix graph coordinates to the moving coordinates
    # The application points of the displacement map are thus the transformed "moving image"-grid
    #   and the displacement vectors are reversed
    if backwards:
        coords = np.hstack([grid_i[..., np.newaxis], grid_j[..., np.newaxis], grid_k[..., np.newaxis]])
        return LinearNDInterpolator(coords, np.reshape(disp_map, [-1, 3]))
    else:
        grid_i = (grid_i + disp_map[..., 0].flatten())
        grid_j = (grid_j + disp_map[..., 1].flatten())
        grid_k = (grid_k + disp_map[..., 2].flatten())

        coords = np.hstack([grid_i[..., np.newaxis], grid_j[..., np.newaxis], grid_k[..., np.newaxis]])
        return LinearNDInterpolator(coords, -np.reshape(disp_map, [-1, 3]))


def resample_segmentation(img, output_shape, preserve_range, threshold=None, gpu=True):
    # Preserve range can be a bool (keep or not the original dyn. range) or a list with a new dyn. range
    zoom_f = np.divide(np.asarray(output_shape), np.asarray(img.shape))

    if gpu:
        out_img = zoom_gpu(cupy.asarray(img), zoom_f, order=1)  # order = 0 or 1
    else:
        out_img = zoom(img, zoom_f)
    if isinstance(preserve_range, bool):
        if preserve_range:
            range_min, range_max = np.amin(img), np.amax(img)
            out_img = min_max_norm(out_img)
            out_img = out_img * (range_max - range_min) + range_min
    elif isinstance(preserve_range, list):
        range_min, range_max = preserve_range
        out_img = min_max_norm(out_img)
        out_img = out_img * (range_max - range_min) + range_min

    if threshold is not None and out_img.min() < threshold < out_img.max():
        range_min, range_max = np.amin(out_img), np.amax(out_img)
        out_img[out_img > threshold] = range_max
        out_img[out_img < range_max] = range_min
    return cupy.asnumpy(out_img) if gpu else out_img


if __name__ == '__main__':
    for dataset_name in DATASET_NAMES:
        dataset_loc = os.path.join(DATASET_LOCATION, dataset_name)
        dataset_files = os.listdir(dataset_loc)
        dataset_files.sort()
        dataset_files = [os.path.join(dataset_loc, f) for f in dataset_files if DATASET_FILENAME in f]

        iterator = tqdm(dataset_files)
        for file_path in iterator:
            file_num = int(re.findall('(\d+)', os.path.split(file_path)[-1])[0])

            iterator.set_description('{} ({}): laoding data'.format(file_num, dataset_name))
            vol_file = h5py.File(file_path, 'r')
            # fix_vessels = vol_file[C.H5_FIX_VESSELS_MASK][..., 0]
            disp_map = vol_file[C.H5_GT_DISP][:]
            bbox = vol_file['parameters/bbox'][:]
            bbox_min = bbox[:3]
            bbox_max = bbox[3:] + bbox_min

            # Load vessel segmentation mask and resize to 64^3
            fix_labels = nib.load(os.path.join(DATASTE_RAW_FILES, 'segmentation-{:04d}.nii.gz'.format(file_num)))
            fix_vessels = fix_labels.slicer[..., 1]
            fix_vessels = resample_img(fix_vessels, np.eye(3))
            fix_vessels = np.asarray(fix_vessels.dataobj)
            fix_vessels = preprocess_image(fix_vessels)
            fix_vessels = resample_segmentation(fix_vessels, vol_file['parameters/first_reshape'][:], [0, 1], 0.3,
                                                gpu=True)
            fix_vessels = fix_vessels[bbox_min[0]:bbox_max[0], bbox_min[1]:bbox_max[1], bbox_min[2]:bbox_max[2]]
            fix_vessels = resample_segmentation(fix_vessels, [64] * 3, [0, 1], 0.3, gpu=True)
            fix_vessels = preprocess_image(fix_vessels)

            mov_vessels = preprocess_image(warp_volume(fix_vessels, disp_map))
            mov_skel = skeletonize_3d(mov_vessels)
            ### Fix the incorrect scaling ###
            disp_map *= 2
            bbox_size = np.asarray(bbox[3:])  # Only load the bbox size
            rescale_factors = 64 / bbox_size

            disp_map[..., 0] = np.multiply(disp_map[..., 0], rescale_factors[0])
            disp_map[..., 1] = np.multiply(disp_map[..., 1], rescale_factors[1])
            disp_map[..., 2] = np.multiply(disp_map[..., 2], rescale_factors[2])
            #################################

            iterator.set_description('{} ({}): getting graphs'.format(file_num, dataset_name))
            # Prepare displacement map
            disp_map_interpolator = build_displacement_map_interpolator(disp_map, backwards=False)

            # Get skeleton and graph
            fix_skel = skeletonize_3d(fix_vessels)
            fix_graph = get_graph_from_skeleton(fix_skel, subsample=True)
            mov_graph = get_graph_from_skeleton(mov_skel, subsample=True) # deform_graph(fix_graph, disp_map_interpolator)

            ##### TODO: ERASE Check the mov graph ######
            # check_mov_vessels = vol_file[C.H5_MOV_VESSELS_MASK][..., 0]
            # check_mov_vessels = preprocess_image(check_mov_vessels)
            # check_mov_skel = skeletonize_3d(check_mov_vessels)
            # check_mov_graph = get_graph_from_skeleton(check_mov_skel, subsample=True)
            ###########
            fix_pts, fix_nodes, fix_edges = graph_to_ndarray(fix_graph)
            mov_pts, mov_nodes, mov_edges = graph_to_ndarray(mov_graph)

            fix_bifur_loc, fix_bifur_id = get_bifurcation_nodes(fix_graph)
            mov_bifur_loc, mov_bifur_id = get_bifurcation_nodes(mov_graph)

            iterator.set_description('{} ({}): saving data'.format(file_num, dataset_name))
            pts_file_path, pts_file_name = os.path.split(file_path)
            pts_file_name = pts_file_name.replace(DATASET_FILENAME, 'points')
            pts_file_path = os.path.join(pts_file_path, pts_file_name)
            pts_file = h5py.File(pts_file_path, 'w')

            pts_file.create_dataset('fix/points', data=fix_pts)
            pts_file.create_dataset('fix/nodes', data=fix_nodes)
            pts_file.create_dataset('fix/edges', data=fix_edges)
            pts_file.create_dataset('fix/bifurcations', data=fix_bifur_loc)
            pts_file.create_dataset('fix/graph', data=fix_graph)
            pts_file.create_dataset('fix/img', data=fix_vessels)
            pts_file.create_dataset('fix/skeleton', data=fix_skel)
            pts_file.create_dataset('fix/centroid', data=vol_file[C.H5_FIX_CENTROID][:])

            pts_file.create_dataset('mov/points', data=mov_pts)
            pts_file.create_dataset('mov/nodes', data=mov_nodes)
            pts_file.create_dataset('mov/edges', data=mov_edges)
            pts_file.create_dataset('mov/bifurcations', data=mov_bifur_loc)
            pts_file.create_dataset('mov/graph', data=mov_graph)
            pts_file.create_dataset('mov/img', data=mov_vessels)
            pts_file.create_dataset('mov/skeleton', data=mov_skel)
            pts_file.create_dataset('mov/centroid', data=vol_file[C.H5_MOV_CENTROID][:])

            pts_file.create_dataset('parameters/voxel_size', data=vol_file['parameters/voxel_size'][:])
            pts_file.create_dataset('parameters/original_affine', data=vol_file['parameters/original_affine'][:])
            pts_file.create_dataset('parameters/isotropic_affine', data=vol_file['parameters/isotropic_affine'][:])
            pts_file.create_dataset('parameters/original_shape', data=vol_file['parameters/original_shape'][:])
            pts_file.create_dataset('parameters/isotropic_shape', data=vol_file['parameters/isotropic_shape'][:])
            pts_file.create_dataset('parameters/first_reshape', data=vol_file['parameters/first_reshape'][:])
            pts_file.create_dataset('parameters/bbox', data=vol_file['parameters/bbox'][:])
            pts_file.create_dataset('parameters/last_reshape', data=vol_file['parameters/last_reshape'][:])

            pts_file.create_dataset('displacement_map', data=disp_map)

            vol_file.close()
            pts_file.close()

            iterator.set_description('{} ({}): drawing plots'.format(file_num, dataset_name))
            num = pts_file_name.split('-')[-1].split('.hd5')[0]
            imgs_folder = os.path.join(IMGS_FOLDER, dataset_name, num)
            os.makedirs(imgs_folder, exist_ok=True)
            plot_skeleton(fix_vessels, fix_skel, fix_graph, os.path.join(imgs_folder, 'fix'), ['.pdf', '.png'])
            plot_skeleton(mov_vessels, mov_skel, mov_graph, os.path.join(imgs_folder, 'mov'), ['.pdf', '.png'])
            iterator.set_description('{} ({})'.format(file_num, dataset_name))
