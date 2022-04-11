import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)  # PYTHON > 3.3 does not allow relative referencing

import h5py
from tqdm import tqdm
from functools import partial
import numpy as np
from scipy.spatial.distance import euclidean
import pandas as pd
from EvaluationScripts.Evaluate_class import resize_img_to_original_space, resize_pts_to_original_space
from Centerline.visualization_utils import plot_cpd_registration_step, plot_cpd
from Centerline.cpd_utils import cpd_non_rigid_transform_pt, radial_basis_function, deform_registration, rigid_registration
from scipy.spatial.distance import cdist
import re

DATASET_LOCATION = '/mnt/EncryptedData1/Users/javier/vessel_registration/3Dirca/dataset/EVAL'
DATASET_NAMES = ['Affine', 'None', 'Translation']
DATASET_FILENAME = 'points'

OUT_IMG_FOLDER = '/mnt/EncryptedData1/Users/javier/vessel_registration/Centerline/cpd/nodes_final'

SCALE = 1e-2  # mm to cm

# CPD PARAMS (deform)
MAX_ITER = 200
ALPHA = 2.
BETA = 2.  # None = Use default
TOLERANCE = 1e-8
RBF_FUNCTION='thin-plate'

if __name__ == '__main__':
    for dataset_name in DATASET_NAMES:
        dataset_loc = os.path.join(DATASET_LOCATION, dataset_name)
        dataset_files = os.listdir(dataset_loc)
        dataset_files.sort()
        dataset_files = [os.path.join(dataset_loc, f) for f in dataset_files if DATASET_FILENAME in f]

        iterator = tqdm(dataset_files)
        df = pd.DataFrame(columns=['DATASET',
                                   'ITERATIONS_DEF', 'ITERATIONS_R_DEF__R', 'ITERATIONS_R_DEF__DEF',
                                   'TIME_DEF', 'TIME_R_DEF',
                                   'Q_DEF', 'Q_R_DEF__R', 'Q_R_DEF__DEF',
                                   'TRE_DEF', 'TRE_R_DEF',
                                   'DS_DISP',
                                   'DATA_PATH',
                                   'DIST_CENTR', 'DIST_CENTR_DEF_95', 'SAMPLE_NUM'])
        for i, file_path in enumerate(iterator):
            fn = os.path.split(file_path)[-1].split('.hd5')[0]
            fnum = int(re.findall('(\d+)', fn)[0])
            iterator.set_description('{}: start'.format(fn))
            pts_file = h5py.File(file_path, 'r')
            fix_pts = pts_file['fix/points'][:]
            fix_nodes = pts_file['fix/nodes'][:]
            fix_centroid = pts_file['fix/centroid'][:]

            mov_pts = pts_file['mov/points'][:]
            mov_nodes = pts_file['mov/nodes'][:]
            mov_centroid = pts_file['mov/centroid'][:]

            bbox = pts_file['parameters/bbox'][:]
            first_reshape = pts_file['parameters/first_reshape'][:]
            isotropic_shape = pts_file['parameters/isotropic_shape'][:]
            iterator.set_description('{}: Loaded data'.format(fn))
            # TODO: bring back to original shape!
            # Reshape to original_shape
            fix_nodes = resize_pts_to_original_space(fix_nodes, bbox, [64] * 3, first_reshape, isotropic_shape)
            fix_pts = resize_pts_to_original_space(fix_pts, bbox, [64] * 3, first_reshape, isotropic_shape)
            fix_centroid = resize_pts_to_original_space(fix_centroid, bbox, [64] * 3, first_reshape, isotropic_shape)
            mov_nodes = resize_pts_to_original_space(mov_nodes, bbox, [64] * 3, first_reshape, isotropic_shape)
            mov_pts = resize_pts_to_original_space(mov_pts, bbox, [64] * 3, first_reshape, isotropic_shape)
            mov_centroid = resize_pts_to_original_space(mov_centroid, bbox, [64] * 3, first_reshape, isotropic_shape)
            iterator.set_description('{}: reshaped data'.format(fn))

            if mov_nodes.shape[0] == 1:
                # Otherwise we only have a point, and CPD can't handle that... absurd!
                fix_nodes = fix_pts
                mov_nodes = mov_pts

            ill_cond_def = False
            ill_cond_r_def = False
            # Deformable only
            iterator.set_description('{}: Computing only deformable reg.'.format(fn))

            # deform_cb = partial(plot_cpd_registration_step,
            # out_folder=os.path.join(OUT_IMG_FOLDER, '{}/{:04d}/DEF'.format(dataset_name, fnum)))

            # _, _, deform_reg_def = deform_registration(fix_nodes, mov_nodes, deform_cb)
            time_def, deform_reg_def = deform_registration(fix_nodes*SCALE, mov_nodes*SCALE, time_it=True,
                                                           tolerance=TOLERANCE, max_iterations=MAX_ITER,
                                                           alpha=ALPHA, beta=BETA)
            if np.isnan(deform_reg_def.diff):
                tre_def = np.nan
                pred_mov_centroid = np.zeros((3,))
            else:
                tps, ill_cond_def = radial_basis_function(mov_nodes, np.dot(*deform_reg_def.get_registration_parameters()) / SCALE, RBF_FUNCTION)
                displacement_mov_centroid = tps(mov_centroid)
                pred_mov_centroid = mov_centroid + displacement_mov_centroid

                tre_def = euclidean(pred_mov_centroid, fix_centroid)

            plot_file = os.path.join(OUT_IMG_FOLDER, '{}/{:04d}/DEF'.format(dataset_name, fnum))
            os.makedirs(plot_file, exist_ok=True)
            plot_cpd(fix_nodes, mov_nodes, fix_centroid, mov_centroid, plot_file + '/before_registration')
            plot_cpd(fix_nodes, deform_reg_def.TY/SCALE, fix_centroid, pred_mov_centroid, plot_file + '/after_registration')

            # Rigid followed by deformable
            iterator.set_description('{}: Computing rigid and deform. reg.'.format(fn))

            # rigid_cb = partial(plot_cpd_registration_step, out_folder=os.path.join(OUT_IMG_FOLDER, '{}/{:04d}/RIGID_DEF/rigid'.format(dataset_name, fnum)))
            # deform_cb = partial(plot_cpd_registration_step, out_folder=os.path.join(OUT_IMG_FOLDER, '{}/{:04d}/RIGID_DEF/deform'.format(dataset_name, fnum)))
            # rigid_yt, rigid_p, rigid_reg_r_def = rigid_registration(fix_nodes, mov_nodes, rigid_cb)
            # deform_yt, deform_p, deform_reg_r_def = deform_registration(fix_nodes, rigid_yt, deform_cb)

            time_r_def__r, rigid_reg_r_def = rigid_registration(fix_nodes*SCALE, mov_nodes*SCALE, time_it=True)
            rigid_yt = rigid_reg_r_def.TY
            time_r_def__def, deform_reg_r_def = deform_registration(fix_nodes*SCALE, rigid_yt, time_it=True,
                                                                    tolerance=TOLERANCE, max_iterations=MAX_ITER,
                                                                    alpha=ALPHA, beta=BETA)

            if np.isnan(deform_reg_r_def.diff):
                pred_mov_centroid = rigid_reg_r_def.transform_point_cloud(mov_centroid*SCALE)/SCALE
            else:
                mov_centroid_t = rigid_reg_r_def.transform_point_cloud(mov_centroid*SCALE)/SCALE
                tps, ill_cond_r_def = radial_basis_function(rigid_yt / SCALE, np.dot(*deform_reg_r_def.get_registration_parameters()) / SCALE, RBF_FUNCTION)
                displacement_mov_centroid_t = tps(mov_centroid_t)
                pred_mov_centroid = mov_centroid_t + displacement_mov_centroid_t

            tre_r_def = euclidean(pred_mov_centroid, fix_centroid)
            dist_centroid_to_pts = cdist(mov_centroid[np.newaxis, ...], mov_nodes)

            plot_file = os.path.join(OUT_IMG_FOLDER, '{}/{:04d}/RIGID_DEF'.format(dataset_name, fnum))
            os.makedirs(plot_file, exist_ok=True)
            plot_cpd(fix_nodes, mov_nodes, fix_centroid, mov_centroid, plot_file + '/before_registration')
            plot_cpd(fix_nodes, deform_reg_r_def.TY/SCALE, fix_centroid, pred_mov_centroid, plot_file + '/after_registration')

            iterator.set_description('{}: Saving data'.format(fn))
            df = df.append({'DATASET':dataset_name,
                       'ITERATIONS_DEF': deform_reg_def.iteration,
                       'ITERATIONS_R_DEF__R': rigid_reg_r_def.iteration,
                       'ITERATIONS_R_DEF__DEF': deform_reg_r_def.iteration,
                       'TIME_DEF': time_def,
                       'TIME_R_DEF': time_r_def__r + time_r_def__def,
                       'Q_DEF': deform_reg_def.diff,
                       'Q_R_DEF__R': rigid_reg_r_def.q,
                       'Q_R_DEF__DEF': deform_reg_r_def.diff,
                       'ILL_COND_DEF': ill_cond_def,
                       'ILL_COND_R_DEF': ill_cond_r_def,
                       'TRE_DEF':tre_def, 'TRE_R_DEF':tre_r_def,
                       'DS_DISP':euclidean(mov_centroid, fix_centroid),
                       'DATA_PATH':file_path,
                       'DIST_CENTR':np.min(dist_centroid_to_pts),
                       'DIST_CENTR_DEF_95':np.percentile(dist_centroid_to_pts, 95),
                       'SAMPLE_NUM':fnum}, ignore_index=True)
            pts_file.close()

        df.to_csv(os.path.join(OUT_IMG_FOLDER, 'cpd_{}.csv'.format(dataset_name)))
