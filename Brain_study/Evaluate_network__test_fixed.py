import os, sys

import shutil
import time

import h5py
import matplotlib.pyplot as plt

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)  # PYTHON > 3.3 does not allow relative referencing

import tensorflow as tf
# tf.enable_eager_execution(config=config)

import numpy as np
import pandas as pd
import voxelmorph as vxm

import DeepDeformationMapRegistration.utils.constants as C
from DeepDeformationMapRegistration.utils.nifti_utils import save_nifti
from DeepDeformationMapRegistration.layers import AugmentationLayer, UncertaintyWeighting
from DeepDeformationMapRegistration.losses import StructuralSimilarity_simplified, NCC, GeneralizedDICEScore, HausdorffDistanceErosion, target_registration_error
from DeepDeformationMapRegistration.ms_ssim_tf import MultiScaleStructuralSimilarity
from DeepDeformationMapRegistration.utils.acummulated_optimizer import AdamAccumulated
from DeepDeformationMapRegistration.utils.visualization import save_disp_map_img, plot_predictions
from DeepDeformationMapRegistration.utils.misc import DisplacementMapInterpolator, get_segmentations_centroids, segmentation_ohe_to_cardinal
from EvaluationScripts.Evaluate_class import EvaluationFigures, resize_pts_to_original_space, resize_img_to_original_space, resize_transformation
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm

import h5py
import re
from Brain_study.data_generator import BatchGenerator

import argparse

from skimage.transform import warp
import neurite as ne

DATASET = '/mnt/EncryptedData1/Users/javier/ext_datasets/IXI_dataset/T1/training'
MODEL_FILE = '/mnt/EncryptedData1/Users/javier/train_output/Brain_study/ERASE/MS_SSIM/BASELINE_L_ssim__MET_mse_ncc_ssim_162756-29062021/checkpoints/best_model.h5'
DATA_ROOT_DIR = '/mnt/EncryptedData1/Users/javier/train_output/Brain_study/ERASE/MS_SSIM/BASELINE_L_ssim__MET_mse_ncc_ssim_162756-29062021/'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', nargs='+', type=str, help='.h5 of the model', default=None)
    parser.add_argument('-d', '--dir', nargs='+', type=str, help='Directory where ./checkpoints/best_model.h5 is located', default=None)
    parser.add_argument('--gpu', type=int, help='GPU', default=0)
    parser.add_argument('--dataset', type=str, help='Dataset to run predictions on',
                        default='/mnt/EncryptedData1/Users/javier/ext_datasets/IXI_dataset/T1/training')
    parser.add_argument('--erase', type=bool, help='Erase the content of the output folder', default=False)
    parser.add_argument('--outdirname', type=str, default='Evaluate')
    args = parser.parse_args()
    if args.model is not None:
        assert '.h5' in args.model[0], 'No checkpoint file provided, use -d/--dir instead'
        MODEL_FILE_LIST = args.model
        DATA_ROOT_DIR_LIST = [os.path.split(model_path)[0] for model_path in args.model]
    elif args.dir is not None:
        assert '.h5' not in args.dir[0], 'Provided checkpoint file, user -m/--model instead'
        MODEL_FILE_LIST = [os.path.join(dir_path, 'checkpoints', 'best_model.h5') for dir_path in args.dir]
        DATA_ROOT_DIR_LIST = args.dir
    else:
        raise ValueError("Provide either the model file or the directory ./containing checkpoints/best_model.h5")

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)  # Check availability before running using 'nvidia-smi'
    DATASET = args.dataset
    list_test_files = [os.path.join(DATASET, f) for f in os.listdir(DATASET) if f.endswith('h5') and 'dm' not in f]
    list_test_files.sort()

    with h5py.File(list_test_files[0], 'r') as f:
        image_input_shape = image_output_shape = list(f['fix_image'][:].shape[:-1])
        nb_labels = f['fix_segmentations'][:].shape[-1]

    # Header of the metrics csv file
    csv_header = ['File', 'SSIM', 'MS-SSIM', 'NCC', 'MSE', 'DICE', 'DICE_MACRO', 'HD', 'Time', 'TRE', 'No_missing_lbls', 'Missing_lbls']

    # TF stuff
    config = tf.compat.v1.ConfigProto()  # device_count={'GPU':0})
    config.gpu_options.allow_growth = True
    config.log_device_placement = False  ## to log device placement (on which device the operation ran)
    config.allow_soft_placement = True

    sess = tf.Session(config=config)
    tf.keras.backend.set_session(sess)

    # Loss and metric functions. Common to all models
    loss_fncs = [StructuralSimilarity_simplified(patch_size=2, dim=3, dynamic_range=1.).loss,
                 NCC(image_input_shape).loss,
                 vxm.losses.MSE().loss,
                 MultiScaleStructuralSimilarity(max_val=1., filter_size=3).loss]

    metric_fncs = [StructuralSimilarity_simplified(patch_size=2, dim=3, dynamic_range=1.).metric,
                   NCC(image_input_shape).metric,
                   vxm.losses.MSE().loss,
                   MultiScaleStructuralSimilarity(max_val=1., filter_size=3).metric,
                   GeneralizedDICEScore(image_output_shape + [nb_labels], num_labels=nb_labels).metric,
                   HausdorffDistanceErosion(3, 10, im_shape=image_output_shape + [nb_labels]).metric,
                   GeneralizedDICEScore(image_output_shape + [nb_labels], num_labels=nb_labels).metric_macro]

    ### METRICS GRAPH ###
    fix_img_ph = tf.placeholder(tf.float32, (1, *image_output_shape, 1), name='fix_img')
    pred_img_ph = tf.placeholder(tf.float32, (1, *image_output_shape, 1), name='pred_img')
    fix_seg_ph = tf.placeholder(tf.float32, (1, *image_output_shape, nb_labels), name='fix_seg')
    pred_seg_ph = tf.placeholder(tf.float32, (1, *image_output_shape, nb_labels), name='pred_seg')

    ssim_tf = metric_fncs[0](fix_img_ph, pred_img_ph)
    ncc_tf = metric_fncs[1](fix_img_ph, pred_img_ph)
    mse_tf = metric_fncs[2](fix_img_ph, pred_img_ph)
    ms_ssim_tf = metric_fncs[3](fix_img_ph, pred_img_ph)
    dice_tf = metric_fncs[4](fix_seg_ph, pred_seg_ph)
    hd_tf = metric_fncs[5](fix_seg_ph, pred_seg_ph)
    dice_macro_tf = metric_fncs[6](fix_seg_ph, pred_seg_ph)
    # hd_exact_tf = HausdorffDistance_exact(fix_seg_ph, pred_seg_ph, ohe=True)

    # Needed for VxmDense type of network
    warp_segmentation = vxm.networks.Transform(image_output_shape, interp_method='nearest', nb_feats=nb_labels)

    dm_interp = DisplacementMapInterpolator(image_output_shape, 'griddata')

    for MODEL_FILE, DATA_ROOT_DIR in zip(MODEL_FILE_LIST, DATA_ROOT_DIR_LIST):
        print('MODEL LOCATION: ', MODEL_FILE)

        # data_folder = '/mnt/EncryptedData1/Users/javier/train_output/DDMR/THESIS/BASELINE_Affine_ncc___mse_ncc_160606-25022021'
        output_folder = os.path.join(DATA_ROOT_DIR, args.outdirname)  # '/mnt/EncryptedData1/Users/javier/train_output/DDMR/THESIS/eval/BASELINE_TRAIN_Affine_ncc_EVAL_Affine'
        # os.makedirs(os.path.join(output_folder, 'images'), exist_ok=True)
        if args.erase:
            shutil.rmtree(output_folder, ignore_errors=True)
        os.makedirs(output_folder, exist_ok=True)
        print('DESTINATION FOLDER: ', output_folder)

        try:
            network = tf.keras.models.load_model(MODEL_FILE, {'VxmDenseSemiSupervisedSeg': vxm.networks.VxmDenseSemiSupervisedSeg,
                                                              'VxmDense': vxm.networks.VxmDense,
                                                              'AdamAccumulated': AdamAccumulated,
                                                              'loss': loss_fncs,
                                                              'metric': metric_fncs},
                                                 compile=False)
        except ValueError as e:
            enc_features = [16, 32, 32, 32]     # const.ENCODER_FILTERS
            dec_features = [32, 32, 32, 32, 32, 16, 16]     # const.ENCODER_FILTERS[::-1]
            nb_features = [enc_features, dec_features]
            if re.search('^UW|SEGGUIDED_', MODEL_FILE):
                network = vxm.networks.VxmDenseSemiSupervisedSeg(inshape=image_output_shape,
                                                                 nb_labels=nb_labels,
                                                                 nb_unet_features=nb_features,
                                                                 int_steps=0,
                                                                 int_downsize=1,
                                                                 seg_downsize=1)
            else:
                network = vxm.networks.VxmDense(inshape=image_output_shape,
                                                nb_unet_features=nb_features,
                                                int_steps=0)
            network.load_weights(MODEL_FILE, by_name=True)
        # Record metrics
        metrics_file = os.path.join(output_folder, 'metrics.csv')
        with open(metrics_file, 'w') as f:
            f.write(';'.join(csv_header)+'\n')

        ssim = ncc = mse = ms_ssim = dice = hd = 0
        with sess.as_default():
            sess.run(tf.global_variables_initializer())
            network.load_weights(MODEL_FILE, by_name=True)
            progress_bar = tqdm(enumerate(list_test_files, 1), desc='Evaluation', total=len(list_test_files))
            for step, in_batch in progress_bar:
                with h5py.File(in_batch, 'r') as f:
                    fix_img = f['fix_image'][:][np.newaxis, ...]    # Add batch axis
                    mov_img = f['mov_image'][:][np.newaxis, ...]
                    fix_seg = f['fix_segmentations'][:][np.newaxis, ...].astype(np.float32)
                    mov_seg = f['mov_segmentations'][:][np.newaxis, ...].astype(np.float32)
                    fix_centroids = f['fix_centroids'][:]

                if network.name == 'vxm_dense_semi_supervised_seg':
                    t0 = time.time()
                    pred_img, disp_map, pred_seg = network.predict([mov_img, fix_img, mov_seg, fix_seg])    # predict([source, target])
                    t1 = time.time()
                else:
                    t0 = time.time()
                    pred_img, disp_map = network.predict([mov_img, fix_img])
                    pred_seg = warp_segmentation.predict([mov_seg, disp_map])
                    t1 = time.time()

                mov_centroids, missing_lbls = get_segmentations_centroids(mov_seg[0, ...], ohe=True, expected_lbls=range(0, 28))
                # pred_centroids = dm_interp(disp_map, mov_centroids, backwards=True)  # with tps, it returns the pred_centroids directly
                pred_centroids = dm_interp(disp_map, mov_centroids, backwards=True) + mov_centroids

                # I need the labels to be OHE to compute the segmentation metrics.
                dice, hd, dice_macro = sess.run([dice_tf, hd_tf, dice_macro_tf], {'fix_seg:0': fix_seg, 'pred_seg:0': pred_seg})

                pred_seg_card = segmentation_ohe_to_cardinal(pred_seg).astype(np.float32)
                mov_seg_card = segmentation_ohe_to_cardinal(mov_seg).astype(np.float32)
                fix_seg_card = segmentation_ohe_to_cardinal(fix_seg).astype(np.float32)

                ssim, ncc, mse, ms_ssim = sess.run([ssim_tf, ncc_tf, mse_tf, ms_ssim_tf], {'fix_img:0': fix_img, 'pred_img:0': pred_img})
                ms_ssim = ms_ssim[0]

                # Rescale the points back to isotropic space, where we have a correspondence voxel <-> mm
                upsample_scale = 128 / 64
                fix_centroids_isotropic = fix_centroids * upsample_scale
                # mov_centroids_isotropic = mov_centroids * upsample_scale
                pred_centroids_isotropic = pred_centroids * upsample_scale

                fix_centroids_isotropic = np.divide(fix_centroids_isotropic, C.IXI_DATASET_iso_to_cubic_scales)
                # mov_centroids_isotropic = np.divide(mov_centroids_isotropic, C.IXI_DATASET_iso_to_cubic_scales)
                pred_centroids_isotropic = np.divide(pred_centroids_isotropic, C.IXI_DATASET_iso_to_cubic_scales)
                # Now we can measure the TRE in mm
                tre_array = target_registration_error(fix_centroids_isotropic, pred_centroids_isotropic, False).eval()
                tre = np.mean([v for v in tre_array if not np.isnan(v)])
                # ['File', 'SSIM', 'MS-SSIM', 'NCC', 'MSE', 'DICE', 'HD', 'Time', 'TRE', 'No_missing_lbls', 'Missing_lbls']

                new_line = [step, ssim, ms_ssim, ncc, mse, dice, dice_macro, hd, t1-t0, tre, len(missing_lbls), missing_lbls]
                with open(metrics_file, 'a') as f:
                    f.write(';'.join(map(str, new_line))+'\n')

                save_nifti(fix_img[0, ...], os.path.join(output_folder, '{:03d}_fix_img_ssim_{:.03f}_dice_{:.03f}.nii.gz'.format(step, ssim, dice)), verbose=False)
                save_nifti(mov_img[0, ...], os.path.join(output_folder, '{:03d}_mov_img_ssim_{:.03f}_dice_{:.03f}.nii.gz'.format(step, ssim, dice)), verbose=False)
                save_nifti(pred_img[0, ...], os.path.join(output_folder, '{:03d}_pred_img_ssim_{:.03f}_dice_{:.03f}.nii.gz'.format(step, ssim, dice)), verbose=False)
                save_nifti(fix_seg[0, ...], os.path.join(output_folder, '{:03d}_fix_seg_ssim_{:.03f}_dice_{:.03f}.nii.gz'.format(step, ssim, dice)), verbose=False)
                save_nifti(mov_seg[0, ...], os.path.join(output_folder, '{:03d}_mov_seg_ssim_{:.03f}_dice_{:.03f}.nii.gz'.format(step, ssim, dice)), verbose=False)
                save_nifti(pred_seg[0, ...], os.path.join(output_folder, '{:03d}_pred_seg_ssim_{:.03f}_dice_{:.03f}.nii.gz'.format(step, ssim, dice)), verbose=False)

                # with h5py.File(os.path.join(output_folder, '{:03d}_centroids.h5'.format(step)), 'w') as f:
                    # f.create_dataset('fix_centroids', dtype=np.float32, data=fix_centroids)
                    # f.create_dataset('mov_centroids', dtype=np.float32, data=mov_centroids)
                    # f.create_dataset('pred_centroids', dtype=np.float32, data=pred_centroids)
                    # f.create_dataset('fix_centroids_isotropic', dtype=np.float32, data=fix_centroids_isotropic)
                    # f.create_dataset('mov_centroids_isotropic', dtype=np.float32, data=mov_centroids_isotropic)

                # magnitude = np.sqrt(np.sum(disp_map[0, ...] ** 2, axis=-1))
                # _ = plt.hist(magnitude.flatten())
                # plt.title('Histogram of disp. magnitudes')
                # plt.show(block=False)
                # plt.savefig(os.path.join(output_folder, '{:03d}_hist_mag_ssim_{:.03f}_dice_{:.03f}.png'.format(step, ssim, dice)))
                # plt.close()

                plot_predictions(fix_img, mov_img, disp_map, pred_img, os.path.join(output_folder, '{:03d}_figures.png'.format(step)), show=False)
                plot_predictions(fix_seg, mov_seg, disp_map, pred_seg, os.path.join(output_folder, '{:03d}_figures_seg.png'.format(step)), show=False)
                save_disp_map_img(disp_map, 'Displacement map', os.path.join(output_folder, '{:03d}_disp_map_fig.png'.format(step)), show=False)

                progress_bar.set_description('SSIM {:.04f}\tDICE: {:.04f}'.format(ssim, dice))

        print('Summary\n=======\n')
        print(pd.read_csv(metrics_file, sep=';', header=0).mean(axis=0))
        print('\n=======\n')
        tf.keras.backend.clear_session()
        # sess.close()
        del network
    print('Done')
