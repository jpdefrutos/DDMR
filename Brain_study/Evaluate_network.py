import os, sys

import shutil

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

import ddmr.utils.constants as C
from ddmr.utils.nifti_utils import save_nifti
from ddmr.layers import AugmentationLayer
from ddmr.losses import StructuralSimilarity_simplified, NCC, GeneralizedDICEScore, HausdorffDistanceErosion
from ddmr.ms_ssim_tf import MultiScaleStructuralSimilarity
from ddmr.utils.acummulated_optimizer import AdamAccumulated
from ddmr.utils.visualization import save_disp_map_img, plot_predictions
from ddmr.utils.misc import segmentation_ohe_to_cardinal
from EvaluationScripts.Evaluate_class import EvaluationFigures, resize_pts_to_original_space, resize_img_to_original_space, resize_transformation
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm

import h5py

from Brain_study.data_generator import BatchGenerator

import argparse

from skimage.transform import warp
import neurite as ne

DATASET = '/mnt/EncryptedData1/Users/javier/ext_datasets/IXI_dataset/T1/training'
MODEL_FILE = '/mnt/EncryptedData1/Users/javier/train_output/Brain_study/ERASE/MS_SSIM/BASELINE_L_ssim__MET_mse_ncc_ssim_162756-29062021/checkpoints/best_model.h5'
DATA_ROOT_DIR = '/mnt/EncryptedData1/Users/javier/train_output/Brain_study/ERASE/MS_SSIM/BASELINE_L_ssim__MET_mse_ncc_ssim_162756-29062021/'

OUTPUT_FOLDER_NAME = 'Evaluate'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, help='.h5 of the model', default='')
    parser.add_argument('-d', '--dir', type=str, help='Directory where ./checkpoints/best_model.h5 is located', default='')
    parser.add_argument('--gpu', type=int, help='GPU', default=0)
    parser.add_argument('--dataset', type=str, help='Dataset to run predictions on',
                        default='/mnt/EncryptedData1/Users/javier/ext_datasets/IXI_dataset/T1/training')
    parser.add_argument('--erase', type=bool, help='Erase the content of the output folder', default=False)
    args = parser.parse_args()
    if args.model != '':
        assert '.h5' in args.model, 'No checkpoint file provided, use -d/--dir instead'
        MODEL_FILE = args.model
        DATA_ROOT_DIR = os.path.split(args.model)[0]
    elif args.dir != '':
        assert '.h5' not in args.model, 'Provided checkpoint file, user -m/--model instead'
        MODEL_FILE = os.path.join(args.dir, 'checkpoints', 'best_model.h5')
        DATA_ROOT_DIR = args.dir
    else:
        raise ValueError("Provide either the model file or the directory ./containing checkpoints/best_model.h5")

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)  # Check availability before running using 'nvidia-smi'
    DATASET = args.dataset

    print('MODEL LOCATION: ', MODEL_FILE)

    # data_folder = '/mnt/EncryptedData1/Users/javier/train_output/DDMR/THESIS/BASELINE_Affine_ncc___mse_ncc_160606-25022021'
    output_folder = os.path.join(DATA_ROOT_DIR, OUTPUT_FOLDER_NAME)  # '/mnt/EncryptedData1/Users/javier/train_output/DDMR/THESIS/eval/BASELINE_TRAIN_Affine_ncc_EVAL_Affine'
    # os.makedirs(os.path.join(output_folder, 'images'), exist_ok=True)
    if args.erase:
        shutil.rmtree(output_folder, ignore_errors=True)
    os.makedirs(output_folder, exist_ok=True)
    print('DESTINATION FOLDER: ', output_folder)

    data_generator = BatchGenerator(DATASET, 1, False, 1.0, False, ['all'])

    img_generator = data_generator.get_train_generator()
    nb_labels = len(img_generator.get_segmentation_labels())
    image_input_shape = img_generator.get_data_shape()[-1][:-1]
    image_output_shape = [64] * 3

    # Build model

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
                                   trainable=False)
    augm_model = tf.keras.Model(inputs=input_augm, outputs=augm_layer(input_augm))

    loss_fncs = [StructuralSimilarity_simplified(patch_size=2, dim=3, dynamic_range=1.).loss,
                 NCC(image_input_shape).loss,
                 vxm.losses.MSE().loss,
                 MultiScaleStructuralSimilarity(max_val=1., filter_size=3).loss]

    metric_fncs = [StructuralSimilarity_simplified(patch_size=2, dim=3, dynamic_range=1.).metric,
                   NCC(image_input_shape).metric,
                   vxm.losses.MSE().loss,
                   MultiScaleStructuralSimilarity(max_val=1., filter_size=3).metric,
                   GeneralizedDICEScore(image_output_shape + [img_generator.get_data_shape()[2][-1]], num_labels=nb_labels).loss,
                   HausdorffDistanceErosion(3, 10, im_shape=image_output_shape + [img_generator.get_data_shape()[2][-1]]).loss]

    network = tf.keras.models.load_model(MODEL_FILE, {'VxmDenseSemiSupervisedSeg': vxm.networks.VxmDenseSemiSupervisedSeg,
                                                      'VxmDense': vxm.networks.VxmDense,
                                                      'AdamAccumulated': AdamAccumulated,
                                                      'loss': loss_fncs,
                                                      'metric': metric_fncs},
                                         compile=False)

    # Needed for VxmDense type of network
    warp_segmentation = vxm.networks.Transform(image_output_shape, interp_method='nearest', nb_feats=nb_labels)

    # Record metrics
    metrics = pd.DataFrame(columns=['File', 'SSIM', 'MS-SSIM', 'MSE', 'DICE', 'HD'])
    config = tf.compat.v1.ConfigProto()  # device_count={'GPU':0})
    config.gpu_options.allow_growth = True
    config.log_device_placement = False  ## to log device placement (on which device the operation ran)

    sess = tf.Session(config=config)
    tf.keras.backend.set_session(sess)
    with sess.as_default():
        sess.run(tf.global_variables_initializer())
        network.load_weights(MODEL_FILE, by_name=True)
        progress_bar = tqdm(enumerate(img_generator, 1), desc='Evaluation', total=len(img_generator))
        for step, (in_batch, _) in progress_bar:
            fix_img, mov_img, fix_seg, mov_seg = augm_model.predict(in_batch)

            if network.name == 'vxm_dense_semi_supervised_seg':
                pred_img, disp_map, pred_seg = network.predict([mov_img, fix_img, mov_seg, fix_seg])    # predict([source, target])
            else:
                pred_img, disp_map = network.predict([mov_img, fix_img])
                pred_seg = warp_segmentation.predict([mov_seg, disp_map])

            # I need the labels to be OHE to compute the segmentation metrics.
            dice = GeneralizedDICEScore(image_output_shape + [img_generator.get_data_shape()[2][-1]], num_labels=nb_labels).metric(fix_seg, pred_seg).eval()
            hd = HausdorffDistanceErosion(3, 10, im_shape=image_output_shape + [img_generator.get_data_shape()[2][-1]]).metric(fix_seg, pred_seg).eval()

            pred_seg = segmentation_ohe_to_cardinal(pred_seg).astype(np.float32)
            mov_seg = segmentation_ohe_to_cardinal(mov_seg).astype(np.float32)
            fix_seg = segmentation_ohe_to_cardinal(fix_seg).astype(np.float32)

            mov_coords = np.stack(np.meshgrid(*[np.arange(0, 64)]*3), axis=-1)
            dest_coords = mov_coords + disp_map[0, ...]

            ssim = StructuralSimilarity_simplified(patch_size=2, dim=3, dynamic_range=1.).metric(fix_img, pred_img).eval()
            ms_ssim = MultiScaleStructuralSimilarity(max_val=1., filter_size=3).metric(fix_img, pred_img).eval()[0]
            mse = vxm.losses.MSE().loss(fix_img, pred_img).eval()

            metrics.append({'File': step,
                            'SSIM': ssim,
                            'MS-SSIM': ms_ssim,
                            'MSE': mse,
                            'DICE': dice,
                            'HD': hd}, ignore_index=True)
            save_nifti(fix_img[0, ...], os.path.join(output_folder, '{:03d}_fix_img_ssim_{:.03f}_dice_{:.03f}.nii.gz'.format(step, ssim, dice)), verbose=False)
            save_nifti(mov_img[0, ...], os.path.join(output_folder, '{:03d}_mov_img_ssim_{:.03f}_dice_{:.03f}.nii.gz'.format(step, ssim, dice)), verbose=False)
            save_nifti(pred_img[0, ...], os.path.join(output_folder, '{:03d}_pred_img_ssim_{:.03f}_dice_{:.03f}.nii.gz'.format(step, ssim, dice)), verbose=False)
            save_nifti(fix_seg[0, ...], os.path.join(output_folder, '{:03d}_fix_seg_ssim_{:.03f}_dice_{:.03f}.nii.gz'.format(step, ssim, dice)), verbose=False)
            save_nifti(mov_seg[0, ...], os.path.join(output_folder, '{:03d}_mov_seg_ssim_{:.03f}_dice_{:.03f}.nii.gz'.format(step, ssim, dice)), verbose=False)
            save_nifti(pred_seg[0, ...], os.path.join(output_folder, '{:03d}_pred_seg_ssim_{:.03f}_dice_{:.03f}.nii.gz'.format(step, ssim, dice)), verbose=False)

            magnitude = np.sqrt(np.sum(disp_map[0, ...] ** 2, axis=-1))
            _ = plt.hist(magnitude.flatten())
            plt.title('Histogram of disp. magnitudes')
            # plt.show(block=False)
            plt.savefig(os.path.join(output_folder, '{:03d}_hist_mag_ssim_{:.03f}_dice_{:.03f}.png'.format(step, ssim, dice)))
            plt.close()

            plot_predictions(img_batches=[fix_img, mov_img, pred_img], disp_map_batch=disp_map, seg_batches=[fix_seg, mov_seg, pred_seg], filename=os.path.join(output_folder, '{:03d}_figures_seg.png'.format(step)), show=False)
            plot_predictions(img_batches=[fix_img, mov_img, pred_img], disp_map_batch=disp_map, filename=os.path.join(output_folder, '{:03d}_figures_img.png'.format(step)), show=False)
            save_disp_map_img(disp_map, 'Displacement map', os.path.join(output_folder, '{:03d}_disp_map_fig.png'.format(step)), show=False)

            progress_bar.set_description('SSIM {:.04f}\tDICE: {:.04f}'.format(ssim, dice))

    metrics.to_csv(os.path.join(output_folder, 'metrics.csv'))
    print('Done')
