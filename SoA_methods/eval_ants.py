import h5py
import ants
import numpy as np
import nibabel as nb
import os, sys
from tqdm import tqdm
import re
import time
import pandas as pd

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)  # PYTHON > 3.3 does not allow relative referencing

from DeepDeformationMapRegistration.losses import StructuralSimilarity_simplified, NCC, GeneralizedDICEScore, HausdorffDistanceErosion, target_registration_error
from DeepDeformationMapRegistration.ms_ssim_tf import MultiScaleStructuralSimilarity
from DeepDeformationMapRegistration.utils.misc import DisplacementMapInterpolator, segmentation_ohe_to_cardinal
from DeepDeformationMapRegistration.utils.nifti_utils import save_nifti
from DeepDeformationMapRegistration.utils.visualization import save_disp_map_img, plot_predictions
import DeepDeformationMapRegistration.utils.constants as C

import medpy.metric as medpy_metrics

import voxelmorph as vxm

from argparse import ArgumentParser

import tensorflow as tf

DATASET_LOCATION = '/mnt/EncryptedData1/Users/javier/vessel_registration/3Dirca/dataset/EVAL'
DATASET_NAMES = 'test_sample_\d{4}.h5'
DATASET_FILENAME = 'volume'
IMGS_FOLDER = '/home/jpdefrutos/workspace/DeepDeformationMapRegistration/Centerline/imgs'

WARPED_MOV = 'warpedmovout'
WARPED_FIX = 'warpedfixout'
FWD_TRFS = 'fwdtransforms'
INV_TRFS = 'invtransforms'


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, help='Directory with the images')
    parser.add_argument('--outdir', type=str, help='Output directory')
    parser.add_argument('--gpu', type=int, help='GPU')
    args = parser.parse_args()

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(os.path.join(args.outdir, 'SyN'), exist_ok=True)
    os.makedirs(os.path.join(args.outdir, 'SyNCC'), exist_ok=True)
    dataset_files = os.listdir(args.dataset)
    dataset_files.sort()
    dataset_files = [os.path.join(args.dataset, f) for f in dataset_files if re.match(DATASET_NAMES, f)]
    dataset_files.sort()
    dataset_iterator = tqdm(enumerate(dataset_files), desc="Running ANTs")

    f = h5py.File(dataset_files[0], 'r')
    image_shape = list(f['fix_image'][:].shape[:-1])
    nb_labels = f['fix_segmentations'][:].shape[-1]
    f.close()

    #### TF prep
    metric_fncs = [StructuralSimilarity_simplified(patch_size=2, dim=3, dynamic_range=1.).metric,
                   NCC(image_shape).metric,
                   vxm.losses.MSE().loss,
                   MultiScaleStructuralSimilarity(max_val=1., filter_size=3).metric,
                   GeneralizedDICEScore(image_shape + [nb_labels], num_labels=nb_labels).metric,
                   HausdorffDistanceErosion(3, 10, im_shape=image_shape + [nb_labels]).metric,
                   GeneralizedDICEScore(image_shape + [nb_labels], num_labels=nb_labels).metric_macro]

    fix_img_ph = tf.placeholder(tf.float32, (1, None, None, None, 1), name='fix_img')
    pred_img_ph = tf.placeholder(tf.float32, (1, None, None, None, 1), name='pred_img')
    fix_seg_ph = tf.placeholder(tf.float32, (1, None, None, None, nb_labels), name='fix_seg')
    pred_seg_ph = tf.placeholder(tf.float32, (1, None, None, None, nb_labels), name='pred_seg')

    ssim_tf = metric_fncs[0](fix_img_ph, pred_img_ph)
    ncc_tf = metric_fncs[1](fix_img_ph, pred_img_ph)
    mse_tf = metric_fncs[2](fix_img_ph, pred_img_ph)
    ms_ssim_tf = metric_fncs[3](fix_img_ph, pred_img_ph)
    # dice_tf = metric_fncs[4](fix_seg_ph, pred_seg_ph)
    # hd_tf = metric_fncs[5](fix_seg_ph, pred_seg_ph)
    # dice_macro_tf = metric_fncs[6](fix_seg_ph, pred_seg_ph)

    config = tf.compat.v1.ConfigProto()  # device_count={'GPU':0})
    config.gpu_options.allow_growth = True
    config.log_device_placement = False  ## to log device placement (on which device the operation ran)
    config.allow_soft_placement = True

    sess = tf.Session(config=config)
    tf.keras.backend.set_session(sess)
    ####
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "{:d}".format(os.cpu_count())  #https://github.com/ANTsX/ANTsPy/issues/261
    print("Running ANTs using {} threads".format(os.environ.get("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS")))
    # dm_interp = DisplacementMapInterpolator(image_shape, 'griddata')
    # Header of the metrics csv file
    csv_header = ['File', 'SSIM', 'MS-SSIM', 'NCC', 'MSE', 'DICE', 'DICE_MACRO', 'HD', 'Time', 'TRE']

    metrics_file = {'SyN': os.path.join(args.outdir, 'SyN', 'metrics.csv'),
                    'SyNCC': os.path.join(args.outdir, 'SyNCC', 'metrics.csv')}
    for k in metrics_file.keys():
        with open(metrics_file[k], 'w') as f:
            f.write(';'.join(csv_header)+'\n')

    print('Starting the loop')
    for step, file_path in dataset_iterator:
        file_num = int(re.findall('(\d+)', os.path.split(file_path)[-1])[0])

        dataset_iterator.set_description('{} ({}): loading data'.format(file_num, file_path))
        with h5py.File(file_path, 'r') as vol_file:
            fix_img = vol_file['fix_image'][:]
            mov_img = vol_file['mov_image'][:]

            fix_seg = vol_file['fix_segmentations'][:]
            mov_seg = vol_file['mov_segmentations'][:]

            fix_centroids = vol_file['fix_centroids'][:]
            mov_centroids = vol_file['mov_centroids'][:]

        # ndarray to ANTsImage
        fix_img_ants = ants.make_image(fix_img.shape[:-1], np.squeeze(fix_img))  # SoA doesn't work fine with 1-ch images
        mov_img_ants = ants.make_image(mov_img.shape[:-1], np.squeeze(mov_img))  # SoA doesn't work fine with 1-ch images

        dataset_iterator.set_description('{} ({}): running ANTs SyN'.format(file_num, file_path))
        t0_syn = time.time()
        reg_output_syn = ants.registration(fix_img_ants, mov_img_ants, 'SyN')
        t1_syn = time.time()

        dataset_iterator.set_description('{} ({}): running ANTs SyN'.format(file_num, file_path))
        t0_syncc = time.time()
        reg_output_syncc = ants.registration(fix_img_ants, mov_img_ants, 'SyNCC')
        t1_syncc = time.time()

        mov_to_fix_trf_syn = reg_output_syn[FWD_TRFS]
        mov_to_fix_trf_syncc = reg_output_syn[FWD_TRFS]
        if not len(mov_to_fix_trf_syn) and not len(mov_to_fix_trf_syncc):
            print('ERR: Registration failed for: '+file_path)
        else:
            for reg_method, reg_output in zip(['SyN', 'SyNCC'], [reg_output_syn, reg_output_syncc]):
                mov_to_fix_trf_list = reg_output[FWD_TRFS]
                pred_img = reg_output[WARPED_MOV].numpy()
                pred_img = pred_img[..., np.newaxis]  # SoA doesn't work fine with 1-ch images

                fix_seg_ants = ants.make_image(fix_seg.shape, np.squeeze(fix_seg))
                mov_seg_ants = ants.make_image(mov_seg.shape, np.squeeze(mov_seg))
                pred_seg = ants.apply_transforms(fixed=fix_seg_ants, moving=mov_seg_ants,
                                                 transformlist=mov_to_fix_trf_list).numpy()
                pred_seg = np.squeeze(pred_seg)  # SoA adds an extra axis which shouldn't be there

                dataset_iterator.set_description('{} ({}): Getting metrics {}'.format(file_num, file_path, reg_method))
                with sess.as_default():
                    dice = np.mean([medpy_metrics.dc(pred_seg[np.newaxis, ..., l], fix_seg[np.newaxis,..., l]) / np.sum(
                        fix_seg[np.newaxis,..., l]) for l in range(nb_labels)])
                    hd = np.mean(
                        [medpy_metrics.hd(pred_seg[np.newaxis,..., l], fix_seg[np.newaxis,..., l]) for l in range(nb_labels)])
                    dice_macro = np.mean(
                        [medpy_metrics.dc(pred_seg[np.newaxis,..., l], fix_seg[np.newaxis,..., l]) for l in range(nb_labels)])

                    # dice, hd, dice_macro = sess.run([dice_tf, hd_tf, dice_macro_tf],
                    #                                 {'fix_seg:0': fix_seg[np.newaxis, ...],  # Batch axis
                    #                                  'pred_seg:0': pred_seg[np.newaxis, ...]  # Batch axis
                    #                                  })

                    pred_seg_card = segmentation_ohe_to_cardinal(pred_seg).astype(np.float32)
                    mov_seg_card = segmentation_ohe_to_cardinal(mov_seg).astype(np.float32)
                    fix_seg_card = segmentation_ohe_to_cardinal(fix_seg).astype(np.float32)

                    ssim, ncc, mse, ms_ssim = sess.run([ssim_tf, ncc_tf, mse_tf, ms_ssim_tf],
                                                       {'fix_img:0': fix_img[np.newaxis, ...],  # Batch axis
                                                        'pred_img:0': pred_img[np.newaxis, ...]  # Batch axis
                                                        })
                    ssim = np.mean(ssim)
                    ms_ssim = ms_ssim[0]

                    # TRE
                    disp_map = np.squeeze(np.asarray(nb.load(mov_to_fix_trf_list[0]).dataobj))
                    dm_interp = DisplacementMapInterpolator(fix_img.shape[:-1], 'griddata', step=2)
                    pred_centroids = dm_interp(disp_map, mov_centroids, backwards=True) + mov_centroids
                    # upsample_scale = 128 / 64
                    # fix_centroids_isotropic = fix_centroids * upsample_scale
                    # pred_centroids_isotropic = pred_centroids * upsample_scale

                    # fix_centroids_isotropic = np.divide(fix_centroids_isotropic, C.COMET_DATASET_iso_to_cubic_scales)
                    # pred_centroids_isotropic = np.divide(pred_centroids_isotropic, C.COMET_DATASET_iso_to_cubic_scales)
                    tre_array = target_registration_error(fix_centroids, pred_centroids, False).eval()
                    tre = np.mean([v for v in tre_array if not np.isnan(v)])

                # dataset_iterator.set_description('{} ({}): Saving data {}'.format(file_num, file_path, reg_method))
                # new_line = [step, ssim, ms_ssim, ncc, mse, dice, dice_macro, hd,
                #             t1_syn-t0_syn if reg_method == 'SyN' else t1_syncc-t0_syncc,
                #             tre]
                # with open(metrics_file[reg_method], 'a') as f:
                #     f.write(';'.join(map(str, new_line))+'\n')

                save_nifti(fix_img[0, ...], os.path.join(args.outdir, reg_method, '{:03d}_fix_img_ssim_{:.03f}_dice_{:.03f}.nii.gz'.format(step, ssim, dice)), verbose=False)
                save_nifti(mov_img[0, ...], os.path.join(args.outdir, reg_method, '{:03d}_mov_img_ssim_{:.03f}_dice_{:.03f}.nii.gz'.format(step, ssim, dice)), verbose=False)
                save_nifti(pred_img[0, ...], os.path.join(args.outdir, reg_method, '{:03d}_pred_img_ssim_{:.03f}_dice_{:.03f}.nii.gz'.format(step, ssim, dice)), verbose=False)
                save_nifti(fix_seg_card[0, ...], os.path.join(args.outdir, reg_method, '{:03d}_fix_seg_ssim_{:.03f}_dice_{:.03f}.nii.gz'.format(step, ssim, dice)), verbose=False)
                save_nifti(mov_seg_card[0, ...], os.path.join(args.outdir, reg_method, '{:03d}_mov_seg_ssim_{:.03f}_dice_{:.03f}.nii.gz'.format(step, ssim, dice)), verbose=False)
                save_nifti(pred_seg_card[0, ...], os.path.join(args.outdir, reg_method, '{:03d}_pred_seg_ssim_{:.03f}_dice_{:.03f}.nii.gz'.format(step, ssim, dice)), verbose=False)

                plot_predictions(img_batches=[fix_img[np.newaxis, ...], mov_img[np.newaxis, ...], pred_img[np.newaxis, ...]], disp_map_batch=disp_map[np.newaxis, ...], seg_batches=[fix_seg_card[np.newaxis, ...], mov_seg_card[np.newaxis, ...], pred_seg_card[np.newaxis, ...]], filename=os.path.join(args.outdir, reg_method, '{:03d}_figures_seg.png'.format(step)), show=False)
                plot_predictions(img_batches=[fix_img[np.newaxis, ...], mov_img[np.newaxis, ...], pred_img[np.newaxis, ...]], disp_map_batch=disp_map[np.newaxis, ...], filename=os.path.join(args.outdir, reg_method, '{:03d}_figures_img.png'.format(step)), show=False)
                save_disp_map_img(disp_map[np.newaxis, ...], 'Displacement map', os.path.join(args.outdir, reg_method, '{:03d}_disp_map_fig.png'.format(step)), show=False)

    for k in metrics_file.keys():
        print('Summary {}\n=======\n'.format(k))
        metrics_df = pd.read_csv(metrics_file[k], sep=';', header=0)
        print('\nAVG:\n')
        print(metrics_df.mean(axis=0))
        print('\nSTD:\n')
        print(metrics_df.std(axis=0))
        print('\nHD95perc:\n')
        print(metrics_df['HD'].describe(percentiles=[.95]))
        print('\n=======\n')