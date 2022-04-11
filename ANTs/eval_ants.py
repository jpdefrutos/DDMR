import h5py
import ants
import numpy as np
import DeepDeformationMapRegistration.utils.constants as C
import os
from tqdm import  tqdm
import re

from DeepDeformationMapRegistration.losses import StructuralSimilarity_simplified, NCC, GeneralizedDICEScore, HausdorffDistanceErosion, target_registration_error
from DeepDeformationMapRegistration.ms_ssim_tf import MultiScaleStructuralSimilarity
from DeepDeformationMapRegistration.utils.misc import DisplacementMapInterpolator, segmentation_ohe_to_cardinal

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
    args = parser.parse_args()

    dataset_files = os.listdir(args.dataset)
    dataset_files.sort()
    dataset_files = [os.path.join(args.dataset, f) for f in dataset_files if re.match(DATASET_NAMES, f)]

    dataset_iterator = tqdm(dataset_files)

    f = h5py.File(dataset_files[0], 'r')
    image_output_shape = list(f['fix_image'][:].shape[:-1])
    f.close()

    #### TF prep
    metric_fncs = [StructuralSimilarity_simplified(patch_size=2, dim=3, dynamic_range=1.).metric,
                   NCC(image_input_shape).metric,
                   vxm.losses.MSE().loss,
                   MultiScaleStructuralSimilarity(max_val=1., filter_size=3).metric,
                   GeneralizedDICEScore(image_output_shape + [nb_labels], num_labels=nb_labels).metric,
                   HausdorffDistanceErosion(3, 10, im_shape=image_output_shape + [nb_labels]).metric,
                   GeneralizedDICEScore(image_output_shape + [nb_labels], num_labels=nb_labels).metric_macro]

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

    config = tf.compat.v1.ConfigProto()  # device_count={'GPU':0})
    config.gpu_options.allow_growth = True
    config.log_device_placement = False  ## to log device placement (on which device the operation ran)
    config.allow_soft_placement = True

    sess = tf.Session(config=config)
    tf.keras.backend.set_session(sess)
    ####
    dm_interp = DisplacementMapInterpolator(image_output_shape, 'griddata')

    metrics_file = os.path.join(output_folder, 'metrics.csv')

    for file_path in dataset_iterator:
        file_num = int(re.findall('(\d+)', os.path.split(file_path)[-1])[0])

        dataset_iterator.set_description('{} ({}): laoding data'.format(file_num, dataset_name))
        with h5py.File(file_path, 'r') as vol_file:
            fix_img = vol_file['fix_image'][:]
            mov_img = vol_file['mov_image'][:]

            fix_seg = vol_file['fix_segmentations'][:]
            mov_seg = vol_file['mov_segmentations'][:]

            fix_centroid = vol_file['fix_centroids'][:]
            mov_centroid = vol_file['mov_centroids'][:]

        # ndarray to ANTsImage
        fix_img = ants.make_image(fix_img.shape, fix_img)
        mov_img = ants.make_image(mov_img.shape, mov_img)

        reg_output_syn = ants.registration(fix_img, mov_img, 'SyN')
        reg_output_syncc = ants.registration(fix_img, mov_img, 'SyNCC')
        mov_to_fix_trf_syn = reg_output_syn[FWD_TRFS]
        mov_to_fix_trf_syncc = reg_output_syn[FWD_TRFS]
        if not len(mov_to_fix_trf_syn) and not len(mov_to_fix_trf_syncc):
            print('ERR: Registration failed for: '+file_path)
        else:
            for reg_output in [reg_output_syn, reg_output_syncc]:
                mov_to_fix_trf = reg_output[FWD_TRFS]
                pred_img = reg_output[WARPED_MOV].numpy()
                pred_seg = mov_to_fix_trf.apply_to_image(ants.make_image(mov_seg.shape, mov_seg)).numpy()

                with sess.as_default():
                    dice, hd, dice_macro = sess.run([dice_tf, hd_tf, dice_macro_tf],
                                                    {'fix_seg:0': fix_seg, 'pred_seg:0': pred_seg})

                    pred_seg_card = segmentation_ohe_to_cardinal(pred_seg).astype(np.float32)
                    mov_seg_card = segmentation_ohe_to_cardinal(mov_seg).astype(np.float32)
                    fix_seg_card = segmentation_ohe_to_cardinal(fix_seg).astype(np.float32)

                    ssim, ncc, mse, ms_ssim = sess.run([ssim_tf, ncc_tf, mse_tf, ms_ssim_tf],
                                                       {'fix_img:0': fix_img, 'pred_img:0': pred_img})
                    ms_ssim = ms_ssim[0]
                tf.keras.backend.clear_session()

                # TRE
                pred_centroids = dm_interp(mov_to_fix_trf.numpy(), mov_centroid, backwards=True) + mov_centroid
                upsample_scale = 128 / 64
                fix_centroids_isotropic = fix_centroids * upsample_scale
                pred_centroids_isotropic = pred_centroids * upsample_scale

                fix_centroids_isotropic = np.divide(fix_centroids_isotropic, C.IXI_DATASET_iso_to_cubic_scales)
                pred_centroids_isotropic = np.divide(pred_centroids_isotropic, C.IXI_DATASET_iso_to_cubic_scales)
                tre_array = target_registration_error(fix_centroids_isotropic, pred_centroids_isotropic, False).eval()
                tre = np.mean([v for v in tre_array if not np.isnan(v)])

                new_line = [step, ssim, ms_ssim, ncc, mse, dice, dice_macro, hd, t1-t0, tre, len(missing_lbls), missing_lbls]
                with open(metrics_file, 'a') as f:
                    f.write(';'.join(map(str, new_line))+'\n')

                save_nifti(fix_img[0, ...], os.path.join(output_folder, '{:03d}_fix_img_ssim_{:.03f}_dice_{:.03f}.nii.gz'.format(step, ssim, dice)), verbose=False)
                save_nifti(mov_img[0, ...], os.path.join(output_folder, '{:03d}_mov_img_ssim_{:.03f}_dice_{:.03f}.nii.gz'.format(step, ssim, dice)), verbose=False)
                save_nifti(pred_img[0, ...], os.path.join(output_folder, '{:03d}_pred_img_ssim_{:.03f}_dice_{:.03f}.nii.gz'.format(step, ssim, dice)), verbose=False)
                save_nifti(fix_seg_card[0, ...], os.path.join(output_folder, '{:03d}_fix_seg_ssim_{:.03f}_dice_{:.03f}.nii.gz'.format(step, ssim, dice)), verbose=False)
                save_nifti(mov_seg_card[0, ...], os.path.join(output_folder, '{:03d}_mov_seg_ssim_{:.03f}_dice_{:.03f}.nii.gz'.format(step, ssim, dice)), verbose=False)
                save_nifti(pred_seg_card[0, ...], os.path.join(output_folder, '{:03d}_pred_seg_ssim_{:.03f}_dice_{:.03f}.nii.gz'.format(step, ssim, dice)), verbose=False)

                plot_predictions(fix_img, mov_img, disp_map, pred_img, os.path.join(output_folder, '{:03d}_figures_img.png'.format(step)), show=False)
                plot_predictions(fix_seg, mov_seg, disp_map, pred_seg, os.path.join(output_folder, '{:03d}_figures_seg.png'.format(step)), show=False)
                save_disp_map_img(disp_map, 'Displacement map', os.path.join(output_folder, '{:03d}_disp_map_fig.png'.format(step)), show=False)

    print('Summary\n=======\n')
    print('\nAVG:\n' + str(pd.read_csv(metrics_file, sep=';', header=0).mean(axis=0)) + '\nSTD:\n' + str(
        pd.read_csv(metrics_file, sep=';', header=0).std(axis=0)))
    print('\n=======\n')