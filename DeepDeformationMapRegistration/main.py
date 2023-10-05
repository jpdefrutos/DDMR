import datetime
import os, sys
import shutil
import re
import argparse
import subprocess
import logging
import time

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)  # PYTHON > 3.3 does not allow relative referencing

import tensorflow as tf

import numpy as np
import nibabel as nib
from scipy.ndimage import gaussian_filter, zoom
from skimage.measure import regionprops
import SimpleITK as sitk

import voxelmorph as vxm
from voxelmorph.tf.layers import SpatialTransformer

import DeepDeformationMapRegistration.utils.constants as C
from DeepDeformationMapRegistration.utils.nifti_utils import save_nifti
from DeepDeformationMapRegistration.losses import StructuralSimilarity_simplified, NCC
from DeepDeformationMapRegistration.ms_ssim_tf import MultiScaleStructuralSimilarity
from DeepDeformationMapRegistration.utils.operators import min_max_norm
from DeepDeformationMapRegistration.utils.misc import resize_displacement_map


MODELS_FILE = {'L': {'BL-N': './models/liver/bl_ncc.h5',
                     'BL-S': './models/liver/bl_ncc_ssim.h5',
                     'SG-ND': './models/liver/sg_ncc_dsc.h5',
                     'SD-NSD': './models/liver/sg_ncc_ssim_dsc.h5',
                     'UW-NSD': './models/liver/uw_ncc_ssim_dsc.h5',
                     'UW-NSDH': './models/liver/uw_ncc_ssim_dsc_hd.h5',
                     },
               'B': {'BL-N': './models/brain/bl_ncc.h5',
                     'BL-S': './models/brain/bl_ncc_ssim.h5',
                     'SG-ND': './models/brain/sg_ncc_dsc.h5',
                     'SD-NSD': './models/brain/sg_ncc_ssim_dsc.h5',
                     'UW-NSD': './models/brain/uw_ncc_ssim_dsc.h5',
                     'UW-NSDH': './models/brain/uw_ncc_ssim_dsc_hd.h5',
                     }
               }

IMAGE_INTPUT_SHAPE = np.asarray([128, 128, 128, 1])


def rigidly_align_images(image_1: str, image_2: str) -> nib.Nifti1Image:
    """
    Rigidly align the images and resample to the same array size, to the dense displacement map is correct

    """
    def resample_to_isotropic(image: sitk.Image) -> sitk.Image:
        spacing = image.GetSpacing()
        spacing = min(spacing)
        resamp_spacing = [spacing] * image.GetDimension()
        resamp_size = [int(round(or_size*or_space/spacing)) for or_size, or_space in zip(image.GetSize(), image.GetSpacing())]
        return sitk.Resample(image,
                             resamp_size, sitk.Transform(), sitk.sitkLinear,image.GetOrigin(),
                             resamp_spacing, image.GetDirection(), 0, image.GetPixelID())

    image_1 = sitk.ReadImage(image_1, sitk.sitkFloat32)
    image_2 = sitk.ReadImage(image_2, sitk.sitkFloat32)

    image_1 = resample_to_isotropic(image_1)
    image_2 = resample_to_isotropic(image_2)

    rig_reg = sitk.ImageRegistrationMethod()
    rig_reg.SetMetricAsMeanSquares()
    rig_reg.SetOptimizerAsRegularStepGradientDescent(4.0, 0.01, 200)
    rig_reg.SetInitialTransform(sitk.TranslationTransform(image_1.GetDimension()))
    rig_reg.SetInterpolator(sitk.sitkLinear)

    print('Running rigid registration...')
    rig_reg_trf = rig_reg.Execute(image_1, image_2)
    print('Rigid registration completed\n----------------------------')
    print('Optimizer stop condition: {}'.format(rig_reg.GetOptimizerStopConditionDescription()))
    print('Iteration: {}'.format(rig_reg.GetOptimizerIteration()))
    print('Metric value: {}'.format(rig_reg.GetMetricValue()))

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(image_1)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(100)
    resampler.SetTransform(rig_reg_trf)

    image_2 = resampler.Execute(image_2)

    # TODO: Build a common image to hold both image_1 and image_2


def pad_images(image_1: nib.Nifti1Image, image_2: nib.Nifti1Image):
    """
    Align image_1 and image_2 by the top left corner and pad them to the largest dimensions along the three axes
    """
    joint_image_shape = np.maximum(image_1.shape, image_2.shape)
    pad_1 = [[0, p] for p in joint_image_shape - image_1.shape]
    pad_2 = [[0, p] for p in joint_image_shape - image_2.shape]
    image_1_padded = np.pad(image_1.dataobj, pad_1, mode='edge')
    image_2_padded = np.pad(image_2.dataobj, pad_2, mode='edge')

    return image_1_padded, image_2_padded


def pad_displacement_map(disp_map: np.ndarray, crop_min: np.ndarray, crop_max: np.ndarray, output_shape: (np.ndarray, list)) -> np.ndarray:
    padding = [[crop_min[i], image_shape_or[i] - crop_max[i]] for i in range(3)] + [[0, 0]]
    return np.pad(disp_map, padding, mode='constant')


def run_livermask(input_image_path, outputdir, filename: str = 'segmentation') -> np.ndarray:
    logger.info('Getting parenchyma segmentations...')
    shutil.copy2(input_image_path, os.path.join(outputdir, f'{filename}.nii.gz'))
    livermask_cmd = "{} -m livermask.livermask --input {} --output {}".format(sys.executable,
                                                                              input_image_path,
                                                                              os.path.join(outputdir,
                                                                                           f'{filename}.nii.gz'))
    subprocess.run(livermask_cmd)
    logger.info('done!')
    segmentation_path = os.path.join(outputdir, f'{filename}.nii.gz')
    return np.asarray(nib.load(segmentation_path).dataobj, dtype=int)


def debug_save_image(image: (np.ndarray, nib.Nifti1Image), filename: str, outputdir: str, debug: bool = True):
    def disp_map_modulus(disp_map, scale: float = None):
        disp_map_mod = np.sqrt(np.sum(np.power(disp_map, 2), -1))
        if scale:
            min_disp = np.min(disp_map_mod)
            max_disp = np.max(disp_map_mod)
            disp_map_mod = disp_map_mod - min_disp / (max_disp - min_disp)
            disp_map_mod *= scale
            logger.debug('Scaled displacement map to [0., 1.] range')
        return disp_map_mod

    if debug:
        os.makedirs(os.path.join(outputdir, 'debug'), exist_ok=True)
        if image.shape[-1] > 1:
            image = disp_map_modulus(image, 1.)
        save_nifti(image, os.path.join(outputdir, 'debug', filename+'.nii.gz'), verbose=False)
        logger.debug(f'Saved {filename} at {os.path.join(outputdir, filename+".nii.gz")}')


def get_roi(image_filepath: str,
            anatomy: str,
            outputdir: str,
            filename_filepath: str = 'segmentation',
            segmentation_file: str = None,
            debug: bool = False) -> list:
    segm = None
    if segmentation_file is None and anatomy == 'L':
        segm = run_livermask(image_filepath, outputdir, filename_filepath)
        logger.info(f'Loaded segmentation using livermask from {os.path.join(outputdir, filename_filepath)}')
    elif segmentation_file is not None:
        segm = np.asarray(nib.load(segmentation_file).dataobj, dtype=int)
        logger.info(f'Loaded fixed segmentation from {segmentation_file}')
    else:
        logger.warning('No segmentation provided! Using the full volume')
    if segm is not None:
        segm[segm > 0] = 1
        ret_val = regionprops(segm)[0].bbox
        debug_save_image(segm, f'img_1_{filename_filepath}', outputdir, debug)
    else:
        ret_val = [0, 0, 0] + list(nib.load(image_filepath).shape[:3])
    logger.debug(f'ROI found at coordinates {ret_val}')
    return ret_val


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--fixed', type=str, help='Path to fixed image file (NIfTI)')
    parser.add_argument('-m', '--moving', type=str, help='Path to moving segmentation image file (NIfTI)', default=None)
    parser.add_argument('-F', '--fixedsegm', type=str, help='Path to fixed image segmentation file(NIfTI)',
                        default=None)
    parser.add_argument('-M', '--movingsegm', type=str, help='Path to moving image file (NIfTI)')
    parser.add_argument('-o', '--outputdir', type=str, help='Output directory', default='./Registration_output')
    parser.add_argument('-a', '--anatomy', type=str, help='Anatomical structure: liver (L) (Default) or brain (B)',
                        default='L')
    parser.add_argument('--gpu', type=int,
                        help='In case of multi-GPU systems, limits the execution to the defined GPU number',
                        default=None)
    parser.add_argument('--model', type=str, help='Which model to use: BL-N, BL-S, SG-ND, SG-NSD, UW-NSD, UW-NSDH',
                        default='UW-NSD')
    parser.add_argument('--debug', '-d', action='store_true', help='Produce additional debug information', default=False)
    parser.add_argument('-y', action='store_true', help='Erase output folder if this has content', default=False)
    # parser.add_argument('--brain', type=bool, action='store_true', help='Perform brain MRi registration', default=False)
    args = parser.parse_args()

    assert os.path.exists(args.fixed), 'Fixed image not found'
    assert os.path.exists(args.moving), 'Moving image not found'
    assert args.model in ['BL-N', 'BL-S', 'SG-ND', 'SG-NSD', 'UW-NSD', 'UW-NSDH'], 'Invalid model type'
    assert args.anatomy in ['L', 'B'], 'Invalid anatomy option'

    if os.path.exists(args.outputdir) and len(os.listdir(args.outputdir)):
        if args.y:
            erase = 'y'
        else:
            erase = input('Output directory is not empty, erase content? (y/n)')
        if erase.lower() in ['y', 'yes']:
            shutil.rmtree(args.outputdir, ignore_errors=True)
            print('Erased directory: ' + args.outputdir)
        elif erase.lower() in ['n', 'no']:
            args.outputdir = os.path.join(args.outputdir, datetime.datetime.now().strftime('%H%M%S_%Y%m%d'))
            print('New output directory: ' + args.outputdir)
    os.makedirs(args.outputdir, exist_ok=True)

    log_format = '%(asctime)s [%(levelname)s]:\t%(message)s'
    logging.basicConfig(filename=os.path.join(args.outputdir, 'log.log'), filemode='w',
                        format=log_format, datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger(__name__)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(logging.Formatter(log_format, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(stdout_handler)
    if isinstance(args.gpu, int):
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)  # Check availability before running using 'nvidia-smi'
    if args.debug:
        logger.setLevel('DEBUG')
        logger.debug('DEBUG MODE ENABLED')

    # Load the file and preprocess it
    logger.info('Loading image files')
    fixed_image_or = nib.load(args.fixed)
    moving_image_or = nib.load(args.moving)
    image_shape_or = np.asarray(fixed_image_or.shape)
    fixed_image_or, moving_image_or = pad_images(fixed_image_or, moving_image_or)
    fixed_image_or = fixed_image_or[..., np.newaxis]  # add channel dim
    moving_image_or = moving_image_or[..., np.newaxis]  # add channel dim
    debug_save_image(fixed_image_or, 'img_0_loaded_fix_image', args.outputdir, args.debug)
    debug_save_image(moving_image_or, 'img_0_loaded_moving_image', args.outputdir, args.debug)

    # TF stuff
    logger.info('Setting up configuration')
    config = tf.compat.v1.ConfigProto()  # device_count={'GPU':0})
    config.gpu_options.allow_growth = True
    config.log_device_placement = False  ## to log device placement (on which device the operation ran)
    config.allow_soft_placement = True

    sess = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(sess)

    # Preprocess data
    # 1. Run Livermask to get the mask around the liver in both the fixed and moving image
    logger.info('Getting ROI')
    fixed_segm_bbox = get_roi(args.fixed, args.anatomy, args.outputdir,
                              'fixed_segmentation', args.fixedsegm, args.debug)
    moving_segm_bbox = get_roi(args.moving, args.anatomy, args.outputdir,
                               'moving_segmentation', args.movingsegm, args.debug)

    crop_min = np.amin(np.vstack([fixed_segm_bbox[:3], moving_segm_bbox[:3]]), axis=0)
    crop_max = np.amax(np.vstack([fixed_segm_bbox[3:], moving_segm_bbox[3:]]), axis=0)

    # 2.2 Crop the fixed and moving images using such boxes
    fixed_image = fixed_image_or[crop_min[0]: crop_max[0],
                                 crop_min[1]: crop_max[1],
                                 crop_min[2]: crop_max[2], ...]
    debug_save_image(fixed_image, 'img_2_cropped_fixed_image', args.outputdir, args.debug)

    moving_image = moving_image_or[crop_min[0]: crop_max[0],
                                   crop_min[1]: crop_max[1],
                                   crop_min[2]: crop_max[2], ...]
    debug_save_image(moving_image, 'img_2_cropped_moving_image', args.outputdir, args.debug)

    image_shape_crop = fixed_image.shape
    # 2.3 Resize the images to the expected input size
    zoom_factors = IMAGE_INTPUT_SHAPE / image_shape_crop
    fixed_image = zoom(fixed_image, zoom_factors)
    moving_image = zoom(moving_image, zoom_factors)
    fixed_image = min_max_norm(fixed_image)
    moving_image = min_max_norm(moving_image)
    debug_save_image(fixed_image, 'img_3_preproc_fixed_image', args.outputdir, args.debug)
    debug_save_image(moving_image, 'img_3_preproc_moving_image', args.outputdir, args.debug)

    # 3. Build the whole graph
    logger.info('Building TF graph')
    ### METRICS GRAPH ###
    fix_img_ph = tf.compat.v1.placeholder(tf.float32, (1, None, None, None, 1), name='fix_img')
    pred_img_ph = tf.compat.v1.placeholder(tf.float32, (1, None, None, None, 1), name='pred_img')

    ssim_tf = StructuralSimilarity_simplified(patch_size=2, dim=3, dynamic_range=1.).metric(fix_img_ph, pred_img_ph)
    ncc_tf = NCC(image_shape_or).metric(fix_img_ph, pred_img_ph)
    mse_tf = vxm.losses.MSE().loss(fix_img_ph, pred_img_ph)
    ms_ssim_tf = MultiScaleStructuralSimilarity(max_val=1., filter_size=3).metric(fix_img_ph, pred_img_ph)

    logger.info(f'Using model: {"Brain" if args.anatomy == "B" else "Liver"} -> {args.model}')
    MODEL_FILE = MODELS_FILE[args.anatomy][args.model]

    # try:
    #     network = tf.keras.models.load_model(MODEL_FILE,
    #                                          {'VxmDense': vxm.networks.VxmDense,
    #                                           # 'VxmDenseSemiSupervisedSeg': vxm.networks.VxmDenseSemiSupervisedSeg,
    #                                           'AdamAccumulated': AdamAccumulated
    #                                           },
    #                                          compile=False)
    # except ValueError as e:
    #     enc_features = [32, 64, 128, 256, 512, 1024]  # const.ENCODER_FILTERS
    #     dec_features = enc_features[::-1] + [16, 16]  # const.ENCODER_FILTERS[::-1]
    #     nb_features = [enc_features, dec_features]
    #     if re.search('^UW|SEGGUIDED_', MODEL_FILE):
    #         network = vxm.networks.VxmDense(inshape=IMAGE_INTPUT_SHAPE[:-1],
    #                                                    nb_unet_features=nb_features,
    #                                                    int_steps=0,
    #                                                    int_downsize=1,
    #                                                    seg_downsize=1)
    #     else:
    #         network = vxm.networks.VxmDense(inshape=IMAGE_INTPUT_SHAPE[:-1],
    #                                                    nb_unet_features=nb_features,
    #                                                    int_steps=0)
    #     network.load_weights(MODEL_FILE, by_name=True)

    enc_features = [32, 64, 128, 256, 512, 1024]  # const.ENCODER_FILTERS
    dec_features = enc_features[::-1] + [16, 16]  # const.ENCODER_FILTERS[::-1]
    nb_features = [enc_features, dec_features]
    network = vxm.networks.VxmDense(inshape=IMAGE_INTPUT_SHAPE[:-1],
                                    nb_unet_features=nb_features,
                                    int_steps=0)
    network.load_weights(MODEL_FILE, by_name=True)
    network.trainable = False

    registration_model = network.get_registration_model()
    deb_model = network.apply_transform

    logger.info('Performing registration')
    with sess.as_default():
        if args.debug:
            registration_model.summary(line_length=C.SUMMARY_LINE_LENGTH)
        time_disp_map_start = time.time()
        # disp_map = registration_model.predict([moving_image[np.newaxis, ...], fixed_image[np.newaxis, ...]])
        p, disp_map = network.predict([moving_image[np.newaxis, ...], fixed_image[np.newaxis, ...]])
        time_disp_map_end = time.time()
        debug_save_image(np.squeeze(disp_map), 'disp_map_0_raw', args.outputdir, args.debug)
        debug_save_image(p[0, ...], 'img_4_net_pred_image', args.outputdir, args.debug)
        # pred_image = min_max_norm(pred_image)
        # pred_image_isot = zoom(pred_image[0, ...], zoom_factors, order=3)[np.newaxis, ...]
        # fixed_image_isot = zoom(fixed_image[0, ...], zoom_factors, order=3)[np.newaxis, ...]

        # Up sample the displacement map to the full res
        trf = np.eye(4)
        np.fill_diagonal(trf, 1/zoom_factors)
        disp_map = resize_displacement_map(np.squeeze(disp_map), None, trf)
        debug_save_image(np.squeeze(disp_map), 'disp_map_1_upsampled', args.outputdir, args.debug)
        disp_map_or = pad_displacement_map(disp_map, crop_min, crop_max, image_shape_or)
        debug_save_image(np.squeeze(disp_map_or), 'disp_map_2_padded', args.outputdir, args.debug)
        disp_map_or = gaussian_filter(disp_map_or, 5)
        debug_save_image(np.squeeze(disp_map_or), 'disp_map_3_smoothed', args.outputdir, args.debug)

        time_pred_img_start = time.time()
        pred_image = SpatialTransformer(interp_method='linear', indexing='ij', single_transform=False)([moving_image_or[np.newaxis, ...], disp_map_or[np.newaxis, ...]]).eval()
        time_pred_img_end = time.time()
        ssim, ncc, mse, ms_ssim = sess.run([ssim_tf, ncc_tf, mse_tf, ms_ssim_tf],
                                           {'fix_img:0': fixed_image_or[np.newaxis, ...], 'pred_img:0': pred_image})
        ssim = np.mean(ssim)
        ms_ssim = ms_ssim[0]
        pred_image = pred_image[0, ...]

        save_nifti(pred_image, os.path.join(args.outputdir, 'pred_image.nii.gz'))
        np.savez_compressed(os.path.join(args.outputdir, 'displacement_map.npz'), disp_map_or)
        logger.info('Predicted image (full image) and displacement map saved in: '.format(args.outputdir))
        logger.info(f'Displacement map prediction time: {time_disp_map_end - time_disp_map_start} s')
        logger.info(f'Predicted image time: {time_pred_img_end - time_pred_img_start} s')

        logger.info('Similarity metrics (Full image)\n------------------')
        logger.info('SSIM: {:.03f}'.format(ssim))
        logger.info('NCC: {:.03f}'.format(ncc))
        logger.info('MSE: {:.03f}'.format(mse))
        logger.info('MS SSIM: {:.03f}'.format(ms_ssim))

        ssim, ncc, mse, ms_ssim = sess.run([ssim_tf, ncc_tf, mse_tf, ms_ssim_tf],
                                           {'fix_img:0': fixed_image[np.newaxis, ...], 'pred_img:0': p})
        ssim = np.mean(ssim)
        ms_ssim = ms_ssim[0]
        logger.info('\nSimilarity metrics (ROI)\n------------------')
        logger.info('SSIM: {:.03f}'.format(ssim))
        logger.info('NCC: {:.03f}'.format(ncc))
        logger.info('MSE: {:.03f}'.format(mse))
        logger.info('MS SSIM: {:.03f}'.format(ms_ssim))

    del registration_model
    logger.info('Done')