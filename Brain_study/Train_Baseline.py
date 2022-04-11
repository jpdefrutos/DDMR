import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)  # PYTHON > 3.3 does not allow relative referencing

import numpy as np
import tensorflow as tf

from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.python.keras.utils import Progbar
from tensorflow.python.framework.errors import InvalidArgumentError

import voxelmorph as vxm
import neurite as ne
import h5py
import pickle

import DeepDeformationMapRegistration.utils.constants as C
from DeepDeformationMapRegistration.losses import NCC, StructuralSimilarity, StructuralSimilarity_simplified
from DeepDeformationMapRegistration.utils.misc import try_mkdir, DatasetCopy, function_decorator
from DeepDeformationMapRegistration.utils.acummulated_optimizer import AdamAccumulated
from DeepDeformationMapRegistration.layers import AugmentationLayer
from DeepDeformationMapRegistration.utils.nifti_utils import save_nifti
from DeepDeformationMapRegistration.ms_ssim_tf import MultiScaleStructuralSimilarity, _MSSSIM_WEIGHTS

from Brain_study.data_generator import BatchGenerator
from Brain_study.utils import SummaryDictionary, named_logs

from tqdm import tqdm
from datetime import datetime


def launch_train(dataset_folder, validation_folder, output_folder, gpu_num=0, lr=1e-4, rw=5e-3, simil='ssim'):
    assert dataset_folder is not None and output_folder is not None

    os.environ['CUDA_DEVICE_ORDER'] = C.DEV_ORDER
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_num) # Check availability before running using 'nvidia-smi'
    C.GPU_NUM = str(gpu_num)

    output_folder = os.path.join(output_folder + '_' + datetime.now().strftime("%H%M%S-%d%m%Y"))
    os.makedirs(output_folder, exist_ok=True)
    # dataset_copy = DatasetCopy(dataset_folder, os.path.join(output_folder, 'temp'))
    log_file = open(os.path.join(output_folder, 'log.txt'), 'w')
    C.TRAINING_DATASET = dataset_folder #dataset_copy.copy_dataset()
    C.VALIDATION_DATASET = validation_folder
    C.ACCUM_GRADIENT_STEP = 16
    C.BATCH_SIZE = 16 if C.ACCUM_GRADIENT_STEP == 1 else C.ACCUM_GRADIENT_STEP
    C.EARLY_STOP_PATIENCE = 5 * (C.ACCUM_GRADIENT_STEP / 2 if C.ACCUM_GRADIENT_STEP != 1 else 1)
    C.LEARNING_RATE = lr
    C.LIMIT_NUM_SAMPLES = None
    C.EPOCHS = 10000

    aux = "[{}]\tINFO:\nTRAIN DATASET: {}\nVALIDATION DATASET: {}\n" \
          "GPU: {}\n" \
          "BATCH SIZE: {}\n" \
          "LR: {}\n" \
          "SIMILARITY: {}\n" \
          "REG. WEIGHT: {}\n" \
          "EPOCHS: {:d}\n" \
          "ACCUM. GRAD: {}\n" \
          "EARLY STOP PATIENCE: {}".format(datetime.now().strftime('%H:%M:%S\t%d/%m/%Y'),
                                           C.TRAINING_DATASET,
                                           C.VALIDATION_DATASET,
                                           C.GPU_NUM,
                                           C.BATCH_SIZE,
                                           C.LEARNING_RATE,
                                           simil,
                                           rw,
                                           C.EPOCHS,
                                           C.ACCUM_GRADIENT_STEP,
                                           C.EARLY_STOP_PATIENCE)

    log_file.write(aux)
    print(aux)

    # Load data
    # Build data generator
    data_generator = BatchGenerator(C.TRAINING_DATASET, C.BATCH_SIZE if C.ACCUM_GRADIENT_STEP == 1 else 1, True,
                                    C.TRAINING_PERC, True, ['none'], directory_val=C.VALIDATION_DATASET)

    train_generator = data_generator.get_train_generator()
    validation_generator = data_generator.get_validation_generator()

    image_input_shape = train_generator.get_data_shape()[-1][:-1]
    image_output_shape = [64] * 3

    # Config the training sessions
    config = tf.compat.v1.ConfigProto()  # device_count={'GPU':0})
    config.gpu_options.allow_growth = True
    config.log_device_placement = False  ## to log device placement (on which device the operation ran)
    sess = tf.Session(config=config)
    tf.keras.backend.set_session(sess)

    # Build model
    input_layer_train = Input(shape=train_generator.get_data_shape()[-1], name='input_train')
    augm_layer_train = AugmentationLayer(max_displacement=C.MAX_AUG_DISP,   # Max 30 mm in isotropic space
                                   max_deformation=C.MAX_AUG_DEF,  # Max 6 mm in isotropic space
                                   max_rotation=C.MAX_AUG_ANGLE,   # Max 10 deg in isotropic space
                                   num_control_points=C.NUM_CONTROL_PTS_AUG,
                                   num_augmentations=C.NUM_AUGMENTATIONS,
                                   gamma_augmentation=C.GAMMA_AUGMENTATION,
                                   brightness_augmentation=C.BRIGHTNESS_AUGMENTATION,
                                   in_img_shape=image_input_shape,
                                   out_img_shape=image_output_shape,
                                   only_image=True,
                                   only_resize=False,
                                   trainable=False)
    augm_model_train = Model(inputs=input_layer_train, outputs=augm_layer_train(input_layer_train))

    input_layer_valid = Input(shape=validation_generator.get_data_shape()[0], name='input_valid')
    augm_layer_valid = AugmentationLayer(max_displacement=C.MAX_AUG_DISP,   # Max 30 mm in isotropic space
                                   max_deformation=C.MAX_AUG_DEF,  # Max 6 mm in isotropic space
                                   max_rotation=C.MAX_AUG_ANGLE,   # Max 10 deg in isotropic space
                                   num_control_points=C.NUM_CONTROL_PTS_AUG,
                                   num_augmentations=C.NUM_AUGMENTATIONS,
                                   gamma_augmentation=C.GAMMA_AUGMENTATION,
                                   brightness_augmentation=C.BRIGHTNESS_AUGMENTATION,
                                   in_img_shape=image_input_shape,
                                   out_img_shape=image_output_shape,
                                   only_image=False,
                                   only_resize=False,
                                   trainable=False)
    augm_model_valid = Model(inputs=input_layer_valid, outputs=augm_layer_valid(input_layer_valid))

    # Build model
    enc_features = [16, 32, 32, 32]     # const.ENCODER_FILTERS
    dec_features = [32, 32, 32, 32, 32, 16, 16]     # const.ENCODER_FILTERS[::-1]
    nb_features = [enc_features, dec_features]
    network = vxm.networks.VxmDense(inshape=image_output_shape,
                                    nb_unet_features=nb_features,
                                    int_steps=0)

    # Losses and loss weights
    SSIM_KER_SIZE = 5
    MS_SSIM_WEIGHTS = _MSSSIM_WEIGHTS[:3]
    MS_SSIM_WEIGHTS /= np.sum(MS_SSIM_WEIGHTS)
    if simil.lower() == 'mse':
        loss_fnc = vxm.losses.MSE().loss
    elif simil.lower() == 'ncc':
        loss_fnc = NCC(image_input_shape).loss
    elif simil.lower() == 'ssim':
        loss_fnc = StructuralSimilarity_simplified(patch_size=SSIM_KER_SIZE, dim=3, dynamic_range=1.).loss
    elif simil.lower() == 'ms_ssim':
        loss_fnc = MultiScaleStructuralSimilarity(max_val=1., filter_size=SSIM_KER_SIZE, power_factors=MS_SSIM_WEIGHTS).loss
    elif simil.lower() == 'mse__ms_ssim' or simil.lower() == 'ms_ssim__mse':
        @function_decorator('MSSSIM_MSE__loss')
        def loss_fnc(y_true, y_pred):
            return vxm.losses.MSE().loss(y_true, y_pred) +\
                   MultiScaleStructuralSimilarity(max_val=1., filter_size=SSIM_KER_SIZE, power_factors=MS_SSIM_WEIGHTS).loss(y_true, y_pred)
    elif simil.lower() == 'ncc__ms_ssim' or simil.lower() == 'ms_ssim__ncc':
        @function_decorator('MSSSIM_NCC__loss')
        def loss_fnc(y_true, y_pred):
            return NCC(image_input_shape).loss(y_true, y_pred) +\
                   MultiScaleStructuralSimilarity(max_val=1., filter_size=SSIM_KER_SIZE, power_factors=MS_SSIM_WEIGHTS).loss(y_true, y_pred)
    elif simil.lower() == 'mse__ssim' or simil.lower() == 'ssim__mse':
        @function_decorator('SSIM_MSE__loss')
        def loss_fnc(y_true, y_pred):
            return vxm.losses.MSE().loss(y_true, y_pred) +\
                   StructuralSimilarity_simplified(patch_size=SSIM_KER_SIZE, dim=3, dynamic_range=1.).loss(y_true, y_pred)
    elif simil.lower() == 'ncc__ssim' or simil.lower() == 'ssim__ncc':
        @function_decorator('SSIM_NCC__loss')
        def loss_fnc(y_true, y_pred):
            return NCC(image_input_shape).loss(y_true, y_pred) +\
                   StructuralSimilarity_simplified(patch_size=SSIM_KER_SIZE, dim=3, dynamic_range=1.).loss(y_true, y_pred)
    else:
        raise ValueError('Unknown similarity metric: ' + simil)

    # Train
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'tensorboard'), exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'history'), exist_ok=True)

    callback_best_model = ModelCheckpoint(os.path.join(output_folder, 'checkpoints', 'best_model.h5'),
                            save_best_only=True, monitor='val_loss', verbose=1, mode='min')
    # callback_save_model = ModelCheckpoint(os.path.join(output_folder, 'checkpoints', 'weights.{epoch:05d}-{val_loss:.2f}.h5'),
    #                save_best_only=True, save_weights_only=True, monitor='val_loss', verbose=0, mode='min')
    # CSVLogger(train_log_name, ';'),
    # UpdateLossweights([haus_weight, dice_weight], [const.MODEL+'_resampler_seg', const.MODEL+'_resampler_seg'])
    callback_tensorboard = TensorBoard(log_dir=os.path.join(output_folder, 'tensorboard'),
                                       batch_size=C.BATCH_SIZE, write_images=False, histogram_freq=0,
                                       update_freq='epoch',     # or 'batch' or integer
                                       write_graph=True, write_grads=True
                                       )
    callback_early_stop = EarlyStopping(monitor='val_loss', verbose=1, patience=C.EARLY_STOP_PATIENCE, min_delta=0.00001)

    losses = {'transformer': loss_fnc,
              'flow': vxm.losses.Grad('l2').loss}
    metrics = {'transformer': [StructuralSimilarity_simplified(patch_size=SSIM_KER_SIZE, dim=3, dynamic_range=1.).metric,
                               MultiScaleStructuralSimilarity(max_val=1., filter_size=SSIM_KER_SIZE, power_factors=MS_SSIM_WEIGHTS).metric,
                               tf.keras.losses.MSE,
                               NCC(image_input_shape).metric],
               #'flow': vxm.losses.Grad('l2').loss
               }
    loss_weights = {'transformer': 1.,
                    'flow': rw}

    # Compile the model
    optimizer = AdamAccumulated(C.ACCUM_GRADIENT_STEP, C.LEARNING_RATE)
    network.compile(optimizer=optimizer,
                    loss=losses,
                    loss_weights=loss_weights,
                    metrics=metrics)

    callback_tensorboard.set_model(network)
    callback_best_model.set_model(network)
    # callback_save_model.set_model(network)
    callback_early_stop.set_model(network)
    # TODO: https://towardsdatascience.com/writing-tensorflow-2-custom-loops-438b1ab6eb6c

    summary = SummaryDictionary(network, C.BATCH_SIZE)
    names = network.metrics_names  # It give both the loss and metric names
    log_file.write('\n\n[{}]\tINFO:\tStart training\n\n'.format(datetime.now().strftime('%H:%M:%S\t%d/%m/%Y')))
    with sess.as_default():
        callback_tensorboard.on_train_begin()
        callback_early_stop.on_train_begin()
        callback_best_model.on_train_begin()
        # callback_save_model.on_train_begin()
        for epoch in range(C.EPOCHS):
            callback_tensorboard.on_epoch_begin(epoch)
            callback_early_stop.on_epoch_begin(epoch)
            callback_best_model.on_epoch_begin(epoch)
            # callback_save_model.on_epoch_begin(epoch)

            print("\nEpoch {}/{}".format(epoch, C.EPOCHS))
            print('TRAINING')

            log_file.write('\n\n[{}]\tINFO:\tTraining epoch {}\n\n'.format(datetime.now().strftime('%H:%M:%S\t%d/%m/%Y'), epoch))
            progress_bar = Progbar(len(train_generator), width=30, verbose=1)
            for step, (in_batch, _) in enumerate(train_generator, 1):
                # callback_tensorboard.on_train_batch_begin(step)
                callback_best_model.on_train_batch_begin(step)
                # callback_save_model.on_train_batch_begin(step)
                callback_early_stop.on_train_batch_begin(step)

                try:
                    fix_img, mov_img, *_ = augm_model_train.predict(in_batch)
                    np.nan_to_num(fix_img, copy=False)
                    np.nan_to_num(mov_img, copy=False)
                    if np.isnan(np.sum(mov_img)) or np.isnan(np.sum(fix_img)) or np.isinf(np.sum(mov_img)) or np.isinf(np.sum(fix_img)):
                        msg = 'CORRUPTED DATA!! Unique: Fix: {}\tMoving: {}'.format(np.unique(fix_img),
                                                                                          np.unique(mov_img))
                        print(msg)
                        log_file.write('\n\n[{}]\tWAR: {}'.format(datetime.now().strftime('%H:%M:%S\t%d/%m/%Y'), msg))

                except InvalidArgumentError as err:
                    print('TF Error : {}'.format(str(err)))
                    continue

                ret = network.train_on_batch(x=(mov_img, fix_img),
                                             y=(fix_img, fix_img))  # The second element doesn't matter
                if np.isnan(ret).any():
                    os.makedirs(os.path.join(output_folder, 'corrupted'), exist_ok=True)
                    save_nifti(mov_img, os.path.join(output_folder, 'corrupted', 'mov_img_nan.nii.gz'))
                    save_nifti(fix_img, os.path.join(output_folder, 'corrupted', 'fix_img_nan.nii.gz'))
                    pred_img, dm = network((mov_img, fix_img))
                    save_nifti(pred_img, os.path.join(output_folder, 'corrupted', 'pred_img_nan.nii.gz'))
                    save_nifti(dm, os.path.join(output_folder, 'corrupted', 'dm_nan.nii.gz'))
                    log_file.write('\n\n[{}]\tERR: Corruption error'.format(datetime.now().strftime('%H:%M:%S\t%d/%m/%Y')))
                    raise ValueError('CORRUPTION ERROR: Halting training')

                summary.on_train_batch_end(ret)
                # callback_tensorboard.on_train_batch_end(step, named_logs(network, ret))
                callback_best_model.on_train_batch_end(step, named_logs(network, ret))
                # callback_save_model.on_train_batch_end(step, named_logs(network, ret))
                callback_early_stop.on_train_batch_end(step, named_logs(network, ret))
                progress_bar.update(step, zip(names, ret))
                log_file.write('\t\tStep {:03d}: {}'.format(step, ret))
            val_values = progress_bar._values.copy()
            ret = [val_values[x][0]/val_values[x][1] for x in names]

            print('\nVALIDATION')
            log_file.write('\n\n[{}]\tINFO:\tValidation epoch {}\n\n'.format(datetime.now().strftime('%H:%M:%S\t%d/%m/%Y'), epoch))
            progress_bar = Progbar(len(validation_generator), width=30, verbose=1)
            for step, (in_batch, _) in enumerate(validation_generator, 1):
                # callback_tensorboard.on_test_batch_begin(step)    # This is cursed, don't do it again
                # callback_early_stop.on_test_batch_begin(step)
                try:
                    fix_img, mov_img, *_ = augm_model_valid.predict(in_batch)
                except InvalidArgumentError as err:
                    print('TF Error : {}'.format(str(err)))
                    continue

                ret = network.test_on_batch(x=(mov_img, fix_img),
                                            y=(fix_img, fix_img))
                # pred_segm = network.register(mov_segm, fix_segm)
                summary.on_validation_batch_end(ret)
                # callback_early_stop.on_test_batch_end(step, named_logs(network, ret))
                # callback_tensorboard.on_test_batch_end(step, named_logs(network, ret))    # This is cursed, don't do it again
                progress_bar.update(step, zip(names, ret))
                log_file.write('\t\tStep {:03d}: {}'.format(step, ret))
            val_values = progress_bar._values.copy()
            ret = [val_values[x][0]/val_values[x][1] for x in names]

            train_generator.on_epoch_end()
            validation_generator.on_epoch_end()
            epoch_summary = summary.on_epoch_end()  # summary resets after on_epoch_end() call
            callback_tensorboard.on_epoch_end(epoch, epoch_summary)
            callback_early_stop.on_epoch_end(epoch, epoch_summary)
            callback_best_model.on_epoch_end(epoch, epoch_summary)
            # callback_save_model.on_epoch_end(epoch, epoch_summary)
            print('End of epoch {}: '.format(epoch), ret, '\n')

        callback_tensorboard.on_train_end()
        # callback_save_model.on_train_end()
        callback_best_model.on_train_end()
        callback_early_stop.on_train_end()


if __name__ == '__main__':
    os.environ['CUDA_DEVICE_ORDER'] = C.DEV_ORDER
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Check availability before running using 'nvidia-smi'

    config = tf.compat.v1.ConfigProto()  # device_count={'GPU':0})
    config.gpu_options.allow_growth = True
    config.log_device_placement = False  ## to log device placement (on which device the operation ran)
    tf.keras.backend.set_session(tf.Session(config=config))

    launch_train('/mnt/EncryptedData1/Users/javier/vessel_registration/LiTS/None',
                 'TrainOutput/THESIS/UW_None_mse_ssim_haus', 0, mse=True)
