import os, sys

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)  # PYTHON > 3.3 does not allow relative referencing

from datetime import datetime

from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from tensorflow.python.keras.utils import Progbar
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.python.framework.errors import InvalidArgumentError

import DeepDeformationMapRegistration.utils.constants as C
from DeepDeformationMapRegistration.losses import StructuralSimilarity_simplified, NCC, GeneralizedDICEScore, \
    HausdorffDistanceErosion
from DeepDeformationMapRegistration.ms_ssim_tf import MultiScaleStructuralSimilarity
from DeepDeformationMapRegistration.ms_ssim_tf import _MSSSIM_WEIGHTS
from DeepDeformationMapRegistration.utils.acummulated_optimizer import AdamAccumulated
from DeepDeformationMapRegistration.utils.misc import function_decorator
from DeepDeformationMapRegistration.layers import AugmentationLayer, UncertaintyWeighting
from DeepDeformationMapRegistration.utils.nifti_utils import save_nifti

from Brain_study.data_generator import BatchGenerator
from Brain_study.utils import SummaryDictionary, named_logs

import COMET.augmentation_constants as COMET_C

import numpy as np
import tensorflow as tf
import voxelmorph as vxm
import h5py
import re
import itertools


def launch_train(dataset_folder, validation_folder, output_folder, model_file, gpu_num=0, lr=1e-4, rw=5e-3,
                 simil=['ssim'], segm=['dice'], max_epochs=C.EPOCHS, prior_reg_w=5e-3, freeze_layers=None,
                 acc_gradients=1, batch_size=16):
    # 0. Input checks
    assert dataset_folder is not None and output_folder is not None and model_file is not None
    assert '.h5' in model_file, 'The model must be an H5 file'

    # 1. Load variables
    os.environ['CUDA_DEVICE_ORDER'] = C.DEV_ORDER
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_num)  # Check availability before running using 'nvidia-smi'
    C.GPU_NUM = str(gpu_num)

    if freeze_layers is not None:
        assert all(s in ['INPUT', 'OUTPUT', 'ENCODER', 'DECODER', 'TOP', 'BOTTOM'] for s in freeze_layers), \
            'Invalid option for "freeze". Expected one or several of: INPUT, OUTPUT, ENCODER, DECODER, TOP, BOTTOM'
        multiple_ranges = 'TOP' in freeze_layers
        freeze_layers = [list(COMET_C.LAYER_RANGES[l]) for l in list(set(freeze_layers))]
        freeze_layers = freeze_layers[0] if multiple_ranges else freeze_layers

        # if len(freeze_layers) > 1:
        #     freeze_layers = list(itertools.chain.from_iterable(freeze_layers))

    os.makedirs(output_folder, exist_ok=True)
    # dataset_copy = DatasetCopy(dataset_folder, os.path.join(output_folder, 'temp'))
    log_file = open(os.path.join(output_folder, 'log.txt'), 'w')
    C.TRAINING_DATASET = dataset_folder  # dataset_copy.copy_dataset()
    C.VALIDATION_DATASET = validation_folder
    C.ACCUM_GRADIENT_STEP = acc_gradients
    C.BATCH_SIZE = batch_size if C.ACCUM_GRADIENT_STEP == 1 else C.ACCUM_GRADIENT_STEP
    C.EARLY_STOP_PATIENCE = 5 * (C.ACCUM_GRADIENT_STEP / 2 if C.ACCUM_GRADIENT_STEP != 1 else 1)
    C.LEARNING_RATE = lr
    C.LIMIT_NUM_SAMPLES = None
    C.EPOCHS = max_epochs

    aux = "[{}]\tINFO:\nTRAIN DATASET: {}\nVALIDATION DATASET: {}\n" \
          "GPU: {}\n" \
          "BATCH SIZE: {}\n" \
          "LR: {}\n" \
          "SIMILARITY: {}\n" \
          "SEGMENTATION: {}\n" \
          "REG. WEIGHT: {}\n" \
          "EPOCHS: {:d}\n" \
          "ACCUM. GRAD: {}\n" \
          "EARLY STOP PATIENCE: {}\n" \
          "FROZEN LAYERS: {}".format(datetime.now().strftime('%H:%M:%S\t%d/%m/%Y'),
                                     C.TRAINING_DATASET,
                                     C.VALIDATION_DATASET,
                                     C.GPU_NUM,
                                     C.BATCH_SIZE,
                                     C.LEARNING_RATE,
                                     simil,
                                     segm,
                                     rw,
                                     C.EPOCHS,
                                     C.ACCUM_GRADIENT_STEP,
                                     C.EARLY_STOP_PATIENCE,
                                     freeze_layers)

    log_file.write(aux)
    print(aux)

    # 2. Data generator
    data_generator = BatchGenerator(C.TRAINING_DATASET, C.BATCH_SIZE if C.ACCUM_GRADIENT_STEP == 1 else 1, True,
                                    C.TRAINING_PERC, labels=['all'], combine_segmentations=False,
                                    directory_val=C.VALIDATION_DATASET)

    train_generator = data_generator.get_train_generator()
    validation_generator = data_generator.get_validation_generator()

    image_input_shape = train_generator.get_data_shape()[-1][:-1]
    image_output_shape = [64] * 3
    nb_labels = len(train_generator.get_segmentation_labels())

    # 3. Load model
    # IMPORTANT: the mode MUST be loaded AFTER setting up the session configuration
    config = tf.compat.v1.ConfigProto()  # device_count={'GPU':0})
    config.gpu_options.allow_growth = True
    config.log_device_placement = False  ## to log device placement (on which device the operation ran)
    sess = tf.Session(config=config)
    tf.keras.backend.set_session(sess)

    print('MODEL LOCATION: ', model_file)

    enc_features = [16, 32, 32, 32]  # const.ENCODER_FILTERS
    dec_features = [32, 32, 32, 32, 32, 16, 16]  # const.ENCODER_FILTERS[::-1]
    nb_features = [enc_features, dec_features]
    network = vxm.networks.VxmDenseSemiSupervisedSeg(inshape=image_output_shape,
                                                     nb_labels=nb_labels,
                                                     nb_unet_features=nb_features,
                                                     int_steps=0,
                                                     int_downsize=1,
                                                     seg_downsize=1)
    network.load_weights(model_file, by_name=True)

    # 4. Freeze/unfreeze model layers
    if freeze_layers is not None:
        aux = list()
        for r in freeze_layers:
            for l in range(*r):
                network.layers[l].trainable = False
                aux.append(l)
        aux.sort()
        msg = "[INF]: Frozen layers {}".format(', '.join([str(a) for a in aux]))
    else:
        msg = "[INF] None frozen layers"
    print(msg)
    log_file.write(msg)

    network.summary()
    network.summary(print_fn=log_file.write)
    #   Complete the model with the augmentation layer
    input_layer_train = Input(shape=train_generator.get_data_shape()[0], name='input_train')
    augm_layer = AugmentationLayer(max_displacement=COMET_C.MAX_AUG_DISP,  # Max 30 mm in isotropic space
                                   max_deformation=COMET_C.MAX_AUG_DEF,  # Max 6 mm in isotropic space
                                   max_rotation=COMET_C.MAX_AUG_ANGLE,  # Max 10 deg in isotropic space
                                   num_control_points=COMET_C.NUM_CONTROL_PTS_AUG,
                                   num_augmentations=COMET_C.NUM_AUGMENTATIONS,
                                   gamma_augmentation=COMET_C.GAMMA_AUGMENTATION,
                                   brightness_augmentation=COMET_C.BRIGHTNESS_AUGMENTATION,
                                   in_img_shape=image_input_shape,
                                   out_img_shape=image_output_shape,
                                   only_image=False,  # If baseline then True
                                   only_resize=False,
                                   trainable=False)
    augm_model = Model(inputs=input_layer_train, outputs=augm_layer(input_layer_train))

    # 5. Setup training environment: loss, optimizer, callbacks, evaluation

    # Losses and loss weights
    SSIM_KER_SIZE = 5
    MS_SSIM_WEIGHTS = _MSSSIM_WEIGHTS[:3]
    MS_SSIM_WEIGHTS /= np.sum(MS_SSIM_WEIGHTS)
    loss_simil = []
    prior_loss_w = []
    for s in simil:
        if s == 'ssim':
            loss_simil.append(StructuralSimilarity_simplified(patch_size=SSIM_KER_SIZE, dim=3, dynamic_range=1.).loss)
            prior_loss_w.append(1.)
        elif s == 'ms_ssim':
            loss_simil.append(MultiScaleStructuralSimilarity(max_val=1., filter_size=SSIM_KER_SIZE,
                                                             power_factors=MS_SSIM_WEIGHTS).loss)
            prior_loss_w.append(1.)
        elif s == 'ncc':
            loss_simil.append(NCC(image_input_shape).loss)
            prior_loss_w.append(1.)
        elif s == 'mse':
            loss_simil.append(vxm.losses.MSE().loss)
            prior_loss_w.append(1.)
        else:
            raise ValueError('Unknown similarity function: ', s)

    loss_segm = []
    for s in segm:
        if s == 'dice':
            loss_segm.append(GeneralizedDICEScore(image_output_shape + [train_generator.get_data_shape()[2][-1]],
                                                  num_labels=nb_labels).loss)
            prior_loss_w.append(1.)
        elif s == 'dice_macro':
            loss_segm.append(GeneralizedDICEScore(image_output_shape + [train_generator.get_data_shape()[2][-1]],
                                                  num_labels=nb_labels).loss_macro)
            prior_loss_w.append(1.)
        elif s == 'hd':
            loss_segm.append(HausdorffDistanceErosion(3, 10, im_shape=image_output_shape + [
                train_generator.get_data_shape()[2][-1]]).loss)
            prior_loss_w.append(1.)
        else:
            raise ValueError('Unknown similarity function: ', s)

    # Uncertainty weigthing module
    grad = tf.keras.Input(shape=(*image_output_shape, 3), name='multiLoss_grad_input', dtype=tf.float32)
    fix_seg = tf.keras.Input(shape=(*image_output_shape, len(train_generator.get_segmentation_labels())),
                             name='multiLoss_fix_seg_input', dtype=tf.float32)

    multiLoss = UncertaintyWeighting(num_loss_fns=len(loss_simil) + len(loss_segm),
                                     num_reg_fns=1,
                                     loss_fns=[*loss_simil,
                                               *loss_segm],
                                     reg_fns=[vxm.losses.Grad('l2').loss],
                                     prior_loss_w=prior_loss_w,
                                     # prior_loss_w=[1., 0.1, 1., 1.],
                                     prior_reg_w=[prior_reg_w],
                                     name='MultiLossLayer')
    loss = multiLoss([*[network.inputs[1]] * len(loss_simil), *[fix_seg] * len(loss_segm),
                      *[network.outputs[0]] * len(loss_simil), *[network.outputs[2]] * len(loss_simil),
                      grad,
                      network.outputs[1]])

    # inputs = [mov_img, fix_img, mov_segm, fix_segm, zero_grads]
    # outputs = [pred_img, flow, pred_segm, loss]
    full_model = tf.keras.Model(inputs=network.inputs + [fix_seg, grad],
                                outputs=network.outputs + [loss])

    os.makedirs(os.path.join(output_folder, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'tensorboard'), exist_ok=True)
    callback_tensorboard = TensorBoard(log_dir=os.path.join(output_folder, 'tensorboard'),
                                       batch_size=C.BATCH_SIZE, write_images=False, histogram_freq=0,
                                       update_freq='epoch',  # or 'batch' or integer
                                       write_graph=True, write_grads=True
                                       )
    callback_early_stop = EarlyStopping(monitor='val_loss', verbose=1, patience=C.EARLY_STOP_PATIENCE,
                                        min_delta=0.00001)

    callback_best_model = ModelCheckpoint(os.path.join(output_folder, 'checkpoints', 'best_model.h5'),
                                          save_best_only=True, monitor='val_loss', verbose=1, mode='min')
    callback_save_checkpoint = ModelCheckpoint(os.path.join(output_folder, 'checkpoints', 'checkpoint.h5'),
                                               save_weights_only=True, monitor='val_loss', verbose=0, mode='min')

    optimizer = AdamAccumulated(C.ACCUM_GRADIENT_STEP, C.LEARNING_RATE)
    full_model.compile(optimizer=optimizer,
                       loss=None, )

    # 6. Training loop
    callback_tensorboard.set_model(full_model)
    callback_early_stop.set_model(full_model)
    callback_best_model.set_model(network)  # ONLY SAVE THE NETWORK!!!
    callback_save_checkpoint.set_model(network)  # ONLY SAVE THE NETWORK!!!

    summary = SummaryDictionary(full_model, C.BATCH_SIZE)
    names = full_model.metrics_names
    zero_grads = tf.zeros_like(network.references.pos_flow, name='dummy_zero_grads')  # Dummy zeros-tensor
    log_file.write('\n\n[{}]\tINFO:\tStart training\n\n'.format(datetime.now().strftime('%H:%M:%S\t%d/%m/%Y')))

    with sess.as_default():
        # tf.global_variables_initializer()
        callback_tensorboard.on_train_begin()
        callback_early_stop.on_train_begin()
        callback_best_model.on_train_begin()
        callback_save_checkpoint.on_train_begin()

        for epoch in range(C.EPOCHS):
            callback_tensorboard.on_epoch_begin(epoch)
            callback_early_stop.on_epoch_begin(epoch)
            callback_best_model.on_epoch_begin(epoch)
            callback_save_checkpoint.on_epoch_begin(epoch)
            print("\nEpoch {}/{}".format(epoch, C.EPOCHS))
            print("TRAIN")

            log_file.write(
                '\n\n[{}]\tINFO:\tTraining epoch {}\n\n'.format(datetime.now().strftime('%H:%M:%S\t%d/%m/%Y'), epoch))
            progress_bar = Progbar(len(train_generator), width=30, verbose=1)
            for step, (in_batch, _) in enumerate(train_generator, 1):
                callback_best_model.on_train_batch_begin(step)
                callback_save_checkpoint.on_train_batch_begin(step)
                callback_early_stop.on_train_batch_begin(step)

                try:
                    fix_img, mov_img, fix_seg, mov_seg = augm_model.predict(in_batch)
                    np.nan_to_num(fix_img, copy=False)
                    np.nan_to_num(mov_img, copy=False)
                    if np.isnan(np.sum(mov_img)) or np.isnan(np.sum(fix_img)) or np.isinf(np.sum(mov_img)) or np.isinf(
                            np.sum(fix_img)):
                        msg = 'CORRUPTED DATA!! Unique: Fix: {}\tMoving: {}'.format(np.unique(fix_img),
                                                                                    np.unique(mov_img))
                        print(msg)
                        log_file.write('\n\n[{}]\tWAR: {}'.format(datetime.now().strftime('%H:%M:%S\t%d/%m/%Y'), msg))

                except InvalidArgumentError as err:
                    print('TF Error : {}'.format(str(err)))
                    continue

                ret = full_model.train_on_batch(
                    x=(mov_img, fix_img, mov_seg, fix_seg, zero_grads))  # The second element doesn't matter

                summary.on_train_batch_end(ret)
                callback_best_model.on_train_batch_end(step, named_logs(full_model, ret))
                callback_save_checkpoint.on_train_batch_end(step, named_logs(full_model, ret))
                callback_early_stop.on_train_batch_end(step, named_logs(full_model, ret))
                progress_bar.update(step, zip(names, ret))
                log_file.write('\t\tStep {:03d}: {}'.format(step, ret))
            val_values = progress_bar._values.copy()
            ret = [val_values[x][0] / val_values[x][1] for x in names]

            print('\nVALIDATION')
            log_file.write(
                '\n\n[{}]\tINFO:\tValidation epoch {}\n\n'.format(datetime.now().strftime('%H:%M:%S\t%d/%m/%Y'), epoch))
            progress_bar = Progbar(len(validation_generator), width=30, verbose=1)
            for step, (in_batch, _) in enumerate(validation_generator, 1):
                try:
                    fix_img, mov_img, fix_seg, mov_seg = augm_model.predict(in_batch)
                except InvalidArgumentError as err:
                    print('TF Error : {}'.format(str(err)))
                    continue

                ret = full_model.test_on_batch(x=(mov_img, fix_img, mov_seg, fix_seg, zero_grads))

                summary.on_validation_batch_end(ret)
                progress_bar.update(step, zip(names, ret))
                log_file.write('\t\tStep {:03d}: {}'.format(step, ret))
            val_values = progress_bar._values.copy()
            ret = [val_values[x][0] / val_values[x][1] for x in names]

            train_generator.on_epoch_end()
            validation_generator.on_epoch_end()
            epoch_summary = summary.on_epoch_end()  # summary resets after on_epoch_end() call
            callback_tensorboard.on_epoch_end(epoch, epoch_summary)
            callback_best_model.on_epoch_end(epoch, epoch_summary)
            callback_save_checkpoint.on_epoch_end(epoch, epoch_summary)
            callback_early_stop.on_epoch_end(epoch, epoch_summary)
            print('End of epoch {}: '.format(epoch), ret, '\n')

        callback_tensorboard.on_train_end()
        callback_best_model.on_train_end()
        callback_save_checkpoint.on_train_end()
        callback_early_stop.on_train_end()
# 7. Wrap up
