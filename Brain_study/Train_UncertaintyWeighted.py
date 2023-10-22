import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)  # PYTHON > 3.3 does not allow relative referencing

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.python.keras.utils import Progbar
from tensorflow.python.framework.errors import InvalidArgumentError
import voxelmorph as vxm
import neurite as ne
import h5py
from datetime import datetime
import pickle

import ddmr.utils.constants as C
from ddmr.utils.misc import try_mkdir, DatasetCopy, function_decorator
from ddmr.networks import WeaklySupervised
from ddmr.losses import HausdorffDistanceErosion, NCC, StructuralSimilarity_simplified, GeneralizedDICEScore
from ddmr.ms_ssim_tf import MultiScaleStructuralSimilarity, _MSSSIM_WEIGHTS
from ddmr.layers import UncertaintyWeighting, AugmentationLayer
from ddmr.utils.acummulated_optimizer import AdamAccumulated

from Brain_study.data_generator import BatchGenerator
from Brain_study.utils import SummaryDictionary, named_logs
import warnings
import re


def launch_train(dataset_folder, validation_folder, output_folder, prior_reg_w=5e-3, lr=1e-4, rw=5e-3,
                 gpu_num=0, simil=['mse'], segm=['dice'], acc_gradients=16, batch_size=1, max_epochs=10000,
                 early_stop_patience=1000, image_size=64, unet=[16, 32, 64, 128, 256], head=[16, 16], resume=None):
    assert dataset_folder is not None and output_folder is not None

    os.environ['CUDA_DEVICE_ORDER'] = C.DEV_ORDER
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_num) # Check availability before running using 'nvidia-smi'
    C.GPU_NUM = str(gpu_num)

    if batch_size != 1 and acc_gradients != 1:
        warnings.warn('WARNING: Batch size and Accumulative gradient step are set!')

    if resume is not None:
        try:
            assert os.path.exists(resume) and len(os.listdir(os.path.join(resume, 'checkpoints'))), 'Invalid directory: ' + resume
            output_folder = resume
            resume = True
        except AssertionError:
            output_folder = os.path.join(output_folder + '_' + datetime.now().strftime("%H%M%S-%d%m%Y"))
            resume = False
    else:
        resume = False
    os.makedirs(output_folder, exist_ok=True)
    log_file = open(os.path.join(output_folder, 'log.txt'), 'w')
    C.TRAINING_DATASET = dataset_folder #dataset_copy.copy_dataset()
    C.VALIDATION_DATASET = validation_folder
    C.ACCUM_GRADIENT_STEP = acc_gradients
    C.BATCH_SIZE = batch_size if C.ACCUM_GRADIENT_STEP == 1 else 1
    C.EARLY_STOP_PATIENCE = early_stop_patience
    C.LEARNING_RATE = lr
    C.LIMIT_NUM_SAMPLES = None
    C.EPOCHS = max_epochs

    aux = "[{}]\tINFO:\nTRAIN DATASET: {}\nVALIDATION DATASET: {}\n" \
          "GPU: {}\n" \
          "BATCH SIZE: {}\n" \
          "LR: {}\n" \
          "SIMILARITY {:d}: {}\n" \
          "SEGMENTATION {:d}: {}\n" \
          "EPOCHS: {:d}" \
          "ACCUM. GRAD: {}" \
          "EARLY STOP PATIENCE: {}".format(datetime.now().strftime('%H:%M:%S\t%d/%m/%Y'),
                                           C.TRAINING_DATASET,
                                           C.VALIDATION_DATASET,
                                           C.GPU_NUM,
                                           C.BATCH_SIZE,
                                           C.LEARNING_RATE,
                                           len(simil), ', '.join(simil),
                                           len(segm), ', '.join(segm),
                                           C.EPOCHS,
                                           C.ACCUM_GRADIENT_STEP,
                                           C.EARLY_STOP_PATIENCE)
    log_file.write(aux)
    print(aux)

    # Load data
    # Build data generator
    data_generator = BatchGenerator(C.TRAINING_DATASET, C.BATCH_SIZE if C.ACCUM_GRADIENT_STEP == 1 else 1, True,
                                    C.TRAINING_PERC, labels=['all'], combine_segmentations=False,
                                    directory_val=C.VALIDATION_DATASET)

    train_generator = data_generator.get_train_generator()
    validation_generator = data_generator.get_validation_generator()

    image_input_shape = train_generator.get_data_shape()[-1][:-1]
    image_output_shape = [image_size] * 3

    nb_labels = len(train_generator.get_segmentation_labels())

    # Config the training sessions
    config = tf.compat.v1.ConfigProto()  # device_count={'GPU':0})
    config.gpu_options.allow_growth = True
    config.log_device_placement = False  ## to log device placement (on which device the operation ran)
    # config.allow_soft_placement = True
    sess = tf.Session(config=config)
    tf.keras.backend.set_session(sess)

    # Losses and loss weights
    SSIM_KER_SIZE = 5
    MS_SSIM_WEIGHTS = _MSSSIM_WEIGHTS[:3]
    MS_SSIM_WEIGHTS /= np.sum(MS_SSIM_WEIGHTS)

    loss_simil = []
    prior_loss_w = []
    for s in simil:
        if s=='ssim':
            loss_simil.append(StructuralSimilarity_simplified(patch_size=SSIM_KER_SIZE, dim=3, dynamic_range=1.).loss)
            prior_loss_w.append(1.)
        elif s=='ms_ssim':
            loss_simil.append(MultiScaleStructuralSimilarity(max_val=1., filter_size=SSIM_KER_SIZE, power_factors=MS_SSIM_WEIGHTS).loss)
            prior_loss_w.append(1.)
        elif s=='ncc':
            loss_simil.append(NCC(image_input_shape).loss)
            prior_loss_w.append(1.)
        elif s=='mse':
            loss_simil.append(vxm.losses.MSE().loss)
            prior_loss_w.append(1.)
        else:
            raise ValueError('Unknown similarity function: ', s)

    loss_segm = []
    for s in segm:
        if s=='dice':
            loss_segm.append(GeneralizedDICEScore(image_output_shape + [nb_labels], num_labels=nb_labels).loss)
            prior_loss_w.append(1.)
        elif s=='hd':
            loss_segm.append(HausdorffDistanceErosion(3, 10, im_shape=image_output_shape + [nb_labels]).loss)
            prior_loss_w.append(1.)
        elif s == 'dice_macro':
            loss_segm.append(GeneralizedDICEScore(image_output_shape + [nb_labels], num_labels=nb_labels).loss_macro)
            prior_loss_w.append(1.)
        else:
            raise ValueError('Unknown similarity function: ' + s)

    # Build augmentation layer model
    input_layer_augm = Input(shape=train_generator.get_data_shape()[0], name='input_augmentation')
    augm_layer = AugmentationLayer(max_displacement=C.MAX_AUG_DISP,   # Max 30 mm in isotropic space
                                   max_deformation=C.MAX_AUG_DEF,  # Max 6 mm in isotropic space
                                   max_rotation=C.MAX_AUG_ANGLE,   # Max 10 deg in isotropic space
                                   num_control_points=C.NUM_CONTROL_PTS_AUG,
                                   num_augmentations=C.NUM_AUGMENTATIONS,
                                   gamma_augmentation=C.GAMMA_AUGMENTATION,
                                   in_img_shape=image_input_shape,
                                   out_img_shape=image_output_shape,
                                   only_image=False,
                                   only_resize=False,
                                   trainable=False)
    augmentation_model = Model(inputs=input_layer_augm, outputs=augm_layer(input_layer_augm))

    # enc_features = [16, 32, 32, 32]     # const.ENCODER_FILTERS
    # dec_features = [32, 32, 32, 32, 32, 16, 16]     # const.ENCODER_FILTERS[::-1]
    enc_features = unet     # const.ENCODER_FILTERS
    dec_features = enc_features[::-1] + head   # const.ENCODER_FILTERS[::-1]
    nb_features = [enc_features, dec_features]
    network = vxm.networks.VxmDenseSemiSupervisedSeg(inshape=image_output_shape,
                                                     nb_labels=nb_labels,
                                                     nb_unet_features=nb_features,
                                                     int_steps=0,
                                                     int_downsize=1,
                                                     seg_downsize=1)

    resume_epoch = 0
    if resume:
        cp_dir = os.path.join(output_folder, 'checkpoints')
        cp_file_list = [os.path.join(cp_dir, f) for f in os.listdir(cp_dir) if (f.startswith('checkpoint') and f.endswith('.h5'))]
        if len(cp_file_list):
            cp_file_list.sort()
            checkpoint_file = cp_file_list[-1]
            if os.path.exists(checkpoint_file):
                network.load_weights(checkpoint_file, by_name=True)
                print('Loaded checkpoint file: ' + checkpoint_file)
                try:
                    resume_epoch = int(re.match('checkpoint\.(\d+)-*.h5', os.path.split(checkpoint_file)[-1])[1])
                except TypeError:
                    # Checkpoint file has no epoch number in the name
                    resume_epoch = 0
                print('Resuming from epoch: {:d}'.format(resume_epoch))
            else:
                warnings.warn('Checkpoint file NOT found. Training from scratch')

    # Network inputs: mov_img, fix_img, mov_seg
    # Network outputs: pred_img, disp_map, pred_seg
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
    loss = multiLoss([*[network.inputs[1]]*len(loss_simil), *[fix_seg]*len(loss_segm),
                      *[network.outputs[0]]*len(loss_simil), *[network.outputs[2]]*len(loss_simil),
                      grad,
                      network.outputs[1]])

    # inputs = [mov_img, fix_img, mov_segm, fix_segm, zero_grads]
    # outputs = [pred_img, flow, pred_segm, loss]
    full_model = tf.keras.Model(inputs=network.inputs + [fix_seg, grad],
                                outputs=network.outputs + [loss])

    # Train
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'tensorboard'), exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'history'), exist_ok=True)

    callback_best_model = ModelCheckpoint(os.path.join(output_folder, 'checkpoints', 'best_model.h5'),
                            save_best_only=True, monitor='val_loss', verbose=1, mode='min')
    callback_save_model = ModelCheckpoint(os.path.join(output_folder, 'checkpoints', 'checkpoint.{epoch:05d}-{val_loss:.2f}.h5'),
                   save_weights_only=True, monitor='val_loss', verbose=0, mode='min')
    # CSVLogger(train_log_name, ';'),
    # UpdateLossweights([haus_weight, dice_weight], [const.MODEL+'_resampler_seg', const.MODEL+'_resampler_seg'])
    callback_tensorboard = TensorBoard(log_dir=os.path.join(output_folder, 'tensorboard'),
                                       batch_size=C.BATCH_SIZE, write_images=False, histogram_freq=0,
                                       update_freq='epoch',     # or 'batch' or integer
                                       write_graph=True, write_grads=True
                                       )
    callback_early_stop = EarlyStopping(monitor='val_loss', verbose=1, patience=C.EARLY_STOP_PATIENCE, min_delta=0.00001)
    callback_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10)

    # Compile the model
    optimizer = AdamAccumulated(C.ACCUM_GRADIENT_STEP, lr=C.LEARNING_RATE)
    full_model.compile(optimizer=optimizer, loss=None)

    callback_tensorboard.set_model(full_model)
    callback_best_model.set_model(network)  # ONLY SAVE THE NETWORK!!!
    callback_save_model.set_model(network)
    callback_early_stop.set_model(full_model)
    callback_lr.set_model(full_model)

    # TODO: https://towardsdatascience.com/writing-tensorflow-2-custom-loops-438b1ab6eb6c

    summary = SummaryDictionary(full_model, C.BATCH_SIZE)
    names = full_model.metrics_names  # It give both the loss and metric names
    zero_grads = tf.zeros_like(network.references.pos_flow, name='dummy_zero_grads')  # Dummy zeros-tensor
    log_file.write('\n\n[{}]\tINFO:\tStart training\n\n'.format(datetime.now().strftime('%H:%M:%S\t%d/%m/%Y')))
    with sess.as_default():
        callback_tensorboard.on_train_begin()
        callback_early_stop.on_train_begin()
        callback_best_model.on_train_begin()
        callback_save_model.on_train_begin()
        callback_lr.on_train_begin()

        for epoch in range(resume_epoch, C.EPOCHS):
            callback_tensorboard.on_epoch_begin(epoch)
            callback_early_stop.on_epoch_begin(epoch)
            callback_best_model.on_epoch_begin(epoch)
            callback_save_model.on_epoch_begin(epoch)
            callback_lr.on_epoch_begin(epoch)

            print("\nEpoch {}/{}".format(epoch, C.EPOCHS))
            print('TRAINING')

            log_file.write('\n\n[{}]\tINFO:\tTraining epoch {}\n\n'.format(datetime.now().strftime('%H:%M:%S\t%d/%m/%Y'), epoch))
            progress_bar = Progbar(len(train_generator), width=30, verbose=1)
            for step, (in_batch, _) in enumerate(train_generator, 1):
                # callback_tensorboard.on_train_batch_begin(step)
                callback_best_model.on_train_batch_begin(step)
                callback_save_model.on_train_batch_begin(step)
                callback_early_stop.on_train_batch_begin(step)
                callback_lr.on_train_batch_begin(step)

                try:
                    fix_img, mov_img, fix_seg, mov_seg = augmentation_model.predict(in_batch)
                    np.nan_to_num(fix_img, copy=False)
                    np.nan_to_num(mov_img, copy=False)
                except InvalidArgumentError as err:
                    print('TF Error : {}'.format(str(err)))
                    continue
                # inputs = [mov_img, fix_img, mov_segm, fix_segm, zero_grads]
                # outputs = [pred_img, flow, pred_segm, loss]
                ret = full_model.train_on_batch(x=(mov_img, fix_img, mov_seg, fix_seg, zero_grads))

                summary.on_train_batch_end(ret)
                # callback_tensorboard.on_train_batch_end(step, named_logs(full_model, ret))
                callback_best_model.on_train_batch_end(step, named_logs(full_model, ret))
                callback_save_model.on_train_batch_end(step, named_logs(full_model, ret))
                callback_early_stop.on_train_batch_end(step, named_logs(full_model, ret))
                callback_lr.on_train_batch_end(step, named_logs(full_model, ret))
                log_file.write('\t\tStep {:03d}: {}'.format(step, ret))
                # print(ret, '\n')
                progress_bar.update(step, zip(names, ret))
            print('End of epoch{}: '.format(step), ret, '\n')
            val_values = progress_bar._values.copy()
            ret = [val_values[x][0]/val_values[x][1] for x in names]

            print('\nVALIDATION')
            log_file.write('\n\n[{}]\tINFO:\tValidation epoch {}\n\n'.format(datetime.now().strftime('%H:%M:%S\t%d/%m/%Y'), epoch))
            progress_bar = Progbar(len(validation_generator), width=30, verbose=1)
            for step, (in_batch, _) in enumerate(validation_generator, 1):
                # callback_tensorboard.on_test_batch_begin(step)    # This is cursed, don't do it again
                # callback_early_stop.on_test_batch_begin(step)
                try:
                    fix_img, mov_img, fix_seg, mov_seg = augmentation_model.predict(in_batch)
                except InvalidArgumentError as err:
                    print('TF Error : {}'.format(str(err)))
                    continue

                ret = full_model.test_on_batch(x=(mov_img, fix_img, mov_seg, fix_seg, zero_grads))
                # pred_segm = network.register(mov_segm, fix_segm)
                summary.on_validation_batch_end(ret)
                # callback_early_stop.on_test_batch_end(step, named_logs(full_model, ret))
                # callback_tensorboard.on_test_batch_end(step, named_logs(network, ret))    # This is cursed, don't do it again
                progress_bar.update(step, zip(names, ret))
                log_file.write('\t\tStep {:03d}: {}'.format(step, ret))
            val_values = progress_bar._values.copy()
            ret = [val_values[x][0]/val_values[x][1] for x in names]

            train_generator.on_epoch_end()
            validation_generator.on_epoch_end()
            epoch_summary = summary.on_epoch_end()  # summary resets after on_epoch_end() call
            callback_tensorboard.on_epoch_end(epoch, epoch_summary)
            callback_best_model.on_epoch_end(epoch, epoch_summary)
            callback_early_stop.on_epoch_end(epoch, epoch_summary)
            callback_save_model.on_epoch_end(epoch, epoch_summary)
            callback_lr.on_epoch_end(epoch, epoch_summary)

        callback_tensorboard.on_train_end()
        callback_save_model.on_train_end()
        callback_best_model.on_train_end()
        callback_early_stop.on_train_end()
        callback_lr.on_train_end()


if __name__ == '__main__':
    os.environ['CUDA_DEVICE_ORDER'] = C.DEV_ORDER
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Check availability before running using 'nvidia-smi'

    config = tf.compat.v1.ConfigProto()  # device_count={'GPU':0})
    config.gpu_options.allow_growth = True
    config.log_device_placement = False  ## to log device placement (on which device the operation ran)
    tf.keras.backend.set_session(tf.Session(config=config))

    launch_train('/mnt/EncryptedData1/Users/javier/Brain_study/ERASE',
                 'TrainOutput/THESIS/UW_None_mse_ssim_haus',
                 0)
