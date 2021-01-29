import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)  # PYTHON > 3.3 does not allow relative referencing

PYCHARM_EXEC = os.getenv('PYCHARM_EXEC') == 'True'

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
import voxelmorph as vxm
import neurite as ne
import h5py
from datetime import datetime

import DeepDeformationMapRegistration.utils.constants as C
from DeepDeformationMapRegistration.data_generator import DataGeneratorManager
from DeepDeformationMapRegistration.utils.misc import try_mkdir
from DeepDeformationMapRegistration.networks import WeaklySupervised
from DeepDeformationMapRegistration.losses import HausdorffDistance
from DeepDeformationMapRegistration.layers import UncertaintyWeighting


os.environ['CUDA_DEVICE_ORDER'] = C.DEV_ORDER
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Check availability before running using 'nvidia-smi'

C.TRAINING_DATASET = '/mnt/EncryptedData1/Users/javier/vessel_registration/sanity_dataset_vessels'
C.BATCH_SIZE = 2
C.LIMIT_NUM_SAMPLES = None
C.EPOCHS = 10000

# Load data
# Build data generator

data_generator = DataGeneratorManager(C.TRAINING_DATASET, C.BATCH_SIZE, True, C.LIMIT_NUM_SAMPLES,
                                      1 - C.TRAINING_PERC, voxelmorph=True, segmentations=True)

train_generator = data_generator.get_generator('train')
validation_generator = data_generator.get_generator('validation')


# Build model
in_shape = train_generator.get_input_shape()[1:-1]
enc_features = [16, 32, 32, 32, 32, 32]# const.ENCODER_FILTERS
dec_features = [32, 32, 32, 32, 32, 32, 32, 16, 16]# const.ENCODER_FILTERS[::-1]
nb_features = [enc_features, dec_features]
vxm_model = WeaklySupervised(inshape=in_shape, all_labels=[1], nb_unet_features=nb_features, int_steps=5)

# Losses and loss weights

grad = tf.keras.Input(shape=(*in_shape, 3), name='multiLoss_grad_input', dtype=tf.float32)
fix_img = tf.keras.Input(shape=(*in_shape, 1), name='multiLoss_fix_img_input', dtype=tf.float32)
def dice_loss(y_true, y_pred):
    # Dice().loss returns -Dice score
    return 1 + vxm.losses.Dice().loss(y_true, y_pred)

multiLoss = UncertaintyWeighting(num_loss_fns=2,
                                 num_reg_fns=1,
                                 loss_fns=[HausdorffDistance(3, 5).loss, dice_loss],
                                 reg_fns=[vxm.losses.Grad('l2').loss],
                                 prior_loss_w=[1., 1., 1.],
                                 prior_reg_w=[0.01],
                                 name='MultiLossLayer')
loss = multiLoss([vxm_model.inputs[1], vxm_model.inputs[1],
                  vxm_model.references.pred_segm, vxm_model.references.pred_segm,
                  grad,
                  vxm_model.references.pos_flow])

full_model = tf.keras.Model(inputs=vxm_model.inputs + [fix_img, grad], outputs=vxm_model.outputs + [loss])

# Compile the model
full_model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4), loss=None)

# Train
output_folder = os.path.join('train_3d_multiloss_segm_haus_dice_ncc_grad_'+datetime.now().strftime("%H%M%S-%d%m%Y"))
try_mkdir(output_folder)
try_mkdir(os.path.join(output_folder, 'checkpoints'))
try_mkdir(os.path.join(output_folder, 'tensorboard'))
my_callbacks = [
        #EarlyStopping(patience=const.EARLY_STOP_PATIENCE, monitor='dice', mode='max', verbose=1),
        ModelCheckpoint(os.path.join(output_folder, 'checkpoints', 'best_model.h5'),
                        save_best_only=True, monitor='val_loss', verbose=0, mode='min'),
        ModelCheckpoint(os.path.join(output_folder, 'checkpoints', 'weights.{epoch:05d}-{val_loss:.2f}.h5'),
                        save_best_only=True, save_weights_only=True, monitor='val_loss', verbose=0, mode='min'),
        # CSVLogger(train_log_name, ';'),
        # UpdateLossweights([haus_weight, dice_weight], [const.MODEL+'_resampler_seg', const.MODEL+'_resampler_seg'])
        TensorBoard(log_dir=os.path.join(output_folder, 'tensorboard'),
                    batch_size=C.BATCH_SIZE, write_images=False, histogram_freq=10, update_freq='epoch',
                    write_grads=True),
        EarlyStopping(monitor='val_loss', verbose=1, patience=50, min_delta=0.0001)
    ]
hist = full_model.fit(train_generator, epochs=C.EPOCHS, validation_data=validation_generator, verbose=2, callbacks=my_callbacks)
