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

import ddmr.utils.constants as C
from ddmr.data_generator import DataGeneratorManager
from ddmr.losses import NCC
from ddmr.utils.misc import try_mkdir


os.environ['CUDA_DEVICE_ORDER'] = C.DEV_ORDER
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # Check availability before running using 'nvidia-smi'

C.TRAINING_DATASET = '/mnt/EncryptedData1/Users/javier/vessel_registration/sanity_dataset_LITS'
C.BATCH_SIZE = 2
C.LIMIT_NUM_SAMPLES = None
C.EPOCHS = 10000

# Load data
# Build data generator
data_generator = DataGeneratorManager(C.TRAINING_DATASET, C.BATCH_SIZE, True, C.LIMIT_NUM_SAMPLES,
                                      1 - C.TRAINING_PERC, voxelmorph=True)

train_generator = data_generator.get_generator('train')
validation_generator = data_generator.get_generator('validation')


# Build model
in_shape = train_generator.get_input_shape()[1:-1]
enc_features = [16, 32, 32, 32, 32, 32]# const.ENCODER_FILTERS
dec_features = [32, 32, 32, 32, 32, 32, 32, 16, 16]# const.ENCODER_FILTERS[::-1]
nb_features = [enc_features, dec_features]
vxm_model = vxm.networks.VxmDense(inshape=in_shape, nb_unet_features=nb_features, int_steps=7)


# Losses and loss weights
losses = [NCC(in_shape).loss, vxm.losses.Grad('l2').loss]
loss_weights = [1., 0.01]

# Compile the model
vxm_model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4), loss=losses, loss_weights=loss_weights)

# Train
output_folder = os.path.join('TrainingScripts/TrainOutput/baseline_LITS_NCC_'+datetime.now().strftime("%H%M%S-%d%m%Y"))
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
hist = vxm_model.fit(train_generator, epochs=C.EPOCHS, validation_data=validation_generator, verbose=2, callbacks=my_callbacks)
