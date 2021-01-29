import os, sys

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)  # PYTHON > 3.3 does not allow relative referencing

PYCHARM_EXEC = os.getenv('PYCHARM_EXEC') == 'True'

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
import voxelmorph as vxm
from datetime import datetime

import DeepDeformationMapRegistration.utils.constants as C
from DeepDeformationMapRegistration.data_generator import DataGeneratorManager2D
from DeepDeformationMapRegistration.utils.misc import try_mkdir
from DeepDeformationMapRegistration.losses import HausdorffDistance


os.environ['CUDA_DEVICE_ORDER'] = C.DEV_ORDER
os.environ['CUDA_VISIBLE_DEVICES'] = C.GPU_NUM  # Check availability before running using 'nvidia-smi'

C.TRAINING_DATASET = '/mnt/EncryptedData1/Users/javier/vessel_registration/ov_dataset/training'
C.BATCH_SIZE = 256
C.LIMIT_NUM_SAMPLES = None
C.EPOCHS = 10000

if PYCHARM_EXEC:
    path_prefix = os.path.join('scripts', 'tf')
else:
    path_prefix = ''

# Load data
# Build data generator
sample_list = [os.path.join(C.TRAINING_DATASET, f) for f in os.listdir(C.TRAINING_DATASET) if
               f.startswith('sample')]
sample_list.sort()

data_generator = DataGeneratorManager2D(sample_list[:C.LIMIT_NUM_SAMPLES],
                                        C.BATCH_SIZE, C.TRAINING_PERC,
                                        (64, 64, 1),
                                        fix_img_tag='dilated/input/fix',
                                        mov_img_tag='dilated/input/mov'
                                        )

# Build model
in_shape = data_generator.train_generator.input_shape[:-1]
enc_features = [32, 32, 32, 32, 32, 32]  # const.ENCODER_FILTERS
dec_features = [32, 32, 32, 32, 32, 32, 32, 16]  # const.ENCODER_FILTERS[::-1]
nb_features = [enc_features, dec_features]
vxm_model = vxm.networks.VxmDense(inshape=in_shape, nb_unet_features=nb_features, int_steps=0)

# Losses and loss weights
def comb_loss(y_true, y_pred):
    return 1e-3 * HausdorffDistance(ndim=2, nerosion=2).loss(y_true, y_pred) + vxm.losses.Dice().loss(y_true, y_pred)


losses = [comb_loss, vxm.losses.Grad('l2').loss]
loss_weights = [1, 0.01]

# Compile the model
vxm_model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4), loss=losses, loss_weights=loss_weights)

# Train
output_folder = os.path.join('train_2d_dice_hausdorff_grad_'+datetime.now().strftime("%H%M%S-%d%m%Y"))
try_mkdir(output_folder)
try_mkdir(os.path.join(output_folder, 'checkpoints'))
try_mkdir(os.path.join(output_folder, 'tensorboard'))
my_callbacks = [
    # EarlyStopping(patience=const.EARLY_STOP_PATIENCE, monitor='dice', mode='max', verbose=1),
    ModelCheckpoint(os.path.join(output_folder, 'checkpoints', 'best_model.h5'),
                    save_best_only=True, monitor='val_loss', verbose=0, mode='min'),
    ModelCheckpoint(os.path.join(output_folder, 'checkpoints', 'weights.{epoch:05d}-{val_loss:.2f}.h5'),
                    save_best_only=True, save_weights_only=True, monitor='val_loss', verbose=0, mode='min'),
    # CSVLogger(train_log_name, ';'),
    # UpdateLossweights([haus_weight, dice_weight], [const.MODEL+'_resampler_seg', const.MODEL+'_resampler_seg'])
    TensorBoard(log_dir=os.path.join(output_folder, 'tensorboard'),
                batch_size=C.BATCH_SIZE, write_images=True, histogram_freq=10, update_freq='epoch',
                write_grads=True),
    EarlyStopping(monitor='val_loss', verbose=1, patience=50, min_delta=0.0001)
]
hist = vxm_model.fit_generator(data_generator.train_generator,
                               epochs=C.EPOCHS,
                               validation_data=data_generator.validation_generator,
                               verbose=2,
                               callbacks=my_callbacks)
