import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)  # PYTHON > 3.3 does not allow relative referencing

PYCHARM_EXEC = os.getenv('PYCHARM_EXEC') == 'True'

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import voxelmorph as vxm
import neurite as ne
import h5py
from datetime import datetime

if PYCHARM_EXEC:
    import scripts.tf.myScript_constants as const
    from scripts.tf.myScript_data_generator import DataGeneratorManager
    from scripts.tf.myScript_utils import save_nifti, try_mkdir
else:
    import myScript_constants as const
    from myScript_data_generator import DataGeneratorManager
    from myScript_utils import save_nifti, try_mkdir

os.environ['CUDA_DEVICE_ORDER'] = const.DEV_ORDER
os.environ['CUDA_VISIBLE_DEVICES'] = const.GPU_NUM  # Check availability before running using 'nvidia-smi'

const.TRAINING_DATASET = '/mnt/EncryptedData1/Users/javier/vessel_registration/sanity_dataset_LITS'
const.BATCH_SIZE = 8
const.LIMIT_NUM_SAMPLES = None
const.EPOCHS = 1000

if PYCHARM_EXEC:
    path_prefix = os.path.join('scripts', 'tf')
else:
    path_prefix = ''

# Load data
# Build data generator
data_generator = DataGeneratorManager(const.TRAINING_DATASET, const.BATCH_SIZE, True, const.LIMIT_NUM_SAMPLES,
                                      1 - const.TRAINING_PERC, voxelmorph=True)

test_generator = data_generator.get_generator('validation')
test_fix_img, test_mov_img, *_ = test_generator.get_random_sample(1)

# Build model
in_shape = test_generator.get_input_shape()[1:-1]
enc_features = [16, 32, 32, 32]# const.ENCODER_FILTERS
dec_features = [32, 32, 32, 32, 32, 16, 16]# const.ENCODER_FILTERS[::-1]
nb_features = [enc_features, dec_features]
vxm_model = vxm.networks.VxmDense(inshape=in_shape, nb_unet_features=nb_features, int_steps=0)

weight_files = [os.path.join(path_prefix, 'checkpoints', f) for f in os.listdir(os.path.join(path_prefix, 'checkpoints')) if 'weights' in f]
weight_files.sort()
pred_folder = os.path.join(path_prefix, 'predictions')
try_mkdir(pred_folder)

# Prepare the images
fix_img = test_fix_img.squeeze()
mid_slice_fix = [np.take(fix_img, fix_img.shape[d]//2, axis=d) for d in range(3)]
mid_slice_fix[1] = np.rot90(mid_slice_fix[1], 1)
mid_slice_fix[2] = np.rot90(mid_slice_fix[2], -1)

mid_mov_slice = list()
mid_disp_slice = list()
# Due to slicing, it can happen that the last file is not tested. So include it always
slice = 5
for f in weight_files[:-1:slice] + [weight_files[-1]]:
    name = os.path.split(f)[-1].split('.h5')[0]
    vxm_model.load_weights(f)
    pred_img, pred_disp = vxm_model.predict([test_mov_img, test_fix_img])
    pred_img = pred_img.squeeze()

    mov_slices = [np.take(pred_img, pred_img.shape[d]//2, axis=d) for d in range(3)]
    mov_slices[1] = np.rot90(mov_slices[1], 1)
    mov_slices[2] = np.rot90(mov_slices[2], -1)
    mid_mov_slice.append(mov_slices)





# Get sample for testing
test_sample = test_generator.get_single_sample()
