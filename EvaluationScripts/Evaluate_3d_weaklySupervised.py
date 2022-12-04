import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)  # PYTHON > 3.3 does not allow relative referencing

PYCHARM_EXEC = os.getenv('PYCHARM_EXEC') == 'True'

import numpy as np
import tensorflow as tf
import voxelmorph as vxm
import neurite as ne
from datetime import datetime

import DeepDeformationMapRegistration.utils.constants as C
from DeepDeformationMapRegistration.data_generator import DataGeneratorManager
from DeepDeformationMapRegistration.utils.misc import try_mkdir
from DeepDeformationMapRegistration.utils.nifti_utils import save_nifti
from DeepDeformationMapRegistration.networks import WeaklySupervised
from DeepDeformationMapRegistration.losses import HausdorffDistanceErosion
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

data_folder = '../train_3d_multiloss_segm_haus_dice_ncc_grad_203925-29012021'

# Build model
in_shape = train_generator.get_input_shape()[1:-1]
enc_features = [16, 32, 32, 32, 32, 32]# const.ENCODER_FILTERS
dec_features = [32, 32, 32, 32, 32, 32, 32, 16, 16]# const.ENCODER_FILTERS[::-1]
nb_features = [enc_features, dec_features]
vxm_model = WeaklySupervised(inshape=in_shape, all_labels=[1], nb_unet_features=nb_features, int_steps=5)
vxm_model.load_weights(os.path.join(data_folder, 'checkpoints', 'best_model.h5'), by_name=True)

# Get some samples and plot them
sample = validation_generator[0]

samp_id = 1
pred_img, pred_seg, pred_flow = vxm_model.predict([sample[0][0][samp_id, ...][np.newaxis, ...],
                                                   sample[0][1][samp_id, ...][np.newaxis, ...],
                                                   sample[0][2][samp_id, ...][np.newaxis, ...]])

save_nifti(np.squeeze(pred_img), os.path.join(data_folder, 'pred_img.nii.gz'))
save_nifti(np.squeeze(pred_seg), os.path.join(data_folder, 'pred_seg.nii.gz'))
save_nifti(sample[0][0][samp_id, ...], os.path.join(data_folder, 'mov_seg.nii.gz'))
save_nifti(sample[0][1][samp_id, ...], os.path.join(data_folder, 'fix_seg.nii.gz'))
save_nifti(sample[0][2][samp_id, ...], os.path.join(data_folder, 'mov_img.nii.gz'))
save_nifti(sample[0][-2][samp_id, ...], os.path.join(data_folder, 'fix_img.nii.gz'))
