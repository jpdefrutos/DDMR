"""
Constants
"""
import tensorflow as tf
import os
import datetime
import numpy as np

# RUN CONFIG
REMOTE = False  # os.popen('hostname').read().encode('utf-8') == 'medtech-beast' #os.environ.get('REMOTE') == 'True'

# Remote execution
DEV_ORDER = 'PCI_BUS_ID'
GPU_NUM = '0'

# Dataset generation constants
# See batchGenerator __next__ method: return [in_mov, in_fix], [disp_map, out_img]
MOVING_IMG = 0
FIXED_IMG = 1
MOVING_PARENCHYMA_MASK = 2
FIXED_PARENCHYMA_MASK = 3
MOVING_VESSELS_MASK = 4
FIXED_VESSELS_MASK = 5
MOVING_TUMORS_MASK = 6
FIXED_TUMORS_MASK = 7
MOVING_SEGMENTATIONS = 8  # Compination of vessels and tumors
FIXED_SEGMENTATIONS = 9  # Compination of vessels and tumors
DISP_MAP_GT = 0
PRED_IMG_GT = 1
DISP_VECT_GT = 2
DISP_VECT_LOC_GT = 3

IMG_SIZE = 64  # Assumed a square image
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, IMG_SIZE, 1)  # (IMG_SIZE, IMG_SIZE, 1)
DISP_MAP_SHAPE = (IMG_SIZE, IMG_SIZE, IMG_SIZE, 3)
BATCH_SHAPE = (None, IMG_SIZE, IMG_SIZE, IMG_SIZE, 2)  # Expected batch shape by the network
BATCH_SHAPE_SEGM = (None, IMG_SIZE, IMG_SIZE, IMG_SIZE, 3)  # Expected batch shape by the network
IMG_BATCH_SHAPE = (None, IMG_SIZE, IMG_SIZE, IMG_SIZE, 1)  # Batch shape for single images

RAW_DATA_BASE_DIR = './data'
DEFORMED_DATA_NAME = 'deformed'
GROUND_TRUTH_DATA_NAME = 'groundTruth'
GROUND_TRUTH_COORDS_FILE = 'centerlineCoords_GT.txt'
DEFORMED_COORDS_FILE = 'centerlineCoords_DF.txt'
H5_MOV_IMG = 'input/{}'.format(MOVING_IMG)
H5_FIX_IMG = 'input/{}'.format(FIXED_IMG)
H5_MOV_PARENCHYMA_MASK = 'input/{}'.format(MOVING_PARENCHYMA_MASK)
H5_FIX_PARENCHYMA_MASK = 'input/{}'.format(FIXED_PARENCHYMA_MASK)
H5_MOV_VESSELS_MASK = 'input/{}'.format(MOVING_VESSELS_MASK)
H5_FIX_VESSELS_MASK = 'input/{}'.format(FIXED_VESSELS_MASK)
H5_MOV_TUMORS_MASK = 'input/{}'.format(MOVING_TUMORS_MASK)
H5_FIX_TUMORS_MASK = 'input/{}'.format(FIXED_TUMORS_MASK)
H5_FIX_SEGMENTATIONS = 'input/{}'.format(FIXED_SEGMENTATIONS)
H5_MOV_SEGMENTATIONS = 'input/{}'.format(MOVING_SEGMENTATIONS)

H5_GT_DISP = 'output/{}'.format(DISP_MAP_GT)
H5_GT_IMG = 'output/{}'.format(PRED_IMG_GT)
H5_GT_DISP_VECT = 'output/{}'.format(DISP_VECT_GT)
H5_GT_DISP_VECT_LOC = 'output/{}'.format(DISP_VECT_LOC_GT)
H5_PARAMS_INTENSITY_RANGE = 'parameters/intensity'
TRAINING_PERC = 0.8
VALIDATION_PERC = 1 - TRAINING_PERC
MAX_ANGLE = 45.0  # degrees
MAX_FLIPS = 2  # Axes to flip over
NUM_ROTATIONS = 5
MAX_WORKERS = 10

# Training constants
MODEL = 'unet'
BATCH_NORM = False
TENSORBOARD = False
LIMIT_NUM_SAMPLES = None  # If you don't want to use all the samples in the training set. None to use all
TRAINING_DATASET = 'data/training.hd5'
TEST_DATASET = 'data/test.hd5'
VALIDATION_DATASET = 'data/validation.hd5'
LOSS_FNC = 'mse'
LOSS_SCHEME = 'unidirectional'
NUM_EPOCHS = 10
DATA_FORMAT = 'channels_last'  # or 'channels_fist'
DATA_DIR = './data'
MODEL_CHECKPOINT = './model_checkpoint'
BATCH_SIZE = 8
EPOCHS = 100
SAVE_EPOCH = EPOCHS // 10  # Epoch when to save the model
VERBOSE_EPOCH = EPOCHS // 10
VALIDATION_ERR_LIMIT = 0.2  # Stop training if reached this limit
VALIDATION_ERR_LIMIT_COUNTER = 10  # Number of successive times the validation error was smaller than the threshold
VALIDATION_ERR_LIMIT_COUNTER_BACKUP = 10
THRESHOLD = 0.5  # Threshold to select the centerline in the interpolated images
RESTORE_TRAINING = True  # look for previously saved models to resume training
EARLY_STOP_PATIENCE = 10
LOG_FIELD_NAMES = ['time', 'epoch', 'step',
                   'training_loss_mean', 'training_loss_std',
                   'training_loss1_mean', 'training_loss1_std',
                   'training_loss2_mean', 'training_loss2_std',
                   'training_loss3_mean', 'training_loss3_std',
                   'training_ncc1_mean', 'training_ncc1_std',
                   'training_ncc2_mean', 'training_ncc2_std',
                   'training_ncc3_mean', 'training_ncc3_std',
                   'validation_loss_mean', 'validation_loss_std',
                   'validation_loss1_mean', 'validation_loss1_std',
                   'validation_loss2_mean', 'validation_loss2_std',
                   'validation_loss3_mean', 'validation_loss3_std',
                   'validation_ncc1_mean', 'validation_ncc1_std',
                   'validation_ncc2_mean', 'validation_ncc2_std',
                   'validation_ncc3_mean', 'validation_ncc3_std']
LOG_FIELD_NAMES_SHORT = ['time', 'epoch', 'step',
                         'training_loss_mean', 'training_loss_std',
                         'training_loss1_mean', 'training_loss1_std',
                         'training_loss2_mean', 'training_loss2_std',
                         'training_ncc1_mean', 'training_ncc1_std',
                         'training_ncc2_mean', 'training_ncc2_std',
                         'validation_loss_mean', 'validation_loss_std',
                         'validation_loss1_mean', 'validation_loss1_std',
                         'validation_loss2_mean', 'validation_loss2_std',
                         'validation_ncc1_mean', 'validation_ncc1_std',
                         'validation_ncc2_mean', 'validation_ncc2_std']
LOG_FIELD_NAMES_UNET = ['time', 'epoch', 'step', 'reg_smooth_coeff', 'reg_jacob_coeff',
                        'training_loss_mean', 'training_loss_std',
                        'training_loss_dissim_mean', 'training_loss_dissim_std',
                        'training_reg_smooth_mean', 'training_reg_smooth_std',
                        'training_reg_jacob_mean', 'training_reg_jacob_std',
                        'training_ncc_mean', 'training_ncc_std',
                        'training_dice_mean', 'training_dice_std',
                        'training_owo_mean', 'training_owo_std',
                        'validation_loss_mean', 'validation_loss_std',
                        'validation_loss_dissim_mean', 'validation_loss_dissim_std',
                        'validation_reg_smooth_mean', 'validation_reg_smooth_std',
                        'validation_reg_jacob_mean', 'validation_reg_jacob_std',
                        'validation_ncc_mean', 'validation_ncc_std',
                        'validation_dice_mean', 'validation_dice_std',
                        'validation_owo_mean', 'validation_owo_std']
CUR_DATETIME = datetime.datetime.now().strftime("%H%M_%d%m%Y")
DESTINATION_FOLDER = 'training_log_' + CUR_DATETIME
CSV_DELIMITER = ";"
CSV_QUOTE_CHAR = '"'
REG_SMOOTH = 0.0
REG_MAG = 1.0
REG_TYPE = 'l2'
MAX_DISP_DM = 10.
MAX_DISP_DM_TF = tf.constant((MAX_DISP_DM,), tf.float32, name='MAX_DISP_DM')
MAX_DISP_DM_PERC = 0.25

W_SIM = 0.7
W_REG = 0.3
W_INV = 0.1

# Loss function parameters
REG_SMOOTH1 = 1 / 100000
REG_SMOOTH2 = 1 / 5000
REG_SMOOTH3 = 1 / 5000
LOSS1 = 1.0
LOSS2 = 0.6
LOSS3 = 0.3
REG_JACOBIAN = 0.1

LOSS_COEFFICIENT = 1.0
REG_COEFFICIENT = 1.0

DICE_SMOOTH = 1.

CC_WINDOW = [9,9,9]

# Adam optimizer
LEARNING_RATE = 1e-3
B1 = 0.9
B2 = 0.999
LEARNING_RATE_DECAY = 0.01
LEARNING_RATE_DECAY_STEP = 10000  # Update the learning rate every LEARNING_RATE_DECAY_STEP steps
OPTIMIZER = 'adam'

# Network architecture constants
LAYER_MAXPOOL = 0
LAYER_UPSAMP = 1
LAYER_CONV = 2
AFFINE_TRANSF = False
OUTPUT_LAYER = 3
DROPOUT = True
DROPOUT_RATE = 0.2
MAX_DATA_SIZE = (1000, 1000, 1)
PLATEAU_THR = 0.01  # A slope between +-PLATEAU_THR will be considered a plateau for the LR updating function
ENCODER_FILTERS = [4, 8, 16, 32, 64]

# SSIM
SSIM_FILTER_SIZE = 11  # Size of Gaussian filter
SSIM_FILTER_SIGMA = 1.5  # Width of Gaussian filter
SSIM_K1 = 0.01  # Def. 0.01
SSIM_K2 = 0.03  # Recommended values 0 < K2 < 0.4
MAX_VALUE = 1.0  # Maximum intensity values

# Mathematic constants
EPS = 1e-8
EPS_tf = tf.constant(EPS, dtype=tf.float32)
LOG2 = tf.math.log(tf.constant(2, dtype=tf.float32))

# Debug constants
VERBOSE = False
DEBUG = False
DEBUG_TRAINING = False
DEBUG_INPUT_DATA = False

# Plotting
FONT_SIZE = 10
DPI = 200  # Dots Per Inch

# Coordinates
B = 0  # Batch dimension
H = 1  # Height dimension
W = 2  # Width dimension
D = 3  # Depth
C = -1  # Channel dimension

D_DISP = 2
W_DISP = 1
H_DISP = 0

DIMENSIONALITY = 3

# Interpolation type
BIL_INTERP = 0
TPS_INTERP = 1
CUADRATIC_C = 0.5

# Data augmentation
MAX_DISP = 5  # Test = 15
NUM_ROT = 5
NUM_FLIPS = 2
MAX_ANGLE = 10

# Thin Plate Splines implementation constants
TPS_NUM_CTRL_PTS_PER_AXIS = 4
TPS_NUM_CTRL_PTS = np.power(TPS_NUM_CTRL_PTS_PER_AXIS, DIMENSIONALITY)
TPS_REG = 0.01
DISP_SCALE = 2  # Scaling of the output of the CNN to increase the range of tanh


class CoordinatesGrid:
    def __init__(self):
        self.__grid = 0
        self.__grid_fl = 0
        self.__norm = False
        self.__num_pts = 0
        self.__batches = False
        self.__shape = None
        self.__shape_flat = None

    def set_coords_grid(self, img_shape: tf.TensorShape, num_ppa: int = None, batches: bool = False,
                        img_type: tf.DType = tf.float32, norm: bool = False):
        self.__batches = batches
        not_batches = not batches  # Just to not make a too complex code when indexing the values
        if num_ppa is None:
            num_ppa = img_shape
        if norm:
            x_coords = tf.linspace(-1., 1.,
                                   num_ppa[W - int(not_batches)])  # np.linspace works fine, tf had some issues...
            y_coords = tf.linspace(-1., 1., num_ppa[H - int(not_batches)])  # num_ppa: number of points per axis
            z_coords = tf.linspace(-1., 1., num_ppa[D - int(not_batches)])
        else:
            x_coords = tf.linspace(0., img_shape[W - int(not_batches)] - 1.,
                                   num_ppa[W - int(not_batches)])  # np.linspace works fine, tf had some issues...
            y_coords = tf.linspace(0., img_shape[H - int(not_batches)] - 1.,
                                   num_ppa[H - int(not_batches)])  # num_ppa: number of points per axis
            z_coords = tf.linspace(0., img_shape[D - int(not_batches)] - 1., num_ppa[D - int(not_batches)])

        coords = tf.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
        self.__num_pts = num_ppa[W - int(not_batches)] * num_ppa[H - int(not_batches)] * num_ppa[D - int(not_batches)]

        grid = tf.stack([coords[0], coords[1], coords[2]], axis=-1)
        grid = tf.cast(grid, img_type)

        grid_fl = tf.stack([tf.reshape(coords[0], [-1]),
                            tf.reshape(coords[1], [-1]),
                            tf.reshape(coords[2], [-1])], axis=-1)
        grid_fl = tf.cast(grid_fl, img_type)

        grid_homogeneous = tf.stack([tf.reshape(coords[0], [-1]),
                                     tf.reshape(coords[1], [-1]),
                                     tf.reshape(coords[2], [-1]),
                                     tf.ones_like(tf.reshape(coords[0], [-1]))], axis=-1)

        self.__shape = np.asarray([num_ppa[W - int(not_batches)], num_ppa[H - int(not_batches)], num_ppa[D - int(not_batches)], 3])
        total_num_pts = np.prod(self.__shape[:-1])
        self.__shape_flat = np.asarray([total_num_pts, 3])
        if batches:
            grid = tf.expand_dims(grid, axis=0)
            grid = tf.tile(grid, [img_shape[B], 1, 1, 1, 1])
            grid_fl = tf.expand_dims(grid_fl, axis=0)
            grid_fl = tf.tile(grid_fl, [img_shape[B], 1, 1])
            grid_homogeneous = tf.expand_dims(grid_homogeneous, axis=0)
            grid_homogeneous = tf.tile(grid_homogeneous, [img_shape[B], 1, 1])
            self.__shape = np.concatenate([np.asarray((img_shape[B],)), self.__shape])
            self.__shape_flat = np.concatenate([np.asarray((img_shape[B],)), self.__shape_flat])

        self.__norm = norm
        self.__grid_fl = grid_fl
        self.__grid = grid
        self.__grid_homogeneous = grid_homogeneous

    @property
    def grid(self):
        return self.__grid

    @property
    def size(self):
        return self.__len__()

    def grid_flat(self, transpose=False):
        if transpose:
            if self.__batches:
                ret = tf.transpose(self.__grid_fl, (0, 2, 1))
            else:
                ret = tf.transpose(self.__grid_fl)
        else:
            ret = self.__grid_fl
        return ret

    def grid_homogeneous(self, transpose=False):
        if transpose:
            if self.__batches:
                ret = tf.transpose(self.__grid_homogeneous, (0, 2, 1))
            else:
                ret = tf.transpose(self.__grid_homogeneous)
        else:
            ret = self.__grid_homogeneous
        return ret

    @property
    def is_normalized(self):
        return self.__norm

    def __len__(self):
        return tf.size(self.__grid)

    @property
    def number_pts(self):
        return self.__num_pts

    @property
    def shape_grid_flat(self):
        return self.__shape_flat

    @property
    def shape(self):
        return self.__shape



COORDS_GRID = CoordinatesGrid()


class VisualizationParameters:
    def __init__(self):
        self.__scale = None  # See https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.axes.Axes.quiver.html
        self.__spacing = 5

    def set_spacing(self, img_shape: tf.TensorShape):
        self.__spacing = int(5 * np.log(img_shape[W]))

    @property
    def spacing(self):
        return self.__spacing

    def set_arrow_scale(self, scale: int):
        self.__scale = scale

    @property
    def arrow_scale(self):
        return self.__scale


QUIVER_PARAMS = VisualizationParameters()

# Configuration file
CONF_FILE_NAME = 'configuration.txt'


def summary():
    return '##### CONFIGURATION: REMOTE {}  DEBUG {} DEBUG TRAINING {}' \
           '\n\t\tLEARNING RATE: {}' \
           '\n\t\tBATCH SIZE: {}' \
           '\n\t\tLIMIT NUM SAMPLES: {}' \
           '\n\t\tLOSS_FNC: {}' \
           '\n\t\tTRAINING_DATASET: {} ({:.1f}%/{:.1f}%)' \
           '\n\t\tTEST_DATASET: {}'.format(REMOTE, DEBUG, DEBUG_TRAINING, LEARNING_RATE, BATCH_SIZE, LIMIT_NUM_SAMPLES,
                                           LOSS_FNC, TRAINING_DATASET, TRAINING_PERC * 100, (1 - TRAINING_PERC) * 100,
                                           TEST_DATASET)


# LOG Severity levers
# https://docs.python.org/2/library/logging.html#logging-levels
INF = 20  # Information
WAR = 30  # Warning
ERR = 40  # Error
DEB = 10  # Debug
CRI = 50  # Critical

SEVERITY_STR = {INF: 'INFO',
                WAR: 'WARNING',
                ERR: 'ERROR',
                DEB: 'DEBUG',
                CRI: 'CRITICAL'}

HL_LOG_FIELD_NAMES = ['Time', 'Epoch', 'Step',
                      'train_loss', 'train_loss_std',
                      'train_loss1', 'train_loss1_std',
                      'train_loss2', 'train_loss2_std',
                      'train_loss3', 'train_loss3_std',
                      'train_NCC', 'train_NCC_std',
                      'val_loss', 'val_loss_std',
                      'val_loss1', 'val_loss1_std',
                      'val_loss2', 'val_loss2_std',
                      'val_loss3', 'val_loss3_std',
                      'val_NCC', 'val_NCC_std']

# Sobel filters
SOBEL_W_2D = tf.constant([[-1., 0., 1.],
                          [-2., 0., 2.],
                          [-1., 0., 1.]], dtype=tf.float32, name='sobel_w_2d')
SOBEL_W_3D = tf.tile(tf.expand_dims(SOBEL_W_2D, axis=-1), [1, 1, 3])
SOBEL_H_3D = tf.transpose(SOBEL_W_3D, [1, 0, 2])
SOBEL_D_3D = tf.transpose(SOBEL_W_3D, [2, 1, 0])

aux = tf.expand_dims(tf.expand_dims(SOBEL_W_3D, axis=-1), axis=-1)
SOBEL_FILTER_W_3D_IMAGE = aux
SOBEL_FILTER_W_3D = tf.tile(aux, [1, 1, 1, 3, 3])
# tf.nn.conv3d expects the filter in [D, H, W, C_in, C_out] order
SOBEL_FILTER_W_3D = tf.transpose(SOBEL_FILTER_W_3D, [2, 0, 1, 3, 4], name='sobel_filter_i_3d')

aux = tf.expand_dims(tf.expand_dims(SOBEL_H_3D, axis=-1), axis=-1)
SOBEL_FILTER_H_3D_IMAGE = aux
SOBEL_FILTER_H_3D = tf.tile(aux, [1, 1, 1, 3, 3])
SOBEL_FILTER_H_3D = tf.transpose(SOBEL_FILTER_H_3D, [2, 0, 1, 3, 4], name='sobel_filter_j_3d')

aux = tf.expand_dims(tf.expand_dims(SOBEL_D_3D, axis=-1), axis=-1)
SOBEL_FILTER_D_3D_IMAGE = aux
SOBEL_FILTER_D_3D = tf.tile(aux, [1, 1, 1, 3, 3])
SOBEL_FILTER_D_3D = tf.transpose(SOBEL_FILTER_D_3D, [2, 1, 0, 3, 4], name='sobel_filter_k_3d')

# Filters for spatial integration of the displacement map
INTEG_WIND_SIZE = IMG_SIZE
INTEG_STEPS = 4  # VoxelMorph default value for the integration of the stationary velocity field. >4 memory alloc issue
INTEG_FILTER_D = tf.ones([INTEG_WIND_SIZE, 1, 1, 1, 1], name='integrate_h_filter')
INTEG_FILTER_H = tf.ones([1, INTEG_WIND_SIZE, 1, 1, 1], name='integrate_w_filter')
INTEG_FILTER_W = tf.ones([1, 1, INTEG_WIND_SIZE, 1, 1], name='integrate_d_filter')

# Laplacian filter
LAPLACIAN_27_P = tf.constant(np.asarray([np.ones((3, 3)),
                                         [[1, 1, 1],
                                          [1, -26, 1],
                                          [1, 1, 1]],
                                         np.ones((3, 3))]), tf.float32)
LAPLACIAN_27_P = tf.expand_dims(tf.expand_dims(LAPLACIAN_27_P, axis=-1), axis=-1)
LAPLACIAN_27_P = tf.tile(LAPLACIAN_27_P, [1, 1, 1, 3, 3], name='laplacian_27_p')


LAPLACIAN_7_P = tf.constant(np.asarray([[[0, 0, 0],
                                         [0, 1, 0],
                                         [0, 0, 0]],
                                        [[0, 1, 0],
                                         [1, -6, 1],
                                         [0, 1, 0]],
                                        [[0, 0, 0],
                                         [0, 1, 0],
                                         [0, 0, 0]]]), tf.float32)
LAPLACIAN_7_P = tf.expand_dims(tf.expand_dims(LAPLACIAN_7_P, axis=-1), axis=-1)
LAPLACIAN_7_P = tf.tile(LAPLACIAN_7_P, [1, 1, 1, 3, 3], name='laplacian_7_p')

# Constants for bias loss
ZERO_WARP = tf.zeros((1,) + DISP_MAP_SHAPE, name='zero_warp')
BIAS_WARP_WEIGHT = 1e-02
BIAS_AFFINE_WEIGHT = 1e-02

# Overlapping score
OS_SCALE = 10
EPS_1 = 1.0
EPS_1_tf = tf.constant(EPS_1)

# LDDMM
GAUSSIAN_KERNEL_SHAPE = (8, 8, 8)

# Constants for MultiLoss layer
PRIOR_W = [1., 1 / 60, 1.]
MANUAL_W = [1.] * len(PRIOR_W)

REG_PRIOR_W = [1e-3]
REG_MANUAL_W = [1.] * len(REG_PRIOR_W)

