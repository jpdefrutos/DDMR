import numpy as np
import re

# Constants for augmentation layer
# .../T1/training/zoom_factors.csv contain the scale factors of all the training samples from isotropic to 128x128x128
#   The augmentation values will be scaled using the average+std
ZOOM_FACTORS = np.asarray([0.5032864535069749, 0.5363100665659675, 0.6292598243796296])
MAX_AUG_DISP_ISOT = 30
MAX_AUG_DEF_ISOT = 6
MAX_AUG_DISP = np.max(MAX_AUG_DISP_ISOT * ZOOM_FACTORS)  # Scaled displacements
MAX_AUG_DEF = np.max(MAX_AUG_DEF_ISOT * ZOOM_FACTORS)  # Scaled deformations
MAX_AUG_ANGLE = np.max([np.arctan(np.tan(10*np.pi/180) * ZOOM_FACTORS[1] / ZOOM_FACTORS[0]) * 180 / np.pi,
                        np.arctan(np.tan(10*np.pi/180) * ZOOM_FACTORS[2] / ZOOM_FACTORS[1]) * 180 / np.pi,
                        np.arctan(np.tan(10*np.pi/180) * ZOOM_FACTORS[2] / ZOOM_FACTORS[0]) * 180 / np.pi])  # Scaled angles
GAMMA_AUGMENTATION = False
BRIGHTNESS_AUGMENTATION = False
NUM_CONTROL_PTS_AUG = 10
NUM_AUGMENTATIONS = 5

IN_LAYERS = (0, 3)
OUT_LAYERS = (33, 39)

ENCONDER_LAYERS = (3, 17)
DECODER_LAYERS = (17, 33)

TOP_LAYERS_ENC = (3, 9)
TOP_LAYERS_DEC = (22, 29)
BOTTOM_LAYERS = (9, 22)

LAYER_RANGES = {'INPUT': (IN_LAYERS),
                'OUTPUT': (OUT_LAYERS),
                'ENCODER': (ENCONDER_LAYERS),
                'DECODER': (DECODER_LAYERS),
                'TOP': (TOP_LAYERS_ENC, TOP_LAYERS_DEC),
                'BOTTOM': (BOTTOM_LAYERS)}

# LAYER names:
IN_LAYER_REGEXP = '.*input'
FC_LAYER_REGEXP = '.*final.*'
OUT_LAYER_REGEXP = '(?:flow|transformer)'
ENC_LAYER_REGEXP = '.*enc_(?:conv|pooling)_(\d).*'
DEC_LAYER_REGEXP = '.*dec_(?:conv|upsample)_(\d).*'
LEVEL_NUMBER = lambda x: re.match('.*(?:enc|dec)_(?:conv|upsample|pooling)_(\d).*', x)
IS_TOP_LEVEL = lambda x: int(LEVEL_NUMBER(x)[1]) < 3 if LEVEL_NUMBER(x) is not None else False or bool(re.match(FC_LAYER_REGEXP, x))
IS_BOTTOM_LEVEL = lambda x: int(LEVEL_NUMBER(x)[1]) >= 3 if LEVEL_NUMBER(x) is not None else False

LAYER_SELECTION = {'INPUT': lambda x: bool(re.match(IN_LAYER_REGEXP, x)),
                   'FULLYCONNECTED': lambda x: bool(re.match(FC_LAYER_REGEXP, x)),
                   'ENCODER': lambda x: bool(re.match(ENC_LAYER_REGEXP, x)),
                   'DECODER': lambda x: bool(re.match(DEC_LAYER_REGEXP, x)),
                   'TOP': lambda x: IS_TOP_LEVEL(x),
                   'BOTTOM': lambda x: IS_BOTTOM_LEVEL(x)
                   }

# STUPID IDEA THAT COMPLICATES THINGS. The points was to allow combinations of the layer groups
# OR_GROUPS = ['ENCODER', 'DECODER', 'INPUT', 'OUTPUT', 'FULLYCONNECTED']     # These groups can be OR'ed with the AND_GROUPS and among them. E.g., Top layers of the encoder and decoder: ENCODER or DECODER and TOP
# AND_GROUPS = ['TOP', 'BOTTOM']      # These groups can be AND'ed with the OR_GROUPS and among them
