import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)  # PYTHON > 3.3 does not allow relative referencing

PYCHARM_EXEC = os.getenv('PYCHARM_EXEC') == 'True'

import tensorflow as tf
from scipy.ndimage import generate_binary_structure

import DeepDeformationMapRegistration.utils.constants as C
from DeepDeformationMapRegistration.utils.operators import soft_threshold


class HausdorffDistance:
    def __init__(self, ndim=3, nerosion=10):
        self.ndims = ndim
        self.conv = getattr(tf.nn, 'conv%dd' % self.ndims)
        self.nerosions = nerosion

    def _erode(self, in_tensor, kernel):
        out = 1. - tf.squeeze(self.conv(tf.expand_dims(1. - in_tensor, 0), kernel, [1] * (self.ndims + 2), 'SAME'), axis=0)
        return soft_threshold(out, 0.5, name='soft_thresholding')

    def _erosion_distance_single(self, y_true, y_pred):
        diff = tf.math.pow(y_pred - y_true, 2)
        alpha = 2.

        norm = 1 / self.ndims * 2 + 1
        kernel = generate_binary_structure(self.ndims, 1).astype(int) * norm
        kernel = tf.constant(kernel, tf.float32)
        kernel = tf.expand_dims(tf.expand_dims(kernel, -1), -1)

        ret = 0.
        for i in range(self.nerosions):
            for j in range(i + 1):
                er = self._erode(diff, kernel)
            ret += tf.reduce_sum(tf.multiply(er, tf.pow(i + 1., alpha)))

        return tf.multiply(C.IMG_SIZE ** -self.ndims, ret)  # Divide by the image size

    def loss(self, y_true, y_pred):
        batched_dist = tf.map_fn(lambda x: self._erosion_distance_single(x[0], x[1]), (y_true, y_pred),
                                 dtype=tf.float32)

        return batched_dist  # tf.reduce_mean(batched_dist)

