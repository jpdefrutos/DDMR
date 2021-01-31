import tensorflow as tf
from scipy.ndimage import generate_binary_structure

from DeepDeformationMapRegistration.utils.operators import soft_threshold
from DeepDeformationMapRegistration.utils.constants import EPS_tf


class HausdorffDistanceErosion:
    def __init__(self, ndim=3, nerosion=10):
        """
        Approximation of the Hausdorff distance based on erosion operations based on the work done by Karimi D., et al.
            Karimi D., et al., "Reducing the Hausdorff Distance in Medical Image Segmentation with Convolutional Neural
            Networks". IEEE Transactions on Medical Imaging, 39, 2020. DOI 10.1109/TMI.2019.2930068

        :param ndim: Dimensionality of the images
        :param nerosion: Number of erosion steps. Defaults to 10.
        """
        self.ndims = ndim
        self.conv = getattr(tf.nn, 'conv%dd' % self.ndims)
        self.nerosions = nerosion

    def _erode(self, in_tensor, kernel):
        out = 1. - tf.squeeze(self.conv(tf.expand_dims(1. - in_tensor, 0), kernel, [1] * (self.ndims + 2), 'SAME'), axis=0)
        return soft_threshold(out, 0.5, name='soft_thresholding')

    def _erosion_distance_single(self, y_true, y_pred):
        diff = tf.math.pow(y_pred - y_true, 2)
        alpha = 2.

        norm = 1 / (self.ndims * 2 + 1)
        kernel = generate_binary_structure(self.ndims, 1).astype(int) * norm
        kernel = tf.constant(kernel, tf.float32)
        kernel = tf.expand_dims(tf.expand_dims(kernel, -1), -1)

        ret = 0.
        for i in range(self.nerosions):
            for j in range(i + 1):
                er = self._erode(diff, kernel)
            ret += tf.reduce_sum(tf.multiply(er, tf.pow(i + 1., alpha)))

        img_vol = tf.cast(tf.reduce_prod(y_true.shape), tf.float32)
        return tf.divide(ret, img_vol)  # Divide by the image size

    def loss(self, y_true, y_pred):
        batched_dist = tf.map_fn(lambda x: self._erosion_distance_single(x[0], x[1]), (y_true, y_pred),
                                 dtype=tf.float32)

        return batched_dist


class NCC:
    def __init__(self, in_shape, eps=EPS_tf):
        self.__shape_size = tf.cast(tf.reduce_prod(in_shape), tf.float32)
        self.__eps = eps

    def ncc(self, y_true, y_pred):
        f_yt = tf.reshape(y_true, [-1])
        f_yp = tf.reshape(y_pred, [-1])
        mean_yt = tf.reduce_mean(f_yt)
        mean_yp = tf.reduce_mean(f_yp)
        std_yt = tf.math.reduce_std(f_yt)
        std_yp = tf.math.reduce_std(f_yp)

        n_f_yt = f_yt - mean_yt
        n_f_yp = f_yp - mean_yp
        numerator = tf.reduce_sum(n_f_yt * n_f_yp)
        denominator = std_yt * std_yp * self.__shape_size + self.__eps
        return tf.math.divide_no_nan(numerator, denominator)

    def loss(self, y_true, y_pred):
        return tf.map_fn(lambda x: 1 - self.ncc(x[0], x[1]), (y_true, y_pred), tf.float32)
