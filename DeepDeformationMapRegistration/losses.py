import tensorflow as tf
from scipy.ndimage import generate_binary_structure

from DeepDeformationMapRegistration.utils.operators import soft_threshold
from DeepDeformationMapRegistration.utils.constants import EPS_tf


class HausdorffDistanceErosion:
    def __init__(self, ndim=3, nerosion=10, value_per_channel=False):
        """
        Approximation of the Hausdorff distance based on erosion operations based on the work done by Karimi D., et al.
            Karimi D., et al., "Reducing the Hausdorff Distance in Medical Image Segmentation with Convolutional Neural
            Networks". IEEE Transactions on Medical Imaging, 39, 2020. DOI 10.1109/TMI.2019.2930068

        :param ndim: Dimensionality of the images
        :param nerosion: Number of erosion steps. Defaults to 10.
        :param value_per_channel: Return an array with the HD distance computed on each channel independently or the sum
        """
        self.ndims = ndim
        self.conv = getattr(tf.nn, 'conv%dd' % self.ndims)
        self.nerosions = nerosion
        self.sum_range = tf.range(0, self.ndims) if value_per_channel else None

    def _erode(self, in_tensor, kernel):
        out = 1. - tf.squeeze(self.conv(tf.expand_dims(1. - in_tensor, 0), kernel, [1] * (self.ndims + 2), 'SAME'), axis=0)
        return soft_threshold(out, 0.5, name='soft_thresholding')

    def _erode_per_channel(self, in_tensor, kernel):
        # In the lambda function we add a fictitious channel and then remove it, so the final shape is [1, H, W, D]
        er_tensor = tf.map_fn(lambda tens: tf.squeeze(self._erode(tf.expand_dims(tens, -1), kernel)),
                         tf.transpose(in_tensor, [3, 0, 1, 2]), tf.float32)  # Iterate along the channel dimension (3)

        return tf.transpose(er_tensor, [1, 2, 3, 0])  # move the channels back to the end

    def _erosion_distance_single(self, y_true, y_pred):
        diff = tf.math.pow(y_pred - y_true, 2)
        alpha = 2

        norm = 1 / (self.ndims * 2 + 1)
        kernel = generate_binary_structure(self.ndims, 1).astype(int) * norm
        kernel = tf.constant(kernel, tf.float32)
        kernel = tf.expand_dims(tf.expand_dims(kernel, -1), -1)   # [H, W, D, C_in, C_out]

        ret = 0.
        for k in range(1, self.nerosions+1):
            er = diff
            # k successive erosions
            for j in range(k):
                er = self._erode_per_channel(er, kernel)
            ret += tf.reduce_sum(tf.multiply(er, tf.cast(tf.pow(k, alpha), tf.float32)), self.sum_range)

        img_vol = tf.cast(tf.reduce_prod(tf.shape(y_true)[:-1]), tf.float32)  # Volume of each channel
        return tf.divide(ret, img_vol)  # Divide by the image size

    def loss(self, y_true, y_pred):
        batched_dist = tf.map_fn(lambda x: self._erosion_distance_single(x[0], x[1]), (y_true, y_pred),
                                 dtype=tf.float32)

        return tf.reduce_mean(batched_dist)


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
        # According to the documentation, the loss returns a scalar
        # Ref: https://www.tensorflow.org/api_docs/python/tf/keras/Model#compile
        return tf.reduce_mean(tf.map_fn(lambda x: 1 - self.ncc(x[0], x[1]), (y_true, y_pred), tf.float32))


class StructuralSimilarity:
    # Based on https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/losses/dssim.py
    def __init__(self, k1=0.01, k2=0.03, patch_size=3, dynamic_range=1., overlap=0.0):
        """
        Structural (Di)Similarity Index Measure:

        :param k1: Internal parameter. Defaults to 0.01
        :param k2: Internal parameter. Defaults to 0.02
        :param patch_size: Size of the extracted patches
        :param dynamic_range: Maximum numerical intensity value (typ. 2^bits_per_pixel - 1). Defaults to 1.
        """
        self.__c1 = (k1 * dynamic_range) ** 2
        self.__c2 = (k2 * dynamic_range) ** 2
        self.__kernel_shape = [1] + [patch_size] * 3 + [1]
        stride = int(patch_size * (1 - overlap))
        self.__stride = [1] + [stride if stride else 1] * 3 + [1]
        self.__max_val = dynamic_range

    def __int_shape(self, x):
        return tf.keras.backend.int_shape(x) if tf.keras.backend.backend() == 'tensorflow' else tf.keras.backend.shape(x)

    def ssim(self, y_true, y_pred):

        patches_true = tf.extract_volume_patches(y_true, self.__kernel_shape, self.__stride, 'VALID',
                                                 'patches_true')
        patches_pred = tf.extract_volume_patches(y_pred, self.__kernel_shape, self.__stride, 'VALID',
                                                 'patches_pred')

        #bs, w, h, d, *c = self.__int_shape(patches_pred)
        #patches_true = tf.reshape(patches_true, [-1, w, h, d, tf.reduce_prod(c)])
        #patches_pred = tf.reshape(patches_pred, [-1, w, h, d, tf.reduce_prod(c)])

        # Mean
        u_true = tf.reduce_mean(patches_true, axis=-1)
        u_pred = tf.reduce_mean(patches_pred, axis=-1)

        # Variance
        v_true = tf.math.reduce_variance(patches_true, axis=-1)
        v_pred = tf.math.reduce_variance(patches_pred, axis=-1)

        # Covariance
        covar = tf.reduce_mean(patches_true * patches_pred, axis=-1) - u_true * u_pred

        # SSIM
        numerator = (2 * u_true * u_pred + self.__c1) * (2 * covar + self.__c2)
        denominator = ((tf.square(u_true) + tf.square(u_pred) + self.__c1) * (v_pred + v_true + self.__c2))
        ssim = numerator / denominator

        return tf.reduce_mean(ssim)

    def dssim(self, y_true, y_pred):
        return tf.reduce_mean((1. - self.ssim(y_true, y_pred)) / 2.0)
