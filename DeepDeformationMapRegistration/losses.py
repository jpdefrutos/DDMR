import tensorflow as tf
import tensorflow.keras.backend as K
from scipy.ndimage import generate_binary_structure
from sklearn.utils.extmath import cartesian

from DeepDeformationMapRegistration.utils.operators import soft_threshold, min_max_norm, hard_threshold
from DeepDeformationMapRegistration.utils.constants import EPS_tf
from DeepDeformationMapRegistration.utils.misc import function_decorator

import numpy as np
import warnings

class HausdorffDistanceErosion:
    def __init__(self, ndim=3, nerosion=10, im_shape: [list, tuple] = (64, 64, 64, 1), alpha=2):
        """
        Approximation of the Hausdorff distance based on erosion operations based on the work done by Karimi D., et al.
            Karimi D., et al., "Reducing the Hausdorff Distance in Medical Image Segmentation with Convolutional Neural
            Networks". IEEE Transactions on Medical Imaging, 39, 2020. DOI 10.1109/TMI.2019.2930068

        :param ndim: Dimensionality of the images
        :param nerosion: Number of erosion steps. Defaults to 10.
        :param alpha: Parameter to penalize large segmentations. Defaults to 2
        """
        assert len(im_shape) == ndim + 1, "im_shape does not match with ndim. Missing channel dimension?"
        self.ndims = ndim
        axes = np.arange(0, self.ndims).tolist()
        self.before_erosion_transp = [axes[-1], *axes[:-1]]  # [H, W, ..., C] -> [C, H, W, ...]
        self.after_erosion_transp = [*axes[1:], axes[0]]  # [C, H, W, ...] -> [H, W, ..., C]
        self.nerosions = nerosion
        self.sum_range = tf.range(0, self.ndims)

        self.im_shape = im_shape
        self.im_vol = np.prod(im_shape[:-1])
        kernel = generate_binary_structure(self.ndims, 1).astype(int)
        kernel = kernel / np.sum(kernel)
        kernel = kernel[..., np.newaxis, np.newaxis]
        self.kernel = tf.convert_to_tensor(kernel, tf.float32)
        self.k_alpha = [np.power(k, alpha).astype(float) for k in range(1, nerosion + 1)]
        self.conv = getattr(tf.nn, 'conv%dd' % self.ndims)

    def _erode(self, in_tensor):
        indiv_channels = tf.split(in_tensor, self.im_shape[-1], -1)
        res = list()
        with tf.compat.v1.variable_scope('erode', reuse=tf.AUTO_REUSE):
            for ch in indiv_channels:
                res.append(self.conv(tf.expand_dims(ch, 0), self.kernel, [1] * (self.ndims + 2), 'SAME'))
        # out = -tf.nn.max_pool3d(-tf.expand_dims(in_tensor, 0), [3]*self.ndims, [1]*self.ndims, 'SAME', name='HDE_erosion')
        out = tf.concat(res, -1)
        out = tf.squeeze(out, axis=0)
        out = hard_threshold(out, 0.5, name='thresholding')  # soft_threshold(out, 0.5, name='thresholding')
        return out

    def _erosion_distance_single(self, y_true, y_pred):
        diff = tf.math.pow(y_pred - y_true, 2, name='HDE_diff')
        alpha = 2

        ret = 0.
        for k in range(1, self.nerosions+1):
            er = diff
            # k successive erosions
            for j in range(k):
                er = self._erode(er)       # er contains the eroded version along the channels
            ret += tf.reduce_sum(tf.multiply(er, self.k_alpha[k - 1]), self.sum_range, name='HDE_ret')

        return tf.divide(ret, self.im_vol)  # Divide by the image size

    @function_decorator('Hausdorff_erosion__loss')
    def loss(self, y_true, y_pred, name='HDE_loss'):
        batched_dist = tf.map_fn(lambda x: self._erosion_distance_single(x[0], x[1]), (y_true, y_pred),
                                 dtype=tf.float32, name=name+'_map_fn')

        return tf.reduce_mean(batched_dist)

    @function_decorator('Hausdorff_erosion__metric')
    def metric(self, y_true, y_pred):
        return self.loss(y_true, y_pred, name='HDE_metric')

    def debug(self, y_true, y_pred):
        return tf.map_fn(lambda x: self._erosion_distance_single(x[0], x[1]), (y_true, y_pred),
                         dtype=tf.float32, name='HDE_loss_map_fn')


# class HausdorffDiatanceConvolution:
#     def __init__(self, ndim=3, im_shape: tuple = (64, 64, 64, 1), max_kernel_size=9, step_kernel_size=3, alpha=2):
#         """
#         Approximation of the Hausdorff distance based on erosion operations based on the work done by Karimi D., et al.
#             Karimi D., et al., "Reducing the Hausdorff Distance in Medical Image Segmentation with Convolutional Neural
#             Networks". IEEE Transactions on Medical Imaging, 39, 2020. DOI 10.1109/TMI.2019.2930068
#
#         :param ndim: Dimensionality of the images
#         :param nerosion: Number of erosion steps. Defaults to 10.
#         :param alpha: Parameter to penalize large segmentations. Defaults to 2
#         """
#         assert len(im_shape) == ndim + 1, "im_shape does not match with ndim. Missing channel dimension?"
#         self.ndims = ndim
#         axes = np.arange(0, self.ndims).tolist()
#         self.before_erosion_transp = [axes[-1], *axes[:-1]]  # [H, W, ..., C] -> [C, H, W, ...]
#         self.after_erosion_transp = [*axes[1:], axes[0]]  # [C, H, W, ...] -> [H, W, ..., C]
#         self.conv = getattr(tf.nn, 'conv%dd' % self.ndims)
#         self.sum_range = tf.range(0, self.ndims)
#
#         self.im_shape = im_shape
#         self.im_vol = np.prod(im_shape[:-1])
#         kernel = generate_binary_structure(self.ndims, 1).astype(int)
#         self.kernel = tf.constant(kernel / np.sum(kernel), tf.float32)
#         self.kernel = tf.expand_dims(tf.expand_dims(self.kernel, -1), -1)   # [H, W, D, C_in, C_out]
#         self.kernel = tf.tile(self.kernel, [*[1]*self.ndims, self.im_shape[-1], self.im_shape[-1]])
#         self.alpha = int(alpha)
#         self.radii = np.arange(1, max_kernel_size, step=step_kernel_size)
#         self.radii_alpha = [np.pow(r, alpha).astype(float) for r in self.radii]
#
#     def soft_diff(self, p, q):
#         return tf.multiply(tf.pow(p - q, 2.), q)
#
#     def body(self, y_true, y_pred):
#

"""
class HausdorffDistanceErosion_2:
    def __init__(self, im_shape, num_erosions, num_dimensions=3, alpha=2., loop_max_iterations=20):
        self.alpha = alpha
        self.ndims = num_dimensions
        self.conv = getattr(tf.nn, 'conv%dd' % self.ndims)

        self.iterator = tf.constant(num_erosions, name='num_erosions')
        self.norm = 1 / np.prod(im_shape)
        self.erosion_kernel = generate_binary_structure(self.ndims, 1).astype(float)
        self.erosion_kernel /= np.sum(self.erosion_kernel)
        self.erosion_kernel = tf.constant( self.erosion_kernel, tf.float32)

        self.loop_max_iterations = loop_max_iterations

    def erosion_sum(self, p, q, k):
        er_tensor = p - q
        er_tensor = tf.pow(er_tensor, 2.)

        def erode(in_tensor):
            # Erosion of in_tensor = Dilation of (1 - in_tensor)
            return self.conv(tf.expand_dims(1. - in_tensor, 0), self.erosion_kernel, [1] * (self.ndims + 2), 'SAME')

        def while_loop_body(i, in_tensor):
            in_tensor = erode(in_tensor)
            i -= 1
            return i, in_tensor

        def while_loop_condition(i, in_tensor):
            return tf.less_equal(i, 1), in_tensor

        er_iterator = tf.constant(k)
        _, er_tensor = tf.while_loop(while_loop_condition, while_loop_body, loop_vars=[er_iterator, er_tensor],
                                     maximum_iterations=self.loop_max_iterations)

        er_tensor *= tf.pow(k, self.alpha)
        return tf.reduce_sum(er_tensor)

    def loss(self, y_true, y_pred):
        hd_distance = tf.constant(0, name='hausdroff_distance')

        def while_loop_body(i, p, q, ret):
            i -= 1
            return i, p, q, ret + self.erosion_sum(p, q, i)

        _, _, _, hd_distance = tf.while_loop(lambda i, p, q, ret: tf.less_equal(i, 1),
                                             while_loop_body,
                                             loop_vars=[self.iterator, y_pred, y_true, hd_distance])
        hd_distance /= self.norm
        return hd_distance

"""


class WeightedHausdorffDistance:
    def __init__(self, input_shape, alpha=-1, threshold=0.5):
        """
        WARNING: Requires a insane amount of memory
        :param input_shape: [H, W, D, C] or [H, W, C]
        :param alpha: Parameter of the generalized mean. Ideally -inf, but then the function becomes less smooth.
        :param threshold: Threshold of segmentations, used in tf.where function
        """
        warnings.warn("This function requires an insane amount of memory")
        self.input_shape = input_shape
        self.dim = len(input_shape[:-1])
        self.ohe_segm = bool(input_shape[-1] > 1)   # One-Hot Encoded segmentations on the channel axis
        aux = np.arange(len(self.input_shape)).tolist()
        self.ohe_transpose = [aux[-1], *aux[:-1]]
        self.alpha = alpha
        self.threshold = threshold
        list_coords = [np.arange(c) for c in self.input_shape[:-1]]
        self.img_loc = tf.convert_to_tensor(cartesian(list_coords), dtype=tf.float32)
        self.max_dist = np.sqrt(np.sum(np.square(self.input_shape[:-1])))      # Largest diagonal

    def pairwise_distance(self, A, B):
        sq_norm_a = tf.reduce_sum(tf.square(A), 1)
        sq_norm_b = tf.reduce_sum(tf.square(B), 1)

        sq_norm_a = tf.reshape(sq_norm_a, [-1, 1])
        sq_norm_b = tf.reshape(sq_norm_b, [1, -1])

        return tf.sqrt(tf.maximum(sq_norm_a - 2 * tf.matmul(A, B, transpose_a=False, transpose_b=True) + sq_norm_b, 0.))

    def hausdorff(self, y_true, y_pred):
        if self.ohe_segm:
            y_true = tf.transpose(y_true, self.ohe_transpose)
            y_pred = tf.transpose(y_pred, self.ohe_transpose)
            hausdorff_per_ch = tf.map_fn(lambda x: self.hausdorff_per_channel(x[0], x[1]), (y_true, y_pred), tf.float32)
            return tf.reduce_mean(hausdorff_per_ch)
        else:
            return self.hausdorff_per_channel(y_true, y_pred)

    def hausdorff_per_channel(self, y_true, y_pred):
        Y = tf.cast(tf.where(y_true > self.threshold), dtype=tf.float32)
        p = K.flatten(y_pred)   # Flatten the predicted segmentation (activation map 'p' in d_WH)

        size_Y = tf.shape(Y)[0]
        S = tf.reduce_sum(p)

        p = tf.squeeze(K.repeat(tf.expand_dims(p, -1), size_Y))
        dist_mat = self.pairwise_distance(self.img_loc, Y)

        term_1 = tf.reduce_sum(p * tf.minimum(dist_mat, 1)) / (S + EPS_tf)

        term_2 = tf.minimum((dist_mat + EPS_tf) / (tf.pow(p, self.alpha) + (EPS_tf / self.max_dist)), 0.)
        term_2 = tf.clip_by_value(term_2, 0., self.max_dist)
        term_2 = tf.reduce_mean(term_2, axis=0)

        return term_1 + term_2

    @function_decorator('Weighted_Hausdorff__loss')
    def loss(self, y_true, y_pred):
        batch_hdist = tf.map_fn(lambda x: self.hausdorff(x[0], x[1]), (y_true, y_pred), dtype=tf.float32)

        return tf.reduce_mean(batch_hdist)

    @function_decorator('Weighted_Hausdorff__metric')
    def metric(self, y_true, y_pred):
        return self.loss(y_true, y_pred)


class NCC:
    def __init__(self, in_shape, eps=EPS_tf):
        self.__shape_size = tf.cast(tf.reduce_prod(in_shape), tf.float32)
        self.__eps = eps

    def ncc(self, y_true, y_pred):
        f_yt = tf.reshape(y_true, [-1])
        f_yp = tf.reshape(y_pred, [-1])
        mean_yt = tf.reduce_mean(f_yt)
        mean_yp = tf.reduce_mean(f_yp)

        n_f_yt = f_yt - mean_yt
        n_f_yp = f_yp - mean_yp
        norm_yt = tf.norm(f_yt, ord='euclidean')
        norm_yp = tf.norm(f_yp, ord='euclidean')
        numerator = tf.reduce_sum(tf.multiply(n_f_yt, n_f_yp))
        denominator = norm_yt * norm_yp + self.__eps
        return tf.math.divide_no_nan(numerator, denominator)

    @function_decorator('NCC__loss')
    def loss(self, y_true, y_pred):
        # According to the documentation, the loss returns a scalar
        # Ref: https://www.tensorflow.org/api_docs/python/tf/keras/Model#compile
        return tf.reduce_mean(tf.map_fn(lambda x: 1 - self.ncc(x[0], x[1]), (y_true, y_pred), tf.float32))

    @function_decorator('NCC__metric')
    def metric(self, y_true, y_pred):
        return tf.reduce_mean(tf.map_fn(lambda x: self.ncc(x[0], x[1]), (y_true, y_pred), tf.float32))


def ncc(y_true, y_pred):
    y_true = K.flatten(K.cast(y_true, 'float32'))
    y_pred = K.flatten(K.cast(y_pred, 'float32'))

    mean_true = K.mean(y_true)
    mean_pred = K.mean(y_pred)

    std_true = K.std(y_true)
    std_pred = K.std(y_pred)

    num = K.mean((y_true - mean_true) * (y_pred - mean_pred))
    den = std_true * std_pred + EPS_tf
    batch_ncc = num / den

    return K.mean(batch_ncc)


class StructuralSimilarity:
    # Based on https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/losses/dssim.py
    def __init__(self, k1=0.01, k2=0.03,
                 patch_size=32, dynamic_range=1., overlap=0.0, dim=3,
                 alpha=1., beta=1., gamma=1.,
                 **kwargs):
        """
        Structural (Di)Similarity Index Measure:

        :param k1: Internal parameter. Defaults to 0.01
        :param k2: Internal parameter. Defaults to 0.02
        :param patch_size: Size of the extracted patches. Defaults to 32. Recommendation: half the image size.
        :param dynamic_range: Maximum numerical intensity value (typ. 2^bits_per_pixel - 1). Defaults to 1.
        :param overlap: Patch overlap ratio. Must be in the range [0., 1.). Defaults to 0.
        :param dim: Data dimensionality. Must be {1, 2, 3}. Defaults to 3.
        :param alpha, beta, gamma: Exponential parameters to balance the contribution of the luminance, contrast and
                                    structure measures. Default to 1.
        """
        assert (dim > 0) and (dim < 4), 'Invalid dimension. It must be 1, 2, or 3'
        assert overlap < 1., 'Invalid overlap. It must be in the range [0., 1.)'
        self.c1 = (k1 * dynamic_range) ** 2
        self.c2 = (k2 * dynamic_range) ** 2
        self.c3 = self.c2 / 2
        self.alpha = tf.cast(alpha, tf.float32)
        self.beta = tf.cast(beta, tf.float32)
        self.gamma = tf.cast(gamma, tf.float32)

        self.kernel_shape = [1] + [patch_size] * dim + [1]
        stride = int(patch_size * (1 - overlap))
        self.stride = [1] + [stride if stride else 1] * dim + [1]
        self.dim = dim
        self.patch_extractor = None
        self.reduce_axis = list()
        if dim == 2:
            self.patch_extractor = tf.extract_image_patches
            self.reduce_axis = [1, 2]
        elif dim == 3:
            self.patch_extractor = tf.extract_volume_patches
            self.reduce_axis = [1, 2, 3]
        else:
            raise ValueError('Invalid dimension value. Expected 2 or 3')

        if patch_size == -1:
            # Don't extract patches
            self.dim = 1

        self.L = None   # Luminance
        self.C = None   # Contrast
        self.S = None   # Structure

    def __int_shape(self, x):
        return tf.keras.backend.int_shape(x) if tf.keras.backend.backend() == 'tensorflow' else tf.keras.backend.shape(x)

    def ssim(self, y_true, y_pred):
        if self.dim > 1:
            # Don't use for training. The gradient doesn't backpropagate through the patch extractors
            # patches: [B, out_rows, out_cols, ..., krows*kcols*...*channels] -> out_rows * out_cols * ... = nb patches
            patches_true = self.patch_extractor(y_true, ksizes=self.kernel_shape, strides=self.stride, padding='VALID', name='patches_true')
            patches_pred = self.patch_extractor(y_pred, ksizes=self.kernel_shape, strides=self.stride, padding='VALID', name='patches_pred')
        else:
            patches_true = y_true
            patches_pred = y_pred

        #bs, w, h, d, *c = self.__int_shape(patches_pred)
        #patches_true = tf.reshape(patches_true, [-1, w, h, d, tf.reduce_prod(c)])
        #patches_pred = tf.reshape(patches_pred, [-1, w, h, d, tf.reduce_prod(c)])

        # Mean
        u_true = tf.reduce_mean(patches_true, axis=-1)
        u_pred = tf.reduce_mean(patches_pred, axis=-1)

        # Variance
        v_true = tf.math.reduce_variance(patches_true, axis=-1)
        v_pred = tf.math.reduce_variance(patches_pred, axis=-1)

        # Standard dev.
        s_true = tf.sqrt(v_true)
        s_pred = tf.sqrt(v_pred)

        # Covariance
        covar = tf.reduce_mean(patches_true * patches_pred, axis=-1) - u_true * u_pred

        # SSIM
        self.L = (2 * u_true * u_pred + self.c1) / (tf.square(u_true) + tf.square(u_pred) + self.c1)
        self.C = (2 * s_true * s_pred + self.c2) / (v_true + v_pred + self.c2)
        self.S = (covar + self.c3) / (s_true * s_pred + self.c3)
        self.L = tf.reduce_mean(self.L, axis=self.reduce_axis)
        self.C = tf.reduce_mean(self.C, axis=self.reduce_axis)
        self.S = tf.reduce_mean(self.S, axis=self.reduce_axis)

        return tf.pow(self.L, self.alpha) * tf.pow(self.C, self.beta) * tf.pow(self.S, self.gamma)

    @function_decorator('SSIM__loss')
    def loss(self, y_true, y_pred):
        return tf.reduce_mean((1. - self.ssim(y_true, y_pred)) / 2.0)

    @function_decorator('SSIM__metric')
    def metric(self, y_true, y_pred):
        return tf.reduce_mean(self.ssim(y_true, y_pred))


class StructuralSimilarity_simplified:
    # Based on https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/losses/dssim.py
    def __init__(self, k1=0.01, k2=0.03,
                 patch_size=32, dynamic_range=1., overlap=0.0, dim=3,
                 alpha=1., beta=1., gamma=1.,
                 **kwargs):
        """
        Structural (Di)Similarity Index Measure:

        :param k1: Internal parameter. Defaults to 0.01
        :param k2: Internal parameter. Defaults to 0.02
        :param patch_size: Size of the extracted patches. Defaults to 32. Recommendation: half the image size.
        :param dynamic_range: Maximum numerical intensity value (typ. 2^bits_per_pixel - 1). Defaults to 1.
        :param overlap: Patch overlap ratio. Must be in the range [0., 1.). Defaults to 0.
        :param dim: Data dimensionality. Must be {1, 2, 3}. Defaults to 3.
        :param alpha, beta, gamma: Exponential parameters to balance the contribution of the luminance, contrast and
                                    structure measures. Default to 1.
        """
        assert (dim > 0) and (dim < 4), 'Invalid dimension. It must be 1, 2, or 3'
        assert overlap < 1., 'Invalid overlap. It must be in the range [0., 1.)'
        self.c1 = (k1 * dynamic_range) ** 2
        self.c2 = (k2 * dynamic_range) ** 2
        self.c3 = self.c2 / 2
        self.alpha = tf.cast(alpha, tf.float32)
        self.beta = tf.cast(beta, tf.float32)
        self.gamma = tf.cast(gamma, tf.float32)

        self.kernel_shape = [1] + [patch_size] * dim + [1]
        stride = int(patch_size * (1 - overlap))
        self.stride = [1] + [stride if stride else 1] * dim + [1]
        self.dim = dim
        self.patch_extractor = None
        if dim == 2:
            self.patch_extractor = tf.extract_image_patches
        elif dim == 3:
            self.patch_extractor = tf.extract_volume_patches

        if patch_size == -1:
            # Don't extract patches
            self.dim = 1

        self.L = None   # Luminance
        self.C = None   # Contrast
        self.S = None   # Structure

    def __int_shape(self, x):
        return tf.keras.backend.int_shape(x) if tf.keras.backend.backend() == 'tensorflow' else tf.keras.backend.shape(x)

    def ssim(self, y_true, y_pred):
        if self.dim > 1:
            # Don't use for training. The gradient doesn't backpropagate through the patch extractors
            # patches: [B, out_rows, out_cols, ..., krows*kcols*...*channels] -> out_rows * out_cols * ... = nb patches
            patches_true = self.patch_extractor(y_true, ksizes=self.kernel_shape, strides=self.stride, padding='VALID', name='patches_true')
            patches_pred = self.patch_extractor(y_pred, ksizes=self.kernel_shape, strides=self.stride, padding='VALID', name='patches_pred')
        else:
            patches_true = y_true
            patches_pred = y_pred

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

        # return tf.pow(self.L, self.alpha) * tf.pow(self.C, self.beta) * tf.pow(self.S, self.gamma)
        num = (2 * u_true * u_pred + self.c1) * (2 * covar + self.c2)
        den = ((tf.square(u_true) + tf.square(u_pred) + self.c1) * (v_pred + v_true + self.c2))
        return num / den

    @function_decorator('SSIM_simple__loss')
    def loss(self, y_true, y_pred):
        return tf.reduce_mean((1. - self.ssim(y_true, y_pred)) / 2.0)

    @function_decorator('SSIM_simple__metric')
    def metric(self, y_true, y_pred):
        return tf.reduce_mean(self.ssim(y_true, y_pred))


class MultiScaleStructuralSimilarity(StructuralSimilarity):
    def __init__(self, k1=0.01, k2=0.03, patch_size=3, dynamic_range=1., overlap=0.0, dim=3, nscales=3, alpha=1., beta=1., gamma=1.):
        """
        Multi Scale Structural (Di)Similarity Index Measure:
        Ref:    [1] https://www.cns.nyu.edu/pub/eero/wang03b.pdf

        :param k1: Internal parameter. Defaults to 0.01
        :param k2: Internal parameter. Defaults to 0.02
        :param patch_size: Size of the extracted patches. Defaults to 32. Recommendation: half the image size.
        :param dynamic_range: Maximum numerical intensity value (typ. 2^bits_per_pixel - 1). Defaults to 1.
        :param overlap: Patch overlap ratio. Must be in the range [0., 1.). Defaults to 0.
        :param dim: Data dimensionality. Must be {2, 3}. Defaults to 3.
        :param nscales: Number of scales to analyze. Defaults to 3.
        :param alpha, beta, gamma: Exponential parameters to balance the contribution of the luminance, contrast and
                                    structure measures. Default to 1.
        """
        assert dim > 1, 'Cannot be used with 1-D data'
        super(MultiScaleStructuralSimilarity, self).__init__(k1=k1, k2=k2, patch_size=patch_size,
                                                             dynamic_range=dynamic_range, overlap=overlap, dim=dim,
                                                             alpha=alpha, beta=beta, gamma=gamma)
        self.num_scales = nscales
        self.avg_pool = getattr(tf.nn, 'avg_pool%dd' % dim)
        self.ds_stride = self.ds_kernel = [1] + [2]*dim + [1]

        # In [1] these are set to the same value at the same scales and normalized across scales
        self.alpha = self.beta = self.gamma = 1 / nscales

    def _cond(self, cs_prod, scale_level, y_true, y_pred):
        return tf.less_equal(scale_level, self.num_scales)

    def _iteration(self, cs_prod, scale_level, y_true, y_pred):
        super(MultiScaleStructuralSimilarity, self).ssim(y_true, y_pred)
        cs_prod *= tf.reduce_mean(tf.pow(self.C, self.beta) * tf.pow(self.S, self.gamma))
        y_true = self.avg_pool(y_true, ksize=self.ds_kernel, strides=self.ds_stride, padding='VALID')
        y_pred = self.avg_pool(y_pred, ksize=self.ds_kernel, strides=self.ds_stride, padding='VALID')
        scale_level += 1
        return cs_prod, scale_level, y_true, y_pred,

    def ssim(self, y_true, y_pred):
        return self.ms_ssim(y_true, y_pred)

    def ms_ssim(self, y_true, y_pred):
        cs_prod = tf.constant(1.)
        scale_level = tf.constant(1.)
        cs_prod, *_ = tf.while_loop(self._cond,
                                    self._iteration,
                                    (cs_prod, scale_level, y_true, y_pred),
                                    (cs_prod.get_shape(), scale_level.get_shape(),
                                     tf.TensorShape(([1] + [None] * self.dim + [1])),
                                     tf.TensorShape(([1] + [None] * self.dim + [1]))))

        ms_ssim = tf.reduce_mean(tf.pow(self.L, self.alpha)) * cs_prod

        return tf.reduce_mean(ms_ssim)

    @function_decorator('MS_SSIM__loss')
    def loss(self, y_true, y_pred):
        return tf.reduce_mean((1. - self.ms_ssim(y_true, y_pred)) / 2.0)


class MultiScaleStructuralSimilarity_v2(StructuralSimilarity):
    def __init__(self, k1=0.01, k2=0.03, patch_size=3, dynamic_range=1., overlap=0.0, dim=3, nscales=3, alpha=1., beta=1., gamma=1.):
        """
        Multi Scale Structural (Di)Similarity Index Measure:
        Ref:    [1] https://www.cns.nyu.edu/pub/eero/wang03b.pdf

        :param k1: Internal parameter. Defaults to 0.01
        :param k2: Internal parameter. Defaults to 0.02
        :param patch_size: Size of the extracted patches. Defaults to 32. Recommendation: half the image size.
        :param dynamic_range: Maximum numerical intensity value (typ. 2^bits_per_pixel - 1). Defaults to 1.
        :param overlap: Patch overlap ratio. Must be in the range [0., 1.). Defaults to 0.
        :param dim: Data dimensionality. Must be {2, 3}. Defaults to 3.
        :param nscales: Number of scales to analyze. Defaults to 3.
        :param alpha, beta, gamma: Exponential parameters to balance the contribution of the luminance, contrast and
                                    structure measures. Default to 1.
        """
        assert dim > 1, 'Cannot be used with 1-D data'
        super(MultiScaleStructuralSimilarity_v2, self).__init__(k1=k1, k2=k2, patch_size=patch_size,
                                                                dynamic_range=dynamic_range, overlap=overlap, dim=dim,
                                                                alpha=alpha, beta=beta, gamma=gamma)
        self.num_scales = nscales
        self.avg_pool = getattr(tf.nn, 'avg_pool%dd' % dim)
        self.ds_stride = self.ds_kernel = [1] + [2]*dim + [1]

        # In [1] these are set to the same value at the same scales and normalized across scales
        self.alpha = self.beta = self.gamma = 1 / nscales

    def _cond(self, cs_prod, scale_level, y_true, y_pred):
        return tf.less_equal(scale_level, self.num_scales)

    def _iteration(self, cs_prod, scale_level, y_true, y_pred):
        super(MultiScaleStructuralSimilarity_v2, self).ssim(y_true, y_pred)
        cs_prod *= tf.reduce_mean(tf.pow(self.C, self.beta) * tf.pow(self.S, self.gamma))
        y_true = self.avg_pool(y_true, ksize=self.ds_kernel, strides=self.ds_stride, padding='VALID')
        y_pred = self.avg_pool(y_pred, ksize=self.ds_kernel, strides=self.ds_stride, padding='VALID')
        scale_level += 1
        return cs_prod, scale_level, y_true, y_pred,

    def ssim(self, y_true, y_pred):
        return self.ms_ssim(y_true, y_pred)

    def ms_ssim(self, y_true, y_pred):
        cs_prod = tf.constant(1.)
        scale_level = tf.constant(1.)
        cs_prod, *_ = tf.while_loop(self._cond,
                                    self._iteration,
                                    (cs_prod, scale_level, y_true, y_pred),
                                    (cs_prod.get_shape(), scale_level.get_shape(),
                                     tf.TensorShape(([1] + [None] * self.dim + [1])),
                                     tf.TensorShape(([1] + [None] * self.dim + [1]))))

        ms_ssim = tf.reduce_mean(tf.pow(self.L, self.alpha)) * cs_prod

        return tf.reduce_mean(ms_ssim)

    @function_decorator('MS_SSIM_v2__loss')
    def loss(self, y_true, y_pred):
        return tf.reduce_mean((1. - self.ms_ssim(y_true, y_pred)) / 2.0)


class StructuralSimilarityGaussian:
    # This is equivalent to StructuralSimilarity(patch_size=img_size)
    def __init__(self, k1=0.01, k2=0.03, dynamic_range=1., gauss_sigma=5., dim=3, alpha=1., beta=1., gamma=1.):
        """
        SSIM using Gaussian filter to approximate the statistics of the images
        Ref:    https://www.cns.nyu.edu/pub/eero/wang03b.pdf
                https://arxiv.org/pdf/1511.08861.pdf
                https://github.com/NVlabs/PL4NN/blob/master/src/loss.py

        :param k1: Internal parameter. Defaults to 0.01
        :param k2: Internal parameter. Defaults to 0.02
        :param dynamic_range: Maximum numerical intensity value (typ. 2^bits_per_pixel - 1). Defaults to 1.
        :param gauss_sigma: Sigma of the Gaussian filter. Defaults to 1.5.
        :param dim: Data dimensionality. Must be {2, 3}. Defaults to 3.
        :param alpha, beta, gamma: Exponential parameters to balance the contribution of the luminance, contrast and
                                    structure measures. Default to 1.
        """
        self.c1 = (k1 * dynamic_range) ** 2
        self.c2 = (k2 * dynamic_range) ** 2
        self.c3 = self.c2 / 2
        self.alpha = tf.cast(alpha, tf.float32)
        self.beta = tf.cast(beta, tf.float32)
        self.gamma = tf.cast(gamma, tf.float32)
        self.dim = dim
        self.convDN = getattr(tf.nn, 'conv%dd' % dim)
        self.sigma = gauss_sigma

    def build_gaussian_filter(self, size, sigma, num_channels=1):
        range_1d = tf.range(-(size/2) + 1, size//2 + 1)
        g_1d = tf.math.exp(-1.0 * tf.pow(range_1d, 2) / (2. * tf.pow(sigma, 2)))
        g_1d_expanded = tf.expand_dims(g_1d, -1)
        iterator = tf.constant(1)
        self.__GF = tf.while_loop(lambda iterator, g_1d: tf.less(iterator, self.dim),
                                  lambda iterator, g_1d: (iterator + 1, tf.expand_dims(g_1d, -1) * tf.transpose(g_1d_expanded)),
                                  [iterator, g_1d],
                                  [iterator.get_shape(), tf.TensorShape([None]*self.dim)],  # Shape invariants
                                  back_prop=False,
                                  )[-1]

        self.__GF = tf.divide(self.__GF, tf.reduce_sum(self.__GF))  # Normalization
        self.__GF = tf.reshape(self.__GF, (*[size]*self.dim, 1, 1))  # Add Ch_in and Ch_out for convolution
        self.__GF = tf.tile(self.__GF, (*[1] * self.dim, num_channels, num_channels,))

    def format_data(self, in_data):
        ret_val = in_data
        if self.dim == 3:
            ret_val = tf.transpose(ret_val, [0, 3, 1, 2, 4])
        return ret_val

    def ssim(self, y_true, y_pred):
        self.build_gaussian_filter(y_pred.shape[1], self.sigma)
        y_true_tr = self.format_data(y_true)
        y_pred_tr = self.format_data(y_pred)

        u_true = self.convDN(y_true_tr, self.__GF, [1] * (self.dim + 2), 'SAME')
        u_pred = self.convDN(y_pred_tr, self.__GF, [1] * (self.dim + 2), 'SAME')

        v_true = self.convDN(tf.pow(y_true_tr, 2), self.__GF, [1] * (self.dim + 2), 'SAME') - tf.pow(u_true, 2)
        v_pred = self.convDN(tf.pow(y_pred_tr, 2), self.__GF, [1] * (self.dim + 2), 'SAME') - tf.pow(u_pred, 2)
        covar = self.convDN(tf.multiply(y_true_tr, y_pred_tr), self.__GF, [1] * (self.dim + 2), 'SAME') - u_true * u_pred

        self.L = (2 * u_true * u_pred + self.c1) / (tf.square(u_true) + tf.square(u_pred) + self.c1)
        self.C = (2 * tf.sqrt(v_true) * tf.sqrt(v_pred) + self.c2) / (v_true + v_pred + self.c2)
        self.S = (covar + self.c3) / (tf.sqrt(v_true) * tf.sqrt(v_pred) + self.c3)
        ssim = tf.pow(self.L, self.alpha) * tf.pow(self.C, self.beta) * tf.pow(self.S, self.gamma)

        return tf.reduce_mean(ssim)

    @function_decorator('SSIM_Gaus__loss')
    def loss(self, y_true, y_pred):
        return tf.reduce_mean((1. - self.ssim(y_true, y_pred))/2.)

    @function_decorator('SSIM_Gaus__metric')
    def metric(self, y_true, y_pred):
        return tf.reduce_mean(self.ssim(y_true, y_pred))


class MultiScaleStructuralSimilarityGaussian(StructuralSimilarityGaussian):
    def __init__(self, k1=0.01, k2=0.03, dynamic_range=1., gauss_sigma=5., dim=3, nscales=3, alpha=1., beta=1., gamma=1.):
        """
        Multi Scale SSIM inheriting from StructuralSimilarityGaussian classed
        Ref:    https://www.cns.nyu.edu/pub/eero/wang03b.pdf
                https://arxiv.org/pdf/1511.08861.pdf
                https://github.com/NVlabs/PL4NN/blob/master/src/loss.py

        :param k1: Internal parameter. Defaults to 0.01
        :param k2: Internal parameter. Defaults to 0.02
        :param dynamic_range: Maximum numerical intensity value (typ. 2^bits_per_pixel - 1). Defaults to 1.
        :param gauss_sigma: Sigma of the Gaussian filter. Defaults to 1.5.
        :param dim: Data dimensionality. Must be {2, 3}. Defaults to 3.
        :param nscales: Number of scales to analyze. Defaults to 3.
        :param alpha, beta, gamma: Exponential parameters to balance the contribution of the luminance, contrast and
                                    structure measures. Default to 1.
        """
        super(MultiScaleStructuralSimilarityGaussian, self).__init__(k1=k1, k2=k2, dynamic_range=dynamic_range,
                                                                     gauss_sigma=gauss_sigma, dim=dim,
                                                                     alpha=alpha, beta=beta, gamma=gamma)
        self.__num_scales = nscales

    # # If using the Gaussian approximation of the pyramid MS approach described in https://arxiv.org/pdf/1511.08861.pdf
    # def build_sigma_scales(self):
    #     iterator = tf.constant(0)
    #     scales = tf.expand_dims(self.sigma, -1)
    #     last_sigma = scales
    #     self.sigma_scales = tf.while_loop(lambda iterator, last_sigma, scales: tf.less_equal(iterator, self.__num_scales),
    #                                        lambda iterator, last_sigma, scales: (iterator + 1, tf.concat([scales, last_sigma/2], 0), last_sigma/2),
    #                                        [iterator, last_sigma, scales])[-1]
    #
    # def build_gaussian_filters_scales(self, size):
    #    self.__GFS = tf.map_fn(lambda sigma: self.build_gaussian_filter(size, sigma), self.sigma, tf.float32)

    def _iteration(self, cs_prod, scale_level, y_true, y_pred):
        # Compute the SSIM, so CS and L have the correct value
        self.ssim(y_true, y_pred)

        cs_prod *= tf.reduce_mean(tf.pow(self.C, self.beta) * tf.pow(self.S, self.gamma))
        scale_level += 1

        # Downsample the images to half the resolution for the next iteration
        y_true = tf.nn.avg_pool(y_true, [1] + [2]*self.dim + [1], [1] + [2]*self.dim + [1], 'SAME')
        y_pred = tf.nn.avg_pool(y_true, [1] + [2]*self.dim + [1], [1] + [2]*self.dim + [1], 'SAME')
        return cs_prod, scale_level, y_true, y_pred

    def ms_ssim(self, y_true, y_pred):
        scale_level = tf.constant(0.)
        cs_prod = tf.constant(1.)
        cs_prod, *_ = tf.while_loop(tf.less(scale_level, self.__num_scales),
                                    self._iteration,
                                    (cs_prod, scale_level, y_true, y_pred),
                                    (cs_prod.get_shape(), scale_level.get_shape(),
                                     tf.TensorShape(([1] + [None]*self.dim + [1])),
                                     tf.TensorShape(([1] + [None]*self.dim + [1]))))
        # L is taken from the last scale
        return tf.reduce_mean(tf.pow(self.L, self.alfa)) * cs_prod

    @function_decorator('MS_SSIM_Gaus__metric')
    def loss(self, y_true, y_pred):
        return tf.reduce_mean((1. - self.ms_ssim(y_true, y_pred))/2.)


class DICEScore:
    def __init__(self, input_shape: list):
        """
        DICE Score.
        :param input_shape: Shape of the input image, without the batch dimension, e.g., 2D: [H, W, C], 3D: [H, W, D, C]
        """
        self.axes = list(range(1, len(input_shape)))  # The list will not include the channel axis [1, ..., num_dims)

    def dice(self, y_true, y_pred):
        numerator = 2 * tf.reduce_sum(y_true * y_pred, self.axes)
        denominator = tf.reduce_sum(y_true + y_pred, self.axes)
        return tf.reduce_mean(tf.div_no_nan(numerator, denominator))

    @function_decorator('DICE__loss')
    def loss(self, y_true, y_pred):
        return 1 - 2 * tf.reduce_mean(self.dice(y_true, y_pred))

    @function_decorator('DICE__metric')
    def metric(self, y_true, y_pred):
        return tf.reduce_mean(self.dice(y_true, y_pred))


class GeneralizedDICEScore:
    def __init__(self, input_shape: list, num_labels: int=None):
        """
        Generalized DICE Score. Implementation based on Carole H. Sudre, et al., "Generalised DIce Overlap as a Deep
        Learning Los Function for Highly Unbalanced Segmentations" https://arxiv.org/abs/1707.03237
        :param input_shape: Shape of the input image, without the batch dimension, e.g., 2D: [H, W, C], 3D: [H, W, D, C]
        """
        self.smooth = 1e-10  # If y_pred = y_true = null -> dice should be 1
        self.num_labels = num_labels
        if input_shape[-1] > 1:
            try:
                self.flat_shape = [-1, np.prod(np.asarray(input_shape[:-1])), input_shape[-1]]
            except TypeError as err:
                self.flat_shape = [-1, None, input_shape[-1]]
            self.cardinal_encoded = False
        elif num_labels is not None:
            try:
                self.flat_shape = [-1, np.prod(np.asarray(input_shape[:-1])), input_shape[-1]]
            except TypeError as err:
                self.flat_shape = [-1, None, input_shape[-1]]
            self.cardinal_enc_shape = [-1, *input_shape[:-1]]
            self.cardinal_encoded = True
            warnings.warn('Differentiable cardinal encoding not yet implemented')
        else:
            raise ValueError('If input_shape does not correspond to cardinally encoded,'
                             'then num_labels must be provided')

    def one_hot_encoding(self, in_img, name=''):
        # TODO: Test if differentiable!
        labels, indices = tf.unique(tf.reshape(in_img, [-1]), tf.int32, name=name+'_unique')
        one_hot = tf.one_hot(indices, self.num_labels, name=name + '_one_hot')
        one_hot = tf.reshape(one_hot, self.cardinal_enc_shape + [self.num_labels], name=name + '_reshape')
        one_hot = tf.slice(one_hot, [0] * len(self.cardinal_enc_shape) + [1], [-1] * (len(self.cardinal_enc_shape) + 1),
                           name=name + '_remove_bg')
        return one_hot

    def weigthed_dice(self, y_true, y_pred):
        # y_true = [B, -1, L]
        # y_pred = [B, -1, L]
        # if self.cardinal_encoded:
        #     y_true = self.one_hot_encoding(y_true, name='GDICE_one_hot_encoding_y_true')
        #     y_pred = self.one_hot_encoding(y_pred, name='GDICE_one_hot_encoding_y_pred')
        y_true = tf.reshape(y_true, self.flat_shape, name='GDICE_reshape_y_true')    # Flatten along the volume dimensions
        y_pred = tf.reshape(y_pred, self.flat_shape, name='GDICE_reshape_y_pred')    # Flatten along the volume dimensions

        size_y_true = tf.reduce_sum(y_true, axis=1, name='GDICE_size_y_true')
        size_y_pred = tf.reduce_sum(y_pred, axis=1, name='GDICE_size_y_pred')
        w = tf.math.divide_no_nan(1., tf.pow(size_y_true, 2), name='GDICE_weight')
        numerator = w * tf.reduce_sum(y_true * y_pred, axis=1)
        denominator = w * (size_y_true + size_y_pred)
        return tf.div_no_nan(2 * tf.reduce_sum(numerator, axis=-1) + self.smooth, tf.reduce_sum(denominator, axis=-1) + self.smooth)

    def macro_dice(self, y_true, y_pred):
        # y_true = [B, -1, L]
        # y_pred = [B, -1, L]
        # if self.cardinal_encoded:
        #     y_true = self.one_hot_encoding(y_true, name='GDICE_one_hot_encoding_y_true')
        #     y_pred = self.one_hot_encoding(y_pred, name='GDICE_one_hot_encoding_y_pred')
        y_true = tf.reshape(y_true, self.flat_shape, name='GDICE_reshape_y_true')    # Flatten along the volume dimensions
        y_pred = tf.reshape(y_pred, self.flat_shape, name='GDICE_reshape_y_pred')    # Flatten along the volume dimensions

        size_y_true = tf.reduce_sum(y_true, axis=1, name='GDICE_size_y_true')
        size_y_pred = tf.reduce_sum(y_pred, axis=1, name='GDICE_size_y_pred')
        numerator = tf.reduce_sum(y_true * y_pred, axis=1)
        denominator = (size_y_true + size_y_pred)
        return tf.div_no_nan(2 * numerator + self.smooth, denominator + self.smooth)

    @function_decorator('GeneralizeDICE__loss')
    def loss(self, y_true, y_pred):
        return 1 - tf.reduce_mean(self.weigthed_dice(y_true, y_pred))

    @function_decorator('GeneralizeDICE__metric')
    def metric(self, y_true, y_pred):
        return tf.reduce_mean(self.weigthed_dice(y_true, y_pred))

    @function_decorator('GeneralizeDICE__loss_macro')
    def loss_macro(self, y_true, y_pred):
        return 1 - tf.reduce_mean(self.macro_dice(y_true, y_pred))

    @function_decorator('GeneralizeDICE__metric_macro')
    def metric_macro(self, y_true, y_pred):
        return tf.reduce_mean(self.macro_dice(y_true, y_pred))


def target_registration_error(y_true, y_pred, average=True):
    '''
    Target Registration Error measured as the average distance between y_true and y_pred
    :param y_true: [N, D] target points
    :param y_pred: [N, D] predicted points
    :param average: return the average TRE or an [N,] array
    :return: averate TRE or [N,] array of TRE for each point
    '''
    assert y_true.shape == y_pred.shape, "y_true and y_pred must have the same shape"
    if average:
        return tf.reduce_mean(tf.linalg.norm(y_pred - y_true, axis=1))
    else:
        return tf.linalg.norm(y_pred - y_true, axis=1)

# TODO: tensorflow-graphic has an implementation of Hausdorff ditance.
#  However, this is not where it should and I can't find it
# def HausdorffDistance_exact(y_true, y_pred, ohe=False, name='hd_exact'):
#     if ohe:
#         y_true = tf.transpose(y_true, [0, 4, 1, 2, 3])
#         y_pred = tf.transpose(y_pred, [0, 4, 1, 2, 3])
#     y_true_coords = tf.where(y_true)
#     y_pred_coords = tf.where(y_pred)
#
#     return tfg_nn.loss.hausdorff_distance.evaluate(y_true_coords, y_pred_coords, name=name)
