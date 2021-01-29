import numpy as np
import tensorflow as tf


def min_max_norm(img: np.ndarray, out_max_val=1.):
    out_img = img
    max_val = np.amax(img)
    min_val = np.amin(img)
    if (max_val - min_val) != 0:
        out_img = (img - min_val) / (max_val - min_val)
    return out_img * out_max_val


def soft_threshold(x, threshold, name=None):
    # https://www.tensorflow.org/probability/api_docs/python/tfp/math/soft_threshold
    with tf.name_scope(name or 'soft_threshold'):
        x = tf.convert_to_tensor(x, name='x')
        threshold = tf.convert_to_tensor(threshold, dtype=x.dtype, name='threshold')
        return tf.sign(x) * tf.maximum(tf.abs(x) - threshold, 0.)


def binary_activation(x):
    # https://stackoverflow.com/questions/37743574/hard-limiting-threshold-activation-function-in-tensorflow
    cond = tf.less(x, tf.zeros(tf.shape(x)))
    out = tf.where(cond, tf.zeros(tf.shape(x)), tf.ones(tf.shape(x)))

    return out

