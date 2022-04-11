import numpy as np
import tensorflow as tf


def min_max_norm(img: np.ndarray, out_max_val=1.):
    out_img = img
    max_val = np.amax(img)
    min_val = np.amin(img)
    if (max_val - min_val) != 0:
        out_img = (img - min_val) / (max_val - min_val)
    return out_img * out_max_val


def soft_threshold(x, threshold, name='soft_threshold'):
    # https://www.tensorflow.org/probability/api_docs/python/tfp/math/soft_threshold
    # Foucart S., Rauhut H. (2013) Basic Algorithms. In: A Mathematical Introduction to Compressive Sensing.
    #   Applied and Numerical Harmonic Analysis. Birkhäuser, New York, NY. https://doi.org/10.1007/978-0-8176-4948-7_3
    #   Chapter 3, page 72
    with tf.name_scope(name):
        x = tf.convert_to_tensor(x, name='x')
        threshold = tf.convert_to_tensor(threshold, dtype=x.dtype, name='threshold')
        return tf.sign(x) * tf.maximum(tf.abs(x) - threshold, 0.)


def hard_threshold(x, threshold, name='hard_threshold'):
    with tf.name_scope(name):
        threshold = tf.convert_to_tensor(threshold, dtype=x.dtype, name='threshold')
        return tf.sign(tf.maximum(tf.abs(x) - threshold, 0.))


def binary_activation(x):
    # https://stackoverflow.com/questions/37743574/hard-limiting-threshold-activation-function-in-tensorflow
    cond = tf.less(x, tf.zeros(tf.shape(x)))
    out = tf.where(cond, tf.zeros(tf.shape(x)), tf.ones(tf.shape(x)))

    return out


def gaussian_kernel(kernel_size, sigma, in_ch, out_ch, dim, dtype=tf.float32):
    # SRC: https://stackoverflow.com/questions/59286171/gaussian-blur-image-in-dataset-pipeline-in-tensorflow
    x = tf.range(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=dtype)
    g = tf.math.exp(-(tf.pow(x, 2) / (2 * tf.pow(tf.cast(sigma, dtype), 2))))

    g_kernel = tf.identity(g)
    g_kernel = tf.tensordot(g_kernel, g, 0)
    g_kernel = tf.tensordot(g_kernel, g, 0)

    # i = tf.constant(0)
    # cond = lambda i, g_kern: tf.less(i, dim - 1)
    # mult_kern = lambda i, g_kern: [tf.add(i, 1), tf.tensordot(g_kern, g, 0)]
    # _, g_kernel = tf.while_loop(cond, mult_kern,
    #                             loop_vars=[i, g_kernel],
    #                             shape_invariants=[i.get_shape(), tf.TensorShape([kernel_size, None, None])])

    g_kernel = g_kernel / tf.reduce_sum(g_kernel)
    g_kernel = tf.expand_dims(tf.expand_dims(g_kernel, axis=-1), axis=-1)
    return tf.tile(g_kernel, (*(1,)*dim, in_ch, out_ch))


def sample_unique(population, samples, tout=tf.int32):
    # src: https://github.com/tensorflow/tensorflow/issues/9260#issuecomment-437875125
    z = -tf.log(-tf.log(tf.random_uniform((tf.shape(population)[0],), 0, 1)))
    _, indices = tf.nn.top_k(z, samples)
    ret_val = tf.gather(population, indices)
    return tf.cast(ret_val, tout)
