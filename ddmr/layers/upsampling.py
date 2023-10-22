import tensorflow as tf
import numpy as np
from numpy import (zeros, where, diff, floor, minimum, maximum, array, concatenate, logical_or, logical_xor,
                   sqrt)
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.util.tf_export import keras_export # api_export
# SRC: https://github.com/tensorflow/tensorflow/issues/46609
# import functools

# keras_export = functools.partial(api_export, 'keras')  # keras_export is not defined in 1.13 but in 1.15 --> https://github.com/tensorflow/tensorflow/blob/3d6e4f24e32b5dbe0a83aaa6e9d0f6671ba41da8/tensorflow/python/util/tf_export.py

def linear_interpolate(x_fix, y_fix, x_var):
    '''
        Functionality:
            1D linear interpolation
        Author:
            Michael Osthege
        Link:
            https://gist.github.com/michaelosthege/e20d242bc62a434843b586c78ebce6cc
    '''

    x_repeat = tf.tile(x_var[:, None], (len(x_fix), ))
    distances = tf.abs(x_repeat - x_fix)

    x_indices = tf.searchsorted(x_fix, x_var)

    weights = tf.zeros_like(distances)
    idx = tf.arange(len(x_indices))
    weights[idx, x_indices] = distances[idx, x_indices - 1]
    weights[idx, x_indices - 1] = distances[idx, x_indices]
    weights /= np.sum(weights, axis=1)[:, None]

    y_var = np.dot(weights, y_fix.T)

    return y_var


def cubic_interpolate(x, y, x0):
    '''
        Functionliaty:
            1D cubic spline interpolation
        Author:
            Raphael Valentin
        Link:
            https://stackoverflow.com/questions/31543775/how-to-perform-cubic-spline-interpolation-in-python
    '''

    x = np.asfarray(x)
    y = np.asfarray(y)

    # remove non finite values
    # indexes = np.isfinite(x)
    # x = x[indexes]
    # y = y[indexes]

    # check if sorted
    if np.any(np.diff(x) < 0):
        indexes = np.argsort(x)
        x = x[indexes]
        y = y[indexes]

    size = len(x)

    xdiff = np.diff(x)
    ydiff = np.diff(y)

    # allocate buffer matrices
    Li = np.empty(size)
    Li_1 = np.empty(size - 1)
    z = np.empty(size)

    # fill diagonals Li and Li-1 and solve [L][y] = [B]
    Li[0] = sqrt(2 * xdiff[0])
    Li_1[0] = 0.0
    B0 = 0.0  # natural boundary
    z[0] = B0 / Li[0]

    for i in range(1, size - 1, 1):
        Li_1[i] = xdiff[i - 1] / Li[i - 1]
        Li[i] = sqrt(2 * (xdiff[i - 1] + xdiff[i]) - Li_1[i - 1] * Li_1[i - 1])
        Bi = 6 * (ydiff[i] / xdiff[i] - ydiff[i - 1] / xdiff[i - 1])
        z[i] = (Bi - Li_1[i - 1] * z[i - 1]) / Li[i]

    i = size - 1
    Li_1[i - 1] = xdiff[-1] / Li[i - 1]
    Li[i] = sqrt(2 * xdiff[-1] - Li_1[i - 1] * Li_1[i - 1])
    Bi = 0.0  # natural boundary
    z[i] = (Bi - Li_1[i - 1] * z[i - 1]) / Li[i]

    # solve [L.T][x] = [y]
    i = size - 1
    z[i] = z[i] / Li[i]
    for i in range(size - 2, -1, -1):
        z[i] = (z[i] - Li_1[i - 1] * z[i + 1]) / Li[i]

    # find index
    index = x.searchsorted(x0)
    np.clip(index, 1, size - 1, index)

    xi1, xi0 = x[index], x[index - 1]
    yi1, yi0 = y[index], y[index - 1]
    zi1, zi0 = z[index], z[index - 1]
    hi1 = xi1 - xi0

    # calculate cubic
    f0 = zi0/(6*hi1)*(xi1-x0)**3 + \
         zi1/(6*hi1)*(x0-xi0)**3 + \
         (yi1/hi1 - zi1*hi1/6)*(x0-xi0) + \
         (yi0/hi1 - zi0*hi1/6)*(xi1-x0)

    return f0


def pchip_interpolate(xi, yi, x, mode="mono", verbose=False):
    '''
        Functionality:
            1D PCHP interpolation
        Authors:
            Michael Taylor <mtaylor@atlanticsciences.com>
            Mathieu Virbel <mat@meltingrocks.com>
        Link:
            https://gist.github.com/tito/553f1135959921ce6699652bf656150d
    '''

    if mode not in ("mono", "quad"):
        raise ValueError("Unrecognized mode string")

    # Search for [xi,xi+1] interval for each x
    xi = xi.astype("double")
    yi = yi.astype("double")

    x_index = zeros(len(x), dtype="int")
    xi_steps = diff(xi)
    if not all(xi_steps > 0):
        raise ValueError("x-coordinates are not in increasing order.")

    x_steps = diff(x)
    if xi_steps.max() / xi_steps.min() < 1.000001:
        # uniform input grid
        if verbose:
            print("pchip: uniform input grid")
        xi_start = xi[0]
        xi_step = (xi[-1] - xi[0]) / (len(xi) - 1)
        x_index = minimum(maximum(floor((x - xi_start) / xi_step).astype(int), 0), len(xi) - 2)

        # Calculate gradients d
        h = (xi[-1] - xi[0]) / (len(xi) - 1)
        d = zeros(len(xi), dtype="double")
        if mode == "quad":
            # quadratic polynomial fit
            d[[0]] = (yi[1] - yi[0]) / h
            d[[-1]] = (yi[-1] - yi[-2]) / h
            d[1:-1] = (yi[2:] - yi[0:-2]) / 2 / h
        else:
            # mode=='mono', Fritsch-Carlson algorithm from fortran numerical
            # recipe
            delta = diff(yi) / h
            d = concatenate((delta[0:1], 2 / (1 / delta[0:-1] + 1 / delta[1:]), delta[-1:]))
            d[concatenate((array([False]), logical_xor(delta[0:-1] > 0, delta[1:] > 0), array([False])))] = 0
            d[logical_or(concatenate((array([False]), delta == 0)), concatenate(
                (delta == 0, array([False]))))] = 0
        # Calculate output values y
        dxxi = x - xi[x_index]
        dxxid = x - xi[1 + x_index]
        dxxi2 = pow(dxxi, 2)
        dxxid2 = pow(dxxid, 2)
        y = (2 / pow(h, 3) * (yi[x_index] * dxxid2 * (dxxi + h / 2) - yi[1 + x_index] * dxxi2 *
                              (dxxid - h / 2)) + 1 / pow(h, 2) *
             (d[x_index] * dxxid2 * dxxi + d[1 + x_index] * dxxi2 * dxxid))
    else:
        # not uniform input grid
        if (x_steps.max() / x_steps.min() < 1.000001 and x_steps.max() / x_steps.min() > 0.999999):
            # non-uniform input grid, uniform output grid
            if verbose:
                print("pchip: non-uniform input grid, uniform output grid")
            x_decreasing = x[-1] < x[0]
            if x_decreasing:
                x = x[::-1]
            x_start = x[0]
            x_step = (x[-1] - x[0]) / (len(x) - 1)
            x_indexprev = -1
            for xi_loop in range(len(xi) - 2):
                x_indexcur = max(int(floor((xi[1 + xi_loop] - x_start) / x_step)), -1)
                x_index[1 + x_indexprev:1 + x_indexcur] = xi_loop
                x_indexprev = x_indexcur
            x_index[1 + x_indexprev:] = len(xi) - 2
            if x_decreasing:
                x = x[::-1]
                x_index = x_index[::-1]
        elif all(x_steps > 0) or all(x_steps < 0):
            # non-uniform input/output grids, output grid monotonic
            if verbose:
                print("pchip: non-uniform in/out grid, output grid monotonic")
            x_decreasing = x[-1] < x[0]
            if x_decreasing:
                x = x[::-1]
            x_len = len(x)
            x_loop = 0
            for xi_loop in range(len(xi) - 1):
                while x_loop < x_len and x[x_loop] < xi[1 + xi_loop]:
                    x_index[x_loop] = xi_loop
                    x_loop += 1
            x_index[x_loop:] = len(xi) - 2
            if x_decreasing:
                x = x[::-1]
                x_index = x_index[::-1]
        else:
            # non-uniform input/output grids, output grid not monotonic
            if verbose:
                print("pchip: non-uniform in/out grids, " "output grid not monotonic")
            for index in range(len(x)):
                loc = where(x[index] < xi)[0]
                if loc.size == 0:
                    x_index[index] = len(xi) - 2
                elif loc[0] == 0:
                    x_index[index] = 0
                else:
                    x_index[index] = loc[0] - 1
        # Calculate gradients d
        h = diff(xi)
        d = zeros(len(xi), dtype="double")
        delta = diff(yi) / h
        if mode == "quad":
            # quadratic polynomial fit
            d[[0, -1]] = delta[[0, -1]]
            d[1:-1] = (delta[1:] * h[0:-1] + delta[0:-1] * h[1:]) / (h[0:-1] + h[1:])
        else:
            # mode=='mono', Fritsch-Carlson algorithm from fortran numerical
            # recipe
            d = concatenate(
                (delta[0:1], 3 * (h[0:-1] + h[1:]) / ((h[0:-1] + 2 * h[1:]) / delta[0:-1] +
                                                      (2 * h[0:-1] + h[1:]) / delta[1:]), delta[-1:]))
            d[concatenate((array([False]), logical_xor(delta[0:-1] > 0, delta[1:] > 0), array([False])))] = 0
            d[logical_or(concatenate((array([False]), delta == 0)), concatenate(
                (delta == 0, array([False]))))] = 0
        dxxi = x - xi[x_index]
        dxxid = x - xi[1 + x_index]
        dxxi2 = pow(dxxi, 2)
        dxxid2 = pow(dxxid, 2)
        y = (2 / pow(h[x_index], 3) *
             (yi[x_index] * dxxid2 * (dxxi + h[x_index] / 2) - yi[1 + x_index] * dxxi2 *
              (dxxid - h[x_index] / 2)) + 1 / pow(h[x_index], 2) *
             (d[x_index] * dxxid2 * dxxi + d[1 + x_index] * dxxi2 * dxxid))
    return y


def Interpolate1D(x, y, xx, method='nearest'):
    '''
        Functionality:
            1D interpolation with various methods
        Author:
            Kai Gao <nebulaekg@gmail.com>
    '''

    n = len(x)
    nn = len(xx)
    yy = np.zeros(nn)

    # Nearest neighbour interpolation
    if method == 'nearest':
        for i in range(0, nn):
            xi = tf.argmin(tf.abs(xx[i] - x))
            yy[i] = y[xi]

    # Linear interpolation
    elif method == 'linear':

        # # slower version
        # if n == 1:
        #     yy[:-1] = y[0]

        # else:
        #     for i in range(0, nn):

        #         if xx[i] < x[0]:
        #             t = (xx[i] - x[0]) / (x[1] - x[0])
        #             yy[i] = (1.0 - t) * y[0] + t * y[1]

        #         elif x[n - 1] <= xx[i]:
        #             t = (xx[i] - x[n - 2]) / (x[n - 1] - x[n - 2])
        #             yy[i] = (1.0 - t) * y[n - 2] + t * y[n - 1]

        #         else:
        #             for k in range(1, n):
        #                 if x[k - 1] <= xx[i] and xx[i] < x[k]:
        #                     t = (xx[i] - x[k - 1]) / (x[k] - x[k - 1])
        #                     yy[i] = (1.0 - t) * y[k - 1] + t * y[k]
        #                     break

        # # faster version
        yy = linear_interpolate(x, y, xx)

    # Cubic interpolation
    elif method == 'cubic':
        yy = cubic_interpolate(x, y, xx)

    # Piecewise cubic Hermite interpolating polynomial (PCHIP)
    elif method == 'pchip':
        yy = pchip_interpolate(x, y, xx, mode='mono')

    return yy


def Interpolate2D(x, y, f, xx, yy, method='nearest'):
    '''
        Functionality:
            2D interpolation implemented in a separable fashion
            There are methods that do real 2D non-separable interpolation, which are
                more difficult to implement.
        Author:
            Kai Gao <nebulaekg@gmail.com>
    '''

    n1 = len(x)
    n2 = len(y)
    nn1 = len(xx)
    nn2 = len(yy)

    w = np.zeros((nn1, n2))
    ff = np.zeros((nn1, nn2))

    # Interpolate along the 1st dimension
    for j in range(0, n2):
        w[:, j] = Interpolate1D(x, f[:, j], xx, method)

    # Interpolate along the 2nd dimension
    for i in range(0, nn1):
        ff[i, :] = Interpolate1D(y, w[i, :], yy, method)

    return ff


def Interpolate3D(x, y, z, f, xx, yy, zz, method='nearest'):
    '''
        Functionality:
            3D interpolation implemented in a separable fashion
            There are methods that do real 3D non-separable interpolation, which are
                more difficult to implement.
        Author:
            Kai Gao <nebulaekg@gmail.com>
    '''

    n1 = len(x)
    n2 = len(y)
    n3 = len(z)
    nn1 = len(xx)
    nn2 = len(yy)
    nn3 = len(zz)

    w1 = tf.zeros((nn1, n2, n3))
    w2 = tf.zeros((nn1, nn2, n3))
    ff = tf.zeros((nn1, nn2, nn3))

    # Interpolate along the 1st dimension
    for k in range(0, n3):
        for j in range(0, n2):
            w1[:, j, k] = Interpolate1D(x, f[:, j, k], xx, method)

    # Interpolate along the 2nd dimension
    for k in range(0, n3):
        for i in range(0, nn1):
            w2[i, :, k] = Interpolate1D(y, w1[i, :, k], yy, method)

    # Interpolate along the 3rd dimension
    for j in range(0, nn2):
        for i in range(0, nn1):
            ff[i, j, :] = Interpolate1D(z, w2[i, j, :], zz, method)

    return ff


def UpInterpolate1D(x, size=2, interpolation='nearest', data_format='channels_first', align_corners=True):
    '''
        Functionality:
            1D upsampling interpolation for tf
        Author:
            Kai Gao <nebulaekg@gmail.com>
    '''

    x = x.numpy()

    if data_format == 'channels_last':
        nb, nr, nh = x.shape
    elif data_format == 'channels_first':
        nb, nh, nr = x.shape

    r = size
    ir = np.linspace(0.0, nr - 1.0, num=nr)

    if align_corners:
        # align_corners=True assumes that values are sampled at discrete points
        iir = np.linspace(0.0, nr - 1.0, num=nr * r)
    else:
        # aling_corners=False assumes that values are sampled at centers of discrete blocks
        iir = np.linspace(0.0 - 0.5 + 0.5 / r, nr - 1.0 + 0.5 - 0.5 / r, num=nr * r)
        iir = np.clip(iir, 0.0, nr - 1.0)

    if data_format == 'channels_last':
        xx = np.zeros((nb, nr * r, nh))
        for i in range(0, nb):
            for j in range(0, nh):
                t = np.reshape(x[i, :, j], (nr))
                xx[i, :, j] = Interpolate1D(ir, t, iir, interpolation)

    elif data_format == 'channels_first':
        xx = np.zeros((nb, nh, nr * r))
        for i in range(0, nb):
            for j in range(0, nh):
                t = np.reshape(x[i, j, :], (nr))
                xx[i, j, :] = Interpolate1D(ir, t, iir, interpolation)

    return tf.convert_to_tensor(xx, dtype=x.dtype)


def UpInterpolate2D(x,
                    size=(2, 2),
                    interpolation='nearest',
                    data_format='channels_first',
                    align_corners=True):
    '''
        Functionality:
            2D upsampling interpolation for tf
        Author:
            Kai Gao <nebulaekg@gmail.com>
    '''

    x = x.numpy()

    if data_format == 'channels_last':
        nb, nr, nc, nh = x.shape
    elif data_format == 'channels_first':
        nb, nh, nr, nc = x.shape

    r = size[0]
    c = size[1]
    ir = np.linspace(0.0, nr - 1.0, num=nr)
    ic = np.linspace(0.0, nc - 1.0, num=nc)

    if align_corners:
        # align_corners=True assumes that values are sampled at discrete points
        iir = np.linspace(0.0, nr - 1.0, num=nr * r)
        iic = np.linspace(0.0, nc - 1.0, num=nc * c)
    else:
        # aling_corners=False assumes that values are sampled at centers of discrete blocks
        iir = np.linspace(0.0 - 0.5 + 0.5 / r, nr - 1.0 + 0.5 - 0.5 / r, num=nr * r)
        iic = np.linspace(0.0 - 0.5 + 0.5 / c, nc - 1.0 + 0.5 - 0.5 / c, num=nc * c)
        iir = np.clip(iir, 0.0, nr - 1.0)
        iic = np.clip(iic, 0.0, nc - 1.0)

    if data_format == 'channels_last':
        xx = np.zeros((nb, nr * r, nc * c, nh))
        for i in range(0, nb):
            for j in range(0, nh):
                t = np.reshape(x[i, :, :, j], (nr, nc))
                xx[i, :, :, j] = Interpolate2D(ir, ic, t, iir, iic, interpolation)

    elif data_format == 'channels_first':
        xx = np.zeros((nb, nh, nr * r, nc * c))
        for i in range(0, nb):
            for j in range(0, nh):
                t = np.reshape(x[i, j, :, :], (nr, nc))
                xx[i, j, :, :] = Interpolate2D(ir, ic, t, iir, iic, interpolation)

    return tf.convert_to_tensor(xx, dtype=x.dtype)


def UpInterpolate3D(x,
                    size=(2, 2, 2),
                    interpolation='nearest',
                    data_format='channels_first',
                    align_corners=True):
    '''
        Functionality:
            3D upsampling interpolation for tf
        Author:
            Kai Gao <nebulaekg@gmail.com>
    '''

    # x = x.numpy()

    if data_format == 'channels_last':
        nb, nr, nc, nd, nh = tf.TensorShape(x).as_list()
    elif data_format == 'channels_first':
        nb, nh, nr, nc, nd = tf.TensorShape(x).as_list()
    else:
        raise ValueError('Invalid option: ', data_format)

    r = size[0]
    c = size[1]
    d = size[2]
    ir = tf.linspace(0.0, nr - 1.0, num=nr)
    ic = tf.linspace(0.0, nc - 1.0, num=nc)
    id = tf.linspace(0.0, nd - 1.0, num=nd)

    if align_corners:
        # align_corners=True assumes that values are sampled at discrete points
        iir = tf.linspace(0.0, nr - 1.0, num=nr * r)
        iic = tf.linspace(0.0, nc - 1.0, num=nc * c)
        iid = tf.linspace(0.0, nd - 1.0, num=nd * d)
    else:
        # aling_corners=False assumes that values are sampled at centers of discrete blocks
        iir = tf.linspace(0.0 - 0.5 + 0.5 / r, nr - 1.0 + 0.5 - 0.5 / r, num=nr * r)
        iic = tf.linspace(0.0 - 0.5 + 0.5 / c, nc - 1.0 + 0.5 - 0.5 / c, num=nc * c)
        iid = tf.linspace(0.0 - 0.5 + 0.5 / d, nd - 1.0 + 0.5 - 0.5 / d, num=nd * d)
        iir = tf.clip_by_value(iir, 0.0, nr - 1.0)
        iic = tf.clip_by_value(iic, 0.0, nc - 1.0)
        iid = tf.clip_by_value(iid, 0.0, nd - 1.0)

    if data_format == 'channels_last':
        xx = tf.zeros((nb, nr * r, nc * c, nd * d, nh))
        for i in range(0, nb):
            for j in range(0, nh):
                t = tf.reshape(x[i, :, :, :, j], (nr, nc, nd))
                xx[i, :, :, :, j] = Interpolate3D(ir, ic, id, t, iir, iic, iid, interpolation)

    elif data_format == 'channels_first':
        xx = tf.zeros((nb, nh, nr * r, nc * c, nd * d))
        for i in range(0, nb):
            for j in range(0, nh):
                t = tf.reshape(x[i, j, :, :, :], (nr, nc, nd))
                xx[i, j, :, :, :] = Interpolate3D(ir, ic, id, t, iir, iic, iid, interpolation)

    return tf.convert_to_tensor(xx, dtype=x.dtype)


# ################################################################################
@keras_export('keras.layers.UpSampling1D')
class UpSampling1D(Layer):
    """Upsampling layer for 1D inputs.
  Repeats each temporal step `size` times along the time axis.
  Examples:
  >>> input_shape = (2, 2, 3)
  >>> x = np.arange(np.prod(input_shape)).reshape(input_shape)
  >>> print(x)
  [[[ 0  1  2]
    [ 3  4  5]]
    [[ 6  7  8]
    [ 9 10 11]]]
  >>> y = tf.keras.layers.UpSampling1D(size=2)(x)
  >>> print(y)
  tf.Tensor(
    [[[ 0  1  2]
      [ 0  1  2]
      [ 3  4  5]
      [ 3  4  5]]
      [[ 6  7  8]
      [ 6  7  8]
      [ 9 10 11]
      [ 9 10 11]]], shape=(2, 4, 3), dtype=int64)
  Args:
    size: Integer. Upsampling factor.
  Input shape:
    3D tensor with shape: `(batch_size, steps, features)`.
  Output shape:
    3D tensor with shape: `(batch_size, upsampled_steps, features)`.
  """
    def __init__(self, size=2, data_format='None', interpolation='nearest', align_corners=True, **kwargs):
        super(UpSampling1D, self).__init__(**kwargs)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.size = int(size)
        self.input_spec = InputSpec(ndim=3)
        self.interpolation = interpolation
        if self.interpolation not in {'nearest', 'linear', 'cubic', 'pchip'}:
            raise ValueError('`interpolation` argument should be one of `"nearest"` '
                             'or `"linear"` '
                             'or `"cubic"` '
                             'or `"pchip"`.')
        self.align_corners = align_corners

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()
        size = self.size * input_shape[1] if input_shape[1] is not None else None
        return tf.TensorShape([input_shape[0], size, input_shape[2]])

    def call(self, inputs):
        return UpInterpolate1D(inputs,
                               self.size,
                               data_format=self.data_format,
                               interpolation=self.interpolation,
                               align_corners=self.align_corners)

    def get_config(self):
        config = {'size': self.size}
        base_config = super(UpSampling1D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


@keras_export('keras.layers.UpSampling2D')
class UpSampling2D(Layer):
    """Upsampling layer for 2D inputs.
  Repeats the rows and columns of the data
  by `size[0]` and `size[1]` respectively.
  Examples:
  >>> input_shape = (2, 2, 1, 3)
  >>> x = np.arange(np.prod(input_shape)).reshape(input_shape)
  >>> print(x)
  [[[[ 0  1  2]]
    [[ 3  4  5]]]
    [[[ 6  7  8]]
    [[ 9 10 11]]]]
  >>> y = tf.keras.layers.UpSampling2D(size=(1, 2))(x)
  >>> print(y)
  tf.Tensor(
    [[[[ 0  1  2]
        [ 0  1  2]]
      [[ 3  4  5]
        [ 3  4  5]]]
      [[[ 6  7  8]
        [ 6  7  8]]
      [[ 9 10 11]
        [ 9 10 11]]]], shape=(2, 2, 2, 3), dtype=int64)
  Args:
    size: Int, or tuple of 2 integers.
      The upsampling factors for rows and columns.
    data_format: A string,
      one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch_size, height, width, channels)` while `channels_first`
      corresponds to inputs with shape
      `(batch_size, channels, height, width)`.
      It defaults to the `image_data_format` value found in your
      Keras config file at `~/.keras/keras.json`.
      If you never set it, then it will be "channels_last".
    interpolation: A string, one of `nearest` or `bilinear`.
  Input shape:
    4D tensor with shape:
    - If `data_format` is `"channels_last"`:
        `(batch_size, rows, cols, channels)`
    - If `data_format` is `"channels_first"`:
        `(batch_size, channels, rows, cols)`
  Output shape:
    4D tensor with shape:
    - If `data_format` is `"channels_last"`:
        `(batch_size, upsampled_rows, upsampled_cols, channels)`
    - If `data_format` is `"channels_first"`:
        `(batch_size, channels, upsampled_rows, upsampled_cols)`
  """
    def __init__(self, size=(2, 2), data_format=None, interpolation='nearest', align_corners=True, **kwargs):
        super(UpSampling2D, self).__init__(**kwargs)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.size = conv_utils.normalize_tuple(size, 2, 'size')
        self.input_spec = InputSpec(ndim=4)
        self.interpolation = interpolation
        if self.interpolation not in {'nearest', 'bilinear', 'linear', 'cubic', 'pchip'}:
            raise ValueError('`interpolation` argument should be one of `"nearest"` '
                             'or `"bilinear"` '
                             'or `"linear"` '
                             'or `"cubic"` '
                             'or `"pchip"`.')
        if self.interpolation == 'bilinear':
            self.interpolation = 'linear'
        self.align_corners = align_corners

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        if self.data_format == 'channels_first':
            height = self.size[0] * input_shape[2] if input_shape[2] is not None else None
            width = self.size[1] * input_shape[3] if input_shape[3] is not None else None
            return tensor_shape.TensorShape([input_shape[0], input_shape[1], height, width])
        else:
            height = self.size[0] * input_shape[1] if input_shape[1] is not None else None
            width = self.size[1] * input_shape[2] if input_shape[2] is not None else None
            return tensor_shape.TensorShape([input_shape[0], height, width, input_shape[3]])

    def call(self, inputs):
        return UpInterpolate2D(inputs,
                               self.size,
                               data_format=self.data_format,
                               interpolation=self.interpolation,
                               align_corners=self.align_corners)

    def get_config(self):
        config = {'size': self.size, 'data_format': self.data_format, 'interpolation': self.interpolation}
        base_config = super(UpSampling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


@keras_export('keras.layers.UpSampling3D')
class UpSampling3D(Layer):
    """Upsampling layer for 3D inputs.
  Repeats the 1st, 2nd and 3rd dimensions
  of the data by `size[0]`, `size[1]` and `size[2]` respectively.
  Examples:
  >>> input_shape = (2, 1, 2, 1, 3)
  >>> x = tf.constant(1, shape=input_shape)
  >>> y = tf.keras.layers.UpSampling3D(size=2)(x)
  >>> print(y.shape)
  (2, 2, 4, 2, 3)
  Args:
    size: Int, or tuple of 3 integers.
      The upsampling factors for dim1, dim2 and dim3.
    data_format: A string,
      one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
      while `channels_first` corresponds to inputs with shape
      `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)`.
      It defaults to the `image_data_format` value found in your
      Keras config file at `~/.keras/keras.json`.
      If you never set it, then it will be "channels_last".
  Input shape:
    5D tensor with shape:
    - If `data_format` is `"channels_last"`:
        `(batch_size, dim1, dim2, dim3, channels)`
    - If `data_format` is `"channels_first"`:
        `(batch_size, channels, dim1, dim2, dim3)`
  Output shape:
    5D tensor with shape:
    - If `data_format` is `"channels_last"`:
        `(batch_size, upsampled_dim1, upsampled_dim2, upsampled_dim3, channels)`
    - If `data_format` is `"channels_first"`:
        `(batch_size, channels, upsampled_dim1, upsampled_dim2, upsampled_dim3)`
  """
    def __init__(self,
                 size=(2, 2, 2),
                 data_format=None,
                 interpolation='nearest',
                 align_corners=True,
                 **kwargs):
        super(UpSampling3D, self).__init__(**kwargs)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.size = conv_utils.normalize_tuple(size, 3, 'size')
        self.input_spec = InputSpec(ndim=5)
        self.interpolation = interpolation
        if interpolation not in {'nearest', 'trilinear', 'linear', 'cubic', 'pchip'}:
            raise ValueError('`interpolation` argument should be one of `"nearest"` '
                             'or `"trilinear"` '
                             'or `"linear"` '
                             'or `"cubic"` '
                             'or `"pchip"`.')
        if self.interpolation == 'trilinear':
            self.interpolation = 'linear'
        self.align_corners = align_corners

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        if self.data_format == 'channels_first':
            dim1 = self.size[0] * input_shape[2] if input_shape[2] is not None else None
            dim2 = self.size[1] * input_shape[3] if input_shape[3] is not None else None
            dim3 = self.size[2] * input_shape[4] if input_shape[4] is not None else None
            return tensor_shape.TensorShape([input_shape[0], input_shape[1], dim1, dim2, dim3])
        else:
            dim1 = self.size[0] * input_shape[1] if input_shape[1] is not None else None
            dim2 = self.size[1] * input_shape[2] if input_shape[2] is not None else None
            dim3 = self.size[2] * input_shape[3] if input_shape[3] is not None else None
            return tensor_shape.TensorShape([input_shape[0], dim1, dim2, dim3, input_shape[4]])

    def call(self, inputs):
        return UpInterpolate3D(inputs,
                               self.size,
                               data_format=self.data_format,
                               interpolation=self.interpolation,
                               align_corners=self.align_corners)

    def get_config(self):
        config = {'size': self.size, 'data_format': self.data_format}
        base_config = super(UpSampling3D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
