# SRC: https://github.com/tensorflow/tensorflow/blob/a4dfb8d1a71385bd6d122e4f27f86dcebb96712d/tensorflow/python/ops/image_ops_impl.py
from tensorflow.python import nn_ops
from tensorflow.python import math_ops
from tensorflow.python import array_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import nn
from tensorflow.python.util.tf_export import tf_export
from tensorflow.python.util import dispatch
from DeepDeformationMapRegistration.utils.misc import function_decorator


@tf_export('image.convert_image_dtype')
@dispatch.add_dispatch_support
def convert_image_dtype(image, dtype, saturate=False, name=None):
    """Convert `image` to `dtype`, scaling its values if needed.
    The operation supports data types (for `image` and `dtype`) of
    `uint8`, `uint16`, `uint32`, `uint64`, `int8`, `int16`, `int32`, `int64`,
    `float16`, `float32`, `float64`, `bfloat16`.
    Images that are represented using floating point values are expected to have
    values in the range [0,1). Image data stored in integer data types are
    expected to have values in the range `[0,MAX]`, where `MAX` is the largest
    positive representable number for the data type.
    This op converts between data types, scaling the values appropriately before
    casting.
    Usage Example:
    >>> x = [[[1, 2, 3], [4, 5, 6]],
    ...      [[7, 8, 9], [10, 11, 12]]]
    >>> x_int8 = tf.convert_to_tensor(x, dtype=tf.int8)
    >>> tf.image.convert_image_dtype(x_int8, dtype=tf.float16, saturate=False)
    <tf.Tensor: shape=(2, 2, 3), dtype=float16, numpy=
    array([[[0.00787, 0.01575, 0.02362],
            [0.0315 , 0.03937, 0.04724]],
           [[0.0551 , 0.063  , 0.07086],
            [0.07874, 0.0866 , 0.0945 ]]], dtype=float16)>
    Converting integer types to floating point types returns normalized floating
    point values in the range [0, 1); the values are normalized by the `MAX` value
    of the input dtype. Consider the following two examples:
    >>> a = [[[1], [2]], [[3], [4]]]
    >>> a_int8 = tf.convert_to_tensor(a, dtype=tf.int8)
    >>> tf.image.convert_image_dtype(a_int8, dtype=tf.float32)
    <tf.Tensor: shape=(2, 2, 1), dtype=float32, numpy=
    array([[[0.00787402],
            [0.01574803]],
           [[0.02362205],
            [0.03149606]]], dtype=float32)>
    >>> a_int32 = tf.convert_to_tensor(a, dtype=tf.int32)
    >>> tf.image.convert_image_dtype(a_int32, dtype=tf.float32)
    <tf.Tensor: shape=(2, 2, 1), dtype=float32, numpy=
    array([[[4.6566129e-10],
            [9.3132257e-10]],
           [[1.3969839e-09],
            [1.8626451e-09]]], dtype=float32)>
    Despite having identical values of `a` and output dtype of `float32`, the
    outputs differ due to the different input dtypes (`int8` vs. `int32`). This
    is, again, because the values are normalized by the `MAX` value of the input
    dtype.
    Note that converting floating point values to integer type may lose precision.
    In the example below, an image tensor `b` of dtype `float32` is converted to
    `int8` and back to `float32`. The final output, however, is different from
    the original input `b` due to precision loss.
    >>> b = [[[0.12], [0.34]], [[0.56], [0.78]]]
    >>> b_float32 = tf.convert_to_tensor(b, dtype=tf.float32)
    >>> b_int8 = tf.image.convert_image_dtype(b_float32, dtype=tf.int8)
    >>> tf.image.convert_image_dtype(b_int8, dtype=tf.float32)
    <tf.Tensor: shape=(2, 2, 1), dtype=float32, numpy=
    array([[[0.11811024],
            [0.33858266]],
           [[0.5590551 ],
            [0.77952754]]], dtype=float32)>
    Scaling up from an integer type (input dtype) to another integer type (output
    dtype) will not map input dtype's `MAX` to output dtype's `MAX` but converting
    back and forth should result in no change. For example, as shown below, the
    `MAX` value of int8 (=127) is not mapped to the `MAX` value of int16 (=32,767)
    but, when scaled back, we get the same, original values of `c`.
    >>> c = [[[1], [2]], [[127], [127]]]
    >>> c_int8 = tf.convert_to_tensor(c, dtype=tf.int8)
    >>> c_int16 = tf.image.convert_image_dtype(c_int8, dtype=tf.int16)
    >>> print(c_int16)
    tf.Tensor(
    [[[  256]
      [  512]]
     [[32512]
      [32512]]], shape=(2, 2, 1), dtype=int16)
    >>> c_int8_back = tf.image.convert_image_dtype(c_int16, dtype=tf.int8)
    >>> print(c_int8_back)
    tf.Tensor(
    [[[  1]
      [  2]]
     [[127]
      [127]]], shape=(2, 2, 1), dtype=int8)
    Scaling down from an integer type to another integer type can be a lossy
    conversion. Notice in the example below that converting `int16` to `uint8` and
    back to `int16` has lost precision.
    >>> d = [[[1000], [2000]], [[3000], [4000]]]
    >>> d_int16 = tf.convert_to_tensor(d, dtype=tf.int16)
    >>> d_uint8 = tf.image.convert_image_dtype(d_int16, dtype=tf.uint8)
    >>> d_int16_back = tf.image.convert_image_dtype(d_uint8, dtype=tf.int16)
    >>> print(d_int16_back)
    tf.Tensor(
    [[[ 896]
      [1920]]
     [[2944]
      [3968]]], shape=(2, 2, 1), dtype=int16)
    Note that converting from floating point inputs to integer types may lead to
    over/underflow problems. Set saturate to `True` to avoid such problem in
    problematic conversions. If enabled, saturation will clip the output into the
    allowed range before performing a potentially dangerous cast (and only before
    performing such a cast, i.e., when casting from a floating point to an integer
    type, and when casting from a signed to an unsigned type; `saturate` has no
    effect on casts between floats, or on casts that increase the type's range).
    Args:
      image: An image.
      dtype: A `DType` to convert `image` to.
      saturate: If `True`, clip the input before casting (if necessary).
      name: A name for this operation (optional).
    Returns:
      `image`, converted to `dtype`.
    Raises:
      AttributeError: Raises an attribute error when dtype is neither
      float nor integer
    """
    image = ops.convert_to_tensor(image, name='image')
    dtype = dtypes.as_dtype(dtype)
    if not dtype.is_floating and not dtype.is_integer:
        raise AttributeError('dtype must be either floating point or integer')
    if dtype == image.dtype:
        return array_ops.identity(image, name=name)

    with ops.name_scope(name, 'convert_image', [image]) as name:
        # Both integer: use integer multiplication in the larger range
        if image.dtype.is_integer and dtype.is_integer:
            scale_in = image.dtype.max
            scale_out = dtype.max
            if scale_in > scale_out:
                # Scaling down, scale first, then cast. The scaling factor will
                # cause in.max to be mapped to above out.max but below out.max+1,
                # so that the output is safely in the supported range.
                scale = (scale_in + 1) // (scale_out + 1)
                scaled = math_ops.floordiv(image, scale)

                if saturate:
                    return math_ops.saturate_cast(scaled, dtype, name=name)
                else:
                    return math_ops.cast(scaled, dtype, name=name)
            else:
                # Scaling up, cast first, then scale. The scale will not map in.max to
                # out.max, but converting back and forth should result in no change.
                if saturate:
                    cast = math_ops.saturate_cast(image, dtype)
                else:
                    cast = math_ops.cast(image, dtype)
                scale = (scale_out + 1) // (scale_in + 1)
                return math_ops.multiply(cast, scale, name=name)
        elif image.dtype.is_floating and dtype.is_floating:
            # Both float: Just cast, no possible overflows in the allowed ranges.
            # Note: We're ignoring float overflows. If your image dynamic range
            # exceeds float range, you're on your own.
            return math_ops.cast(image, dtype, name=name)
        else:
            if image.dtype.is_integer:
                # Converting to float: first cast, then scale. No saturation possible.
                cast = math_ops.cast(image, dtype)
                scale = 1. / image.dtype.max
                return math_ops.multiply(cast, scale, name=name)
            else:
                # Converting from float: first scale, then cast
                scale = dtype.max + 0.5  # avoid rounding problems in the cast
                scaled = math_ops.multiply(image, scale)
                if saturate:
                    return math_ops.saturate_cast(scaled, dtype, name=name)
                else:
                    return math_ops.cast(scaled, dtype, name=name)


def _verify_compatible_image_shapes(img1, img2):
    """Checks if two image tensors are compatible for applying SSIM or PSNR.
    This function checks if two sets of images have ranks at least 3, and if the
    last three dimensions match.
    Args:
      img1: Tensor containing the first image batch.
      img2: Tensor containing the second image batch.
    Returns:
      A tuple containing: the first tensor shape, the second tensor shape, and a
      list of control_flow_ops.Assert() ops implementing the checks.
    Raises:
      ValueError: When static shape check fails.
    """
    shape1 = img1.get_shape().with_rank_at_least(4)     # at least [H, W, D, C]
    shape2 = img2.get_shape().with_rank_at_least(4)     # at least [H, W, D, C]
    shape1[-4:].assert_is_compatible_with(shape2[-4:])

    if shape1.ndims is not None and shape2.ndims is not None:
        for dim1, dim2 in zip(
                reversed(shape1.dims[:-4]), reversed(shape2.dims[:-4])):
            if not (dim1 == 1 or dim2 == 1 or dim1.is_compatible_with(dim2)):
                raise ValueError('Two images are not compatible: %s and %s' %
                                 (shape1, shape2))

    # Now assign shape tensors.
    shape1, shape2 = array_ops.shape_n([img1, img2])

    # TODO(sjhwang): Check if shape1[:-4] and shape2[:-4] are broadcastable.
    checks = []
    checks.append(
        control_flow_ops.Assert(
            math_ops.greater_equal(array_ops.size(shape1), 4), [shape1, shape2],
            summarize=10))
    checks.append(
        control_flow_ops.Assert(
            math_ops.reduce_all(math_ops.equal(shape1[-4:], shape2[-4:])),
            [shape1, shape2],
            summarize=10))
    return shape1, shape2, checks


def _ssim_helper(x, y, reducer, max_val, compensation=1.0, k1=0.01, k2=0.03):
    r"""Helper function for computing SSIM.
    SSIM estimates covariances with weighted sums.  The default parameters
    use a biased estimate of the covariance:
    Suppose `reducer` is a weighted sum, then the mean estimators are
      \mu_x = \sum_i w_i x_i,
      \mu_y = \sum_i w_i y_i,
    where w_i's are the weighted-sum weights, and covariance estimator is
      cov_{xy} = \sum_i w_i (x_i - \mu_x) (y_i - \mu_y)
    with assumption \sum_i w_i = 1. This covariance estimator is biased, since
      E[cov_{xy}] = (1 - \sum_i w_i ^ 2) Cov(X, Y).
    For SSIM measure with unbiased covariance estimators, pass as `compensation`
    argument (1 - \sum_i w_i ^ 2).
    Args:
      x: First set of images.
      y: Second set of images.
      reducer: Function that computes 'local' averages from the set of images. For
        non-convolutional version, this is usually tf.reduce_mean(x, [1, 2]), and
        for convolutional version, this is usually tf.nn.avg_pool2d or
        tf.nn.conv3d with weighted-sum kernel.
      max_val: The dynamic range (i.e., the difference between the maximum
        possible allowed value and the minimum allowed value).
      compensation: Compensation factor. See above.
      k1: Default value 0.01
      k2: Default value 0.03 (SSIM is less sensitivity to K2 for lower values, so
        it would be better if we took the values in the range of 0 < K2 < 0.4).
    Returns:
      A pair containing the luminance measure, and the contrast-structure measure.
    """

    c1 = (k1 * max_val)**2
    c2 = (k2 * max_val)**2

    # SSIM luminance measure is
    # (2 * mu_x * mu_y + c1) / (mu_x ** 2 + mu_y ** 2 + c1).
    mean0 = reducer(x)
    mean1 = reducer(y)
    num0 = mean0 * mean1 * 2.0
    den0 = math_ops.square(mean0) + math_ops.square(mean1)
    luminance = (num0 + c1) / (den0 + c1)

    # SSIM contrast-structure measure is
    #   (2 * cov_{xy} + c2) / (cov_{xx} + cov_{yy} + c2).
    # Note that `reducer` is a weighted sum with weight w_k, \sum_i w_i = 1, then
    #   cov_{xy} = \sum_i w_i (x_i - \mu_x) (y_i - \mu_y)
    #          = \sum_i w_i x_i y_i - (\sum_i w_i x_i) (\sum_j w_j y_j).
    num1 = reducer(x * y) * 2.0
    den1 = reducer(math_ops.square(x) + math_ops.square(y))
    c2 *= compensation
    cs = (num1 - num0 + c2) / (den1 - den0 + c2)

    # SSIM score is the product of the luminance and contrast-structure measures.
    return luminance, cs


def _fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function."""
    size = ops.convert_to_tensor(size, dtypes.int32)
    sigma = ops.convert_to_tensor(sigma, dtypes.float32)

    coords = math_ops.cast(math_ops.range(size), sigma.dtype)
    coords -= math_ops.cast(size - 1, sigma.dtype) / 2.0

    g = math_ops.square(coords)
    g *= -0.5 / math_ops.square(sigma)

    g = array_ops.reshape(g, shape=[1, -1]) + array_ops.reshape(g, shape=[-1, 1])
    g = array_ops.reshape(g, shape=[size, size, 1]) + array_ops.reshape(g, shape=[1, size, size])
    g = array_ops.reshape(g, shape=[1, -1])  # For tf.nn.softmax().
    g = nn_ops.softmax(g)
    return array_ops.reshape(g, shape=[size, size, size, 1, 1])


def _ssim_per_channel(img1,
                      img2,
                      max_val=1.0,
                      filter_size=11,
                      filter_sigma=1.5,
                      k1=0.01,
                      k2=0.03):
    """Computes SSIM index between img1 and img2 per color channel.
    This function matches the standard SSIM implementation from:
    Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). Image
    quality assessment: from error visibility to structural similarity. IEEE
    transactions on image processing.
    Details:
      - 11x11 Gaussian filter of width 1.5 is used.
      - k1 = 0.01, k2 = 0.03 as in the original paper.
    Args:
      img1: First image batch.
      img2: Second image batch.
      max_val: The dynamic range of the images (i.e., the difference between the
        maximum the and minimum allowed values).
      filter_size: Default value 11 (size of gaussian filter).
      filter_sigma: Default value 1.5 (width of gaussian filter).
      k1: Default value 0.01
      k2: Default value 0.03 (SSIM is less sensitivity to K2 for lower values, so
        it would be better if we took the values in the range of 0 < K2 < 0.4).
    Returns:
      A pair of tensors containing and channel-wise SSIM and contrast-structure
      values. The shape is [..., channels].
    """
    filter_size = constant_op.constant(filter_size, dtype=dtypes.int32)
    filter_sigma = constant_op.constant(filter_sigma, dtype=img1.dtype)

    shape1, shape2 = array_ops.shape_n([img1, img2])
    checks = [
        control_flow_ops.Assert(
            math_ops.reduce_all(
                math_ops.greater_equal(shape1[-4:-1], filter_size)),
            [shape1, filter_size],
            summarize=8),
        control_flow_ops.Assert(
            math_ops.reduce_all(
                math_ops.greater_equal(shape2[-4:-1], filter_size)),
            [shape2, filter_size],
            summarize=8)
    ]

    # Enforce the check to run before computation.
    with ops.control_dependencies(checks):
        img1 = array_ops.identity(img1)

    # TODO(sjhwang): Try to cache kernels and compensation factor.
    kernel = _fspecial_gauss(filter_size, filter_sigma)
    kernel = array_ops.tile(kernel, multiples=[1, 1, 1, shape1[-1], 1])

    # The correct compensation factor is `1.0 - tf.reduce_sum(tf.square(kernel))`,
    # but to match MATLAB implementation of MS-SSIM, we use 1.0 instead.
    compensation = 1.0

    # TODO(sjhwang): Try FFT.
    # TODO(sjhwang): Gaussian kernel is separable in space. Consider applying
    #   1-by-n and n-by-1 Gaussian filters instead of an n-by-n filter.
    def reducer(x):
        shape = array_ops.shape(x)
        x = array_ops.reshape(x, shape=array_ops.concat([[-1], shape[-4:]], 0))
        y = nn.conv3d(x, kernel, strides=[1, 1, 1, 1, 1], padding='VALID')
        return array_ops.reshape(y, array_ops.concat([shape[:-4], array_ops.shape(y)[1:]], 0))

    luminance, cs = _ssim_helper(img1, img2, reducer, max_val, compensation, k1,
                                 k2)

    # Average over the second, third and the fourth from the last: height, width, depth.
    axes = constant_op.constant([-4, -3, -2], dtype=dtypes.int32)
    ssim_val = math_ops.reduce_mean(luminance * cs, axes)
    cs = math_ops.reduce_mean(cs, axes)
    return ssim_val, cs


@tf_export('image.ssim')
@dispatch.add_dispatch_support
def ssim(img1,
         img2,
         max_val,
         filter_size=11,
         filter_sigma=1.5,
         k1=0.01,
         k2=0.03):
    """Computes SSIM index between img1 and img2.
    This function is based on the standard SSIM implementation from:
    Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). Image
    quality assessment: from error visibility to structural similarity. IEEE
    transactions on image processing.
    Note: The true SSIM is only defined on grayscale.  This function does not
    perform any colorspace transform.  (If the input is already YUV, then it will
    compute YUV SSIM average.)
    Details:
      - 11x11 Gaussian filter of width 1.5 is used.
      - k1 = 0.01, k2 = 0.03 as in the original paper.
    The image sizes must be at least 11x11 because of the filter size.
    Example:
    ```python
        # Read images (of size 255 x 255) from file.
        im1 = tf.image.decode_image(tf.io.read_file('path/to/im1.png'))
        im2 = tf.image.decode_image(tf.io.read_file('path/to/im2.png'))
        tf.shape(im1)  # `img1.png` has 3 channels; shape is `(255, 255, 3)`
        tf.shape(im2)  # `img2.png` has 3 channels; shape is `(255, 255, 3)`
        # Add an outer batch for each image.
        im1 = tf.expand_dims(im1, axis=0)
        im2 = tf.expand_dims(im2, axis=0)
        # Compute SSIM over tf.uint8 Tensors.
        ssim1 = tf.image.ssim(im1, im2, max_val=255, filter_size=11,
                              filter_sigma=1.5, k1=0.01, k2=0.03)
        # Compute SSIM over tf.float32 Tensors.
        im1 = tf.image.convert_image_dtype(im1, tf.float32)
        im2 = tf.image.convert_image_dtype(im2, tf.float32)
        ssim2 = tf.image.ssim(im1, im2, max_val=1.0, filter_size=11,
                              filter_sigma=1.5, k1=0.01, k2=0.03)
        # ssim1 and ssim2 both have type tf.float32 and are almost equal.
    ```
    Args:
      img1: First image batch. 4-D Tensor of shape `[batch, height, width,
        channels]` with only Positive Pixel Values.
      img2: Second image batch. 4-D Tensor of shape `[batch, height, width,
        channels]` with only Positive Pixel Values.
      max_val: The dynamic range of the images (i.e., the difference between the
        maximum the and minimum allowed values).
      filter_size: Default value 11 (size of gaussian filter).
      filter_sigma: Default value 1.5 (width of gaussian filter).
      k1: Default value 0.01
      k2: Default value 0.03 (SSIM is less sensitivity to K2 for lower values, so
        it would be better if we took the values in the range of 0 < K2 < 0.4).
    Returns:
      A tensor containing an SSIM value for each image in batch.  Returned SSIM
      values are in range (-1, 1], when pixel values are non-negative. Returns
      a tensor with shape: broadcast(img1.shape[:-3], img2.shape[:-3]).
    """
    with ops.name_scope(None, 'SSIM', [img1, img2]):
        # Convert to tensor if needed.
        img1 = ops.convert_to_tensor(img1, name='img1')
        img2 = ops.convert_to_tensor(img2, name='img2')
        # Shape checking.
        _, _, checks = _verify_compatible_image_shapes(img1, img2)
        with ops.control_dependencies(checks):
            img1 = array_ops.identity(img1)

        # Need to convert the images to float32.  Scale max_val accordingly so that
        # SSIM is computed correctly.
        max_val = math_ops.cast(max_val, img1.dtype)
        max_val = convert_image_dtype(max_val, dtypes.float32)
        img1 = convert_image_dtype(img1, dtypes.float32)
        img2 = convert_image_dtype(img2, dtypes.float32)
        ssim_per_channel, _ = _ssim_per_channel(img1, img2, max_val, filter_size,
                                                filter_sigma, k1, k2)
        # Compute average over color channels.
        return math_ops.reduce_mean(ssim_per_channel, [-1])


# Default values obtained by Wang et al.
_MSSSIM_WEIGHTS = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)


@tf_export('image.ssim_multiscale')
@dispatch.add_dispatch_support
def ssim_multiscale(img1,
                    img2,
                    max_val,
                    power_factors=_MSSSIM_WEIGHTS,
                    filter_size=11,
                    filter_sigma=1.5,
                    k1=0.01,
                    k2=0.03):
    """Computes the MS-SSIM between img1 and img2.
    This function assumes that `img1` and `img2` are image batches, i.e. the last
    three dimensions are [height, width, channels].
    Note: The true SSIM is only defined on grayscale.  This function does not
    perform any colorspace transform.  (If the input is already YUV, then it will
    compute YUV SSIM average.)
    Original paper: Wang, Zhou, Eero P. Simoncelli, and Alan C. Bovik. "Multiscale
    structural similarity for image quality assessment." Signals, Systems and
    Computers, 2004.
    Args:
      img1: First image batch with only Positive Pixel Values.
      img2: Second image batch with only Positive Pixel Values. Must have the
      same rank as img1.
      max_val: The dynamic range of the images (i.e., the difference between the
        maximum the and minimum allowed values).
      power_factors: Iterable of weights for each of the scales. The number of
        scales used is the length of the list. Index 0 is the unscaled
        resolution's weight and each increasing scale corresponds to the image
        being downsampled by 2.  Defaults to (0.0448, 0.2856, 0.3001, 0.2363,
        0.1333), which are the values obtained in the original paper.
      filter_size: Default value 11 (size of gaussian filter).
      filter_sigma: Default value 1.5 (width of gaussian filter).
      k1: Default value 0.01
      k2: Default value 0.03 (SSIM is less sensitivity to K2 for lower values, so
        it would be better if we took the values in the range of 0 < K2 < 0.4).
    Returns:
      A tensor containing an MS-SSIM value for each image in batch.  The values
      are in range [0, 1].  Returns a tensor with shape:
      broadcast(img1.shape[:-3], img2.shape[:-3]).
    """
    with ops.name_scope(None, 'MS-SSIM', [img1, img2]):
        # Convert to tensor if needed.
        img1 = ops.convert_to_tensor(img1, name='img1')
        img2 = ops.convert_to_tensor(img2, name='img2')
        # Shape checking.
        shape1, shape2, checks = _verify_compatible_image_shapes(img1, img2)
        with ops.control_dependencies(checks):
            img1 = array_ops.identity(img1)

        # Need to convert the images to float32.  Scale max_val accordingly so that
        # SSIM is computed correctly.
        max_val = math_ops.cast(max_val, img1.dtype)
        max_val = convert_image_dtype(max_val, dtypes.float32)
        img1 = convert_image_dtype(img1, dtypes.float32)
        img2 = convert_image_dtype(img2, dtypes.float32)

        imgs = [img1, img2]
        shapes = [shape1, shape2]

        # img1 and img2 are assumed to be a (multi-dimensional) batch of
        # 4-dimensional images (height, width, depth, channels). `heads` contain the batch
        # dimensions, and `tails` contain the image dimensions.
        heads = [s[:-4] for s in shapes]
        tails = [s[-4:] for s in shapes]

        divisor = [1, 2, 2, 2, 1]
        divisor_tensor = constant_op.constant(divisor[1:], dtype=dtypes.int32)

        def do_pad(images, remainder):
            padding = array_ops.expand_dims(remainder, -1)
            padding = array_ops.pad(padding, [[1, 0], [1, 0]])
            return [array_ops.pad(x, padding, mode='SYMMETRIC') for x in images]

        mcs = []
        for k in range(len(power_factors)):
            with ops.name_scope(None, 'Scale%d' % k, imgs):
                if k > 0:
                    # Avg pool takes rank 4 tensors. Flatten leading dimensions.
                    flat_imgs = [
                        array_ops.reshape(x, array_ops.concat([[-1], t], 0))
                        for x, t in zip(imgs, tails)
                    ]

                    remainder = tails[0] % divisor_tensor
                    need_padding = math_ops.reduce_any(math_ops.not_equal(remainder, 0))
                    # pylint: disable=cell-var-from-loop
                    padded = control_flow_ops.cond(need_padding,
                                                   lambda: do_pad(flat_imgs, remainder),
                                                   lambda: flat_imgs)
                    # pylint: enable=cell-var-from-loop

                    downscaled = [
                        nn_ops.avg_pool3d(
                            x, ksize=divisor, strides=divisor, padding='VALID', data_format='NDHWC',)
                        for x in padded
                    ]
                    tails = [x[1:] for x in array_ops.shape_n(downscaled)]
                    imgs = [
                        array_ops.reshape(x, array_ops.concat([h, t], 0))
                        for x, h, t in zip(downscaled, heads, tails)
                    ]

                # Overwrite previous ssim value since we only need the last one.
                ssim_per_channel, cs = _ssim_per_channel(
                    *imgs,
                    max_val=max_val,
                    filter_size=filter_size,
                    filter_sigma=filter_sigma,
                    k1=k1,
                    k2=k2)
                mcs.append(nn_ops.relu(cs))

        # Remove the cs score for the last scale. In the MS-SSIM calculation,
        # we use the l(p) at the highest scale. l(p) * cs(p) is ssim(p).
        mcs.pop()  # Remove the cs score for the last scale.
        mcs_and_ssim = array_ops.stack(
            mcs + [nn_ops.relu(ssim_per_channel)], axis=-1)
        # Take weighted geometric mean across the scale axis.
        ms_ssim = math_ops.reduce_prod(
            math_ops.pow(mcs_and_ssim, power_factors), [-1])

        return math_ops.reduce_mean(ms_ssim, [-1])  # Avg over color channels.


class MultiScaleStructuralSimilarity:
    def __init__(self,
                 max_val,
                 power_factors=_MSSSIM_WEIGHTS,
                 filter_size=11,
                 filter_sigma=1.5,
                 k1=0.01,
                 k2=0.03):
        self.max_val = max_val
        self.power_factors = power_factors
        self.filter_size = int(filter_size)
        self.filter_sigma = filter_sigma
        self.k1 = k1
        self.k2 = k2

    @function_decorator('MS_SSIM__loss')
    def loss(self, y_true, y_pred):
        return math_ops.reduce_mean((1 - ssim_multiscale(y_true, y_pred, self.max_val, self.power_factors,
                                                         self.filter_size, self.filter_sigma, self.k1, self.k2))/2)

    @function_decorator('MS_SSIM__metric')
    def metric(self, y_true, y_pred):
        return ssim_multiscale(y_true, y_pred, self.max_val, self.power_factors, self.filter_size, self.filter_sigma,
                               self.k1, self.k2)


if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    import tensorflow as tf
    tf.enable_eager_execution()
    import nibabel as nib
    import numpy as np
    from DeepDeformationMapRegistration.utils.operators import min_max_norm
    from skimage.metrics import structural_similarity

    img1 = nib.load('test_images/ixi_image.nii.gz')
    img1 = np.asarray(img1.dataobj)
    img1 = img1[np.newaxis, ..., np.newaxis]    # Add Batch and Channel dimensions

    img2 = nib.load('test_images/ixi_image2.nii.gz')
    img2 = np.asarray(img2.dataobj)
    img2 = img2[np.newaxis, ..., np.newaxis]

    img1 = min_max_norm(img1)
    img2 = min_max_norm(img2)

    ssim_tf_1_2 = ssim(img1, img2, 1., filter_size=5)
    assert ssim(img1, img1, 1., filter_size=5).numpy()[0] == 1., 'TF SSIM returned an unexpected value'
    ssim_sklearn = structural_similarity(img1[0, ..., 0], img2[0, ..., 0], win_size=5)

    ms_ssim_tf_1_2 = ssim_multiscale(img1, img2, 1., filter_size=5)
    assert ssim_multiscale(img1, img1, 1., filter_size=5).numpy()[0] == 1., 'TF MS-SSIM returned an unexpected value'

    print('SSIM TF: {}\nSSIM SKLEARN: {}\nMS SSIM TF: {}\n'.format(ssim_tf_1_2, ssim_sklearn, ms_ssim_tf_1_2))

    batch_img1 = np.stack([img1, img2], axis=0)
    batch_img2 = np.stack([img2, img2], axis=0)
    batch_ssim_tf = ssim(batch_img1, batch_img2, 1., filter_size=5)
    batch_ms_ssim_tf = ssim_multiscale(batch_img1, batch_img2, 1., filter_size=5)

    print('Batch SSIM TF: {}\nBatch MS SSIM TF: {}\n'.format(batch_ssim_tf, batch_ms_ssim_tf))

    img1 = img1[:, :127, :127, :127, :]
    img2 = img2[:, :127, :127, :127, :]
    MS_SSIM = MultiScaleStructuralSimilarity(1., filter_size=5)
    print('MS SSIM Loss{}'.format(MS_SSIM.loss(img1, img2)))
