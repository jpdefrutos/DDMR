import os
import errno
import shutil
import numpy as np
from scipy.interpolate import griddata, Rbf, LinearNDInterpolator, NearestNDInterpolator
from skimage.measure import regionprops
from DeepDeformationMapRegistration.layers.b_splines import interpolate_spline
from DeepDeformationMapRegistration.utils.thin_plate_splines import ThinPlateSplines
from tensorflow import squeeze
from scipy.ndimage import zoom
import tensorflow as tf


def try_mkdir(dir, verbose=True):
    try:
        os.makedirs(dir)
    except OSError as err:
        if err.errno == errno.EEXIST and verbose:
            print("Directory " + dir + " already exists")
        else:
            raise ValueError("Can't create dir " + dir)
    else:
        print("Created directory " + dir)


def function_decorator(new_name):
    """"
    Change the __name__ property of a function using new_name.
    :param new_name:
    :return:
    """
    def decorator(func):
        func.__name__ = new_name
        return func
    return decorator


class DatasetCopy:
    def __init__(self, dataset_location, copy_location=None, verbose=True):
        self.__copy_loc = os.path.join(os.getcwd(), 'temp_dataset') if copy_location is None else copy_location
        self.__dst_loc = dataset_location
        self.__verbose = verbose

    def copy_dataset(self):
        shutil.copytree(self.__dst_loc, self.__copy_loc)
        if self.__verbose:
            print('{} copied to {}'.format(self.__dst_loc, self.__copy_loc))
        return self.__copy_loc

    def delete_temp(self):
        shutil.rmtree(self.__copy_loc)
        if self.__verbose:
            print('Deleted: ', self.__copy_loc)


class DisplacementMapInterpolator:
    def __init__(self,
                 image_shape=[64, 64, 64],
                 method='rbf',
                 step=1):
        assert method in ['rbf', 'griddata', 'tf', 'tps'], "Method must be 'rbf' or 'griddata'"
        self.method = method
        self.image_shape = image_shape
        self.step = step  # If to use every point or even N-th point

        self.grid = self.__regular_grid()

    def __regular_grid(self):
        xx = np.linspace(0, self.image_shape[0], self.image_shape[0], endpoint=False, dtype=np.uint16)
        yy = np.linspace(0, self.image_shape[1], self.image_shape[1], endpoint=False, dtype=np.uint16)
        zz = np.linspace(0, self.image_shape[2], self.image_shape[2], endpoint=False, dtype=np.uint16)

        xx, yy, zz = np.meshgrid(xx, yy, zz)

        return np.stack([xx[::self.step, ::self.step, ::self.step].flatten(),
                         yy[::self.step, ::self.step, ::self.step].flatten(),
                         zz[::self.step, ::self.step, ::self.step].flatten()], axis=0).T

    def __call__(self, disp_map, interp_points, backwards=False):
        disp_map = disp_map.squeeze()[::self.step, ::self.step, ::self.step, ...].reshape([-1, 3])
        grid_pts = self.grid.copy()
        if backwards:
            grid_pts = np.add(grid_pts, disp_map).astype(np.float32)
            disp_map *= -1

        if self.method == 'rbf':
            interpolator = Rbf(grid_pts[:, 0], grid_pts[:, 1], grid_pts[:, 2], disp_map[:, :],
                               method='thin_plate', mode='N-D')
            disp = interpolator(interp_points)
        elif self.method == 'griddata':
            linear_interp = LinearNDInterpolator(grid_pts, disp_map)
            disp = linear_interp(interp_points).copy()
            del linear_interp

            if np.any(np.isnan(disp)):
                # It might happen (though it shouldn't) that the interpolation point is outside the convex hull of grid points.
                #   in this situation, linear interpolation fails and will put NaN. Nearest can give a value, so we are going to
                #   substitute those unexpected NaNs with the nearest value. Unexpected == not in interp_points
                nan_disp_idx = set(np.unique(np.argwhere(np.isnan(disp))[:, 0]))
                nan_interp_pts_idx = set(np.unique(np.argwhere(np.isnan(interp_points))[:, 0]))
                idx = nan_disp_idx - nan_interp_pts_idx if len(nan_disp_idx) > len(nan_interp_pts_idx) else nan_interp_pts_idx - nan_disp_idx
                idx = list(idx)
                if len(idx):
                    # We have unexpected NaNs
                    near_interp = NearestNDInterpolator(grid_pts, disp_map)
                    near_disp = near_interp(interp_points[idx, ...]).copy()
                    del near_interp
                    for n, i in enumerate(idx):
                        disp[i, ...] = near_disp[n, ...]
        elif self.method == 'tf':
            # Order: 1 -> linear, 2 -> thin plate, 3 -> cubic
            disp = squeeze(interpolate_spline(grid_pts[np.newaxis, ...][::4, :],    # Batch axis
                                              disp_map[np.newaxis, ...][::4, :],
                                              interp_points[np.newaxis, ...], order=2), axis=0)
        else:
            tps_interp = ThinPlateSplines(grid_pts[::8, :], self.grid.copy().astype(np.float32)[::8, :])
            disp = tps_interp.interpolate(interp_points).eval()
            del tps_interp

        return disp


def get_segmentations_centroids(segmentations, ohe=True, expected_lbls=range(1, 28), missing_centroid=[np.nan]*3, brain_study=True):
    segmentations = np.squeeze(segmentations)
    if ohe:
        segmentations = segmentation_ohe_to_cardinal(segmentations)
        lbls = set(np.unique(segmentations)) - {0}  # Remove the 0 value returned by np.unique, no label
        # missing_lbls = set(expected_lbls) - lbls
        # if brain_study:
        #     segmentations += np.ones_like(segmentations)  # Regionsprops neglect the label 0. But we need it, so offset all labels by 1
    else:
        lbls = set(np.unique(segmentations)) if 0 in expected_lbls else set(np.unique(segmentations)) - {0}
    missing_lbls = set(expected_lbls) - lbls

    if 0 in expected_lbls:
        segmentations += np.ones_like(segmentations)  # Regionsprops neglects the label 0. But we need it, so offset all labels by 1

    segmentations = np.squeeze(segmentations)   # remove channel dimension, not needed anyway

    seg_props = regionprops(segmentations)
    centroids = np.asarray([c.centroid for c in seg_props]).astype(np.float32)

    for lbl in missing_lbls:
        idx = expected_lbls.index(lbl)
        centroids = np.insert(centroids, idx, missing_centroid, axis=0)
    return centroids.copy(), missing_lbls


def segmentation_ohe_to_cardinal(segmentation):
    cpy = segmentation.copy()
    for lbl in range(segmentation.shape[-1]):
        cpy[..., lbl] *= (lbl + 1)
    # Add the Background
    cpy = np.concatenate([np.zeros(segmentation.shape[:-1])[..., np.newaxis], cpy], axis=-1)
    return np.argmax(cpy, axis=-1)[..., np.newaxis]


def segmentation_cardinal_to_ohe(segmentation, labels_list: list = None):
    # Keep in mind that we don't handle the overlap between the segmentations!
    #labels_list = np.unique(segmentation)[1:] if labels_list is None else labels_list
    num_labels = len(labels_list)
    expected_shape = segmentation.shape[:-1] + (num_labels,)
    cpy = np.zeros(expected_shape, dtype=np.uint8)
    seg_squeezed = np.squeeze(segmentation, axis=-1)
    for ch, lbl in enumerate(labels_list):
        cpy[seg_squeezed == lbl, ch] = 1
    return cpy


def resize_displacement_map(displacement_map: np.ndarray, dest_shape: [list, np.ndarray, tuple], scale_trf: np.ndarray=None):
    if scale_trf is None:
        scale_trf = scale_transformation(displacement_map.shape, dest_shape)
    else:
        assert isinstance(scale_trf, np.ndarray) and scale_trf.shape == (4, 4), 'Invalid transformation: {}'.format(scale_trf)
    zoom_factors = scale_trf.diagonal()
    # First scale the values, so we cut down the number of multiplications
    dm_resized = np.copy(displacement_map)
    dm_resized[..., 0] *= zoom_factors[0]
    dm_resized[..., 1] *= zoom_factors[1]
    dm_resized[..., 2] *= zoom_factors[2]
    # Then rescale using zoom
    dm_resized = zoom(dm_resized, zoom_factors)
    return dm_resized


def scale_transformation(original_shape: [list, tuple, np.ndarray], dest_shape: [list, tuple, np.ndarray]) -> np.ndarray:
    if isinstance(original_shape, (list, tuple)):
        original_shape = np.asarray(original_shape, dtype=int)
    if isinstance(dest_shape, (list, tuple)):
        dest_shape = np.asarray(dest_shape, dtype=int)
    original_shape = original_shape.astype(int)
    dest_shape = dest_shape.astype(int)

    trf = np.eye(4)
    np.fill_diagonal(trf, [*np.divide(dest_shape, original_shape), 1])

    return trf


class GaussianFilter:
    def __init__(self, size, sigma, dim, num_channels, stride=None, batch: bool=True):
        """
        Gaussian filter
        :param size: Kernel size
        :param sigma: Sigma of the Gaussian filter.
        :param dim: Data dimensionality. Must be {2, 3}.
        :param num_channels: Number of channels of the image to filter.
        """
        self.size = size
        self.dim = dim
        self.sigma = float(sigma)
        self.num_channels = num_channels
        self.stride = size // 2 if stride is None else int(stride)
        if batch:
            self.stride = [1] + [self.stride] * self.dim + [1]   # No support for strides in the batch and channel dims
        else:
            self.stride = [self.stride] * self.dim + [1]    # No support for strides in the batch and channel dims

        self.convDN = getattr(tf.nn, 'conv%dd' % dim)
        self.__GF = None

        self.__build_gaussian_filter()

    def __build_gaussian_filter(self):
        range_1d = tf.range(-(self.size/2) + 1, self.size//2 + 1)
        g_1d = tf.math.exp(-1.0 * tf.pow(range_1d, 2) / (2. * tf.pow(self.sigma, 2)))
        g_1d_expanded = tf.expand_dims(g_1d, -1)
        iterator = tf.constant(1)
        self.__GF = tf.while_loop(lambda iterator, g_1d: tf.less(iterator, self.dim),
                                  lambda iterator, g_1d: (iterator + 1, tf.expand_dims(g_1d, -1) * tf.transpose(g_1d_expanded)),
                                  [iterator, g_1d],
                                  [iterator.get_shape(), tf.TensorShape(None)],  # Shape invariants
                                  back_prop=False
                                  )[-1]

        self.__GF = tf.divide(self.__GF, tf.reduce_sum(self.__GF))  # Normalization
        self.__GF = tf.reshape(self.__GF, (*[self.size]*self.dim, 1, 1))  # Add Ch_in and Ch_out for convolution
        self.__GF = tf.tile(self.__GF, (*[1] * self.dim, self.num_channels, self.num_channels,))

    def apply_filter(self, in_image):
        return self.convDN(in_image, self.__GF, self.stride, 'SAME')

    @property
    def kernel(self):
        return self.__GF