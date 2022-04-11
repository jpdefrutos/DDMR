import os
import errno
import shutil
import numpy as np
from scipy.interpolate import griddata, Rbf, LinearNDInterpolator, NearestNDInterpolator
from skimage.measure import regionprops
from DeepDeformationMapRegistration.layers.b_splines import interpolate_spline
from DeepDeformationMapRegistration.utils.thin_plate_splines import ThinPlateSplines
from tensorflow import squeeze


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
                 method='rbf'):
        assert method in ['rbf', 'griddata', 'tf', 'tps'], "Method must be 'rbf' or 'griddata'"
        self.method = method
        self.image_shape = image_shape

        self.grid = self.__regular_grid()

    def __regular_grid(self):
        xx = np.linspace(0, self.image_shape[0], self.image_shape[0], endpoint=False, dtype=np.uint16)
        yy = np.linspace(0, self.image_shape[0], self.image_shape[0], endpoint=False, dtype=np.uint16)
        zz = np.linspace(0, self.image_shape[0], self.image_shape[0], endpoint=False, dtype=np.uint16)

        xx, yy, zz = np.meshgrid(xx, yy, zz)

        return np.stack([xx.flatten(), yy.flatten(), zz.flatten()], axis=0).T

    def __call__(self, disp_map, interp_points, backwards=False):
        disp_map = disp_map.reshape([-1, 3])
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


def get_segmentations_centroids(segmentations, ohe=True, expected_lbls=range(0, 28), missing_centroid=[np.nan]*3, brain_study=True):
    segmentations = np.squeeze(segmentations)
    if ohe:
        segmentations = np.sum(segmentations, axis=-1).astype(np.uint8)
        missing_lbls = set(expected_lbls) - set(np.unique(segmentations))
        if brain_study:
            segmentations += np.ones_like(segmentations)  # Regionsprops neglect the label 0. But we need it, so offset all labels by 1
    else:
        missing_lbls = set(expected_lbls) - set(np.unique(segmentations))

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


def segmentation_cardinal_to_ohe(segmentation):
    # Keep in mind that we don't handle the overlap between the segmentations!
    cpy = np.tile(np.zeros_like(segmentation), (1, 1, 1, len(np.unique(segmentation)[1:])))
    for ch, lbl in enumerate(np.unique(segmentation)[1:]):
        cpy[segmentation == lbl, ch] = 1
    return cpy
