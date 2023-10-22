from scipy.signal import correlate as cc
from scipy.spatial.distance import euclidean
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import structural_similarity as ssim
from medpy.metric.binary import dc, hd95
import numpy as np
import pandas as pd
import os
from ddmr.utils.constants import EPS
from ddmr.utils.nifti_utils import save_nifti
from skimage.transform import resize
from skimage.measure import regionprops, label


def ncc(y_true, y_pred, eps=EPS):
    f_yt = np.reshape(y_true, [-1])
    f_yp = np.reshape(y_pred, [-1])
    mean_yt = np.mean(f_yt)
    mean_yp = np.mean(f_yp)

    n_f_yt = f_yt - mean_yt
    n_f_yp = f_yp - mean_yp
    norm_yt = np.linalg.norm(f_yt, ord=2)
    norm_yp = np.linalg.norm(f_yp, ord=2)
    numerator = np.sum(np.multiply(n_f_yt, n_f_yp))
    denominator = norm_yt * norm_yp + eps
    return np.divide(numerator, denominator)


class EvaluationFigures:
    def __init__(self, output_folder):
        pd.set_option('display.max_columns', None)
        self.__metrics_df = pd.DataFrame(columns=['Name', 'Train_ds', 'Eval_ds', 'MSE', 'NCC', 'SSIM',
                                                  'DICE_PAR', 'DICE_TUM', 'DICE_VES',
                                                  'HD95_PAR', 'HD95_TUM', 'HD95_VES', 'TRE'])
        self.__output_folder = output_folder

    def add_sample(self, name, train_ds, eval_ds, fix_t, par_t, tum_t, ves_t, centroid_t, fix_p, par_p, tum_p, ves_p, centroid_p, scale_transform=None):

        n_fix_t = self.__mean_centred_img(fix_t)
        n_fix_p = self.__mean_centred_img(fix_p)

        if scale_transform is not None:
            s_centroid_t = self.__scale_point(centroid_t, scale_transform)
            s_centroid_p = self.__scale_point(centroid_p, scale_transform)
        else:
            s_centroid_t = centroid_t
            s_centroid_p = centroid_p

        new_row = {'Name': name,
                   'Train_ds': train_ds,
                   'Eval_ds': eval_ds,
                   'MSE': mse(fix_t, fix_p),
                   'NCC': ncc(n_fix_t, n_fix_p),
                   'SSIM': ssim(fix_t, fix_p, multichannel=True),
                   'DICE_PAR': dc(par_p, par_t),
                   'DICE_TUM': dc(tum_p, tum_t),
                   'DICE_VES': dc(ves_p, ves_t),
                   'HD95_PAR': hd95(par_p, par_t) if np.sum(par_p) else 64,
                   'HD95_TUM': hd95(tum_p, tum_t) if np.sum(tum_p) else 64,
                   'HD95_VES': hd95(ves_p, ves_t) if np.sum(ves_p) else 64,
                   'TRE': euclidean(s_centroid_t, s_centroid_p)}

        self.__metrics_df = self.__metrics_df.append(new_row, ignore_index=True)

    @staticmethod
    def __mean_centred_img(img):
        return img - np.mean(img)

    @staticmethod
    def __scale_point(point, scale_matrix):
        assert scale_matrix.shape == (4, 4), 'Transformation matrix is expected to have shape (4, 4)'
        aux_aug = np.ones((4,))
        aux_aug[:3] = point
        return np.matmul(scale_matrix, aux_aug)[:1]

    def save_metrics(self, dest_folder=None):
        if dest_folder is None:
            dest_folder = self.__output_folder
        self.__metrics_df.to_csv(os.path.join(dest_folder, 'metrics.csv'))
        self.__metrics_df.to_latex(os.path.join(dest_folder, 'table.txt'), sparsify=True)
        print('Metrics saved in: ' + os.path.join(dest_folder))

    def print_summary(self):
        print(self.__metrics_df[['MSE', 'NCC', 'SSIM',
                                 'DICE_PAR', 'DICE_TUM', 'DICE_VES',
                                 'HD95_PAR', 'HD95_TUM', 'HD95_VES', 'TRE']].describe())


def resize_img_to_original_space(img, bb, first_reshape, original_shape, clip_img=False, flow=False):
    first_reshape = first_reshape.astype(int)
    bb = bb.astype(int)
    original_shape = original_shape.astype(int)

    if flow:
        # Multiply before resizing to reduce the number of multiplications
        img = _rescale_flow_values(img, bb, img.shape, first_reshape, original_shape)

    min_i, min_j, min_k, bb_i, bb_j, bb_k = bb
    max_i = min_i + bb_i
    max_j = min_j + bb_j
    max_k = min_k + bb_k

    img_bb = resize(img, (bb_i, bb_j, bb_k))   # Get the original bounding box shape

    # Place the bounding box again in the cubic volume
    img_copy = np.zeros((*first_reshape, img.shape[-1]) if len(img.shape) > 3 else first_reshape)  # Get channels if any
    img_copy[min_i:max_i, min_j:max_j, min_k:max_k, ...] = img_bb

    # Now resize to the original shape
    resized_img = resize(img_copy, original_shape, preserve_range=True, anti_aliasing=False)
    if clip_img or flow:
        # clip_mask = np.zeros(img_copy.shape[:3], np.int)
        # clip_mask[min_i:max_i, min_j:max_j, min_k:max_k] = 1
        # clip_mask = resize(clip_mask, original_shape, preserve_range=True, anti_aliasing=False)
        # clip_mask[clip_mask > 0.5] = 1
        # clip_mask[clip_mask < 1] = 0
        #
        # [min_i, min_j, min_k, max_i, max_j, max_k] = regionprops(label(clip_mask))[0].bbox
        #
        # resized_img = resized_img[min_i:max_i, min_j:max_j, min_k:max_k, ...]

        # Compute the coordinates of the boundix box in the upsampled volume, instead of resizing a mask image
        S = resize_transformation(img.shape, bb=None, first_reshape=first_reshape, original_shape=original_shape, translate=True)
        bb_coords = np.asarray([[min_i, min_j, min_k], [max_i, max_j, max_k]])
        bb_coords = np.hstack([bb_coords, np.ones((2, 1))])

        upsamp_bbox_coords = np.around(np.matmul(S, bb_coords.T)[:-1, :].T).astype(np.int)
        min_i = upsamp_bbox_coords[0][0]
        min_j = upsamp_bbox_coords[0][1]
        min_k = upsamp_bbox_coords[0][2]
        max_i = upsamp_bbox_coords[1][0]
        max_j = upsamp_bbox_coords[1][1]
        max_k = upsamp_bbox_coords[1][2]
        resized_img = resized_img[min_i:max_i, min_j:max_j, min_k:max_k, ...]

        if flow:
            # Return also the origin of the bb in the resized volume for the following interpolation
            return resized_img, np.asarray([min_i, min_j, min_k])
    return resized_img
    # This is supposed to be an isotropic image with voxel size 1 mm


def _rescale_flow_values(flow, bb, current_img_shape, first_reshape, original_shape):
    S = resize_transformation(current_img_shape, bb, first_reshape, original_shape, translate=False)

    [si, sj, sk] = np.diag(S[:3, :3])
    flow[..., 0] *= si
    flow[..., 1] *= sj
    flow[..., 2] *= sk

    return flow


def resize_pts_to_original_space(pt, bb, current_img_shape, first_reshape, original_shape):
    T = resize_transformation(current_img_shape, bb, first_reshape, original_shape)
    if len(pt.shape) > 1:
        pt_aug = np.ones((4, pt.shape[0]))
        pt_aug[0:3, :] = pt.T
    else:
        pt_aug = np.ones((4,))
        pt_aug[0:3] = pt
    trf_pt = np.matmul(T, pt_aug)[:-1, ...].T

    return trf_pt


def resize_transformation(current_img_shape, bb=None, first_reshape=None, original_shape=None, translate=True):
    first_reshape = first_reshape.astype(int)
    original_shape = original_shape.astype(int)

    first_resize_trf = np.eye(4)
    if bb is not None:
        bb = bb.astype(int)
        min_i, min_j, min_k, bb_i, bb_j, bb_k = bb
        np.fill_diagonal(first_resize_trf, [bb_i / current_img_shape[0], bb_j / current_img_shape[1], bb_k / current_img_shape[2], 1])
        if translate:
            first_resize_trf[:3, -1] = np.asarray([min_i, min_j, min_k])

    original_resize_trf = np.eye(4)
    np.fill_diagonal(original_resize_trf, [original_shape[0] / first_reshape[0], original_shape[1] / first_reshape[1], original_shape[2] / first_reshape[2], 1])

    return np.matmul(original_resize_trf, first_resize_trf)

