# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import tensorflow as tf
import numpy as np
import pyVesselRegistration_constants as const
from skimage.exposure import rescale_intensity
import scipy.misc as scpmisc
import os

THRES = 0.9

# COLOR MAPS
chunks = np.linspace(0, 1, 10)
cmap1 = plt.get_cmap('hsv', 4)
# cmaplist = [cmap1(i) for i in range(cmap1.N)]
cmaplist = [(1, 1, 1, 1), (0, 0, 1, 1), (230 / 255, 97 / 255, 1 / 255, 1), (128 / 255, 0 / 255, 32 / 255, 1)]
cmaplist[0] = (1, 1, 1, 1.0)
cmap1 = mcolors.LinearSegmentedColormap.from_list('custom', cmaplist, cmap1.N)

colors = [(0, 0, 1, i) for i in chunks]
cmap2 = mcolors.LinearSegmentedColormap.from_list('mycmap', colors, N=100)

colors = [(230 / 255, 97 / 255, 1 / 255, i) for i in chunks]
cmap3 = mcolors.LinearSegmentedColormap.from_list('mycmap', colors, N=100)

colors = [(128 / 255, 0 / 255, 32 / 255, i) for i in chunks]
cmap4 = mcolors.LinearSegmentedColormap.from_list('mycmap', colors, N=100)

cmap_bin = cm.get_cmap('viridis', 3)  # viridis is the default colormap


def view_centerline_sample(sample: np.ndarray, dimensionality: int, ax=None, c=None, name=None):
    if dimensionality == 2:
        _plot_2d(sample, ax, c, name=name)
    elif dimensionality == 3:
        _plot_3d(sample, ax, c, name=name)
    else:
        raise ValueError('Invalid valud for dimensionality. Expected int 2 or 3')


def matrix_to_orthographicProjection(matrix: np.ndarray, ret_list=False):
    """ Given a 3D matrix, returns the three orthographic projections: top, front, right.
    Top corresponds to dimensions 1 and 2
    Front corresponds to dimensions 0 and 1
    Right corresponds to dimensions 0 and 2

    :param matrix: 3D matrix
    :param ret_list: return a list instead of an array (optional)
    :return: list or array with the three views [top, front, right]
    """
    top = _getProjection(matrix, dim=0)  # YZ
    front = _getProjection(matrix, dim=2)  # XY
    right = _getProjection(matrix, dim=1)  # XZ

    if ret_list:
        return top, front, right
    else:
        return np.asarray([top, front, right])


def _getProjection(matrix: np.ndarray, dim: int):
    orth_view = matrix.sum(axis=dim, dtype=float)
    orth_view = orth_view > 0.0
    orth_view.astype(np.float)

    return orth_view


def orthographicProjection_to_matrix(top: np.ndarray, front: np.ndarray, right: np.ndarray):
    """ Given the three orthographic projections, it returns a 3D-view of the object based on back projection

    :param top: 2D view top view
    :param front: 2D front view
    :param right: 2D right view
    :return: matrix with the 3D-view
    """
    top_mat = np.tile(top, (front.shape[0], 1, 1))
    front_mat = np.tile(top, (right.shape[1], 1, 1))
    right_mat = np.tile(top, (top.shape[0], 1, 1))

    reconstruction = np.zeros((front.shape[0], right.shape[1], top.shape[0]))
    iter = np.nditer([top_mat, front_mat, right_mat, reconstruction], flags=['multi_index'], op_flags=['readwrite'])
    while not iter.finished:
        if iter[0] and iter[1] and iter[2]:
            iter[3] = 1
        iter.iternext()

    return reconstruction


def _plot_2d(sample: np.ndarray, ax=None, c=None, name=None):
    if isinstance(sample, tf.Tensor):
        sample = sample.eval(session=tf.Session())

    x_range = list()
    y_range = list()
    marker_size = list()
    for idx, val in np.ndenumerate(sample):
        if val >= THRES:
            x_range.append(idx[0])
            y_range.append(idx[1])
            marker_size.append(val ** 2)

    if not ax:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    if c:
        ax.scatter(x_range, y_range, c=c, s=marker_size)
    else:
        ax.scatter(x_range, y_range, s=marker_size)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    if name:
        ax.set_title(name)

    return ax


def _plot_3d(sample: np.ndarray, ax=None, c=None, name=None):
    from mpl_toolkits.mplot3d import Axes3D
    if isinstance(sample, tf.Tensor):
        sample = sample.eval(session=tf.Session())

    x_range = list()
    y_range = list()
    z_range = list()
    marker_size = list()
    for idx, val in np.ndenumerate(sample):
        if val >= THRES:
            x_range.append(idx[0])
            y_range.append(idx[1])
            z_range.append(idx[2])
            marker_size.append(val ** 2)

    print('Found ', len(x_range), ' points')
    # x_range = np.linspace(start=0, stop=sample.shape[0], num=sample.shape[0])
    # y_range = np.linspace(start=0, stop=sample.shape[1], num=sample.shape[1])
    # z_range = np.linspace(start=0, stop=sample.shape[2], num=sample.shape[2])
    #
    # sample_flat = sample.flatten(order='C')

    if len(x_range):
        if not ax:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

        if c:
            ax.scatter(x_range, y_range, z_range, c=c, s=marker_size)
        else:
            ax.scatter(x_range, y_range, z_range, s=marker_size)
        # ax.scatter(x_range, y_range, z_range, s=marker_size)#, c=sample_flat)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        if name:
            ax.set_title(name)

        return ax
    else:
        print('Nothing to plot')
        return None


def plot_training(list_imgs: [np.ndarray], affine_transf=True, filename='img', fig=None):
    if fig is not None:
        fig.clear()
        plt.figure(fig.number)
    else:
        fig = plt.figure(dpi=const.DPI)

    ax_fix = fig.add_subplot(231)
    im_fix = ax_fix.imshow(list_imgs[0][:, :, 0])
    ax_fix.set_title('Fix image', fontsize=const.FONT_SIZE)
    ax_fix.tick_params(axis='both',
                       which='both',
                       bottom=False,
                       left=False,
                       labelleft=False,
                       labelbottom=False)
    ax_mov = fig.add_subplot(232)
    im_mov = ax_mov.imshow(list_imgs[1][:, :, 0])
    ax_mov.set_title('Moving image', fontsize=const.FONT_SIZE)
    ax_mov.tick_params(axis='both',
                       which='both',
                       bottom=False,
                       left=False,
                       labelleft=False,
                       labelbottom=False)

    ax_pred_im = fig.add_subplot(233)
    im_pred_im = ax_pred_im.imshow(list_imgs[2][:, :, 0])
    ax_pred_im.set_title('Prediction', fontsize=const.FONT_SIZE)
    ax_pred_im.tick_params(axis='both',
                       which='both',
                       bottom=False,
                       left=False,
                       labelleft=False,
                       labelbottom=False)

    ax_pred_disp = fig.add_subplot(234)
    if affine_transf:
        fake_bg = np.array([[0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0]])

        bottom = np.asarray([0, 0, 0, 1])

        transf_mat = np.reshape(list_imgs[3], (2, 3))
        transf_mat = np.stack([transf_mat, bottom], axis=0)

        im_pred_disp = ax_pred_disp.imshow(fake_bg)
        for i in range(4):
            for j in range(4):
                ax_pred_disp.text(i, j, transf_mat[i, j], ha="center", va="center", color="b")

        ax_pred_disp.set_title('Affine transformation matrix')

    else:
        cx, cy, dx, dy, s = _prepare_quiver_map(list_imgs[3])
        im_pred_disp = ax_pred_disp.imshow(s, interpolation='none', aspect='equal')
        ax_pred_disp.quiver(cx, cy, dx, dy, scale=const.QUIVER_PARAMS.arrow_scale)
        ax_pred_disp.set_title('Pred disp map', fontsize=const.FONT_SIZE)
    ax_pred_disp.tick_params(axis='both',
                               which='both',
                               bottom=False,
                               left=False,
                               labelleft=False,
                               labelbottom=False)

    ax_gt_disp = fig.add_subplot(235)
    if affine_transf:
        fake_bg = np.array([[0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0]])

        bottom = np.asarray([0, 0, 0, 1])

        transf_mat = np.reshape(list_imgs[4], (2, 3))
        transf_mat = np.stack([transf_mat, bottom], axis=0)

        im_gt_disp = ax_pred_disp.imshow(fake_bg)
        for i in range(4):
            for j in range(4):
                ax_pred_disp.text(i, j, transf_mat[i, j], ha="center", va="center", color="b")

        ax_pred_disp.set_title('Affine transformation matrix')

    else:
        cx, cy, dx, dy, s = _prepare_quiver_map(list_imgs[4])
        im_gt_disp = ax_gt_disp.imshow(s, interpolation='none', aspect='equal')
        ax_gt_disp.quiver(cx, cy, dx, dy, scale=const.QUIVER_PARAMS.arrow_scale)
        ax_gt_disp.set_title('GT disp map', fontsize=const.FONT_SIZE)
    ax_gt_disp.tick_params(axis='both',
                       which='both',
                       bottom=False,
                       left=False,
                       labelleft=False,
                       labelbottom=False)

    cb_fix = _set_colorbar(fig, ax_fix, im_fix, False)
    cb_mov = _set_colorbar(fig, ax_mov, im_mov, False)
    cb_pred = _set_colorbar(fig, ax_pred_im, im_pred_im, False)
    cb_pred_disp = _set_colorbar(fig, ax_pred_disp, im_pred_disp, False)
    cd_gt_disp = _set_colorbar(fig, ax_gt_disp, im_gt_disp, False)

    if filename is not None:
        plt.savefig(filename, format='png')  # Call before show
    if not const.REMOTE:
        plt.show()
    else:
        plt.close()
    return fig


def save_centreline_img(img, title, filename, fig=None):
    if fig is not None:
        fig.clear()
        plt.figure(fig.number)
    else:
        fig = plt.figure(dpi=const.DPI)

    dim = len(img.shape[:-1])

    if dim == 2:
        ax = fig.add_subplot(111)
        fig.suptitle(title)
        im = ax.imshow(img[..., 0], cmap=cmap_bin)
        ax.tick_params(axis='both',
                       which='both',
                       bottom=False,
                       left=False,
                       labelleft=False,
                       labelbottom=False)

        #cb = _set_colorbar(fig, ax, im, False)
    else:
        ax = fig.add_subplot(111, projection='3d')
        fig.suptitle(title)
        im = ax.voxels(img[0, ..., 0] > 0.0)
        _square_3d_plot(np.arange(0, 63), np.arange(0, 63), np.arange(0, 63), ax)

        ax.tick_params(axis='both',
                       which='both',
                       bottom=False,
                       left=False,
                       labelleft=False,
                       labelbottom=False)

    plt.savefig(filename, format='png')
    plt.close()


def save_disp_map_img(disp_map, title, filename, affine_transf=False, fig=None):
    if fig is not None:
        fig.clear()
        plt.figure(fig.number)
    else:
        fig = plt.figure(dpi=const.DPI)

    dim = disp_map.shape[-1]

    if dim == 2:
        ax_x = fig.add_subplot(131)
        ax_x.set_title('H displacement')
        im_x = ax_x.imshow(disp_map[..., const.H_DISP])
        ax_x.tick_params(axis='both',
                       which='both',
                       bottom=False,
                       left=False,
                       labelleft=False,
                       labelbottom=False)
        cb_x = _set_colorbar(fig, ax_x, im_x, False)

        ax_y = fig.add_subplot(132)
        ax_y.set_title('W displacement')
        im_y = ax_y.imshow(disp_map[..., const.W_DISP])
        ax_y.tick_params(axis='both',
                         which='both',
                         bottom=False,
                         left=False,
                         labelleft=False,
                         labelbottom=False)
        cb_y = _set_colorbar(fig, ax_y, im_y, False)

        ax = fig.add_subplot(133)
        if affine_transf:
            fake_bg = np.array([[0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0]])

            bottom = np.asarray([0, 0, 0, 1])

            transf_mat = np.reshape(disp_map, (2, 3))
            transf_mat = np.stack([transf_mat, bottom], axis=0)

            im = ax.imshow(fake_bg)
            for i in range(4):
                for j in range(4):
                    ax.text(i, j, transf_mat[i, j], ha="center", va="center", color="b")

        else:
            c, d, s = _prepare_quiver_map(disp_map, dim=dim)
            im = ax.imshow(s, interpolation='none', aspect='equal')
            ax.quiver(c[const.H_DISP], c[const.W_DISP], d[const.H_DISP], d[const.W_DISP],
                      scale=const.QUIVER_PARAMS.arrow_scale)
            cb = _set_colorbar(fig, ax, im, False)
            ax.set_title('Displacement map')
        ax.tick_params(axis='both',
                       which='both',
                       bottom=False,
                       left=False,
                       labelleft=False,
                       labelbottom=False)
        fig.suptitle(title)
    else:
        ax = fig.add_subplot(111, projection='3d')
        c, d, s = _prepare_quiver_map(disp_map[0, ...], dim=dim)
        ax.quiver(c[const.H_DISP], c[const.W_DISP], c[const.D_DISP], d[const.H_DISP], d[const.W_DISP], d[const.D_DISP],
                  norm=True)
        _square_3d_plot(np.arange(0, 63), np.arange(0, 63), np.arange(0, 63), ax)
        fig.suptitle('Displacement map')
        ax.tick_params(axis='both',  # Same parameters as in 2D https://matplotlib.org/api/_as_gen/mpl_toolkits.mplot3d.axes3d.Axes3D.html
                       which='both',
                       bottom=False,
                       left=False,
                       labelleft=False,
                       labelbottom=False)
        fig.suptitle(title)

    plt.savefig(filename, format='png')
    plt.close()


def plot_training_and_validation(list_imgs: [np.ndarray], affine_transf=True, filename='img', fig=None,
                                 title_first_row='TRAINING', title_second_row='VALIDATION'):
    if fig is not None:
        fig.clear()
        plt.figure(fig.number)
    else:
        fig = plt.figure(dpi=const.DPI)

    dim = len(list_imgs[0].shape[:-1])

    if dim == 2:
        # TRAINING
        ax_input = fig.add_subplot(241)
        ax_input.set_ylabel(title_first_row, fontsize=const.FONT_SIZE)
        im_fix = ax_input.imshow(list_imgs[0][:, :, 0])
        ax_input.set_title('Fix image', fontsize=const.FONT_SIZE)
        ax_input.tick_params(axis='both',
                           which='both',
                           bottom=False,
                           left=False,
                           labelleft=False,
                           labelbottom=False)
        ax_mov = fig.add_subplot(242)
        im_mov = ax_mov.imshow(list_imgs[1][:, :, 0])
        ax_mov.set_title('Moving image', fontsize=const.FONT_SIZE)
        ax_mov.tick_params(axis='both',
                           which='both',
                           bottom=False,
                           left=False,
                           labelleft=False,
                           labelbottom=False)

        ax_pred_im = fig.add_subplot(244)
        im_pred_im = ax_pred_im.imshow(list_imgs[2][:, :, 0])
        ax_pred_im.set_title('Predicted fix image', fontsize=const.FONT_SIZE)
        ax_pred_im.tick_params(axis='both',
                           which='both',
                           bottom=False,
                           left=False,
                           labelleft=False,
                           labelbottom=False)

        ax_pred_disp = fig.add_subplot(243)
        if affine_transf:
            fake_bg = np.array([[0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0]])

            bottom = np.asarray([0, 0, 0, 1])

            transf_mat = np.reshape(list_imgs[3], (2, 3))
            transf_mat = np.stack([transf_mat, bottom], axis=0)

            im_pred_disp = ax_pred_disp.imshow(fake_bg)
            for i in range(4):
                for j in range(4):
                    ax_pred_disp.text(i, j, transf_mat[i, j], ha="center", va="center", color="b")

            ax_pred_disp.set_title('Affine transformation matrix')

        else:
            cx, cy, dx, dy, s = _prepare_quiver_map(list_imgs[3], dim=dim)
            im_pred_disp = ax_pred_disp.imshow(s, interpolation='none', aspect='equal')
            ax_pred_disp.quiver(cx, cy, dx, dy, scale=const.QUIVER_PARAMS.arrow_scale)
            ax_pred_disp.set_title('Pred disp map', fontsize=const.FONT_SIZE)
        ax_pred_disp.tick_params(axis='both',
                                   which='both',
                                   bottom=False,
                                   left=False,
                                   labelleft=False,
                                   labelbottom=False)

        # VALIDATION
        axinput_val = fig.add_subplot(245)
        axinput_val.set_ylabel(title_second_row, fontsize=const.FONT_SIZE)
        im_fix_val = axinput_val.imshow(list_imgs[4][:, :, 0])
        axinput_val.set_title('Fix image', fontsize=const.FONT_SIZE)
        axinput_val.tick_params(axis='both',
                           which='both',
                           bottom=False,
                           left=False,
                           labelleft=False,
                           labelbottom=False)
        ax_mov_val = fig.add_subplot(246)
        im_mov_val = ax_mov_val.imshow(list_imgs[5][:, :, 0])
        ax_mov_val.set_title('Moving image', fontsize=const.FONT_SIZE)
        ax_mov_val.tick_params(axis='both',
                           which='both',
                           bottom=False,
                           left=False,
                           labelleft=False,
                           labelbottom=False)

        ax_pred_im_val = fig.add_subplot(248)
        im_pred_im_val = ax_pred_im_val.imshow(list_imgs[6][:, :, 0])
        ax_pred_im_val.set_title('Predicted fix image', fontsize=const.FONT_SIZE)
        ax_pred_im_val.tick_params(axis='both',
                           which='both',
                           bottom=False,
                           left=False,
                           labelleft=False,
                           labelbottom=False)

        ax_pred_disp_val = fig.add_subplot(247)
        if affine_transf:
            fake_bg = np.array([[0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0]])

            bottom = np.asarray([0, 0, 0, 1])

            transf_mat = np.reshape(list_imgs[7], (2, 3))
            transf_mat = np.stack([transf_mat, bottom], axis=0)

            im_pred_disp_val = ax_pred_disp_val.imshow(fake_bg)
            for i in range(4):
                for j in range(4):
                    ax_pred_disp_val.text(i, j, transf_mat[i, j], ha="center", va="center", color="b")

            ax_pred_disp_val.set_title('Affine transformation matrix')

        else:
            c, d, s = _prepare_quiver_map(list_imgs[7], dim=dim)
            im_pred_disp_val = ax_pred_disp_val.imshow(s, interpolation='none', aspect='equal')
            ax_pred_disp_val.quiver(c[0], c[1], d[0], d[1], scale=const.QUIVER_PARAMS.arrow_scale)
            ax_pred_disp_val.set_title('Pred disp map', fontsize=const.FONT_SIZE)
        ax_pred_disp_val.tick_params(axis='both',
                                   which='both',
                                   bottom=False,
                                   left=False,
                                   labelleft=False,
                                   labelbottom=False)

        cb_fix = _set_colorbar(fig, ax_input, im_fix, False)
        cb_mov = _set_colorbar(fig, ax_mov, im_mov, False)
        cb_pred = _set_colorbar(fig, ax_pred_im, im_pred_im, False)
        cb_pred_disp = _set_colorbar(fig, ax_pred_disp, im_pred_disp, False)

        cd_fix_val = _set_colorbar(fig, axinput_val, im_fix_val, False)
        cb_mov_val = _set_colorbar(fig, ax_mov_val, im_mov_val, False)
        cb_pred_val = _set_colorbar(fig, ax_pred_im_val, im_pred_im_val, False)
        cb_pred_disp_val = _set_colorbar(fig, ax_pred_disp_val, im_pred_disp_val, False)

    else:
        # 3D
        # TRAINING
        ax_input = fig.add_subplot(231, projection='3d')
        ax_input.set_ylabel(title_first_row, fontsize=const.FONT_SIZE)
        im_fix = ax_input.voxels(list_imgs[0][..., 0] > 0.0, facecolors='red', edgecolors='red', label='Fixed')
        im_mov = ax_input.voxels(list_imgs[1][..., 0] > 0.0, facecolors='blue', edgecolors='blue', label='Moving')
        ax_input.set_title('Fix image', fontsize=const.FONT_SIZE)
        ax_input.tick_params(axis='both',
                           which='both',
                           bottom=False,
                           left=False,
                           labelleft=False,
                           labelbottom=False)

        ax_pred_im = fig.add_subplot(232, projection='3d')
        im_pred_im = ax_input.voxels(list_imgs[2][..., 0] > 0.0, facecolors='green', edgecolors='green', label='Prediction')
        im_fix = ax_input.voxels(list_imgs[0][..., 0] > 0.0, facecolors='red', edgecolors='red', label='Fixed')
        ax_pred_im.set_title('Predicted fix image', fontsize=const.FONT_SIZE)
        ax_pred_im.tick_params(axis='both',
                               which='both',
                               bottom=False,
                               left=False,
                               labelleft=False,
                               labelbottom=False)

        ax_pred_disp = fig.add_subplot(233, projection='3d')

        c, d, s = _prepare_quiver_map(list_imgs[3], dim=dim)
        im_pred_disp = ax_pred_disp.imshow(s, interpolation='none', aspect='equal')
        ax_pred_disp.quiver(c[const.H_DISP], c[const.W_DISP], c[const.D_DISP],
                            d[const.H_DISP], d[const.W_DISP], d[const.D_DISP], scale=const.QUIVER_PARAMS.arrow_scale)
        ax_pred_disp.set_title('Pred disp map', fontsize=const.FONT_SIZE)
        ax_pred_disp.tick_params(axis='both',
                                 which='both',
                                 bottom=False,
                                 left=False,
                                 labelleft=False,
                                 labelbottom=False)

        # VALIDATION
        axinput_val = fig.add_subplot(234, projection='3d')
        axinput_val.set_ylabel(title_second_row, fontsize=const.FONT_SIZE)
        im_fix_val = ax_input.voxels(list_imgs[4][..., 0] > 0.0, facecolors='red', edgecolors='red', label='Fixed (val)')
        im_mov_val = ax_input.voxels(list_imgs[5][..., 0] > 0.0, facecolors='blue', edgecolors='blue', label='Moving (val)')
        axinput_val.set_title('Fix image', fontsize=const.FONT_SIZE)
        axinput_val.tick_params(axis='both',
                               which='both',
                               bottom=False,
                               left=False,
                               labelleft=False,
                               labelbottom=False)

        ax_pred_im_val = fig.add_subplot(235, projection='3d')
        im_pred_im_val = ax_input.voxels(list_imgs[2][..., 0] > 0.0, facecolors='green', edgecolors='green', label='Prediction (val)')
        im_fix_val = ax_input.voxels(list_imgs[0][..., 0] > 0.0, facecolors='red', edgecolors='red', label='Fixed (val)')
        ax_pred_im_val.set_title('Predicted fix image', fontsize=const.FONT_SIZE)
        ax_pred_im_val.tick_params(axis='both',
                                   which='both',
                                   bottom=False,
                                   left=False,
                                   labelleft=False,
                                   labelbottom=False)

        ax_pred_disp_val = fig.add_subplot(236, projection='3d')
        c, d, s = _prepare_quiver_map(list_imgs[7], dim=dim)
        im_pred_disp_val = ax_pred_disp_val.imshow(s, interpolation='none', aspect='equal')
        ax_pred_disp_val.quiver(c[const.H_DISP], c[const.W_DISP], c[const.D_DISP],
                                d[const.H_DISP], d[const.W_DISP], d[const.D_DISP],
                                scale=const.QUIVER_PARAMS.arrow_scale)
        ax_pred_disp_val.set_title('Pred disp map', fontsize=const.FONT_SIZE)
        ax_pred_disp_val.tick_params(axis='both',
                                     which='both',
                                     bottom=False,
                                     left=False,
                                     labelleft=False,
                                     labelbottom=False)

    if filename is not None:
        plt.savefig(filename, format='png')  # Call before show
    if not const.REMOTE:
        plt.show()
    else:
        plt.close()
    return fig


def _set_colorbar(fig, ax, im, drawedges=True):
    div = make_axes_locatable(ax)
    im_cax = div.append_axes('right', size='5%', pad=0.05)
    im_cb = fig.colorbar(im, cax=im_cax, drawedges=drawedges, shrink=0.5, orientation='vertical')
    im_cb.ax.tick_params(labelsize=5)

    return im_cb


def _prepare_quiver_map(disp_map: np.ndarray, dim=2, spc=const.QUIVER_PARAMS.spacing):
    if isinstance(disp_map, tf.Tensor):
        if tf.executing_eagerly():
            disp_map = disp_map.numpy()
        else:
            disp_map = disp_map.eval()
    dx = disp_map[..., const.H_DISP]
    dy = disp_map[..., const.W_DISP]
    if dim > 2:
        dz = disp_map[..., const.D_DISP]

    img_size_x = disp_map.shape[const.H_DISP]
    img_size_y = disp_map.shape[const.W_DISP]
    if dim > 2:
        img_size_z = disp_map.shape[const.D_DISP]

    if dim > 2:
        s = np.sqrt(np.square(dx) + np.square(dy) + np.square(dz))
        s = np.reshape(s, [img_size_x, img_size_y, img_size_z])

        cx, cy, cz = np.meshgrid(list(range(0, img_size_x)), list(range(0, img_size_y)), list(range(0, img_size_z)),
                                 indexing='ij')
        c = [cx[::spc, ::spc, ::spc], cy[::spc, ::spc, ::spc], cz[::spc, ::spc, ::spc]]
        d = [dx[::spc, ::spc, ::spc], dy[::spc, ::spc, ::spc], dz[::spc, ::spc, ::spc]]
    else:
        s = np.sqrt(np.square(dx) + np.square(dy))
        s = np.reshape(s, [img_size_x, img_size_y])

        cx, cy = np.meshgrid(list(range(0, img_size_x)), list(range(0, img_size_y)))
        c = [cx[::spc, ::spc], cy[::spc, ::spc]]
        d = [dx[::spc, ::spc], dy[::spc, ::spc]]

    return c, d, s


def _prepare_colormap(disp_map: np.ndarray):
    if isinstance(disp_map, tf.Tensor):
        disp_map = disp_map.eval()
    dx = disp_map[:, :, 0]
    dy = disp_map[:, :, 1]

    mod_img = np.zeros_like(dx)

    for i in range(dx.shape[0]):
        for j in range(dx.shape[1]):
            vec = np.asarray([dx[i, j], dy[i, j]])
            mod_img[i, j] = np.linalg.norm(vec, ord=2)

    p_l, p_h = np.percentile(mod_img, (0, 100))
    mod_img = rescale_intensity(mod_img, in_range=(p_l, p_h), out_range=(0, 255))

    return mod_img


def plot_input_data(fix_img, mov_img, img_size=(64, 64), title=None, filename=None):
    num_samples = fix_img.shape[0]

    if num_samples != 16 and num_samples != 32:
        raise ValueError('Only batches of 16 or 32 samples!')

    fig, ax = plt.subplots(nrows=4 if num_samples == 16 else 8, ncols=4)
    ncol = 0
    nrow = 0
    black_col = np.ones([img_size[0], 0])
    for sample in range(num_samples):
        combined_img = np.hstack([fix_img[sample, :, :, 0], black_col, mov_img[sample, :, :, 0]])
        ax[nrow, ncol].imshow(combined_img, cmap='Greys')
        ax[nrow, ncol].set_ylabel('#{}'.format(sample))
        ax[nrow, ncol].tick_params(axis='both',
                                   which='both',
                                   bottom=False,
                                   left=False,
                                   labelleft=False,
                                   labelbottom=False)
        ncol += 1
        if ncol >= 4:
            ncol = 0
            nrow += 1

    if title is not None:
        fig.suptitle(title)

    if filename is not None:
        plt.savefig(filename, format='png')  # Call before show
    if not const.REMOTE:
        plt.show()
    else:
        plt.close()
    return fig


def plot_dataset_orthographic_views(view_sets: [[np.ndarray]]):
    """

    :param views_fix: Expected order: top, front, left
    :param views_mov: Expected order: top, front, left
    :return:
    """
    nrows = len(view_sets)
    fig, ax = plt.subplots(nrows=nrows, ncols=3)
    labels = ['top', 'front', 'left']
    for nrow in range(nrows):
        for ncol in range(3):
            if nrows == 1:
                ax[ncol].imshow(view_sets[nrow][ncol][:, :, 0])
                ax[ncol].set_title('Fix ' + labels[ncol])
                ax[ncol].tick_params(axis='both',
                                     which='both',
                                     bottom=False,
                                     left=False,
                                     labelleft=False,
                                     labelbottom=False)

            else:
                ax[nrow, ncol].imshow(view_sets[nrow][ncol][:, :, 0])
                ax[nrow, ncol].set_title('Fix ' + labels[ncol])
                ax[nrow, ncol].tick_params(axis='both',
                                           which='both',
                                           bottom=False,
                                           left=False,
                                           labelleft=False,
                                           labelbottom=False)

    plt.show()
    return fig


def plot_compare_2d_images(img1, img2, img1_name='img1', img2_name='img2'):
    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].imshow(img1[:, :, 0])
    ax[0].set_title(img1_name)
    ax[0].tick_params(axis='both',
                      which='both',
                      bottom=False,
                      left=False,
                      labelleft=False,
                      labelbottom=False)

    ax[1].imshow(img2[:, :, 0])
    ax[1].set_title(img2_name)
    ax[1].tick_params(axis='both',
                      which='both',
                      bottom=False,
                      left=False,
                      labelleft=False,
                      labelbottom=False)

    plt.show()
    return fig


def plot_dataset_3d(img_sets):
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for idx in range(len(img_sets)):
        ax = _plot_3d(img_sets[idx], ax=ax, name='Set {}'.format(idx))

    plt.show()
    return fig


def plot_predictions(fix_img_batch, mov_img_batch, disp_map_batch, pred_img_batch, filename='predictions', fig=None):
    num_rows = fix_img_batch.shape[0]
    img_size = fix_img_batch.shape[1:3]
    if fig is not None:
        fig.clear()
        plt.figure(fig.number)
        ax = fig.add_subplot(nrows=num_rows, ncols=4, dpi=const.DPI)
    else:
        fig, ax = plt.subplots(nrows=num_rows, ncols=4, dpi=const.DPI)

    for row in range(num_rows):
        fix_img = fix_img_batch[row, :, :, 0]
        mov_img = mov_img_batch[row, :, :, 0]
        disp_map = disp_map_batch[row, :, :, :]
        pred_img = pred_img_batch[row, :, :, 0]
        ax[row, 0].imshow(fix_img)
        ax[row, 0].tick_params(axis='both',
                               which='both',
                               bottom=False,
                               left=False,
                               labelleft=False,
                               labelbottom=False)
        ax[row, 1].imshow(mov_img)
        ax[row, 1].tick_params(axis='both',
                               which='both',
                               bottom=False,
                               left=False,
                               labelleft=False,
                               labelbottom=False)

        cx, cy, dx, dy, s = _prepare_quiver_map(disp_map)
        disp_map_color = _prepare_colormap(disp_map)
        ax[row, 2].imshow(disp_map_color, interpolation='none', aspect='equal')
        ax[row, 2].quiver(cx.eval(), cy.eval(), dx.eval(), dy.eval(), units='xy', scale=const.QUIVER_PARAMS.arrow_scale)
        ax[row, 2].figure.set_size_inches(img_size)
        ax[row, 2].tick_params(axis='both',
                               which='both',
                               bottom=False,
                               left=False,
                               labelleft=False,
                               labelbottom=False)

        ax[row, 3].tick_params(axis='both',
                               which='both',
                               bottom=False,
                               left=False,
                               labelleft=False,
                               labelbottom=False)

    plt.axis('off')
    ax[0, 0].set_title('Fixed img ($I_f$)', fontsize=const.FONT_SIZE)
    ax[0, 1].set_title('Moving img ($I_m$)', fontsize=const.FONT_SIZE)
    ax[0, 2].set_title('Displacement map ($\delta$)', fontsize=const.FONT_SIZE)
    ax[0, 3].set_title('Updated $I_m$', fontsize=const.FONT_SIZE)

    if filename is not None:
        plt.savefig(filename, format='png')  # Call before show
    if not const.REMOTE:
        plt.show()
    else:
        plt.close()
    return fig


def inspect_disp_map_generation(fix_img, mov_img, disp_map, filename=None, fig=None):
    if fig is not None:
        fig.clear()
        plt.figure(fig.number)
    else:
        fig = plt.figure(dpi=const.DPI)

    ax0 = fig.add_subplot(221)
    im0 = ax0.imshow(fix_img[..., 0])
    ax0.tick_params(axis='both',
                      which='both',
                      bottom=False,
                      left=False,
                      labelleft=False,
                      labelbottom=False)
    ax1 = fig.add_subplot(222)
    im1 = ax1.imshow(mov_img[..., 0])
    ax1.tick_params(axis='both',
                      which='both',
                      bottom=False,
                      left=False,
                      labelleft=False,
                      labelbottom=False)

    cx, cy, dx, dy, s = _prepare_quiver_map(disp_map)
    disp_map_color = _prepare_colormap(disp_map)
    ax2 = fig.add_subplot(223)
    im2 = ax2.imshow(s, interpolation='none', aspect='equal')

    ax2.quiver(cx, cy, dx, dy, scale=const.QUIVER_PARAMS.arrow_scale)
    # ax2.figure.set_size_inches(img_size)
    ax2.tick_params(axis='both',
                      which='both',
                      bottom=False,
                      left=False,
                      labelleft=False,
                      labelbottom=False)

    ax3 = fig.add_subplot(224)
    dif = fix_img[..., 0] - mov_img[..., 0]
    im3 = ax3.imshow(dif)
    ax3.quiver(cx, cy, dx, dy, scale=const.QUIVER_PARAMS.arrow_scale)
    ax3.tick_params(axis='both',
                    which='both',
                    bottom=False,
                    left=False,
                    labelleft=False,
                    labelbottom=False)

    plt.axis('off')
    ax0.set_title('Fixed img ($I_f$)', fontsize=const.FONT_SIZE)
    ax1.set_title('Moving img ($I_m$)', fontsize=const.FONT_SIZE)
    ax2.set_title('Displacement map', fontsize=const.FONT_SIZE)
    ax3.set_title('Fix - Mov', fontsize=const.FONT_SIZE)

    im0_cb = _set_colorbar(fig, ax0, im0, False)
    im1_cb = _set_colorbar(fig, ax1, im1, False)
    disp_cb = _set_colorbar(fig, ax2, im2, False)
    im3_cb = _set_colorbar(fig, ax3, im3, False)

    if filename is not None:
        plt.savefig(filename, format='png')  # Call before show
    if not const.REMOTE:
        plt.show()
    else:
        plt.close()

    return fig


def inspect_displacement_grid(ctrl_coords, dense_coords, target_coords, disp_coords, disp_map, mask, fix_img, mov_img,
                              filename=None, fig=None):
    if fig is not None:
        fig.clear()
        plt.figure(fig.number)
    else:
        fig = plt.figure()

    ax_grid = fig.add_subplot(231)
    ax_grid.set_title('Grids', fontsize=const.FONT_SIZE)
    ax_grid.scatter(ctrl_coords[:, 0], ctrl_coords[:, 1], marker='+', c='r', s=20)
    ax_grid.scatter(dense_coords[:, 0], dense_coords[:, 1], marker='.', c='r', s=1)
    ax_grid.tick_params(axis='both',
                        which='both',
                        bottom=False,
                        left=False,
                        labelleft=False,
                        labelbottom=False)

    ax_grid.scatter(target_coords[:, 0], target_coords[:, 1], marker='+', c='b', s=20)
    ax_grid.scatter(disp_coords[:, 0], disp_coords[:, 1], marker='.', c='b', s=1)

    ax_grid.set_aspect('equal')

    ax_disp = fig.add_subplot(232)
    ax_disp.set_title('Displacement map', fontsize=const.FONT_SIZE)
    cx, cy, dx, dy, s = _prepare_quiver_map(disp_map)
    ax_disp.imshow(s, interpolation='none', aspect='equal')
    ax_disp.tick_params(axis='both',
                        which='both',
                        bottom=False,
                        left=False,
                        labelleft=False,
                        labelbottom=False)

    ax_mask = fig.add_subplot(233)
    ax_mask.set_title('Mask', fontsize=const.FONT_SIZE)
    ax_mask.imshow(mask)
    ax_mask.tick_params(axis='both',
                        which='both',
                        bottom=False,
                        left=False,
                        labelleft=False,
                        labelbottom=False)

    ax_fix = fig.add_subplot(234)
    ax_fix.set_title('Fix image', fontsize=const.FONT_SIZE)
    ax_fix.imshow(fix_img[..., 0])
    ax_fix.tick_params(axis='both',
                        which='both',
                        bottom=False,
                        left=False,
                        labelleft=False,
                        labelbottom=False)

    ax_mov = fig.add_subplot(235)
    ax_mov.set_title('Moving image', fontsize=const.FONT_SIZE)
    ax_mov.imshow(mov_img[..., 0])
    ax_mov.tick_params(axis='both',
                        which='both',
                        bottom=False,
                        left=False,
                        labelleft=False,
                        labelbottom=False)

    ax_dif = fig.add_subplot(236)
    ax_dif.set_title('Fix - Moving image', fontsize=const.FONT_SIZE)
    ax_dif.imshow(fix_img[..., 0] - mov_img[..., 0], cmap=cmap_bin)
    ax_dif.tick_params(axis='both',
                        which='both',
                        bottom=False,
                        left=False,
                        labelleft=False,
                        labelbottom=False)
    legend_elems = [Line2D([0], [0], color=cmap_bin(0), lw=2),
                    Line2D([0], [0], color=cmap_bin(2), lw=2)]

    ax_dif.legend(legend_elems, ['Mov', 'Fix'], loc='upper left', bbox_to_anchor=(0., 0., 1., 0.),
                  ncol=2, mode='expand')

    if filename is not None:
        plt.savefig(filename, format='png')  # Call before show
    if not const.REMOTE:
        plt.show()

    return fig


def compare_disp_maps(disp_m_f, disp_f_m, fix_img, mov_img, filename=None, fig=None):
    if fig is not None:
        fig.clear()
        plt.figure(fig.number)
    else:
        fig = plt.figure()

    ax_d_m_f = fig.add_subplot(131)
    ax_d_m_f.set_title('Disp M->F', fontsize=const.FONT_SIZE)
    cx, cy, dx, dy, s = _prepare_quiver_map(disp_m_f)
    ax_d_m_f.imshow(s, interpolation='none', aspect='equal')
    ax_d_m_f.quiver(cx, cy, dx, dy, scale=const.QUIVER_PARAMS.arrow_scale)
    ax_d_m_f.tick_params(axis='both',
                        which='both',
                        bottom=False,
                        left=False,
                        labelleft=False,
                        labelbottom=False)

    ax_d_f_m = fig.add_subplot(132)
    ax_d_f_m.set_title('Disp F->M', fontsize=const.FONT_SIZE)
    cx, cy, dx, dy, s = _prepare_quiver_map(disp_f_m)
    ax_d_f_m.quiver(cx, cy, dx, dy, scale=const.QUIVER_PARAMS.arrow_scale)
    ax_d_f_m.imshow(s, interpolation='none', aspect='equal')
    ax_d_f_m.tick_params(axis='both',
                         which='both',
                         bottom=False,
                         left=False,
                         labelleft=False,
                         labelbottom=False)

    ax_dif = fig.add_subplot(133)
    ax_dif.set_title('Fix - Moving image', fontsize=const.FONT_SIZE)
    ax_dif.imshow(fix_img[..., 0] - mov_img[..., 0], cmap=cmap_bin)
    ax_dif.tick_params(axis='both',
                       which='both',
                       bottom=False,
                       left=False,
                       labelleft=False,
                       labelbottom=False)

    legend_elems = [Line2D([0], [0], color=cmap_bin(0), lw=2),
                    Line2D([0], [0], color=cmap_bin(2), lw=2)]

    ax_dif.legend(legend_elems, ['Mov', 'Fix'], loc='upper left', bbox_to_anchor=(0., 0., 1., 0.),
                  ncol=2, mode='expand')

    if filename is not None:
        plt.savefig(filename, format='png')  # Call before show
    if not const.REMOTE:
        plt.show()
    else:
        plt.close()

    return fig


def plot_train_step(list_imgs: [np.ndarray], fig_title='TRAINING', dest_folder='.', save_file=True):
    # list_imgs[0]: fix image
    # list_imgs[1]: moving image
    # list_imgs[2]: prediction scale 1
    # list_imgs[3]: prediction scale 2
    # list_imgs[4]: prediction scale 3
    # list_imgs[5]: disp map scale 1
    # list_imgs[6]: disp map scale 2
    # list_imgs[7]: disp map scale 3
    num_imgs = len(list_imgs)
    num_preds = (num_imgs - 2) // 2
    num_cols = num_preds + 1
    # 3D
    # TRAINING
    fig = plt.figure(figsize=(12.8, 10.24))
    fig.tight_layout(pad=5.0)
    ax = fig.add_subplot(2, num_cols, 1, projection='3d')
    ax.voxels(list_imgs[0][0, ..., 0] > 0.0, facecolors='red', edgecolors='red', label='Fixed')
    ax.set_title('Fix image', fontsize=const.FONT_SIZE)
    _square_3d_plot(np.arange(0, 63), np.arange(0, 63), np.arange(0, 63), ax)

    for i in range(2, num_preds+2):
        ax = fig.add_subplot(2, num_cols, i, projection='3d')
        ax.voxels(list_imgs[0][0, ..., 0] > 0.0, facecolors='red',  edgecolors='red', label='Fixed')
        ax.voxels(list_imgs[i][0, ..., 0] > 0.0, facecolors='green', edgecolors='green', label='Pred_{}'.format(i - 1))
        ax.set_title('Pred. #{}'.format(i - 1))
        _square_3d_plot(np.arange(0, 63), np.arange(0, 63), np.arange(0, 63), ax)

    ax = fig.add_subplot(2, num_cols, num_preds+2, projection='3d')
    ax.voxels(list_imgs[1][0, ..., 0] > 0.0, facecolors='blue', edgecolors='blue', label='Moving')
    ax.set_title('Fix image', fontsize=const.FONT_SIZE)
    _square_3d_plot(np.arange(0, 63), np.arange(0, 63), np.arange(0, 63), ax)

    for i in range(num_preds+2, 2 * num_preds + 2):
        ax = fig.add_subplot(2, num_cols, i + 1, projection='3d')
        c, d, s = _prepare_quiver_map(list_imgs[i][0, ...], dim=3)
        ax.quiver(c[const.H_DISP], c[const.W_DISP], c[const.D_DISP],
                  d[const.H_DISP], d[const.W_DISP], d[const.D_DISP],
                  norm=True)
        ax.set_title('Disp. #{}'.format(i - 5))
        _square_3d_plot(np.arange(0, 63), np.arange(0, 63), np.arange(0, 63), ax)

    fig.suptitle(fig_title, fontsize=const.FONT_SIZE)

    if save_file:
        plt.savefig(os.path.join(dest_folder, fig_title+'.png'), format='png')  # Call before show
    if not const.REMOTE:
        plt.show()
    else:
        plt.close()
    return fig


def _square_3d_plot(X, Y, Z, ax):
    max_range = np.array([X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]).max() / 2.0

    mid_x = (X.max() + X.min()) * 0.5
    mid_y = (Y.max() + Y.min()) * 0.5
    mid_z = (Z.max() + Z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

