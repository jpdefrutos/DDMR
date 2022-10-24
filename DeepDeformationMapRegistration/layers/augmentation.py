import os, sys

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)  # PYTHON > 3.3 does not allow relative referencing

PYCHARM_EXEC = os.getenv('PYCHARM_EXEC') == 'True'

import tensorflow.keras.layers as kl
import tensorflow as tf
from tensorflow.python.framework.errors import InvalidArgumentError

from DeepDeformationMapRegistration.utils.operators import soft_threshold, gaussian_kernel, sample_unique
import DeepDeformationMapRegistration.utils.constants as C
from DeepDeformationMapRegistration.utils.thin_plate_splines import ThinPlateSplines
from voxelmorph.tf.layers import SpatialTransformer


class AugmentationLayer(kl.Layer):
    def __init__(self,
                 max_deformation,
                 max_displacement,
                 max_rotation,
                 num_control_points,
                 in_img_shape,
                 out_img_shape,
                 num_augmentations=1,
                 gamma_augmentation=True,
                 brightness_augmentation=True,
                 only_image=False,
                 only_resize=True,
                 return_displacement_map=False,
                 **kwargs):
        super(AugmentationLayer, self).__init__(**kwargs)

        self.max_deformation = max_deformation
        self.max_displacement = max_displacement
        self.max_rotation = max_rotation
        self.num_control_points = num_control_points
        self.num_augmentations = num_augmentations
        self.in_img_shape = in_img_shape
        self.out_img_shape = out_img_shape
        self.only_image = only_image
        self.return_disp_map = return_displacement_map

        self.do_gamma_augm = gamma_augmentation
        self.do_brightness_augm = brightness_augmentation

        grid = C.CoordinatesGrid()
        grid.set_coords_grid(in_img_shape, [C.TPS_NUM_CTRL_PTS_PER_AXIS] * 3)
        self.control_grid = tf.identity(grid.grid_flat(), name='control_grid')
        self.target_grid = tf.identity(grid.grid_flat(), name='target_grid')

        grid.set_coords_grid(in_img_shape, in_img_shape)
        self.fine_grid = tf.identity(grid.grid_flat(), 'fine_grid')

        if out_img_shape is not None:
            self.downsample_factor = [i // o for o, i in zip(out_img_shape, in_img_shape)]
            self.img_gauss_filter = gaussian_kernel(3, 0.001, 1, 1, 3)
            # self.resize_transf = tf.diag([*self.downsample_factor, 1])[:-1, :]
            # self.resize_transf = tf.expand_dims(tf.reshape(self.resize_transf, [-1]), 0, name='resize_transformation')  # ST expects a (12,) vector

        self.augment = not only_resize

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()
        img_shape = (input_shape[0], *self.out_img_shape, 1)
        seg_shape = (input_shape[0], *self.out_img_shape, input_shape[-1] - 1)
        disp_shape = (input_shape[0], *self.out_img_shape, 3)
        # Expect the input to have the image and segmentations in the same tensor
        if self.return_disp_map:
            return (img_shape, img_shape, seg_shape, seg_shape, disp_shape)
        else:
            return (img_shape, img_shape, seg_shape, seg_shape)

    #@tf.custom_gradient
    def call(self, in_data, training=None):
        # def custom_grad(in_grad):
        #     return tf.ones_like(in_grad)
        if training is not None:
            self.augment = training
        return self.build_batch(in_data)# , custom_grad

    def build_batch(self, fix_data: tf.Tensor):
        if len(fix_data.get_shape().as_list()) < 5:
            fix_data = tf.expand_dims(fix_data, axis=0)  # Add Batch dimension
        # fix_data = tf.tile(fix_data, (self.num_augmentations, *(1,)*4))
        fix_img_batch, mov_img_batch, fix_seg_batch, mov_seg_batch, disp_map = tf.map_fn(lambda x: self.augment_sample(x),
                                                                                         fix_data,
                                                                                         dtype=(tf.float32, tf.float32, tf.float32, tf.float32, tf.float32))
        # map_fn unstacks elems on axis 0
        if self.return_disp_map:
            return fix_img_batch, mov_img_batch, fix_seg_batch, mov_seg_batch, disp_map
        else:
            return fix_img_batch, mov_img_batch, fix_seg_batch, mov_seg_batch

    def augment_sample(self, fix_data: tf.Tensor):
        if self.only_image or not self.augment:
            fix_img = fix_data
            fix_segm = tf.zeros_like(fix_data, dtype=tf.float32)
        else:
            fix_img = fix_data[..., 0]
            fix_img = tf.expand_dims(fix_img, -1)
            fix_segm = fix_data[..., 1:]        # We expect several segmentation masks

        if self.augment:
            # If we are training, do the full-fledged augmentation
            fix_img = self.min_max_normalization(fix_img)

            mov_img, mov_segm, disp_map = self.deform_image(tf.squeeze(fix_img), fix_segm)
            mov_img = tf.expand_dims(mov_img, -1)  # Add the removed channel axis

        # Resample to output_shape
            if self.out_img_shape is not None:
                fix_img = self.downsize_image(fix_img)
                mov_img = self.downsize_image(mov_img)

                fix_segm = self.downsize_segmentation(fix_segm)
                mov_segm = self.downsize_segmentation(mov_segm)

                disp_map = self.downsize_displacement_map(disp_map)

            if self.do_gamma_augm:
                fix_img = self.gamma_augmentation(fix_img)
                mov_img = self.gamma_augmentation(mov_img)

            if self.do_brightness_augm:
                fix_img = self.brightness_augmentation(fix_img)
                mov_img = self.brightness_augmentation(mov_img)

        else:
            # During inference, just resize the input images
            mov_img = tf.zeros_like(fix_img)
            mov_segm = tf.zeros_like(fix_segm)

            disp_map = tf.tile(tf.zeros_like(fix_img), [1, 1, 1, 1, 3])  # TODO: change, don't use tile!!

            if self.out_img_shape is not None:
                fix_img = self.downsize_image(fix_img)
                mov_img = self.downsize_image(mov_img)

                fix_segm = self.downsize_segmentation(fix_segm)
                mov_segm = self.downsize_segmentation(mov_segm)

                disp_map = self.downsize_displacement_map(disp_map)

        fix_img = self.min_max_normalization(fix_img)
        mov_img = self.min_max_normalization(mov_img)
        return fix_img, mov_img, fix_segm, mov_segm, disp_map

    def downsize_image(self, img):
        img = tf.expand_dims(img, axis=0)
        # The filter is symmetrical along the three axes, hence there is no need for transposing the H and D dims
        img = tf.nn.conv3d(img, self.img_gauss_filter, strides=[1, ] * 5, padding='SAME', data_format='NDHWC')
        img = tf.layers.MaxPooling3D([1]*3, self.downsample_factor, padding='valid', data_format='channels_last')(img)

        return tf.squeeze(img, axis=0)

    def downsize_segmentation(self, segm):
        segm = tf.expand_dims(segm, axis=0)
        segm = tf.layers.MaxPooling3D([1]*3, self.downsample_factor, padding='valid', data_format='channels_last')(segm)

        segm = tf.cast(segm, tf.float32)
        return tf.squeeze(segm, axis=0)

    def downsize_displacement_map(self, disp_map):
        disp_map = tf.expand_dims(disp_map, axis=0)
        # The filter is symmetrical along the three axes, hence there is no need for transposing the H and D dims
        disp_map = tf.layers.AveragePooling3D([1]*3, self.downsample_factor, padding='valid', data_format='channels_last')(disp_map)

        # self.downsample_factor = in_shape / out_shape, but here we need out_shape / in_shape. Hence, 1 / factor
        if self.downsample_factor[0] != self.downsample_factor[1] != self.downsample_factor[2]:
            # Downsize the displacement magnitude along the different axes
            disp_map_x = disp_map[..., 0] * 1 / self.downsample_factor[0]
            disp_map_y = disp_map[..., 1] * 1 / self.downsample_factor[1]
            disp_map_z = disp_map[..., 2] * 1 / self.downsample_factor[2]

            disp_map = tf.stack([disp_map_x, disp_map_y, disp_map_z], axis=-1)
        else:
            disp_map = disp_map * 1 / self.downsample_factor[0]

        return tf.squeeze(disp_map, axis=0)

    def gamma_augmentation(self, in_img: tf.Tensor):
        in_img += 1e-5  # To prevent NaNs
        f = tf.random.uniform((), -1, 1, tf.float32)  # gamma [0.5, 2]
        gamma = tf.pow(2.0, f)

        return tf.clip_by_value(tf.pow(in_img, gamma), 0, 1)

    def brightness_augmentation(self, in_img: tf.Tensor):
        c = tf.random.uniform((), -0.2, 0.2, tf.float32)  # 20% shift
        return tf.clip_by_value(c + in_img, 0, 1)

    def min_max_normalization(self, in_img: tf.Tensor):
        return tf.div(tf.subtract(in_img, tf.reduce_min(in_img)),
                      tf.subtract(tf.reduce_max(in_img), tf.reduce_min(in_img)))

    def deform_image(self, fix_img: tf.Tensor, fix_segm: tf.Tensor):
        # Get locations where the intensity > 0.0
        idx_points_in_label = tf.where(tf.greater(fix_img, 0.0))

        # Randomly select N points
        # random_idx = tf.random.uniform((self.num_control_points,),
        #                                          minval=0, maxval=tf.shape(idx_points_in_label)[0],
        #                                          dtype=tf.int32)
        #
        # disp_location = tf.gather(idx_points_in_label, random_idx)  # And get the coordinates
        # disp_location = tf.cast(disp_location, tf.float32)
        disp_location = sample_unique(idx_points_in_label, self.num_control_points, tf.float32)

        # Get the coordinates of the control point displaces
        rand_disp = tf.random.uniform((self.num_control_points, 3), minval=-1, maxval=1, dtype=tf.float32) * self.max_deformation
        warped_location = disp_location + rand_disp

        # Add the selected locations to the control grid and the warped locations to the target grid
        control_grid = tf.concat([self.control_grid, disp_location], axis=0)
        trg_grid = tf.concat([self.control_grid, warped_location], axis=0)

        # Apply global transformation
        valid_trf = False
        while not valid_trf:
            trg_grid, aff = self.global_transformation(trg_grid)

            # Interpolate the displacement map
            try:
                tps = ThinPlateSplines(control_grid, trg_grid)
                def_grid = tps.interpolate(self.fine_grid)
            except InvalidArgumentError as err:
                # If the transformation raises a non-invertible error,
                # try again until we get a valid transformation
                tf.print('TPS non invertible matrix', output_stream=sys.stdout)
                continue
            else:
                valid_trf = True

        disp_map = self.fine_grid - def_grid
        disp_map = tf.reshape(disp_map, (*self.in_img_shape, -1))

        # Apply the displacement map
        fix_img = tf.expand_dims(tf.expand_dims(fix_img, -1), 0)
        fix_segm = tf.expand_dims(fix_segm, 0)
        disp_map = tf.cast(tf.expand_dims(disp_map, 0), tf.float32)

        mov_img = SpatialTransformer(interp_method='linear', indexing='ij', single_transform=False)([fix_img, disp_map])
        mov_segm = SpatialTransformer(interp_method='nearest', indexing='ij', single_transform=False)([fix_segm, disp_map])

        mov_img = tf.where(tf.is_nan(mov_img), tf.zeros_like(mov_img), mov_img)
        mov_img = tf.where(tf.is_inf(mov_img), tf.zeros_like(mov_img), mov_img)

        mov_segm = tf.where(tf.is_nan(mov_segm), tf.zeros_like(mov_segm), mov_segm)
        mov_segm = tf.where(tf.is_inf(mov_segm), tf.zeros_like(mov_segm), mov_segm)

        return tf.squeeze(mov_img), tf.squeeze(mov_segm, axis=0), tf.squeeze(disp_map, axis=0)

    def global_transformation(self, points: tf.Tensor):
        axis = tf.random.uniform((), 0, 3)

        alpha = C.DEG_TO_RAD * tf.cond(tf.logical_and(tf.greater(axis, 0.), tf.less_equal(axis, 1.)),
                        lambda: tf.random.uniform((), -self.max_rotation, self.max_rotation),
                        lambda: tf.zeros((), tf.float32))
        beta = C.DEG_TO_RAD * tf.cond(tf.logical_and(tf.greater(axis, 1.), tf.less_equal(axis, 2.)),
                       lambda: tf.random.uniform((), -self.max_rotation, self.max_rotation),
                       lambda: tf.zeros((), tf.float32))
        gamma = C.DEG_TO_RAD * tf.cond(tf.logical_and(tf.greater(axis, 2.), tf.less_equal(axis, 3.)),
                        lambda: tf.random.uniform((), -self.max_rotation, self.max_rotation),
                        lambda: tf.zeros((), tf.float32))

        ti = tf.random.uniform((), minval=-1, maxval=1, dtype=tf.float32) * self.max_displacement
        tj = tf.random.uniform((), minval=-1, maxval=1, dtype=tf.float32) * self.max_displacement
        tk = tf.random.uniform((), minval=-1, maxval=1, dtype=tf.float32) * self.max_displacement

        M = self.build_affine_transformation(tf.convert_to_tensor(self.in_img_shape, tf.float32),
                                             alpha, beta, gamma, ti, tj, tk)

        points = tf.transpose(points)
        new_pts = tf.matmul(M[:3, :3], points)
        new_pts = tf.expand_dims(M[:3, -1], -1) + new_pts
        return tf.transpose(new_pts), M

    @staticmethod
    def build_affine_transformation(img_shape, alpha, beta, gamma, ti, tj, tk):
        img_centre = tf.divide(img_shape, 2.)

        # Rotation matrix around the image centre
        # R* = T(p) R(ang) T(-p)
        # tf.cos and tf.sin expect radians

        T = tf.convert_to_tensor([[1, 0, 0, ti],
                                  [0, 1, 0, tj],
                                  [0, 0, 1, tk],
                                  [0, 0, 0, 1]], tf.float32)

        Ri = tf.convert_to_tensor([[1, 0, 0, 0],
                                   [0, tf.math.cos(alpha), -tf.math.sin(alpha), 0],
                                   [0, tf.math.sin(alpha),  tf.math.cos(alpha), 0],
                                   [0, 0, 0, 1]], tf.float32)

        Rj = tf.convert_to_tensor([[ tf.math.cos(beta), 0, tf.math.sin(beta), 0],
                                   [0, 1, 0, 0],
                                   [-tf.math.sin(beta), 0, tf.math.cos(beta), 0],
                                   [0, 0, 0, 1]], tf.float32)

        Rk = tf.convert_to_tensor([[tf.math.cos(gamma), -tf.math.sin(gamma), 0, 0],
                                   [tf.math.sin(gamma),  tf.math.cos(gamma), 0, 0],
                                   [0, 0, 1, 0],
                                   [0, 0, 0, 1]], tf.float32)

        R = tf.matmul(tf.matmul(Ri, Rj), Rk)

        Tc = tf.convert_to_tensor([[1, 0, 0, img_centre[0]],
                                   [0, 1, 0, img_centre[1]],
                                   [0, 0, 1, img_centre[2]],
                                   [0, 0, 0, 1]], tf.float32)

        Tc_ = tf.convert_to_tensor([[1, 0, 0, -img_centre[0]],
                                    [0, 1, 0, -img_centre[1]],
                                    [0, 0, 1, -img_centre[2]],
                                    [0, 0, 0, 1]], tf.float32)

        return tf.matmul(T, tf.matmul(Tc, tf.matmul(R, Tc_)))

    def get_config(self):
        config = super(AugmentationLayer, self).get_config()
        return config



