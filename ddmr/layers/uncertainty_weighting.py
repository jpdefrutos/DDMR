import os, sys

# currentdir = os.path.dirname(os.path.realpath(__file__))
# parentdir = os.path.dirname(currentdir)
# sys.path.append(parentdir)  # PYTHON > 3.3 does not allow relative referencing
#
# PYCHARM_EXEC = os.getenv('PYCHARM_EXEC') == 'True'

import tensorflow.keras.layers as kl
import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np
import random

from ddmr.utils.operators import soft_threshold, gaussian_kernel, sample_unique
import ddmr.utils.constants as C
from ddmr.utils.thin_plate_splines import ThinPlateSplines
from voxelmorph.tf.layers import SpatialTransformer
from neurite.tf.utils import resize
#from cupyx.scipy.ndimage import zoom
#import cupy


class UncertaintyWeighting(kl.Layer):
    def __init__(self, num_loss_fns=1, num_reg_fns=0, loss_fns: list = [tf.keras.losses.mean_squared_error],
                 reg_fns: list = list(), prior_loss_w=[1.], manual_loss_w=[1.], prior_reg_w=[1.], manual_reg_w=[1.],
                 **kwargs):
        assert isinstance(loss_fns, list) and (num_loss_fns == len(loss_fns) or len(loss_fns) == 1)
        assert isinstance(reg_fns, list) and (num_reg_fns == len(reg_fns))
        self.num_loss = num_loss_fns
        if len(loss_fns) == 1 and self.num_loss > 1:
            self.loss_fns = loss_fns * self.num_loss
        else:
            self.loss_fns = loss_fns

        if len(prior_loss_w) == 1:
            self.prior_loss_w = prior_loss_w * num_loss_fns
        else:
            self.prior_loss_w = prior_loss_w
        self.prior_loss_w = np.log(self.prior_loss_w)

        if len(manual_loss_w) == 1:
            self.manual_loss_w = manual_loss_w * num_loss_fns
        else:
            self.manual_loss_w = manual_loss_w

        self.num_reg = num_reg_fns
        if self.num_reg != 0:
            if len(reg_fns) == 1 and self.num_reg > 1:
                self.reg_fns = reg_fns * self.num_reg
            else:
                self.reg_fns = reg_fns

            self.is_placeholder = True
            if self.num_reg != 0:
                if len(prior_reg_w) == 1:
                    self.prior_reg_w = prior_reg_w * num_reg_fns
                else:
                    self.prior_reg_w = prior_reg_w
                self.prior_reg_w = np.log(self.prior_reg_w)

                if len(manual_reg_w) == 1:
                    self.manual_reg_w = manual_reg_w * num_reg_fns
                else:
                    self.manual_reg_w = manual_reg_w

        else:
            self.prior_reg_w = list()
            self.manual_reg_w = list()

        super(UncertaintyWeighting, self).__init__(**kwargs)

    def build(self, input_shape=None):
        self.log_loss_vars = self.add_weight(name='loss_log_vars', shape=(self.num_loss,),
                                             initializer=tf.keras.initializers.Constant(self.prior_loss_w),
                                             trainable=True)
        self.loss_weights = tf.math.softmax(self.log_loss_vars, name='SM_loss_weights')

        if self.num_reg != 0:
            self.log_reg_vars = self.add_weight(name='loss_reg_vars', shape=(self.num_reg,),
                                                initializer=tf.keras.initializers.Constant(self.prior_reg_w),
                                                trainable=True)
            if self.num_reg == 1:
                self.reg_weights = tf.math.exp(self.log_reg_vars, name='EXP_reg_weights')
            else:
                self.reg_weights = tf.math.softmax(self.log_reg_vars, name='SM_reg_weights')

        super(UncertaintyWeighting, self).build(input_shape)

    def multi_loss(self, ys_true, ys_pred, regs_true, regs_pred):
        loss_values = list()
        loss_names_loss = list()
        loss_names_reg = list()

        for y_true, y_pred, loss_fn, man_w in zip(ys_true, ys_pred, self.loss_fns, self.manual_loss_w):
            loss_values.append(tf.keras.backend.mean(man_w * loss_fn(y_true, y_pred)))
            loss_names_loss.append(loss_fn.__name__)

        loss_values = tf.convert_to_tensor(loss_values, dtype=tf.float32, name="step_loss_values")
        loss = tf.math.multiply(self.loss_weights, loss_values, name='step_weighted_loss')

        if self.num_reg != 0:
            loss_reg = list()
            for reg_true, reg_pred, reg_fn, man_w in zip(regs_true, regs_pred, self.reg_fns, self.manual_reg_w):
                loss_reg.append(K.mean(man_w * reg_fn(reg_true, reg_pred)))
                loss_names_reg.append(reg_fn.__name__)

            reg_values = tf.convert_to_tensor(loss_reg, dtype=tf.float32, name="step_reg_values")
            loss = loss + tf.math.multiply(self.reg_weights, reg_values, name='step_weighted_reg')

        for i, loss_name in enumerate(loss_names_loss):
            self.add_metric(tf.slice(self.loss_weights, [i], [1]), name='LOSS_WEIGHT_{}_{}'.format(i, loss_name),
                            aggregation='mean')
            self.add_metric(tf.slice(loss_values, [i], [1]), name='LOSS_VALUE_{}_{}'.format(i, loss_name),
                            aggregation='mean')
        if self.num_reg != 0:
            for i, loss_name in enumerate(loss_names_reg):
                self.add_metric(tf.slice(self.reg_weights, [i], [1]), name='REG_WEIGHT_{}_{}'.format(i, loss_name),
                                aggregation='mean')
                self.add_metric(tf.slice(reg_values, [i], [1]), name='REG_VALUE_{}_{}'.format(i, loss_name),
                                aggregation='mean')

        return K.sum(loss)

    def call(self, inputs):
        ys_true = inputs[:self.num_loss]
        ys_pred = inputs[self.num_loss:self.num_loss*2]
        reg_true = inputs[-self.num_reg*2:-self.num_reg]
        reg_pred = inputs[-self.num_reg:]             # The last terms are the regularization ones which have no GT
        loss = self.multi_loss(ys_true, ys_pred, reg_true, reg_pred)
        self.add_loss(loss, inputs=inputs)
        # We won't actually use the output, but we need something for the TF graph
        return K.concatenate(inputs, -1)

    def get_config(self):
        base_config = super(UncertaintyWeighting, self).get_config()
        base_config['num_loss_fns'] = self.num_loss
        base_config['num_reg_fns'] = self.num_reg

        return base_config


class UncertaintyWeightingWithRollingAverage(kl.Layer):
    def __init__(self, num_loss_fns=1, num_reg_fns=0, loss_fns: list = [tf.keras.losses.mean_squared_error],
                 reg_fns: list = list(), prior_loss_w=[1.], manual_loss_w=[1.], prior_reg_w=[1.], manual_reg_w=[1.],
                 roll_avg_reference=0,  # position in loss_fns of the reference loss function for the rolling avg
                 **kwargs):
        assert isinstance(loss_fns, list) and (num_loss_fns == len(loss_fns) or len(loss_fns) == 1)
        assert isinstance(reg_fns, list) and (num_reg_fns == len(reg_fns))
        # Rolling average attributes
        self.ref_loss = roll_avg_reference
        self.compute_roll_avg = False    # Toogle between computing the average of the losses or updating a know average
        self.scale_factor = [1.] * num_loss_fns
        self.n = 0  # Number of viewed samples
        self.temp_storage = [0.] * num_loss_fns

        self.num_loss = num_loss_fns
        if len(loss_fns) == 1 and self.num_loss > 1:
            self.loss_fns = loss_fns * self.num_loss
        else:
            self.loss_fns = loss_fns

        if len(prior_loss_w) == 1:
            self.prior_loss_w = prior_loss_w * num_loss_fns
        else:
            self.prior_loss_w = prior_loss_w
        self.prior_loss_w = np.log(self.prior_loss_w)

        if len(manual_loss_w) == 1:
            self.manual_loss_w = manual_loss_w * num_loss_fns
        else:
            self.manual_loss_w = manual_loss_w

        self.num_reg = num_reg_fns
        if self.num_reg != 0:
            if len(reg_fns) == 1 and self.num_reg > 1:
                self.reg_fns = reg_fns * self.num_reg
            else:
                self.reg_fns = reg_fns

            self.is_placeholder = True
            if self.num_reg != 0:
                if len(prior_reg_w) == 1:
                    self.prior_reg_w = prior_reg_w * num_reg_fns
                else:
                    self.prior_reg_w = prior_reg_w
                self.prior_reg_w = np.log(self.prior_reg_w)

                if len(manual_reg_w) == 1:
                    self.manual_reg_w = manual_reg_w * num_reg_fns
                else:
                    self.manual_reg_w = manual_reg_w

        else:
            self.prior_reg_w = list()
            self.manual_reg_w = list()

        super(UncertaintyWeightingWithRollingAverage, self).__init__(**kwargs)

    def build(self, input_shape=None):
        self.log_loss_vars = self.add_weight(name='loss_log_vars', shape=(self.num_loss,),
                                             initializer=tf.keras.initializers.Constant(self.prior_loss_w),
                                             trainable=True)
        self.loss_weights = tf.math.softmax(self.log_loss_vars, name='SM_loss_weights')

        if self.num_reg != 0:
            self.log_reg_vars = self.add_weight(name='loss_reg_vars', shape=(self.num_reg,),
                                                initializer=tf.keras.initializers.Constant(self.prior_reg_w),
                                                trainable=True)
            if self.num_reg == 1:
                self.reg_weights = tf.math.exp(self.log_reg_vars, name='EXP_reg_weights')
            else:
                self.reg_weights = tf.math.softmax(self.log_reg_vars, name='SM_reg_weights')

        super(UncertaintyWeightingWithRollingAverage, self).build(input_shape)

    def store_values(self, new_loss_values):
        for i, (t, v) in enumerate(zip(self.temp_storage, new_loss_values)):
            self.temp_storage[i] = t + v
        self.n += 1

    def compute_scale_factors(self):
        for i, val in enumerate(self.temp_storage):
            self.scale_factor[i] = self.n / val  # 1/avg

        self.scale_factor[self.ref_loss] = 1.

        self.temp_storage = [0.] * self.num_loss
        self.n = 0

    @property
    def ref_on_epoch_end_function(self):
        return self.compute_scale_factors

    def multi_loss(self, ys_true, ys_pred, regs_true, regs_pred):
        loss_values = list()
        loss_names_loss = list()
        loss_names_reg = list()

        for y_true, y_pred, loss_fn, man_w, sf in zip(ys_true, ys_pred, self.loss_fns, self.manual_loss_w, self.scale_factor):
            loss_values.append(sf * tf.keras.backend.mean(man_w * loss_fn(y_true, y_pred)))
            loss_names_loss.append(loss_fn.__name__)

        self.store_values(loss_values)
        loss_values = tf.convert_to_tensor(loss_values, dtype=tf.float32, name="step_loss_values")
        loss = tf.math.multiply(self.loss_weights, loss_values, name='step_weighted_loss')

        if self.num_reg != 0:
            loss_reg = list()
            for reg_true, reg_pred, reg_fn, man_w in zip(regs_true, regs_pred, self.reg_fns, self.manual_reg_w):
                loss_reg.append(K.mean(man_w * reg_fn(reg_true, reg_pred)))
                loss_names_reg.append(reg_fn.__name__)

            reg_values = tf.convert_to_tensor(loss_reg, dtype=tf.float32, name="step_reg_values")
            loss = loss + tf.math.multiply(self.reg_weights, reg_values, name='step_weighted_reg')

        for i, loss_name in enumerate(loss_names_loss):
            self.add_metric(tf.slice(self.loss_weights, [i], [1]), name='LOSS_WEIGHT_{}_{}'.format(i, loss_name),
                            aggregation='mean')
            self.add_metric(tf.slice(loss_values, [i], [1]), name='LOSS_VALUE_{}_{}'.format(i, loss_name),
                            aggregation='mean')
        if self.num_reg != 0:
            for i, loss_name in enumerate(loss_names_reg):
                self.add_metric(tf.slice(self.reg_weights, [i], [1]), name='REG_WEIGHT_{}_{}'.format(i, loss_name),
                                aggregation='mean')
                self.add_metric(tf.slice(reg_values, [i], [1]), name='REG_VALUE_{}_{}'.format(i, loss_name),
                                aggregation='mean')
                sc_tf = tf.convert_to_tensor(self.scale_factor, dtype=tf.float32, name='scale_factors_tf')
                self.add_metric(tf.slice(sc_tf, [i], [1]), name='SCALE_FACTOR_{}_{}'.format(i, loss_name),
                                aggregation='mean')

        return K.sum(loss)

    def call(self, inputs):
        ys_true = inputs[:self.num_loss]
        ys_pred = inputs[self.num_loss:self.num_loss*2]
        reg_true = inputs[-self.num_reg*2:-self.num_reg]
        reg_pred = inputs[-self.num_reg:]             # The last terms are the regularization ones which have no GT
        loss = self.multi_loss(ys_true, ys_pred, reg_true, reg_pred)
        self.add_loss(loss, inputs=inputs)
        # We won't actually use the output, but we need something for the TF graph
        return K.concatenate(inputs, -1)

    def get_config(self):
        base_config = super(UncertaintyWeighting, self).get_config()
        base_config['num_loss_fns'] = self.num_loss
        base_config['num_reg_fns'] = self.num_reg

        return base_config


def distance_map(coord1, coord2, dist, img_shape_w_channel=(64, 64, 1)):
    max_dist = np.max(img_shape_w_channel)
    dm_p = np.ones(img_shape_w_channel, np.float32)*max_dist
    dm_n = np.ones(img_shape_w_channel, np.float32)*max_dist

    for c1, c2, d in zip(coord1, coord2, dist):
        dm_p[c1, c2, 0] = d if dm_p[c1, c2, 0] > d else dm_p[c1, c2]
        d_n = 64. - max_dist
        dm_n[c1, c2, 0] = d_n if dm_n[c1, c2, 0] > d_n else dm_n[c1, c2]

    return dm_p/max_dist, dm_n/max_dist


def volume_to_ov_and_dm(in_volume: tf.Tensor):
    # This one is run as a preprocessing step
    def get_ov_projections_and_dm(volume):
        # tf.sign returns -1, 0, 1 depending on the sign of the elements of the input (negative, zero, positive)
        i, j, k, c = tf.where(volume > 0.0)
        top = tf.sign(tf.reduce_sum(volume, axis=0), name='ov_top')
        right = tf.sign(tf.reduce_sum(volume, axis=1), name='ov_right')
        front = tf.sign(tf.reduce_sum(volume, axis=2), name='ov_front')

        top_p, top_n = tf.py_func(distance_map, [j, k, i], tf.float32)
        right_p, right_n = tf.py_func(distance_map, [i, k, j], tf.float32)
        front_p, front_n = tf.py_func(distance_map, [i, j, k], tf.float32)

        return [front, right, top], [front_p, front_n, top_p, top_n, right_p, right_n]

    if len(in_volume.shape.as_list()) > 4:
        return tf.map_fn(get_ov_projections_and_dm, in_volume, [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32])
    else:
        return get_ov_projections_and_dm(in_volume)


def ov_and_dm_to_volume(ov_projections):
    front, right, top = ov_projections

    def get_volume(front: tf.Tensor, right: tf.Tensor, top: tf.Tensor):
        front_shape = front.shape.as_list()  # Assume (H, W, C)
        top_shape = top.shape.as_list()

        front_vol = tf.tile(tf.expand_dims(front, 2), [1, 1, top_shape[0], 1])
        right_vol = tf.tile(tf.expand_dims(right, 1), [1, front_shape[1], 1, 1])
        top_vol = tf.tile(tf.expand_dims(top, 0), [front_shape[0], 1, 1, 1])
        sum = tf.add(tf.add(front_vol, right_vol), top_vol)
        return soft_threshold(sum, 2., 'get_volume')

    if len(front.shape.as_list()) > 3:
        return tf.map_fn(lambda x: get_volume(x[0], x[1], x[2]), ov_projections, tf.float32)
    else:
        return get_volume(front, right, top)

# TODO: Recovering the coordinates from the distance maps to prevent artifacts
#   will the gradients be backpropagated??!?!!?!?!


