import numpy as np
import tensorflow as tf


class ThinPlateSplines:
    def __init__(self, ctrl_pts: tf.Tensor, target_pts: tf.Tensor, reg=0.0):
        """

        :param ctrl_pts: [N, d] tensor of control d-dimensional points
        :param target_pts: [N, d] tensor of target d-dimensional points
        :param reg: regularization coefficient
        """
        self.__ctrl_pts = ctrl_pts
        self.__target_pts = target_pts
        self.__reg = reg
        self.__num_ctrl_pts = ctrl_pts.shape[0]
        self.__dim = ctrl_pts.shape[1]

        self.__compute_coeffs()
        # self.__aff_params = self.__coeffs[self.__num_ctrl_pts:, ...]   # Affine  parameters of the TPS
        self.__non_aff_paramms = self.__coeffs[:self.__num_ctrl_pts, ...]      # Non-affine parameters of  he TPS

    def __compute_coeffs(self):
        target_pts_aug = tf.concat([self.__target_pts,
                                    tf.zeros([self.__dim + 1, self.__dim], dtype=self.__target_pts.dtype)],
                                   axis=0)

        # T = self.__make_T()
        T_i = tf.cast(tf.linalg.inv(self.__make_T()), target_pts_aug.dtype)
        self.__coeffs = tf.cast(tf.matmul(T_i, target_pts_aug), tf.float32)

    def __make_T(self):
        # cp: [K x 2] control points
        # T: [(num_pts+dim+1) x (num_pts+dim+1)]
        num_pts = self.__ctrl_pts.shape[0]

        P = tf.concat([tf.ones([self.__num_ctrl_pts, 1], dtype=tf.float32), self.__ctrl_pts], axis=1)
        zeros = np.zeros([self.__dim + 1, self.__dim + 1], dtype=np.float)
        self.__K = self.__U_dist(self.__ctrl_pts)
        alfa = tf.reduce_mean(self.__K)

        self.__K = self.__K + tf.ones_like(self.__K) * tf.pow(alfa, 2) * self.__reg

        # top = tf.concat([self.__K, P], axis=1)
        # bottom = tf.concat([tf.transpose(P), zeros], axis=1)

        return tf.concat([tf.concat([self.__K, P], axis=1), tf.concat([tf.transpose(P), zeros], axis=1)], axis=0)

    def __U_dist(self, ctrl_pts, int_pts=None):
        if int_pts is None:
            dist = self.__pairwise_distance_equal(ctrl_pts)  # Already squared!
        else:
            dist = self.__pairwise_distance_different(ctrl_pts, int_pts)  # Already squared!


        # U(x, y) = p_w_dist(x, y)^2 * log(p_w_dist(x, y)) (dist() > =0); 0 otw
        if ctrl_pts.shape[-1] == 2:
            u_dist = dist * tf.math.log(dist + 1e-6)
        else:
            # Src: https://github.com/vaipatel/morphops/blob/master/morphops/tps.py
            # In particular, if k = 2, then  U(r) = r^2 * log(r^2), else U(r) = r
            u_dist = tf.sqrt(dist)
        # tf.matrix_set_diag(u_dist, tf.constant(0, dtype=dist_sq.dtype))
        # reg_term = self.__reg * tf.pow(alfa, 2) * tf.eye(self.__num_ctrl_pts)

        return u_dist # + reg_term

    def __pairwise_distance_sq(self, pts_a, pts_b):
        with tf.variable_scope('pairwise_distance'):
            if np.all(pts_a == pts_b):
                # This implementation works better when doing the pairwise distance os a single set of points
                pts_a_ = tf.reshape(pts_a, [-1, 1, 3])
                pts_b_ = tf.reshape(pts_b, [1, -1, 3])
                dist = tf.reduce_sum(tf.square(pts_a_ - pts_b_), 2)   # squared pairwise distance
            else:
                # PwD^2= A_norm^2 - 2*A*B' + B_norm^2
                pts_a_ = tf.reduce_sum(tf.square(pts_a), 1)
                pts_b_ = tf.reduce_sum(tf.square(pts_b), 1)

                pts_a_ = tf.expand_dims(pts_a_, 1)
                pts_b_ = tf.expand_dims(pts_b_, 0)

                pts_a_pts_b_ = tf.matmul(pts_a, pts_b, adjoint_b=True)

                dist = pts_a_ - 2 * pts_a_pts_b_ + pts_b_

            return tf.cast(dist, tf.float32)

    @staticmethod
    def __pairwise_distance_equal(pts):
        # This implementation works better when doing the pairwise distance os a single set of points
        dist = tf.reduce_sum(tf.square(tf.reshape(pts, [-1, 1, 3]) - tf.reshape(pts, [1, -1, 3])), 2)  # squared pairwise distance
        return tf.cast(dist, tf.float32)

    @staticmethod
    def __pairwise_distance_different(pts_a, pts_b):
        pts_a_ = tf.reduce_sum(tf.square(pts_a), 1)
        pts_b_ = tf.reduce_sum(tf.square(pts_b), 1)

        pts_a_ = tf.expand_dims(pts_a_, 1)
        pts_b_ = tf.expand_dims(pts_b_, 0)

        pts_a_pts_b_ = tf.matmul(pts_a, pts_b, adjoint_b=True)

        dist = pts_a_ - 2 * pts_a_pts_b_ + pts_b_
        return tf.cast(dist, tf.float32)

    def __lift_pts(self, int_pts: tf.Tensor, num_pts):
        # int_pts: [N x 2], input points
        # cp: [K x 2], control points
        # pLift: [N x (3+K)], lifted input points

        # u_dist = self.__U_dist(int_pts, self.__ctrl_pts)

        int_pts_lift = tf.concat([self.__U_dist(int_pts, self.__ctrl_pts),
                                 tf.ones([num_pts, 1], dtype=tf.float32),
                                 int_pts], axis=1)
        return int_pts_lift

    @property
    def bending_energy(self):
        aux = tf.matmul(self.__non_aff_paramms, self.__K, transpose_a=True)
        return tf.matmul(aux, self.__non_aff_paramms)

    def interpolate(self, int_points): #, num_pts):
        """

        :param int_points: [K, d] flattened d-points of a mesh
        :return:
        """
        num_pts = tf.shape(int_points)[0]
        int_points_lift = self.__lift_pts(int_points, num_pts)

        return tf.matmul(int_points_lift, self.__coeffs)

    def __call__(self, int_points, num_pts, **kwargs):
        return self.interpolate(int_points) # , num_pts)


def thin_plate_splines_batch(ctrl_pts: tf.Tensor, target_pts: tf.Tensor, int_pts: tf.Tensor, reg=0.0):
    _batches = ctrl_pts.shape[0]

    if tf.get_default_session() is not None:
        print('DEBUG TIME')

    def tps_sample(in_data):
        cp, tp, ip = in_data
        # _num_pts = ip.shape[0]
        tps = ThinPlateSplines(cp, tp, reg)
        interp = tps.interpolate(ip) # , _num_pts)
        return interp

    return tf.map_fn(tps_sample, elems=(ctrl_pts, target_pts, int_pts), dtype=tf.float32)

