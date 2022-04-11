import numpy as np
from scipy.spatial.distance import pdist, cdist, squareform
from sklearn.metrics import pairwise_distances

class ThinPlateSplines:
    def __init__(self, ctrl_pts: np.ndarray, target_pts: np.ndarray, reg=0.0):
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

        self.__K = None
        self.__compute_coeffs()
        self.__aff_params = self.__coeffs[self.__num_ctrl_pts:, ...]   # Affine  parameters of the TPS
        self.__non_aff_paramms = self.__coeffs[:self.__num_ctrl_pts, ...]      # Non-affine parameters of  he TPS

    def __compute_coeffs(self):
        target_pts_aug = np.vstack([self.__target_pts,
                                    np.zeros([self.__dim + 1, self.__dim])]).astype(self.__target_pts.dtype)

        T_i = np.linalg.inv(self.__make_T()).astype(self.__target_pts.dtype)
        self.__coeffs = np.matmul(T_i, target_pts_aug).astype(self.__target_pts.dtype)

    def __make_T(self):
        # cp: [K x 2] control points
        # T: [(K+3) x (K+3)]
        P = np.hstack([np.ones([self.__num_ctrl_pts, 1], dtype=np.float), self.__ctrl_pts])
        zeros = np.zeros([self.__dim + 1, self.__dim + 1], dtype=np.float)
        self.__K = self.__U_dist(self.__ctrl_pts)
        alfa = np.mean(self.__K)

        self.__K = self.__K + np.ones_like(self.__K) * np.power(alfa, 2) * self.__reg

        top = np.hstack([P, self.__K])
        bottom = np.hstack([P.transpose(), zeros])

        return np.vstack([top, bottom])

    def __U_dist(self, ctrl_pts, int_pts=None):
        dist = pairwise_distances(ctrl_pts, int_pts)

        if ctrl_pts.shape[-1] == 2:
            u_dist = np.square(dist) * np.log(dist + 1e-6)
        else:
            u_dist = np.sqrt(dist)

        return u_dist

    def __lift_pts(self, int_pts: np.ndarray, num_pts):
        # int_pts: [N x 2], input points
        # cp: [K x 2], control points
        # pLift: [N x (3+K)], lifted input points

        int_pts_lift = np.hstack([self.__U_dist(self.__ctrl_pts, int_pts),
                                  np.ones([num_pts, 1], dtype=np.float),
                                  int_pts])
        return int_pts_lift

    def _get_coefficients(self):
        return self.__coeffs

    def interpolate(self, int_points):
        """

        :param int_points: [K, d] flattened d-points of a mesh
        :return:
        """
        num_pts = int_points.shape[0]
        int_points_lift = self.__lift_pts(int_points, num_pts)
        return np.dot(int_points_lift, self.__coeffs)

    @property
    def bending_energy(self):
        aux = tf.matmul(self.__non_aff_paramms, self.__K, transpose_a=True)
        return tf.matmul(aux, self.__non_aff_paramms)
