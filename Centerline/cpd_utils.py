from pycpd import DeformableRegistration, RigidRegistration
import numpy as np
import time
from scipy.interpolate import Rbf
import warnings

def cpd_non_rigid_transform_pt(pt, Y, G, W):
    from scipy.interpolate import LinearNDInterpolator
    interp = LinearNDInterpolator(points=Y, values=np.dot(G, W), fill_value=0.)
    return interp(pt)


def radial_basis_function(pts, vals, function='thin-plate'):
    # The Rbf function does not handle n-D hyper-surfaces, so we need an interpolator per displacements. Actually it does mode='N-D'
    pts_unique, idxs = np.unique(pts, return_index=True, axis=0)  # Prevent singular matrices
    ill_conditioned = False
    with warnings.catch_warnings(record=True) as caught_warns:
        warnings.simplefilter('always')
        dx = Rbf(pts_unique[:, 0], pts_unique[:, 1], pts_unique[:, 2], vals[idxs][:, 0], function=function)
        dy = Rbf(pts_unique[:, 0], pts_unique[:, 1], pts_unique[:, 2], vals[idxs][:, 1], function=function)
        dz = Rbf(pts_unique[:, 0], pts_unique[:, 1], pts_unique[:, 2], vals[idxs][:, 2], function=function)
        for w in caught_warns:
            print(w)
            ill_conditioned = ill_conditioned or 'ill-conditioned matrix' in str(w).lower()
    return lambda int_pt: np.asarray([dx(*int_pt), dy(*int_pt), dz(*int_pt)]), ill_conditioned


def deform_registration(fix_pts, mov_pts, callback_fnc=None, time_it=False, max_iterations=100, tolerance=1e-8, alpha=None, beta=None):
    deform_reg = DeformableRegistration(**{'Y': mov_pts, 'X': fix_pts},
                                        alpha=alpha, beta=beta, tolerance=tolerance, max_iterations=max_iterations)
    start_t = time.time()
    trf_mov_pts, deform_p = deform_reg.register(callback_fnc)
    end_t = time.time()
    if time_it:
        return end_t - start_t, deform_reg
    else:
        return trf_mov_pts, deform_p, deform_reg


def rigid_registration(fix_pts, mov_pts, callback_fnc=None, time_it=False):
    rigid_reg = RigidRegistration(**{'Y': mov_pts, 'X': fix_pts})
    start_t = time.time()
    trf_mov_pts, trf_p = rigid_reg.register(callback_fnc)
    end_t = time.time()
    if time_it:
        return end_t - start_t, rigid_reg
    else:
        return trf_mov_pts, trf_p, rigid_reg
