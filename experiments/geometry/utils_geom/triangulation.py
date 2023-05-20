from typing import Union, Optional

import numpy as np
from numba import njit
import cv2


@njit('Tuple((f8[:,:,::1], f8[:,::1]))(f8[:,::1])', cache=True)
def jacobian_dprojection_dy(y):
    """Jacobian of projection function w.r.t. set of points being projected."""
    y_inv = 1.0 / y[2]
    y_inv_sq = y_inv * y_inv

    J = np.zeros((y.shape[1], 2, 3))
    J[:, 0, 0] = y_inv
    J[:, 1, 1] = y_inv
    J[:, 0, 2] = - y_inv_sq * y[0]
    J[:, 1, 2] = - y_inv_sq * y[1]

    u = y[:2] * y_inv

    return J, u


@njit('Tuple((f8[:,:,::1], f8[:,::1]))(f8[:,::1], f8[:,::1], f8[:,::1], f8[:,::1])', cache=True)
def jacobian_dprojection_dX(X, K, R, t):
    """ Jacobian of projection function w.r.t. 3D points

    Defining the projection function as:
        u = \pi(y) = proj(R X + t),
    the following Jacobian is computed:
        d u   d
        ─── = ─── proj(K (R X + t))
        d X   d X

    Args:
        X: (3, n) array of 3d coordinates.
        K: (3, 3) calibration matrix.
        R: (3, 3) rotation matrix.
        t: (3, 1) translation vector.

    Returns:
        J: (n, 2, 3) jacobian of each projection "u" w.r.t. each point in X.
        u: (2, n) projections on image plane.
    """
    KR = K.dot(R)
    Kt = K.dot(t)
    y = KR.dot(X) + Kt

    # dprojection / dy, and image plane projections.
    J, u = jacobian_dprojection_dy(y)

    # dprojection / dX, via chain rule.
    J[:, 0] = J[:, 0].dot(KR)
    J[:, 1] = J[:, 1].dot(KR)

    return J, u


@njit('Tuple((f8[:,:,::1], f8[::1,:]))(f8[:,::1], f8[:,::1], f8[:,::1], '
      'f8[:,::1], f8[:,::1], f8[:,::1], f8[:,::1], f8[:,::1])', cache=True)
def jacobians_and_residuals_triangulation(X, p1, p2, K, R1, t1, R2, t2):
    """ Jacobians and gold-standard residuals for triangulation of points.

    The Jacobian returned is a (n, 4, 3) array for each of the "n" 3D points
    being triangulated. The error being differenciated for each 3D point is
    the gold-standard (reprojection) error, for each of the two 2D features
    corresponding to each 3D point contained in X.

    Returns:
        J: (n, 4, 3) per-(3D)-point Jacobians.
        res: (n, 4) per-(3D)-point residuals.
    """
    J1, u1 = jacobian_dprojection_dX(X, K, R1, t1)
    J2, u2 = jacobian_dprojection_dX(X, K, R2, t2)

    J = np.concatenate((J1, J2), axis=1)
    res = np.concatenate(((u1 - p1).T, (u2 - p2).T), axis=1)

    return J, res


def triangulation_two_view_LM_optimization(
        X, p1, p2, K, R1, t1, R2, t2, sigmas1=None, sigmas2=None, niter=5):
    """LM optimization of 3D points, by minimizig reprojection errors

    Args:
        X: (3, n) 3D coordinates of the n points observed in both views.
        p1: (2, n) array of feature points in view 1.
        p2: (2, n) array of feature points in view 2.
        K: (3, 3) calibration matrix.
        distCoeffs: (n,) n distortion coeffs. given in OpenCV format.
        R1: (3, 3) rotation matrix for view 1.
        t1: (3, 1) translation vector for view 1.
        R2: (3, 3) rotation matrix for view 2.
        t2: (3, 1) translation vector for view 2.
        sigmas1: (n, 2, 2) 2d covariances of features in view 1.
        sigmas2: (n, 2, 2) 2d covariances of features in view 2.

    Returns:
        X_opt: (3, n) optimized 3d coordinates.
        X_cov: (n, 3, 3) 3d covariance of 3d point X.
    """
    eps = 1e-9

    # increase/ decrase factor of influence for the gradient.
    damping_factor_inc = 1e1
    damping_factor_dec = 1.0 / damping_factor_inc

    use_covs = sigmas1 is not None or sigmas2 is not None
    if use_covs:
        # verify that covariances for both views are given.
        assert sigmas1 is not None and sigmas2 is not None
        # confidence weights.
        sigmas1_inv = np.linalg.inv(sigmas1 + eps)
        sigmas2_inv = np.linalg.inv(sigmas2 + eps)

    # LM iterative optimization.
    X_current = X.copy()
    for i in range(niter):
        J, res = jacobians_and_residuals_triangulation(
            X_current, p1, p2, K, R1, t1, R2, t2)

        # approximate per-point Hessians.
        if use_covs:
            JTW = J.transpose(0, 2, 1).copy()
            JTW[:, :, :2] = JTW[:, :, :2] @ sigmas1_inv
            JTW[:, :, 2:] = JTW[:, :, 2:] @ sigmas2_inv
        else:
            JTW = J.transpose(0, 2, 1)
        JTWJ = JTW @ J

        # get writeable view of the per-point Hessians' diagonal.
        diags = np.einsum('ndd -> nd', JTWJ)

        # augment diagonals with damping terms (independent for each point).
        if i == 0:
            # damping initialization based on [Hartley, Zisserman, 2004].
            # 1e-3 * (trace(JTWJ) / d)
            dampings = 2.5e-4 * diags.sum(axis=1)
            # per-point cost.
            cost_prev = np.einsum('nd, nd -> n', res, res)
        else:
            # update dampings based on current cost.
            cost = np.einsum('nd, nd -> n', res, res)
            dampings = np.where(
                cost > cost_prev,
                dampings * damping_factor_inc,
                dampings * damping_factor_dec)
            cost_prev = cost
            if np.any(cost > cost_prev):
                raise ValueError

        # in-place augmentation of the Hessians' diagonals.
        diags += dampings[:, None] * diags

        # add step.
        X_current += np.linalg.solve(
            JTWJ, -(JTW @ res[..., None]).squeeze(axis=-1)).T

        if i == niter - 1:
            # 3d point covariance, taking into account 2d covariances.
            J, _ = jacobians_and_residuals_triangulation(
                X_current, p1, p2, K, R1, t1, R2, t2)
            if use_covs:
                JTW = J.transpose(0, 2, 1).copy()
                JTW[:, :, :2] = JTW[:, :, :2] @ sigmas1_inv
                JTW[:, :, 2:] = JTW[:, :, 2:] @ sigmas2_inv
            else:
                JTW = J.transpose(0, 2, 1)
            sigmas3d = np.linalg.inv(JTW @ J + eps)

    return X_current, sigmas3d


@njit('Tuple((f8[:,:,::1], f8[:,::1]))(f8[:,::1], f8[:,::1], f8[:,::1], '
      'f8[:,:,::1], f8[:,:,::1])', fastmath=True, cache=True)
def jacobian_and_residuals_nview_tiangulation(X, p, K, R, t):
    """ Jacobian and residuals for the projections of a 3D-point, X, viewed
    from n-views.

    Args:
        X: (3, 1) 3D point coordinates.
        p: (2, n) 2d observations.
        K: (3, 3) calib matrix.
        R: (n, 3, 3) rot matrix.
        t: (n, 3, 1) translation vector.

    Returns:
        J: (n, 2, 3) jacobian matrix of the error w.r.t. X ->
            d(proj(X) - p) / dX for each of the n observations.
        res: (n, 2, 1) residuals.
    """
    n = p.shape[1]
    J = np.empty((n, 2, 3))
    res = np.empty((2, n))

    for i in range(n):
        J[i], ui = jacobian_dprojection_dX(
            X, K, R[i], t[i])
        res[:, i] = ui[:, 0] - p[:, i]
    return J, res


@njit('void(f8[:,::1], f8)', fastmath=True, cache=True)
def augment_diag(H, damping):
    """Levenberg Marquardt augmentation of the Hessian's diagonal elements"""
    d, d2 = H.shape
    assert d == d2
    # in-place augmentation.
    for i in range(d):
        H[i, i] += damping * H[i, i]


@njit('Tuple((f8[:,::1], f8[:,::1]))(f8[:,::1], f8[:,::1], f8[:,::1], '
      'f8[:,:,::1], f8[:,:,::1], optional(f8[:,:,::1]), i8)', fastmath=True, cache=True)
def triangulation_nview_1point_LM_optimization(
        X: np.ndarray,
        p: np.ndarray,
        K: np.ndarray,
        R: np.ndarray,
        t: np.ndarray,
        sigmas: Optional[np.ndarray],
        niter: int = 5
):
    """Optimize 1 triangulated point, viewd from "n" different views.

    Args:
        X: (3, 1) 3D coordinated of the 3D point, X_i, that will be optimized.
        p: (2, ni) 2D coordinates corresponding to X_i for each view.
        K: (3, 3) calibration matrix.
        R: (ni, 3, 3) Rotation matrices corresponding to each p_i
        t: (ni, 3, 1) translation vectors corresponding to each p_i
        sigmas: (ni, 2, 2) array of cov matrices corresponding to each p_i
        niter: number of Levenberg-marquardt iterations.

    Returns:
        X: (3, 1) optimized 3D coordinates
        sigma: (3, 3) cov matrix of the estimate X.
    """
    eps = 1e-9
    d = 3
    n = p.shape[1]

    # if needed, define confidence weights.
    use_covs = sigmas is not None
    if use_covs:
        sigmas_inv = np.empty((n, 2, 2))
        for i in range(n):
            sigmas_inv[i] = np.linalg.inv(sigmas[i] + eps)

    # increase/ decrease factor of influence for the gradient.
    damping_factor_inc = 1e1
    damping_factor_dec = 1.0 / damping_factor_inc

    # initialize Hessian and gradient for in-place modification.
    JTWJ = np.empty((d, d))
    JTWr = np.empty((d,))

    # LM iterative optimization.
    X_current = X.copy()
    for i in range(niter):
        # (n, 2, 3) and (2, n)
        J, res = jacobian_and_residuals_nview_tiangulation(
            X_current, p, K, R, t)

        # construct gradient and approximate Hessian.
        JTWJ[:] = 0.0
        JTWr[:] = 0.0
        for j in range(n):
            if use_covs:
                JTW = J[j].T.dot(sigmas_inv[j])
                JTWJ += JTW.dot(J[j])
                JTWr += JTW.dot(res[:, j])
            else:
                JTWJ += J[j].T.dot(J[j])
                JTWr += J[j].T.dot(res[:, j])

        # damping term (lambda).
        if i == 0:
            # damping initialization based on [Hartley, Zisserman, 2004].
            # 1e-3 * (trace(JTWJ) / d)
            damping = 3e-4 * (JTWJ[0, 0] + JTWJ[1, 1] + JTWJ[2, 2])
            cost_prev = (res * res).sum()
        else:
            # update damping based on current cost.
            cost = (res * res).sum()
            if cost < cost_prev:
                damping *= damping_factor_dec
            else:
                damping *= damping_factor_inc
            cost_prev = cost
        # augment approximate Hessian (inplace modification).
        augment_diag(JTWJ, damping)

        # solve step and update state.
        X_current += np.expand_dims(np.linalg.solve(JTWJ, -JTWr), axis=1)

        if i == niter - 1:
            # final covariance matrix of the point coordinates.
            J, _ = jacobian_and_residuals_nview_tiangulation(
                X_current, p, K, R, t)

            JTWJ[:, :] = 0.0
            for j in range(n):
                if use_covs:
                    JTWJ += J[j].T.dot(sigmas_inv[j]).dot(J[j])
                else:
                    JTWJ += J[j].T.dot(J[j])
            sigma3d = np.ascontiguousarray(np.linalg.inv(JTWJ))

    return X_current, sigma3d


# def triangulation_nview_npoints_LM_optimization(
#         X: list[np.ndarray],
#         p: list[np.ndarray],
#         K: list[np.ndarray],
#         R: np.ndarray,
#         t: np.ndarray,
#         idx_pose: list[]
#         sigmas: list[np.ndarray],
#         niter: int = 5
#         ):
#     """Optimize 1 triangulated point, viewd from "n" different views.

#     Args:
#         X: (3, 1) 3D coordinated of the 3D point, X_i, that will be optimized.
#         p: (2, ni) 2D coordinates corresponding to X_i for each view.
#         K: (3, 3) calibration matrix.
#         R: (ni, 3, 3) Rotation matrices corresponding to each p_i
#         t: (ni, 3, 1) translation vectors corresponding to each p_i
#         sigmas: (ni, 2, 2) array of cov matrices corresponding to each p_i
#         niter: number of Levenberg-marquardt iterations.

#     Returns:
#         X: (3, 1) optimized 3D coordinates
#         sigma: (3, 3) cov matrix of the estimate X.
#     """


def DLT_two_view_triangulation(
        p1, p2,
        P1=None, P2=None,
        K=None, distCoeffs=None, R1=None, t1=None, R2=None, t2=None):
    """Point triangulation using DLT method implemented in OpenCV.

    The arguments K, distCoeffs, R1, t1, R2, t2 are only considered in the case
    when P1 and P2 (projection matrices for views 1 and 2) are not given.

    main Args:
        p1: (2, n) array of feature points in view 1.
        p2: (2, n) array of feature points in view 2.
        P1: (3, 4) projection matrix for view 1.
        P2: (3, 4) projection matrix for view 2.
        K: (3, 3) calibration matrix.
        distCoeffs: (n,) n distortion coeffs. given in OpenCV format.
        R1: (3, 3) rotation matrix for view 1.
        t1: (3, 1) translation vector for view 1.
        R2: (3, 3) rotation matrix for view 2.
        t2: (3, 1) translation vector for view 2.

    Returns:
        Pw: (3, n) triangulated coordinates in world reference.
    """
    if P1 is not None and P2 is not None:
        Pw_hom = cv2.triangulatePoints(P1, P2, p1, p2)
        return Pw_hom[:-1] / Pw_hom[-1]

    if R1 is None or t1 is None or R2 is None or t2 is None:
        raise ValueError(
            "When projection matrices are not given, the following arguments "
            "must be provided:\n- Two view rigid transformations, R1, t1, R2, "
            " t2,\n- The calibration matrix, K,\n- and optionally, the "
            " distortion coefficients, distCoeffs.")

    if distCoeffs is not None:
        assert K is not None, 'Undistortion also requires calib. matrix'
        p1 = cv2.undistortPoints(p1, K, distCoeffs, R=None, P=None)[:, 0].T
        p2 = cv2.undistortPoints(p2, K, distCoeffs, R=None, P=None)[:, 0].T

        P1 = np.concatenate((R1, t1), axis=1)
        P2 = np.concatenate((R2, t2), axis=1)

    else:
        P1 = K @ np.concatenate((R1, t1), axis=1)
        P2 = K @ np.concatenate((R2, t2), axis=1)

    # coordinates are now normalized. triangulate them.
    Pw_hom = cv2.triangulatePoints(
        P1,
        P2,
        p1,
        p2)
    return Pw_hom[:-1] / Pw_hom[-1]


@njit('f8[:,::1](f8[:,::1], f8[:,::1], f8[:,::1], f8[:,::1], Tuple((i8, i8)))', cache=True)
def DLT_normalized_two_view_triangulation(
        p1: np.ndarray,
        p2: np.ndarray,
        P1: np.ndarray,
        P2: np.ndarray,
        imsize: tuple[int, int]
):
    """Point triangulation using DLT method for n-views and w. normalization.

    Implementation based on [1] and the code [2].
    [1] Hartley, Zisserman, MVG, 2004.
    [2] https://www.robots.ox.ac.uk/~vgg/hzbook/code/vgg_multiview/vgg_X_from_xP_lin.m

    Returns:
        Pw: (3, 1) triangulated point in world reference.
    """
    n = p1.shape[1]
    n2 = p2.shape[1]
    assert n == n2

    out = np.empty((4, n))

    # these are going to be modified in-place. So, copy first.
    p1 = p1.copy()
    p2 = p2.copy()
    P1 = P1.copy()
    P2 = P2.copy()

    # # normalization, see [2]. move image coordinates to [-1, 1] range.
    p1[0] *= 2.0 / imsize[1]
    p1[1] *= 2.0 / imsize[0]
    p1 -= 1.0

    p2[0] *= 2.0 / imsize[1]
    p2[1] *= 2.0 / imsize[0]
    p2 -= 1.0

    # corresponding transformation of the projection matrices.
    P1[0] *= 2.0 / imsize[1]
    P1[1] *= 2.0 / imsize[0]
    P1[:-1] -= P1[2:]

    P2[0] *= 2.0 / imsize[1]
    P2[1] *= 2.0 / imsize[0]
    P2[:-1] -= P2[2:]

    A = np.empty((4, 4))
    for i in range(n):
        A[0] = p1[0, i] * P1[2] - P1[0]
        A[1] = p1[1, i] * P1[2] - P1[1]
        A[2] = p2[0, i] * P2[2] - P2[0]
        A[3] = p2[1, i] * P2[2] - P2[1]

        # solve for the normalized point.
        _, _, Vt = np.linalg.svd(A)
        out[:, i] = Vt[3] / Vt[3, 3]

        # chierality constraint.
        # if P1[2].dot(out[:, i]) < 0 and P2[2].dot(out[:, i]) < 0:
        #     out[:, i] *= -1

    return out[:3]


@njit('f8[:,::1](f8[:,:], f8[:,:,::1], UniTuple(Optional(i8), 2))', cache=True)
def DLT_normalized_nview_point_triangulation(
        p: np.ndarray, P: np.ndarray,
        imsize: tuple[Optional[int], Optional[int]]
):
    """Point triangulation using DLT method for n-views and w. normalization.

    Implementation based on [1] and the code [2].
    [1] Hartley, Zisserman, MVG, 2004.
    [2] https://www.robots.ox.ac.uk/~vgg/hzbook/code/vgg_multiview/vgg_X_from_xP_lin.m

    main Args:
        p: (2, n) feature point in each of the n-views.
        P: (n, 3, 4) projection matrix for each view.
        imsize: (h, w) spatial dimensions of the image. It is assumed that they
            have the same spatial dimensions. If specified, normalization is
            done, as explained in [2], it transforms the image coordinates to
            be in the range [-1, 1]. If imsize contains at least a None,
            normalization is not done.

    Returns:
        Pw: (3, 1) triangulated point in world reference.
    """
    n = p.shape[1]
    assert n > 1

    do_normalization = (imsize[0] is not None) and (imsize[1] is not None)
    if do_normalization:
        # normalization can be done following [2].
        p = p.copy()
        P = P.copy()
        # move image coordinates to [-1, 1] range.
        p[0] *= 2.0 / imsize[1]
        p[1] *= 2.0 / imsize[0]
        p -= 1.0
        # corresponding transformation of the projection matrices.
        P[:, 0] *= 2.0 / imsize[1]
        P[:, 1] *= 2.0 / imsize[0]
        P[:, :-1] -= P[:, 2:]

    A = np.empty((2 * n, 4))
    for i in range(4):
        # fill columns.
        A[::2, i] = p[0] * P[:, 2, i] - P[:, 0, i]
        A[1::2, i] = p[1] * P[:, 2, i] - P[:, 1, i]

    # solve for the normalized point.
    _, _, Vt = np.linalg.svd(A.T.dot(A))
    out = Vt[3] / Vt[3, 3]

    # # enforce chierality constraint.
    # depths = P[:, 2].dot(out)
    # if np.any(depths) < 0:
    #     out *= -1
    #     if np.any(depths) > 0:
    #         raise ValueError

    # alternative (marginally faster), by precomputing the columns of A:
    # c0u = p[0] * P[:, 2, 0] - P[:, 0, 0]
    # c1u = p[0] * P[:, 2, 1] - P[:, 0, 1]
    # c2u = p[0] * P[:, 2, 2] - P[:, 0, 2]
    # c3u = p[0] * P[:, 2, 3] - P[:, 0, 3]

    # c0v = p[1] * P[:, 2, 0] - P[:, 1, 0]
    # c1v = p[1] * P[:, 2, 1] - P[:, 1, 1]
    # c2v = p[1] * P[:, 2, 2] - P[:, 1, 2]
    # c3v = p[1] * P[:, 2, 3] - P[:, 1, 3]

    # ATA = np.empty((4, 4))
    # ATA[0, 0] = c0u.dot(c0u) + c0v.dot(c0v)
    # ATA[0, 1] = c0u.dot(c1u) + c0v.dot(c1v)
    # ATA[0, 2] = c0u.dot(c2u) + c0v.dot(c2v)
    # ATA[0, 3] = c0u.dot(c3u) + c0v.dot(c3v)

    # ATA[1, 1] = c1u.dot(c1u) + c1v.dot(c1v)
    # ATA[1, 2] = c1u.dot(c2u) + c1v.dot(c2v)
    # ATA[1, 3] = c1u.dot(c3u) + c1v.dot(c3v)

    # ATA[2, 2] = c2u.dot(c2u) + c2v.dot(c2v)
    # ATA[2, 3] = c2u.dot(c3u) + c2v.dot(c3v)

    # ATA[3, 3] = c3u.dot(c3u) + c3v.dot(c3v)

    # ATA[1, 0] = ATA[0, 1]
    # ATA[2, 0] = ATA[0, 2]
    # ATA[2, 1] = ATA[1, 2]
    # ATA[3, 0] = ATA[0, 3]
    # ATA[3, 1] = ATA[1, 3]
    # ATA[3, 2] = ATA[2, 3]

    return np.expand_dims(out[:-1], axis=1)


def DLT_normalized_nview_points_triangulation(
        p: Union[list[np.ndarray], np.ndarray],
        P: Union[list[np.ndarray], np.ndarray],
        imsize: tuple[int, int]
):
    """DLT triangulation of m points, for n-views and w. normalization.

    Implementation based on Hartley, Zisserman, MVG, 2004, sec. 12.2.

    main Args:
        p: (m, 2, n) array, or list of (2, n_i) feature point in neach of the
            n-views.
        P: (m, n, 3, 4) array, or list of (n_i, 3, 4) projection matrices for
            each view i (of n-views).
        imsize: (h, w) spatial dimensions of the image. It is assumed that they
            have the same spatial dimensions. If specified, normalization is
            done, as explained in [2], it transforms the image coordinates to
            be in the range [-1, 1].

    Returns:
        Pw: (3, m) array of triangulated points in world reference.
    """
    return np.array([
        DLT_normalized_nview_point_triangulation(pi, Pi, imsize)
        for pi, Pi in zip(p, P)]).T


def geometric_triangulation(u_1, u_2, R_wc1, t_xc1, R_wc2, t_wc2):
    raise NotImplementedError()
    # # ground truth 3D point
    # p = np.random.rand(3, 1)

    # # intrinsic parameters (assume the same camera in two views).
    # fx, fy = 500.0, 600.0
    # cx, cy = 240.0, 380.0
    # K = np.array([
    #     [fx, 0.0, cx],
    #     [0.0, fy, cy],
    #     [0.0, 0.0, 1.0]
    #     ])
    # Kinv = np.linalg.inv(K)

    # # transformations
    # R_wc1 = SO(3).rvs()
    # t_wc1 = np.random.rand(3, 1)

    # R_wc2 = SO(3).rvs()
    # t_wc2 = np.random.rand(3, 1)

    # # 2d point observations.
    # x_1 = K.dot(R_wc1.dot(p) + t_wc1)
    # x_2 = K.dot(R_wc2.dot(p) + t_wc2)

    # x_1 = x_1[:-1] / x_1[-1]
    # x_2 = x_2[:-1] / x_2[-1]

    # # ALGORITHM
    # # 1) get unit rays in each camera reference.
    # u_1 = Kinv[:, :2].dot(x_1) + Kinv[:, 2:]
    # u_2 = Kinv[:, :2].dot(x_2) + Kinv[:, 2:]

    # u_1 /= np.linalg.norm(u_1)
    # u_2 /= np.linalg.norm(u_2)

    # # 2) linear estimation of the depths in each reference of the cameras.
    # A = np.concatenate(
    #     (R_wc2 @ R_wc1.T.dot(u_1), -u_2), axis=1)
    # b = R_wc2 @ R_wc1.T.dot(t_wc1) - t_wc2

    # lambdas_12_est = np.linalg.solve(A.T.dot(A), A.T.dot(b))

    # # 3) final estimation.
    # p_est = 0.5 * (
    #     R_wc1.T.dot(lambdas_12_est[0]*u_1 - t_wc1)
    #     + R_wc2.T.dot(lambdas_12_est[1]*u_2 - t_wc2))
