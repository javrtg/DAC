"""EPnPU implementation"""

import sys
from pathlib import Path
from typing import Union, Optional
from pprint import pprint
import cProfile

import numpy as np
from numba import njit
import cv2

ENV_PATH = str(Path(__file__).parents[3])
if ENV_PATH not in sys.path:
    print(f"inserting {ENV_PATH} to sys.path.")
    sys.path.insert(0, ENV_PATH)
from experiments.geometry.models import epnp, geometry as geom


class EPnPU:
    """Implementation of EPnPU (points only) from [1].

    [1] Uncertainty-Aware Camera Pose Estimation from Points and Lines,
        Vakhitov et. al, 2021

    Args:
        cfg: dict containing customized parameters.

    Keywords of cfg:
        use_pca: If True, the centroid and principal directions of the point
            cloud are used to create the control points (as described in
            paper). Otherwise, the world basis is used. Default: True.
        th_reproj_error: Reprojection error threshold in [pixels] to decide
                    when a closed-form solution of the betas is good enough,
                    to stop checking for more cases (see Sec.3.3). If all cases
                    want to be checked, set it, e.g., to -1. Default: 20 (as
                    in author's Matlab implementation').
        use_median: If True, median is used over all reprojection errors to
                compute the overal reprojection error. Otherwise, mean is used.
                Default: False.
        refine_w_cps: If True pairwise distances between the control points
                are used to compute the betas (as in paper). Otherwise, the
                relative distances of 3D points to their centroid are
                used (as in authors' Matlab implementation). Default: True.
        optimize_best_only: If True, only the best betas (based on reprojection
                    error) closed-form solution is refined with Gauss-Newton
                    (as in authors' Matlab version). Otherwise, all
                    combinations are refined prior to selecting the best
                    overall solution (as in authors' c++ implementation).
                    Default: True.
        gn_iters: number of Gauss-Newton refinement iterations. Default: 10 (as
            in paper).
        verbose: If True, log relevant information. Default: False.
        p3p: dict with custom parameters when computing a first pose estimate
            with P3P ransac. Currently, using OpenCV functions.
    """

    default_cfg: dict[str, Union[bool, int], dict] = {
        "use_pca": True,
        "th_reproj_error": -1,
        "use_median": False,
        "refine_w_cps": True,
        "optimize_best_only": False,
        "gn_iters": 5,
        "verbose": False,
        "p3p": {
            'method': 'ap3p',
            'reproj_th': 6.0,
            'confidence': 0.999,
            'max_iter': 1_000
        },
    }

    def __init__(self, cfg=None):
        if cfg is None:
            cfg = {}

        for ki, vi in cfg.items():
            # check if input key is valid.
            if ki not in self.default_cfg:
                raise ValueError(
                    f"Input config key '{ki}' is not valid. "
                    f"Valid keys:\n\t\t\t{self.default_cfg.keys()}")

            # check input value has correct type.
            if not isinstance(vi, type(self.default_cfg[ki])):
                raise TypeError(
                    "'cfg[{}]' is expected to be {} but type {} was given."
                    .format(
                        ki,
                        type(self.default_cfg[ki]).__name__,
                        type(vi).__name__))

            if ki == "p3p":
                # update ransac dict to avoid overriding.
                cfg['p3p'] = {**self.default_cfg['p3p'], **cfg['p3p']}

        self.cfg = {**self.default_cfg, **cfg}

        if self.cfg['verbose']:
            print("\nEPnPU configuration:")
            pprint(self.cfg)

    def __call__(
            self,
            Pw: np.ndarray,
            U: np.ndarray,
            A: np.ndarray,
            sigmas2d: Optional[np.ndarray] = None,
            sigmas3d: Optional[np.ndarray] = None,
            R_rough: Optional[np.ndarray] = None,
            t_rough: Optional[np.ndarray] = None
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Camera localization given 2D-3D correspondences with EPnPu method.

        Args:
            Pw: (3, n) given 3D coordinates of n points.
            U: (2, n) corresponding 2D coordinates in the image plane.
            A: (3, 3) camera calibration matrix of the form
                [[fu, 0, u0],
                 [0, fv, v0],
                 [0,  0, 1]],
            sigmas2d: (n, 2, 2) covariance matrices of 2D local-features.
            sigmas3d: (n, 3, 3) covariance matrices of 3D points locations.
            R: (3, 3) rough estimation of covariance matrix.
            t: (3, 1) rough estimation of translation vector.

        Returns:
            R_cw: (3, 3) rotation matrix that rotates points from world
                to camera coordinate system, i.e. p_c = R_cw * p_w + t_cw
            t_cw: (3, 1) translation vector.
            error: reprojection error of the returned solution.
        """
        # check c-contiguity and type (to comply with numba methods).
        Pw = _check_input(Pw)
        U = _check_input(U)
        A = _check_input(A)

        if sigmas2d is not None:
            use_2d_noise = True
            sigmas2d = _check_input(sigmas2d)
        else:
            use_2d_noise = False

        if sigmas3d is not None:
            use_3d_noise = True
            sigmas3d = _check_input(sigmas3d)
        else:
            use_3d_noise = False

        # when using covariances, a rough estimation of the pose is needed.
        if use_2d_noise or use_3d_noise:
            if R_rough is None or t_rough is None:
                # obtain rough pose with P3P ransac and inliers.
                R_rough, t_rough, inliers = _get_rough_pose(
                    Pw, U, A, self.cfg['p3p'])
                Pw = np.ascontiguousarray(Pw[:, inliers[:, 0]])
                U = np.ascontiguousarray(U[:, inliers[:, 0]])

                if use_2d_noise:
                    sigmas2d = np.ascontiguousarray(sigmas2d[inliers[:, 0]])
                if use_3d_noise:
                    sigmas3d = np.ascontiguousarray(sigmas3d[inliers[:, 0]])

            else:
                R_rough = _check_input(R_rough)
                t_rough = _check_input(t_rough)

        # control points in world reference.
        if use_3d_noise:
            Cw = _define_control_points(Pw, np.linalg.inv(sigmas3d + 1e-12))
        else:
            Cw = epnp._define_control_points(self.cfg['use_pca'], Pw)

        # homogeneous barycentric coordinates (independent of the reference).
        alphas = epnp._compute_alphas(Pw, Cw)

        # 4 right-most sigular vectors (conservative null space of M).
        if use_3d_noise and use_2d_noise:
            # covariance of 3D points in camera reference.
            sigmas3d_c = A.dot(R_rough) @ sigmas3d @ R_rough.T.dot(A.T)

            # slower version.
            # V = np.ascontiguousarray(_get_rsvs_2d3d(
            #     U, A, alphas, sigmas2d, sigmas3d_c, Pw, R_rough, t_rough))

            # faster version.
            M, W = _get_MW_2d3d(
                U, A, alphas, sigmas2d, sigmas3d_c, Pw, R_rough, t_rough)
            V = _rsvs_given_MW(M, W)

        elif use_3d_noise:
            sigmas3d_c = A.dot(R_rough) @ sigmas3d @ R_rough.T.dot(A.T)

            # V = np.ascontiguousarray(_get_rsvs_3d(
            #     U, A, alphas, sigmas3d_c, R_rough, t_rough))

            # faster version.
            M, W = _get_MW_3d(U, A, alphas, sigmas3d_c, Pw)
            V = _rsvs_given_MW(M, W)

        elif use_2d_noise:
            # V = np.ascontiguousarray(_get_rsvs_2d(
            #     U, A, alphas, sigmas2d, Pw, R_rough, t_rough))

            # faster version.
            M, W = _get_MW_2d(U, A, alphas, sigmas2d, Pw, R_rough, t_rough)
            V = _rsvs_given_MW(M, W)

        else:
            V = np.ascontiguousarray(epnp._get_rsv(
                epnp._compute_M(U, A, alphas), 4, asc_order=True))

        # distances to compare with when refining the scale:
        if self.cfg["refine_w_cps"]:
            # pairwise euclidean distances (as in paper).
            dist_w = geom._euc_pdist(Cw.T, apply_sqrt=True)
        else:
            # euclidean distances w.r.t. the centroid.
            dist_w = geom._euc_cdist(geom._mean_axis1(
                Pw).T, Pw.T, apply_sqrt=True)[0]

        # reference values for the optimization.
        if self.cfg["gn_iters"] > 0:
            dist_cw = geom._euc_pdist(Cw.T, apply_sqrt=False)

        # Solve case N=1.
        beta, Cc, Pc = epnp._solve_caseN1_or_refine(
            self.cfg["refine_w_cps"], V[:, 0], alphas, dist_w)
        betas = np.array([beta, 0.0, 0.0, 0.0])

        R_cw, t_cw = geom.umeyama_alignment_se3(Pc, Pw)
        error = geom.reproj_error_nb(
            Pw, U, R_cw, t_cw, A, self.cfg["use_median"])

        if (not self.cfg["optimize_best_only"]) & (self.cfg["gn_iters"] > 0):
            betas_opt = epnp._gauss_newton_betas(
                V, betas, dist_cw, self.cfg["gn_iters"])
            Pc = epnp._solve_for_sign_and_pointcloud(V, betas_opt, alphas)

            R_cw_opt, t_cw_opt = geom.umeyama_alignment_se3(Pc, Pw)
            error_opt = geom.reproj_error_nb(
                Pw, U, R_cw_opt, t_cw_opt, A, self.cfg["use_median"])

            # Avoid bad convergence/minima.
            if error_opt < error:
                betas = betas_opt
                R_cw = R_cw_opt
                t_cw = t_cw_opt
                error = error_opt

        if self.cfg['verbose']:
            print("\n==> Case N=1:")
            print(f"Betas\t= {betas}")
            print(f"Error\t= {error}")

        # record for selecting the best case.
        best_case = 1
        best_error = error
        best_case_data = [R_cw, t_cw, betas]

        # case N=2.
        if error >= self.cfg["th_reproj_error"]:
            # Form system for solving the linearized betas (e.g. eq. 13).
            L6_10 = epnp._compute_L6_10(V)
            # We may have already computed rho. Check.
            if self.cfg["gn_iters"] <= 0:
                rho = geom._euc_pdist(Cw.T, apply_sqrt=False)
            else:
                rho = dist_cw

            # raw solution.
            betas = epnp._solve_caseN2(L6_10, rho)
            Cc = V.dot(betas)

            # refinement.
            beta, Cc, Pc = epnp._solve_caseN1_or_refine(
                self.cfg["refine_w_cps"], Cc, alphas, dist_w)
            betas *= beta

            R_cw, t_cw = geom.umeyama_alignment_se3(Pc, Pw)
            error = geom.reproj_error_nb(
                Pw, U, R_cw, t_cw, A, self.cfg["use_median"])

            if (not self.cfg["optimize_best_only"]) & (self.cfg["gn_iters"] > 0):
                betas_opt = epnp._gauss_newton_betas(
                    V, betas, dist_cw, self.cfg["gn_iters"])
                Pc = epnp._solve_for_sign_and_pointcloud(V, betas_opt, alphas)

                R_cw_opt, t_cw_opt = geom.umeyama_alignment_se3(Pc, Pw)
                error_opt = geom.reproj_error_nb(
                    Pw, U, R_cw_opt, t_cw_opt, A, self.cfg["use_median"])

                # Avoid bad convergence/minima.
                if error_opt < error:
                    betas = betas_opt
                    R_cw = R_cw_opt
                    t_cw = t_cw_opt
                    error = error_opt

            if self.cfg['verbose']:
                print("\n==> Case N=2:")
                print(f"Betas\t= {betas}")
                print(f"Error\t= {error}")

            if error < best_error:
                best_case = 2
                best_error = error
                best_case_data = [R_cw, t_cw, betas]

        # case N=3.
        if best_error >= self.cfg["th_reproj_error"]:
            # raw solution.
            betas = epnp._solve_caseN3(L6_10, rho)
            Cc = V.dot(betas)

            # refinement.
            beta, Cc, Pc = epnp._solve_caseN1_or_refine(
                self.cfg["refine_w_cps"], Cc, alphas, dist_w)
            betas *= beta

            R_cw, t_cw = geom.umeyama_alignment_se3(Pc, Pw)
            error = geom.reproj_error_nb(
                Pw, U, R_cw, t_cw, A, self.cfg["use_median"])

            if (not self.cfg["optimize_best_only"]) & (self.cfg["gn_iters"] > 0):
                betas_opt = epnp._gauss_newton_betas(
                    V, betas, dist_cw, self.cfg["gn_iters"])
                Pc = epnp._solve_for_sign_and_pointcloud(V, betas_opt, alphas)

                R_cw_opt, t_cw_opt = geom.umeyama_alignment_se3(Pc, Pw)
                error_opt = geom.reproj_error_nb(
                    Pw, U, R_cw_opt, t_cw_opt, A, self.cfg["use_median"])

                # Avoid bad convergence/minima.
                if error_opt < error:
                    betas = betas_opt
                    R_cw = R_cw_opt
                    t_cw = t_cw_opt
                    error = error_opt

            if self.cfg['verbose']:
                print("\n==> Case N=3:")
                print(f"Betas\t= {betas}")
                print(f"Error\t= {error}")

            if error < best_error:
                best_case = 3
                best_error = error
                best_case_data = [R_cw, t_cw, betas]

        # case N=4.
        if best_error >= self.cfg["th_reproj_error"]:
            # raw solution.
            betas = epnp._solve_caseN4(L6_10, rho)
            Cc = V.dot(betas)

            # refinement.
            beta, Cc, Pc = epnp._solve_caseN1_or_refine(
                self.cfg["refine_w_cps"], Cc, alphas, dist_w)
            betas *= beta

            R_cw, t_cw = geom.umeyama_alignment_se3(Pc, Pw)
            error = geom.reproj_error_nb(
                Pw, U, R_cw, t_cw, A, self.cfg["use_median"])

            if (not self.cfg["optimize_best_only"]) & (self.cfg["gn_iters"] > 0):
                betas_opt = epnp._gauss_newton_betas(
                    V, betas, dist_cw, self.cfg["gn_iters"])
                Pc = epnp._solve_for_sign_and_pointcloud(V, betas_opt, alphas)

                R_cw_opt, t_cw_opt = geom.umeyama_alignment_se3(Pc, Pw)
                error_opt = geom.reproj_error_nb(
                    Pw, U, R_cw_opt, t_cw_opt, A, self.cfg["use_median"])

                # Avoid bad convergence/minima.
                if error_opt < error:
                    betas = betas_opt
                    R_cw = R_cw_opt
                    t_cw = t_cw_opt
                    error = error_opt

            if self.cfg['verbose']:
                print("\n==> Case N=4:")
                print(f"Betas\t= {betas}")
                print(f"Error\t= {error}")

            if error < best_error:
                best_case = 4
                best_error = error
                best_case_data = [R_cw, t_cw, betas]

        # Final optimization.
        if self.cfg["optimize_best_only"] & (self.cfg["gn_iters"] > 0):
            betas = epnp._gauss_newton_betas(
                V, best_case_data[-1], dist_cw, self.cfg["gn_iters"])
            Pc = epnp._solve_for_sign_and_pointcloud(V, betas, alphas)

            R_cw, t_cw = geom.umeyama_alignment_se3(Pc, Pw)
            error = geom.reproj_error_nb(
                Pw, U, R_cw, t_cw, A, self.cfg["use_median"])

            # Avoid bad convergence/minima (when prior has less reproj. error).
            if error > best_error:
                R_cw, t_cw, betas = best_case_data
                error = best_error
        else:
            R_cw, t_cw, betas = best_case_data
            error = best_error

        if self.cfg["verbose"]:
            print("\n\n---------------------------------------")
            print(f"==> Case with best solution:\tN={best_case}.")
            print("---------------------------------------")
            print(f"\n\tR_cw:\n{R_cw}")
            print(f"\n\tt_cw:\n{t_cw}")
            print(f"\n\tBetas:\n{betas}")
            print(f"\n\tError:\n{error}")
            print("---------------------------------------")

        return R_cw, t_cw, error


def _check_input(arg):
    """Ensure c-contiguity and float type to comply w. numba methods."""
    if not arg.flags['C_CONTIGUOUS']:
        arg = np.ascontiguousarray(arg)
    if arg.dtype != float:
        arg = arg.astype(float)
    return arg


def _get_rough_pose(Pw, U, A, p3p_cfg):
    """Pose estimate with P3P ransac scheme."""

    if p3p_cfg['method'] in {'ap3p', 'p3p'}:
        # P3P from [Ke, 2017] or [Gao, 2003] in ransac scheme.
        # + final refinement w. all the inliers w. epnp.
        flag = (
            cv2.SOLVEPNP_AP3P if p3p_cfg['method'] == "ap3p"
            else cv2.SOLVEPNP_P3P)

        ret, qest, test, inliers = cv2.solvePnPRansac(
            objectPoints=Pw.T,
            imagePoints=U.T,
            cameraMatrix=A,
            distCoeffs=None,
            reprojectionError=p3p_cfg['reproj_th'],
            confidence=p3p_cfg['confidence'],
            flags=flag,
            iterationsCount=p3p_cfg['max_iter']
        )

    elif p3p_cfg['method'] == 'usac':
        # P3P [Pajdla, 2013, p.52]
        # + LO (local optim.) w. subset of inliers and w. DLS [Hesch, 2011]
        # + final refinement with all the inliers (again w. DLS).

        # custom usac_params (using a usac flag would lead to predefined ones).
        usac_params = cv2.UsacParams()
        usac_params.confidence = p3p_cfg['confidence']  # default=0.99
        usac_params.maxIterations = p3p_cfg['max_iter']  # default=5_000
        # sample used for LO refinement.
        usac_params.loSampleSize = 10  # hardcoded maximum = 15
        usac_params.loIterations = 4  # hardcoded maximum = 15
        # DLS is not iterative refinement (otherwise LOCAL_OPTIM_INNER_AND_ITER_LO)
        usac_params.loMethod = cv2.LOCAL_OPTIM_INNER_LO
        usac_params.sampler = cv2.SAMPLING_UNIFORM
        usac_params.score = cv2.SCORE_METHOD_MSAC
        usac_params.threshold = p3p_cfg['reproj_th']  # default=1.5

        ret, _, qest, test, inliers = cv2.solvePnPRansac(
            objectPoints=Pw.T,
            imagePoints=U.T,
            cameraMatrix=A,
            distCoeffs=None,
            params=usac_params)

    else:
        raise ValueError

    if not ret:
        raise RuntimeError
    if len(inliers) < 6:
        raise RuntimeError

    return (
        cv2.Rodrigues(qest)[0],
        test,
        inliers,
    )


def _define_control_points(Pw: np.ndarray, sigmas3d_inv: np.ndarray):
    """Control points definition based on covariance-weighted PCA

    Args:
        Pw: (3, n), points in world reference.
        sigmas_3d_inv: (n, 3, 3) inverse cov. matrices for each 3D point.

    Returns:
        (3, 4), control points of the barycentric system.
    """
    _, n = Pw.shape

    # normalizationt term
    sigmas_sum = np.linalg.inv(sigmas3d_inv.sum(axis=0) + 1e-6)

    # weighted mean:
    wmean = sigmas_sum @ np.einsum('nij, jn -> i', sigmas3d_inv, Pw)[:, None]

    # SVD of covariance-weighted sample covariance matrix.
    Pw = Pw[None].transpose(2, 1, 0)
    v, s, _ = np.linalg.svd(
        n * sigmas_sum @ (
            (wcov := sigmas3d_inv @ (Pw - wmean))
            @ wcov.transpose(0, 2, 1)
        ).sum(axis=0)
        @ sigmas_sum)

    return np.concatenate(
        (v * np.sqrt(s)[None] + wmean, wmean),
        axis=1)


@njit('f8[:,:](f8[:,::1], f8[:,:,::1])', cache=True)
def _define_control_points_nb(Pw, sigmas3d_inv):
    d, n = Pw.shape

    sigmas_sum = np.linalg.inv(sigmas3d_inv.sum(axis=0))

    wmean = np.zeros((d, 1))
    for i in range(n):
        wmean += sigmas3d_inv[i] @ np.expand_dims(Pw[:, i], axis=1)
    wmean = sigmas_sum @ wmean

    w_sigmas_inv = np.zeros((3, 3))
    for i in range(n):
        temp = sigmas3d_inv[i] @ (np.expand_dims(Pw[:, i], axis=1) - wmean)
        w_sigmas_inv += temp @ temp.T

    v, s, _ = np.linalg.svd(n * sigmas_sum @ w_sigmas_inv @ sigmas_sum)

    return np.concatenate(
        (v * np.expand_dims(np.sqrt(s), axis=0) + wmean, wmean),
        axis=1)


@njit(['f8[:,::1](f8[::1], f8[:], f8[:,::1])',
       'f8[:,::1](f8[:], f8[:], f8[:,::1])'], cache=True)
def _outer2d(a, b, out):
    """Fast 2d outer product when constructing M"""
    out[0, 0] = a[0] * b[0]
    out[0, 1] = a[0] * b[1]
    out[1, 0] = a[1] * b[0]
    out[1, 1] = a[1] * b[1]
    return out


@njit('f8[:,::1](f8[:,::1], f8[:,::1])', cache=True)
def _fast2x2inv(C, out):
    """Fast 2d matrix inverse when constructing M"""
    ((a, b), (c, d)) = C
    # determinant.
    div = (a * d - b * c)
    out[0, 0] = d / div
    out[0, 1] = -b / div
    out[1, 0] = -c / div
    out[1, 1] = a / div
    return out


def _rsvs_given_MW(M, W):
    return np.ascontiguousarray(np.linalg.svd(
        M.T @ (W @ M.reshape(len(W), 2, 12)).reshape(len(W) * 2, 12)
    )[0][:, -1:-5:-1])


@njit("Tuple((f8[:,::1], f8[:,:,::1]))"
      "(f8[:,::1], f8[:,::1], f8[:,::1], f8[:,:,::1], f8[:,:,::1], "
      "f8[:,::1], f8[:,::1], f8[:,::1])", fastmath=True, cache=True)
def _get_MW_2d3d(U, A, alphas, sigmas2d, sigmas3d_c, Pw, R, t):
    """Get 4-rightmost singular vectors of the weighted-constraints matrix.

    Args:
        U: (2,n) image plane coordinates.
        A: (3,3) calibration matrix:
            [[fu, 0, u0],
             [0, fv, v0],
             [0,  0, 1]]
        alphas: (4,n) homogeneous baricentric coordinates.
        sigmas2d: (n, 2, 2) covariance matrices of 2D local-features locations.
        sigmas3d_c: (n, 3, 3) covariance matrices of 3D points locations in
            *camera* reference.
        Pw: (3, n) 3D points coordinates in world reference.
        R: (3, 3) rough estimation of covariance matrix.
        t: (3, 1) rough estimation of translation vector.

    Returns:
        (12, 4) 4 right-most singular vectors of weighted-constraints matrix
    """
    n = U.shape[1]
    # U = np.asfortranarray(U)

    # rough estimation of the depth of the points in camera reference.
    depths_c = R[2].dot(Pw) + t[2]

    fu, fv = A[0, 0], A[1, 1]
    u0, v0 = A[0, 2], A[1, 2]

    # temporal variables for speed.
    # uut = np.empty((2, 2))
    # wut = np.empty((2, 2))
    inv_tmp = np.empty((2, 2))

    # initialize weighted matrix.
    M = np.empty((2 * n, 12))
    W = np.empty((n, 2, 2))

    for i in range(n):
        # unweighted terms.
        row = 2 * i
        row_n = row + 1

        M[row, 0] = alphas[0, i] * fu
        M[row, 1] = 0.0
        M[row, 2] = alphas[0, i] * (u0 - U[0, i])
        M[row, 3] = alphas[1, i] * fu
        M[row, 4] = 0.0
        M[row, 5] = alphas[1, i] * (u0 - U[0, i])
        M[row, 6] = alphas[2, i] * fu
        M[row, 7] = 0.0
        M[row, 8] = alphas[2, i] * (u0 - U[0, i])
        M[row, 9] = alphas[3, i] * fu
        M[row, 10] = 0.0
        M[row, 11] = alphas[3, i] * (u0 - U[0, i])

        M[row_n, 0] = 0.0
        M[row_n, 1] = alphas[0, i] * fv
        M[row_n, 2] = alphas[0, i] * (v0 - U[1, i])
        M[row_n, 3] = 0.0
        M[row_n, 4] = alphas[1, i] * fv
        M[row_n, 5] = alphas[1, i] * (v0 - U[1, i])
        M[row_n, 6] = 0.0
        M[row_n, 7] = alphas[2, i] * fv
        M[row_n, 8] = alphas[2, i] * (v0 - U[1, i])
        M[row_n, 9] = 0.0
        M[row_n, 10] = alphas[3, i] * fv
        M[row_n, 11] = alphas[3, i] * (v0 - U[1, i])

        # store weighting of current observation.
        # fmt: off
        W[i] = _fast2x2inv(
            # S
            sigmas3d_c[i, :2, :2]

            # \gamma uu^T
            # + sigmas3d_c[i, 2, 2] * _outer2d(U[:, i], U[:, i], uut)
            + sigmas3d_c[i, 2, 2] * (np.expand_dims(U[:, i], axis=1) * U[:, i])

            # x_3^2 Sigma_u
            + depths_c[i] * depths_c[i] * sigmas2d[i]

            # - wu^T
            # + _outer2d(sigmas3d_c[i, 2, :2], -U[:, i], wut)
            - (np.expand_dims(sigmas3d_c[i, 2, :2], axis=1) * U[:, i])

            # - uw^T
            # + wut.T
            - (np.expand_dims(U[:, i], axis=1) * sigmas3d_c[i, 2, :2])

            # \gamma * Sigma_u (does not appear in [Vakhitov, 2021])
            + sigmas3d_c[i, 2, 2] * sigmas2d[i]
            + 1e-12, inv_tmp)
        # fmt: on

    return M, W


@njit("Tuple((f8[:,::1], f8[:,:,::1]))"
      "(f8[:,::1], f8[:,::1], f8[:,::1], f8[:,:,::1], f8[:,::1])",
      fastmath=True, cache=True)
def _get_MW_3d(U, A, alphas, sigmas3d_c, Pw):
    """Get 4-rightmost singular vectors of the weighted-constraints matrix.

    Args:
        U: (2,n) image plane coordinates.
        A: (3,3) calibration matrix:
            [[fu, 0, u0],
             [0, fv, v0],
             [0,  0, 1]]
        alphas: (4,n) homogeneous baricentric coordinates.
        sigmas3d_c: (n, 3, 3) covariance matrices of 3D points locations in
            *camera* reference.
        Pw: (3, n) 3D points coordinates in world reference.

    Returns:
        (12, 4) 4 right-most singular vectors of weighted-constraints matrix
    """
    n = U.shape[1]
    # U = np.asfortranarray(U)

    fu, fv = A[0, 0], A[1, 1]
    u0, v0 = A[0, 2], A[1, 2]

    # temporal variables for speed.
    # uut = np.empty((2, 2))
    # wut = np.empty((2, 2))
    inv_tmp = np.empty((2, 2))

    # initialize weighted matrix.
    M = np.empty((2 * n, 12))
    W = np.empty((n, 2, 2))

    for i in range(n):
        # unweighted terms.
        row = 2 * i
        row_n = row + 1

        M[row, 0] = alphas[0, i] * fu
        M[row, 1] = 0.0
        M[row, 2] = alphas[0, i] * (u0 - U[0, i])
        M[row, 3] = alphas[1, i] * fu
        M[row, 4] = 0.0
        M[row, 5] = alphas[1, i] * (u0 - U[0, i])
        M[row, 6] = alphas[2, i] * fu
        M[row, 7] = 0.0
        M[row, 8] = alphas[2, i] * (u0 - U[0, i])
        M[row, 9] = alphas[3, i] * fu
        M[row, 10] = 0.0
        M[row, 11] = alphas[3, i] * (u0 - U[0, i])

        M[row_n, 0] = 0.0
        M[row_n, 1] = alphas[0, i] * fv
        M[row_n, 2] = alphas[0, i] * (v0 - U[1, i])
        M[row_n, 3] = 0.0
        M[row_n, 4] = alphas[1, i] * fv
        M[row_n, 5] = alphas[1, i] * (v0 - U[1, i])
        M[row_n, 6] = 0.0
        M[row_n, 7] = alphas[2, i] * fv
        M[row_n, 8] = alphas[2, i] * (v0 - U[1, i])
        M[row_n, 9] = 0.0
        M[row_n, 10] = alphas[3, i] * fv
        M[row_n, 11] = alphas[3, i] * (v0 - U[1, i])

        # store weighting of current observation.
        # fmt: off
        W[i] = _fast2x2inv(
            # S
            sigmas3d_c[i, :2, :2]
            # \gamma uu^T
            # + sigmas3d_c[i, 2, 2] * _outer2d(U[:, i], U[:, i], uut)
            + sigmas3d_c[i, 2, 2] * (np.expand_dims(U[:, i], axis=1) * U[:, i])

            # - wu^T
            # + _outer2d(sigmas3d_c[i, 2, :2], -U[:, i], wut)
            - (np.expand_dims(sigmas3d_c[i, 2, :2], axis=1) * U[:, i])

            # - uw^T
            # + wut.T
            - (np.expand_dims(U[:, i], axis=1) * sigmas3d_c[i, 2, :2])
            + 1e-12, inv_tmp)
        # fmt: on

    return M, W


@njit("Tuple((f8[:,::1], f8[:,:,::1]))"
      "(f8[:,::1], f8[:,::1], f8[:,::1], f8[:,:,::1], "
      "f8[:,::1], f8[:,::1], f8[:,::1])", fastmath=True, cache=True)
def _get_MW_2d(U, A, alphas, sigmas2d, Pw, R, t):
    """Get 4-rightmost singular vectors of the weighted-constraints matrix.

    Args:
        U: (2,n) image plane coordinates.
        A: (3,3) calibration matrix:
            [[fu, 0, u0],
             [0, fv, v0],
             [0,  0, 1]]
        alphas: (4,n) homogeneous baricentric coordinates.
        sigmas2d: (n, 2, 2) covariance matrices of 2D local-features locations.
        Pw: (3, n) 3D points coordinates in world reference.
        R: (3, 3) rough estimation of covariance matrix.
        t: (3, 1) rough estimation of translation vector.

    Returns:
        (12, 4) 4 right-most singular vectors of weighted-constraints matrix
    """
    n = U.shape[1]

    # rough estimation of the depth of the points in camera reference.
    depths_c = R[2].dot(Pw) + t[2]

    fu, fv = A[0, 0], A[1, 1]
    u0, v0 = A[0, 2], A[1, 2]

    # temporal variable for speed.
    inv_tmp = np.empty((2, 2))

    # initialize weighted matrix.
    M = np.empty((2 * n, 12))
    W = np.empty((n, 2, 2))

    for i in range(n):
        # unweighted terms.
        row = 2 * i
        row_n = row + 1

        M[row, 0] = alphas[0, i] * fu
        M[row, 1] = 0.0
        M[row, 2] = alphas[0, i] * (u0 - U[0, i])
        M[row, 3] = alphas[1, i] * fu
        M[row, 4] = 0.0
        M[row, 5] = alphas[1, i] * (u0 - U[0, i])
        M[row, 6] = alphas[2, i] * fu
        M[row, 7] = 0.0
        M[row, 8] = alphas[2, i] * (u0 - U[0, i])
        M[row, 9] = alphas[3, i] * fu
        M[row, 10] = 0.0
        M[row, 11] = alphas[3, i] * (u0 - U[0, i])

        M[row_n, 0] = 0.0
        M[row_n, 1] = alphas[0, i] * fv
        M[row_n, 2] = alphas[0, i] * (v0 - U[1, i])
        M[row_n, 3] = 0.0
        M[row_n, 4] = alphas[1, i] * fv
        M[row_n, 5] = alphas[1, i] * (v0 - U[1, i])
        M[row_n, 6] = 0.0
        M[row_n, 7] = alphas[2, i] * fv
        M[row_n, 8] = alphas[2, i] * (v0 - U[1, i])
        M[row_n, 9] = 0.0
        M[row_n, 10] = alphas[3, i] * fv
        M[row_n, 11] = alphas[3, i] * (v0 - U[1, i])

        # store weighting of current observation.
        W[i] = _fast2x2inv(
            # x_3^2 Sigma_u
            + depths_c[i] * depths_c[i] * sigmas2d[i] + 1e-12, inv_tmp)

    return M, W


@njit("f8[:,:](f8[:,::1], f8[:,::1], f8[:,::1], f8[:,:,::1], f8[:,:,::1], "
      "f8[:,::1], f8[:,::1], f8[:,::1])", fastmath=True, cache=True)
def _get_rsvs_2d3d(U, A, alphas, sigmas2d, sigmas3d_c, Pw, R, t):
    """Get 4-rightmost singular vectors of the weighted-constraints matrix.

    Args:
        U: (2,n) image plane coordinates.
        A: (3,3) calibration matrix:
            [[fu, 0, u0],
             [0, fv, v0],
             [0,  0, 1]]
        alphas: (4,n) homogeneous baricentric coordinates.
        sigmas2d: (n, 2, 2) covariance matrices of 2D local-features locations.
        sigmas3d_c: (n, 3, 3) covariance matrices of 3D points locations in
            *camera* reference.
        Pw: (3, n) 3D points coordinates in world reference.
        R: (3, 3) rough estimation of covariance matrix.
        t: (3, 1) rough estimation of translation vector.

    Returns:
        (12, 4) 4 right-most singular vectors of weighted-constraints matrix
    """
    n = U.shape[1]
    # U = np.asfortranarray(U)

    # rough estimation of the depth of the points in camera reference.
    depths_c = R[2].dot(Pw) + t[2]

    fu, fv = A[0, 0], A[1, 1]
    u0, v0 = A[0, 2], A[1, 2]

    # temporal variables for speed.
    # uut = np.empty((2, 2))
    # wut = np.empty((2, 2))
    inv_tmp = np.empty((2, 2))

    # initialize weighted matrix.
    M = np.empty((2 * n, 12))

    for i in range(n):
        # unweighted terms.
        row = 2 * i
        row_n = row + 1

        M[row, 0] = alphas[0, i] * fu
        M[row, 1] = 0.0
        M[row, 2] = alphas[0, i] * (u0 - U[0, i])
        M[row, 3] = alphas[1, i] * fu
        M[row, 4] = 0.0
        M[row, 5] = alphas[1, i] * (u0 - U[0, i])
        M[row, 6] = alphas[2, i] * fu
        M[row, 7] = 0.0
        M[row, 8] = alphas[2, i] * (u0 - U[0, i])
        M[row, 9] = alphas[3, i] * fu
        M[row, 10] = 0.0
        M[row, 11] = alphas[3, i] * (u0 - U[0, i])

        M[row_n, 0] = 0.0
        M[row_n, 1] = alphas[0, i] * fv
        M[row_n, 2] = alphas[0, i] * (v0 - U[1, i])
        M[row_n, 3] = 0.0
        M[row_n, 4] = alphas[1, i] * fv
        M[row_n, 5] = alphas[1, i] * (v0 - U[1, i])
        M[row_n, 6] = 0.0
        M[row_n, 7] = alphas[2, i] * fv
        M[row_n, 8] = alphas[2, i] * (v0 - U[1, i])
        M[row_n, 9] = 0.0
        M[row_n, 10] = alphas[3, i] * fv
        M[row_n, 11] = alphas[3, i] * (v0 - U[1, i])

        # weighting with full covariance -its square root with Cholesky decomp.
        # fmt: off
        M[row : row+2] = np.linalg.cholesky(_fast2x2inv(
            # S
            sigmas3d_c[i, :2, :2]
            # \gamma uu^T
            # + sigmas3d_c[i, 2, 2] * _outer2d(U[:, i], U[:, i], uut)
            + sigmas3d_c[i, 2, 2] * (np.expand_dims(U[:, i], axis=1) * U[:, i])

            # x_3^2 Sigma_u
            + depths_c[i] * depths_c[i] * sigmas2d[i]

            # - wu^T
            # + _outer2d(sigmas3d_c[i, 2, :2], -U[:, i], wut)
            - (np.expand_dims(sigmas3d_c[i, 2, :2], axis=1) * U[:, i])

            # - uw^T
            # + wut.T
            - (np.expand_dims(U[:, i], axis=1) * sigmas3d_c[i, 2, :2])

            # \gamma * Sigma_u (does not appear in [Vakhitov, 2021])
            # + sigmas3d_c[i, 2, 2] * sigmas2d[i]
            + 1e-12, inv_tmp)
            ).T.dot(M[row : row+2])
        # fmt: on

    # 4 right-most singular vectors.
    return np.linalg.svd(M.T.dot(M))[0][:, -1:-5:-1]


@njit("f8[:,:](f8[:,::1], f8[:,::1], f8[:,::1], f8[:,:,::1], "
      "f8[:,::1], f8[:,::1], f8[:,::1])", fastmath=True, cache=True)
def _get_rsvs_2d(U, A, alphas, sigmas2d, Pw, R, t):
    """Get 4-rightmost singular vectors of the weighted-constraints matrix.

    Args:
        U: (2,n) image plane coordinates.
        A: (3,3) calibration matrix:
            [[fu, 0, u0],
             [0, fv, v0],
             [0,  0, 1]]
        alphas: (4,n) homogeneous baricentric coordinates.
        sigmas2d: (n, 2, 2) covariance matrices of 2D local-features locations.
        Pw: (3, n) 3D points coordinates in world reference.
        R: (3, 3) rough estimation of covariance matrix.
        t: (3, 1) rough estimation of translation vector.

    Returns:
        (12, 4) 4 right-most singular vectors of weighted-constraints matrix
    """

    n = U.shape[1]

    # rough estimation of the depth of the points in camera reference.
    depths_c = R[2].dot(Pw) + t[2]

    fu, fv = A[0, 0], A[1, 1]
    u0, v0 = A[0, 2], A[1, 2]

    M = np.empty((2 * n, 12))

    for i in range(n):
        # unweighted terms.
        row = 2 * i
        row_n = row + 1

        M[row, 0] = alphas[0, i] * fu
        M[row, 1] = 0.0
        M[row, 2] = alphas[0, i] * (u0 - U[0, i])
        M[row, 3] = alphas[1, i] * fu
        M[row, 4] = 0.0
        M[row, 5] = alphas[1, i] * (u0 - U[0, i])
        M[row, 6] = alphas[2, i] * fu
        M[row, 7] = 0.0
        M[row, 8] = alphas[2, i] * (u0 - U[0, i])
        M[row, 9] = alphas[3, i] * fu
        M[row, 10] = 0.0
        M[row, 11] = alphas[3, i] * (u0 - U[0, i])

        M[row_n, 0] = 0.0
        M[row_n, 1] = alphas[0, i] * fv
        M[row_n, 2] = alphas[0, i] * (v0 - U[1, i])
        M[row_n, 3] = 0.0
        M[row_n, 4] = alphas[1, i] * fv
        M[row_n, 5] = alphas[1, i] * (v0 - U[1, i])
        M[row_n, 6] = 0.0
        M[row_n, 7] = alphas[2, i] * fv
        M[row_n, 8] = alphas[2, i] * (v0 - U[1, i])
        M[row_n, 9] = 0.0
        M[row_n, 10] = alphas[3, i] * fv
        M[row_n, 11] = alphas[3, i] * (v0 - U[1, i])

        # weighting with full covariance -its square root with Cholesky decomp.
        M[row: row + 2] = np.linalg.cholesky(np.linalg.inv(
            depths_c[i] * depths_c[i] * sigmas2d[i] + 1e-12)
        ).T.dot(M[row: row + 2])

    # 4 right-most singular vectors.
    return np.linalg.svd(M.T.dot(M))[0][:, -1:-5:-1]


@njit("f8[:,:](f8[:,::1], f8[:,::1], f8[:,::1], f8[:,:,::1], "
      "f8[:,::1], f8[:,::1])", fastmath=True, cache=True)
def _get_rsvs_3d(U, A, alphas, sigmas3d_c, R, t):
    """Get 4-rightmost singular vectors of the weighted-constraints matrix.

    Args:
        U: (2,n) image plane coordinates.
        A: (3,3) calibration matrix:
            [[fu, 0, u0],
             [0, fv, v0],
             [0,  0, 1]]
        alphas: (4,n) homogeneous baricentric coordinates.
        sigmas3d_c: (n, 3, 3) covariance matrices of 3D points locations in
            *camera* reference.
        Pw: (3, n) 3D points coordinates in world reference.
        R: (3, 3) rough estimation of covariance matrix.
        t: (3, 1) rough estimation of translation vector.

    Returns:
        (12, 4) 4 right-most singular vectors of weighted-constraints matrix
    """

    n = U.shape[1]

    fu, fv = A[0, 0], A[1, 1]
    u0, v0 = A[0, 2], A[1, 2]

    M = np.empty((2 * n, 12))

    for i in range(n):
        # unweighted terms.
        row = 2 * i
        row_n = row + 1

        M[row, 0] = alphas[0, i] * fu
        M[row, 1] = 0.0
        M[row, 2] = alphas[0, i] * (u0 - U[0, i])
        M[row, 3] = alphas[1, i] * fu
        M[row, 4] = 0.0
        M[row, 5] = alphas[1, i] * (u0 - U[0, i])
        M[row, 6] = alphas[2, i] * fu
        M[row, 7] = 0.0
        M[row, 8] = alphas[2, i] * (u0 - U[0, i])
        M[row, 9] = alphas[3, i] * fu
        M[row, 10] = 0.0
        M[row, 11] = alphas[3, i] * (u0 - U[0, i])

        M[row_n, 0] = 0.0
        M[row_n, 1] = alphas[0, i] * fv
        M[row_n, 2] = alphas[0, i] * (v0 - U[1, i])
        M[row_n, 3] = 0.0
        M[row_n, 4] = alphas[1, i] * fv
        M[row_n, 5] = alphas[1, i] * (v0 - U[1, i])
        M[row_n, 6] = 0.0
        M[row_n, 7] = alphas[2, i] * fv
        M[row_n, 8] = alphas[2, i] * (v0 - U[1, i])
        M[row_n, 9] = 0.0
        M[row_n, 10] = alphas[3, i] * fv
        M[row_n, 11] = alphas[3, i] * (v0 - U[1, i])

        # weighting with full covariance -its square root with Cholesky decomp.
        # fmt: off
        M[row : row+2] = np.linalg.cholesky(np.linalg.inv(
            # S
            sigmas3d_c[i, :2, :2]
            # \gamma uu^T
            + sigmas3d_c[i, 2, 2] * (np.expand_dims(U[:, i], axis=1) * U[:, i])
            # - wu^T
            - (np.expand_dims(sigmas3d_c[i, 2, :2], axis=1) * U[:, i])
            # - uw^T
            - (np.expand_dims(U[:, i], axis=1) * sigmas3d_c[i, 2, :2])
            + 1e-12)
            ).T.dot(M[row : row+2])
        # fmt: on

    # 4 right-most singular vectors.
    return np.linalg.svd(M.T.dot(M))[0][:, -1:-5:-1]


if __name__ == "__main__":
    from scipy.stats import special_ortho_group as SO

    def generate_pc(n):
        U = np.random.rand(2, n) * 500
        d = np.random.rand(n,) * 100
        Pc = np.linalg.inv(A) @ np.r_[U, np.ones((1, n))] * d
        Pw = R_test.T @ Pc - R_test.T.dot(t_test)[:, None]

        sigmas2d_test = np.repeat(np.eye(2)[None], n, axis=0)
        sigmas3d_test = np.repeat(np.eye(3)[None], n, axis=0)
        return U, Pc, Pw, sigmas2d_test, sigmas3d_test

    A = np.array([
        [480., 0, 100],
        [0, 350., 200],
        [0., 0, 1],
    ])
    R_test = SO.rvs(3)
    t_test = np.random.rand(3,)

    U, Pc, Pw, sigmas2d_test, sigmas3d_test = generate_pc(1000)

    cfg_epnpu = {
        "use_pca": True,
        "th_reproj_error": -1,
        "use_median": False,
        "refine_w_cps": True,
        "optimize_best_only": False,
        "gn_iters": 5,
        "verbose": False,
        "p3p": {
            'method': 'ap3p',
            'reproj_th': 6.0,
            'confidence': 0.999,
            'max_iter': 1_000
        },
    }

    cfg_epnp = {
        "use_pca": True,
        "th_reproj_error": -1,
        "use_median": False,
        "refine_w_cps": True,
        "optimize_best_only": False,
        "gn_iters": 5,
        "verbose": False,
    }

    model_epnp = epnp.EPnP(cfg_epnp)
    model_epnpu = EPnPU(cfg_epnpu)

    # model_epnp(Pw, U, A)
    # model_epnpu(Pw, U, A, sigmas2d_test, sigmas3d_test)

    cProfile.run(
        'for _ in range(1000): model_epnpu(Pw, U, A, sigmas2d_test, sigmas3d_test)',
        sort='cumtime')
