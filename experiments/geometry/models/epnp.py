"""EPnP implementation"""

import sys
from pathlib import Path
from math import sqrt
from typing import Union
from pprint import pprint
import cProfile

import numpy as np
from numba import njit

ENV_PATH = str(Path(__file__).parents[3])
if ENV_PATH not in sys.path:
    print(f"inserting {ENV_PATH} to sys.path.")
    sys.path.insert(0, ENV_PATH)
from experiments.geometry.models import geometry as geom


class EPnP:
    """Implementation of EPnP: An accurate O(n) solution to the PnP problem.

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
    """

    default_cfg: dict[str, Union[bool, int]] = {
        "use_pca": True,
        "th_reproj_error": -1,
        "use_median": False,
        "refine_w_cps": True,
        "optimize_best_only": False,
        "gn_iters": 5,
        "verbose": False,
    }

    def __init__(self, cfg=None):
        if cfg is None:
            cfg = {}
        self.cfg = {**self.default_cfg, **cfg}

        # safety checks for given configuration.
        msg = "'{}' is expected to be {} but type {} was given."
        for d_key in self.default_cfg.keys():
            if not isinstance(self.cfg[d_key], type(self.default_cfg[d_key])):
                raise TypeError(msg.format(
                    d_key,
                    type(self.default_cfg[d_key]).__name__,
                    type(self.cfg[d_key]).__name__
                ))

        if self.cfg['verbose']:
            print("\nEPnP configuration:")
            pprint(self.cfg)

    def __call__(
            self, Pw: np.ndarray, U: np.ndarray, A: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Camera localization given 2D-3D correspondences with EPnP method.

        Args:
            Pw: (3, n) given 3D coordinates of n points.
            U: (2, n) corresponding 2D coordinates in the image plane.
            A: (3, 3) camera calibration matrix of the form
                [[fu, 0, u0],
                 [0, fv, v0],
                 [0,  0, 1]]

        Returns:
            R_cw: (3, 3) rotation matrix that rotates points from world
                to camera coordinate system, i.e. p_c = R_cw * p_w + t_cw
            t_cw: (3, 1) translation vector.
            error: reprojection error of the returned solution.
        """
        # check c-contiguity and type of data (to comply with numba methods).
        if not Pw.flags['C_CONTIGUOUS']:
            Pw = np.ascontiguousarray(Pw)
        if not U.flags['C_CONTIGUOUS']:
            U = np.ascontiguousarray(U)
        if not A.flags['C_CONTIGUOUS']:
            A = np.ascontiguousarray(A)

        if Pw.dtype != float:
            Pw = Pw.astype(float)
        if U.dtype != float:
            U = U.astype(float)
        if A.dtype != float:
            A = A.astype(float)

        # control points in world reference.
        Cw = _define_control_points(self.cfg['use_pca'], Pw)

        # homogeneous barycentric coordinates (independent of the reference).
        alphas = _compute_alphas(Pw, Cw)

        # construct M (matrix of projection constraints).
        M = _compute_M(U, A, alphas)

        # 4 right-most sigular vectors (conservative null space of M).
        V = np.ascontiguousarray(_get_rsv(M, 4, asc_order=True))

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
        beta, Cc, Pc = _solve_caseN1_or_refine(
            self.cfg["refine_w_cps"], V[:, 0], alphas, dist_w)
        betas = np.array([beta, 0.0, 0.0, 0.0])

        R_cw, t_cw = geom.umeyama_alignment_se3(Pc, Pw)
        error = geom.reproj_error_nb(
            Pw, U, R_cw, t_cw, A, self.cfg["use_median"])

        if (not self.cfg["optimize_best_only"]) & (self.cfg["gn_iters"] > 0):
            betas_opt = _gauss_newton_betas(
                V, betas, dist_cw, self.cfg["gn_iters"])
            Pc = _solve_for_sign_and_pointcloud(V, betas_opt, alphas)

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
            L6_10 = _compute_L6_10(V)
            # We may have already computed rho. Check.
            if self.cfg["gn_iters"] <= 0:
                rho = geom._euc_pdist(Cw.T, apply_sqrt=False)
            else:
                rho = dist_cw

            # raw solution.
            betas = _solve_caseN2(L6_10, rho)
            Cc = V.dot(betas)

            # refinement.
            beta, Cc, Pc = _solve_caseN1_or_refine(
                self.cfg["refine_w_cps"], Cc, alphas, dist_w)
            betas *= beta

            R_cw, t_cw = geom.umeyama_alignment_se3(Pc, Pw)
            error = geom.reproj_error_nb(
                Pw, U, R_cw, t_cw, A, self.cfg["use_median"])

            if (not self.cfg["optimize_best_only"]) & (self.cfg["gn_iters"] > 0):
                betas_opt = _gauss_newton_betas(
                    V, betas, dist_cw, self.cfg["gn_iters"])
                Pc = _solve_for_sign_and_pointcloud(V, betas_opt, alphas)

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
            betas = _solve_caseN3(L6_10, rho)
            Cc = V.dot(betas)

            # refinement.
            beta, Cc, Pc = _solve_caseN1_or_refine(
                self.cfg["refine_w_cps"], Cc, alphas, dist_w)
            betas *= beta

            R_cw, t_cw = geom.umeyama_alignment_se3(Pc, Pw)
            error = geom.reproj_error_nb(
                Pw, U, R_cw, t_cw, A, self.cfg["use_median"])

            if (not self.cfg["optimize_best_only"]) & (self.cfg["gn_iters"] > 0):
                betas_opt = _gauss_newton_betas(
                    V, betas, dist_cw, self.cfg["gn_iters"])
                Pc = _solve_for_sign_and_pointcloud(V, betas_opt, alphas)

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
            betas = _solve_caseN4(L6_10, rho)
            Cc = V.dot(betas)

            # refinement.
            beta, Cc, Pc = _solve_caseN1_or_refine(
                self.cfg["refine_w_cps"], Cc, alphas, dist_w)
            betas *= beta

            R_cw, t_cw = geom.umeyama_alignment_se3(Pc, Pw)
            error = geom.reproj_error_nb(
                Pw, U, R_cw, t_cw, A, self.cfg["use_median"])

            if (not self.cfg["optimize_best_only"]) & (self.cfg["gn_iters"] > 0):
                betas_opt = _gauss_newton_betas(
                    V, betas, dist_cw, self.cfg["gn_iters"])
                Pc = _solve_for_sign_and_pointcloud(V, betas_opt, alphas)

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
            betas = _gauss_newton_betas(
                V, best_case_data[-1], dist_cw, self.cfg["gn_iters"])
            Pc = _solve_for_sign_and_pointcloud(V, betas, alphas)

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


@njit('f8[:,::1](b1, f8[:,::1])', fastmath=True, cache=True)
def _define_control_points(use_pca: bool, Pw: np.ndarray) -> np.ndarray:
    """Define the control points of the barycentric coordinate system.

    If use_pca is True, the control points correspond to 1) the centroid
    of the point cloud and 2), 3), 4) form a basis aligned with the
    principal directions of the pointcloud (w. centroid as origin). Their
    "lengths" correspond to the std found in the p. directions. This is
    the option recommended in the paper. Otherwise, the control points
    correspond to the world origin and basis.

    Args:
        use_pca: if True, the first option explained above is used.
        Pw: (3, n), points in world reference. Ignored if use_pca is False.

    Returns:
        (3, 4), control points of the barycentric system.
    """
    if use_pca:
        centroid = geom._mean_axis1(Pw)
        Pw_cent = Pw - centroid
        v, s, _ = np.linalg.svd(Pw_cent @ Pw_cent.T)
        return np.concatenate(
            (v * np.expand_dims(np.sqrt(s / Pw.shape[1]), axis=0) + centroid, centroid), axis=1
        )

    return np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
    )


@njit('f8[:,::1](f8[:,:], f8[:,:])', cache=True)
def _compute_alphas(Pw: np.ndarray, Cw: np.ndarray) -> np.ndarray:
    """Compute alphas (4,n) such that Pw = Cw * alphas

    Matrices are augmented with ones to add the constaint sum(alphas) = 1.
    The system is solved with PLU factorization for greater stability.

    Args:
        Pw: (3, n), points in world reference.
        Cw: (3, 4), control points in world reference.

    Returns:
        (4, n) array with the homogeneous barycentric coordinates of Pw.
    """
    return np.ascontiguousarray(
        np.linalg.solve(
            np.concatenate((Cw, np.ones((1, 4)))),
            np.concatenate((Pw, np.ones((1, Pw.shape[1]))))
        ))


@njit("f8[:,::1](f8[:,::1], f8[:,::1], f8[:,::1])", fastmath=True, cache=True)
def _compute_M(U, A, alphas):
    """Fast construction of the matrix of constraints M

    Args:
        U: (2,n) image plane coordinates.
        A: (3,3) calibration matrix:
            [[fu, 0, u0],
             [0, fv, v0],
             [0,  0, 1]]
        alphas: (4,n) homogeneous baricentric coordinates.

    Returns:
        (2 * n, 12) matrix M (see eq. 7 of paper).
    """
    n = U.shape[1]

    fu, fv = A[0, 0], A[1, 1]
    u0, v0 = A[0, 2], A[1, 2]

    M = np.empty((2 * n, 12))

    for i in range(n):

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

    return M


@njit('f8[:,:](f8[:,::1], i8, b1)', cache=True)
def _get_rsv(M: np.ndarray, d: int, asc_order: bool = False) -> np.ndarray:
    """Get last d right-singular vectors of matrix A.

    why compute M.T @ M?
    As we know, svd(M) = U @ S @ V.T, and svd(M.T @ M) = V @ S^2 @ V.T
    i.e. same right-singular vectors. OTOH svd has a O(n^2 * m) time
    complexity for a tall (n, m) matrix, and multiplication M.T @ M is
    O(m*n*m). Thereby is more efficient to apply svd on the latter.

    Args:
        M: (2 * n, 12) matrix of constraints (see eq. 7 of paper).
        d: number of right-singular vectors to return.
        asc_order: if True the singular vectors are returned in ascending order
            based on their corresponding singular values.

    Returns:
        (12, d), right-most singular vectors
    """
    if asc_order:
        return np.linalg.svd(M.T.dot(M))[0][:, -1: -d - 1: -1]
    return np.linalg.svd(M.T.dot(M))[0][:, -d:]


@njit('Tuple((f8, f8[:,:], f8[:,::1]))(b1, f8[:], f8[:,::1], f8[::1])', fastmath=True, cache=True)
def _solve_caseN1_or_refine(
        refine_w_cps: bool, v: np.ndarray, alphas: np.ndarray, dist_w: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    """Least-squares solution for case N=1 or refinement for other cases

    Two versions are considered:
        1) if refine_w_cps: The way that the paper describes (Eq. 11).
            The constraints correspond to the norm of the pairwise relative
            differences between the control points in both references.
        2) else: The way implemented at [1] (see
            function compute_norm_sign_scaling_factor.m). The constraints
            correspond to the norm of the relative differences between the
            3D points and their centroid in both coordinate systems.
    [1] https://github.com/cvlab-epfl/EPnP/blob/master/matlab/EPnP

    Args:
        refine_w_cps: defines the constraints used.
        v: (12,) right-most singular vector of M or vector to refine its scale.
        alphas: (4, n) barycentric coordinates of the 3d points.
        dist_w: (6,) or (n,) values w.r.t. optimization is done, depending
                in refine_w_cps.
    Returns:
        beta: optimized scale factor of the input vector.
        Cc: (3, 4) estimated control-point coordinates in cam. ref.
        Pc: (3, n) Observed points in camera reference.
    """
    # singular vector shape to control points shape.
    Cc = v.copy().reshape(4, 3).T

    if refine_w_cps:
        # pairwise euclidean distances.
        dist_c = geom._euc_pdist(Cc.T, apply_sqrt=True)
        # least square estimate of the scale.
        beta = dist_c.dot(dist_w) / dist_c.dot(dist_c)
        # solution.
        Cc *= beta
        Pc = Cc @ alphas
    else:
        # current estimate of cartesian point cloud coordinates.
        Pc = Cc @ alphas
        # euclidean distances w.r.t. the centroid.
        dist_c = geom._euc_cdist(geom._mean_axis1(
            Pc).T, Pc.T, apply_sqrt=True)[0]
        # least square estimate of the scale.
        beta = dist_c.dot(dist_w) / dist_c.dot(dist_c)
        # scaled control points and new point coordinates.
        Cc *= beta
        Pc *= beta

    # norm constraints are always > 0 -> negative depths can arise. Check.
    if np.any(Pc[2] < 0):
        # print('uaaaa')
        Pc *= -1
        Cc *= -1
        beta *= -1
    return beta, Cc, Pc


@njit("f8[::1](f8[:,::1], f8[::1], f8[::1], i8)", fastmath=True, cache=True)
def _gauss_newton_betas(
        kernel: np.ndarray, betas: np.ndarray, dist_cw: np.ndarray, niter: int
) -> np.ndarray:
    """Gauss-Newton optimization of the betas.

    As defined in eq. 15, each residual is defined as:
        || c_i^c - c_j^c ||^2 - || c_i^w - c_j^w ||^2,
    the first term depends on the betas:
        c_i^c - c_j^c = dC_ij @ betas,
    where dC_ij is a 4x4 matrix with the substraction of reshaped singular
    vectors as columns. Therefore the error can be rewritten as:
        betas^T @ dC_ij.T @ dC_ij @ betas - || c_i^w - c_j^w ||^2 =
        betas^T @ dCTdC_ij @ betas - || c_i^w - c_j^w ||^2.
    From which the Jacobian w.r.t. betas, J_ij, is derived:
        J_ij = 2 * betas^T @ dCTdC_ij.
    For efficiency, we can precompute dCTdC, as it is constant.
    The complete Jacobian, J is a 6x4 matrix that results from stacking
    all the J_ij.

    Args:
        kernel: (12, 4) right-most singular vectors.
        betas: (4,) initial value of the betas.
        dist_cw: (6,) contains each || c_i^w - c_j^w ||^2 value.

    Returns:
        betas: refined betas.
    """
    kt = kernel.T
    betas = betas.copy()

    dCTdC = np.empty((6, 4, 4))
    row = 0

    for i in range(3):
        for j in range(i + 1, 4):
            dc_ij = kt[:, 3 * i: 3 * i + 3] - kt[:, 3 * j: 3 * j + 3]

            # equivalent line -> shorter but slower.
            # dCTdC[row] = dc_ij @ dc_ij.T

            dCTdC[row, 0, 0] = dc_ij[0].dot(dc_ij[0])
            dCTdC[row, 0, 1] = dc_ij[0].dot(dc_ij[1])
            dCTdC[row, 0, 2] = dc_ij[0].dot(dc_ij[2])
            dCTdC[row, 0, 3] = dc_ij[0].dot(dc_ij[3])
            dCTdC[row, 1, 1] = dc_ij[1].dot(dc_ij[1])
            dCTdC[row, 1, 2] = dc_ij[1].dot(dc_ij[2])
            dCTdC[row, 1, 3] = dc_ij[1].dot(dc_ij[3])
            dCTdC[row, 2, 2] = dc_ij[2].dot(dc_ij[2])
            dCTdC[row, 2, 3] = dc_ij[2].dot(dc_ij[3])
            dCTdC[row, 3, 3] = dc_ij[3].dot(dc_ij[3])

            # symmetric terms.
            dCTdC[row, 1, 0] = dCTdC[row, 0, 1]
            dCTdC[row, 2, 0] = dCTdC[row, 0, 2]
            dCTdC[row, 3, 0] = dCTdC[row, 0, 3]
            dCTdC[row, 2, 1] = dCTdC[row, 1, 2]
            dCTdC[row, 3, 1] = dCTdC[row, 1, 3]
            dCTdC[row, 3, 2] = dCTdC[row, 2, 3]

            row += 1

    # Auxiliar jacobian / 2 (the div by 2 is not done for reusing purposes).
    Jdiv2 = np.empty((6, 4))
    for k in range(niter):
        # Loop to fill Jdiv2 since numba does not support 3D multiplications.
        for m in range(6):
            Jdiv2[m] = dCTdC[m].dot(betas)

        # gauss newton step.
        betas += np.linalg.solve(
            4.0 * Jdiv2.T.dot(Jdiv2), 2.0
            * Jdiv2.T.dot(dist_cw - Jdiv2.dot(betas))
        )
    return betas


@njit("f8[:,::1](f8[:,::1], f8[::1], f8[:,::1])", fastmath=True, cache=True)
def _solve_for_sign_and_pointcloud(V, betas, alphas):
    """Points in camera ref. and (possible) inplace sign correction of betas.

    Args:
        V: (12, 4) 4 right-most singular vectors.
        betas: (4,) estimated linear combination of V.
        alphas: (4, n) barycentric coordinates.
    Returns:
        Pc: (3, n) point cloud in camera reference.
    """
    # Cc (control points) are not needed. Thereby we compute Pc directly.
    Pc = V.dot(betas).reshape(4, 3).T @ alphas
    if np.any(Pc[2] < 0):
        Pc *= -1
        betas *= -1
    return Pc


@njit("f8[:,::1](f8[:,::1])", fastmath=True, cache=True)
def _compute_L6_10(kernel: np.ndarray) -> np.ndarray:
    """Compute matrix L, of size (6, 10), corresponding to case N=4

    This is the matrix of the system (see Sec. 3.3):
        L * betas = rho,
    For the case N=4. It arises from the "linearization" of the
    squared-norm constraints mentioned in Eq. 10.
    The cases N = {2, 3} are a subset of this case. Therefore their
    corresponding matrix L is formed by blocks of the one computed here.

    Args:
        kernel: (12, 4) nullspace of M. Expected order:
            [v_L, v_(L-1), v_(L-2), v_(L-3)],
            "L" is the index of the last (smallest) singular value/vector.
    Returns:
        L: (6, 10), matrix of constraints. The corresponding order of
        the linearized betas (parameters to solve) is:
            betas_lin = [B11, B12, B13, B14, B22, B23, B24, B33, B34, B44]
    """
    vs = kernel.T

    L = np.empty((6, 10))
    row = 0

    for i in range(3):
        for j in range(i + 1, 4):
            vdiff = vs[:, 3 * i: 3 * i + 3] - vs[:, 3 * j: 3 * j + 3]

            L[row, 0] = vdiff[0].dot(vdiff[0])
            L[row, 1] = 2 * vdiff[0].dot(vdiff[1])
            L[row, 2] = 2 * vdiff[0].dot(vdiff[2])
            L[row, 3] = 2 * vdiff[0].dot(vdiff[3])
            L[row, 4] = vdiff[1].dot(vdiff[1])
            L[row, 5] = 2 * vdiff[1].dot(vdiff[2])
            L[row, 6] = 2 * vdiff[1].dot(vdiff[3])
            L[row, 7] = vdiff[2].dot(vdiff[2])
            L[row, 8] = 2 * vdiff[2].dot(vdiff[3])
            L[row, 9] = vdiff[3].dot(vdiff[3])

            row += 1

    return L


@njit('f8[::1](f8[:,::1], f8[::1])', fastmath=True, cache=True)
def _solve_caseN2(L6_10: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """Case N=2.

    Solving for [beta_{11}, beta_{12}, beta_{22}].
    NOTE: after linearization, it is possible that the returned values for
    beta_{11} = beta1**2, beta_{22} = beta2**2 (positive by definition)
    become negative, which is not a valid solution to the original
    system. This can happen since the linearization process destroys
    information. As a consequence, if a solution to the  original system
    exists, then  it would also be a solution to the  linearized one.
    However the converse is false [Bard, 2009].
    To handle this, the official and OpenCV implementations, keep the
    absolute value. However, personally I believe they should
    be considered spurious.
    Here we follow the original implementation.

    Args:
        L6_10: (6, 10) matrix of constraints.
        rho: (6,) squared pairwise distances between control points in
            world reference.
    Returns:
        betas: (4,) estimates [beta_{1}, beta_{2}, 0, 0]
    """
    # submatrix of interest (comnstraints of beta11, beta12, beta22).
    L = np.vstack((L6_10[:, 0], L6_10[:, 1], L6_10[:, 4])).T

    # solve overdetermined linearized system.
    betas_lin, _, _, _ = np.linalg.lstsq(L, rho)

    # corresponding solution for betas.
    betas = np.zeros((4,))
    betas[0] = sqrt(abs(betas_lin[0]))

    sign2 = -1.0 if (betas_lin[0] > 0) ^ (betas_lin[1] > 0) else 1.0
    betas[1] = sign2 * sqrt(abs(betas_lin[2]))

    return betas


@njit('f8[::1](f8[:,::1], f8[::1])', fastmath=True, cache=True)
def _solve_caseN3(L6_10: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """Case N=3.

    Solving for [beta_{11}, beta_{12}, beta_{13}, beta_{22}, beta_{23},
    beta_{33}]. The NOTE in solve_caseN2() also applies here.

    Args:
        L6_10: (6, 10) matrix of constraints.
        rho: (6,) squared pairwise distances between control points in
            world reference.

    Returns:
        betas: (4,) estimates [beta_{1}, beta_{2}, beta_{3}, 0]
    """
    # submatrix of interest.
    L = np.vstack((
        L6_10[:, 0], L6_10[:, 1], L6_10[:, 2],
        L6_10[:, 4], L6_10[:, 5], L6_10[:, 7]
    )).T

    # solve linearized system.
    betas_lin = np.linalg.solve(L, rho)

    # corresponding solution for betas.
    betas = np.zeros((4,))
    betas[0] = sqrt(abs(betas_lin[0]))

    sign2 = -1.0 if (betas_lin[0] > 0) ^ (betas_lin[1] > 0) else 1.0
    betas[1] = sign2 * sqrt(abs(betas_lin[3]))

    sign3 = -1.0 if (betas_lin[0] > 0) ^ (betas_lin[2] > 0) else 1.0
    betas[2] = sign3 * sqrt(abs(betas_lin[5]))

    return betas


@njit("f8[::1](f8[:,::1])", fastmath=True, cache=True)
def _get_linear_combination_of_rsv(V: np.ndarray) -> np.ndarray:
    """Right linear combination of the null space with relinearization.

    We know that the solution of the betas for N=4 lie in the null space,
    ker([L6_10, -rho]) = (v1, v2, v3, v4, v5), with each v_i in R^11,
    formed with the original constraints. This solution can be expressed as
    [betas_lin, 1] = lambda1 * v1 + ... + lambda2 * v5.
    This method computes each lambda. For that, we impose algebraic
    constraints (see Eq. 14 of the paper). For instance:
    betas_lin[0] * betas_lin[4] -  betas_lin[1] * betas_lin[1] = 0.
    Extending it, we end up with a constraint that depends quadratically
    on the lambdas, which is later linearized (as we did with the betas).
    Concretely, there are 15 quadratic combinations of lambdas:
    [lambda_11, lambda_12, ..., lambda_15, lambda_21, ..., lambda_55]

    Args:
        V: (11, 5), right-singular vectors conforming the null space.

    Returns:
        lambdas: (5,) linear combination of the null-space, according to the
            algebraic constraints.
    """
    lambdas = np.empty((5,))

    N = 5  # dimension of the kernel
    n = 4  # dimension of Bij
    idx = np.array([[0, 1, 2, 3], [1, 4, 5, 6], [2, 5, 7, 8], [3, 6, 8, 9]])

    nrowsK = n * (n - 1) // 2 + n * (n - 1) * n // 2
    ncolsK = N * (N + 1) // 2
    K = np.empty((nrowsK, ncolsK))

    # constraints Bii.Bjj - Bij.Bij = 0, (n(n-1)/2 eqs).
    t = 0
    for i in range(n):
        for j in range(i + 1, n):
            offset = 0
            for a in range(N):
                for b in range(a, N):
                    if a == b:
                        K[t, offset] = (
                            V[idx[i, i], a] * V[idx[j, j], a] -
                            V[idx[i, j], a] * V[idx[i, j], a]
                        )
                    else:
                        K[t, offset] = (
                            V[idx[i, i], a] * V[idx[j, j], b] -
                            V[idx[i, j], a] * V[idx[i, j], b] +
                            V[idx[i, i], b] * V[idx[j, j], a] -
                            V[idx[i, j], b] * V[idx[i, j], a]
                        )
                    offset += 1
            t += 1

    for k in range(n):
        for j in range(k, n):
            for i in range(n):
                if (i != j) & (i != k):
                    offset = 0
                    for a in range(N):
                        for b in range(a, N):
                            if a == b:
                                K[t, offset] = (
                                    V[idx[i, j], a] * V[idx[i, k], a] -
                                    V[idx[i, i], a] * V[idx[j, k], a]
                                )
                            else:
                                K[t, offset] = (
                                    V[idx[i, j], a] * V[idx[i, k], b] -
                                    V[idx[i, i], a] * V[idx[j, k], b] +
                                    V[idx[i, j], b] * V[idx[i, k], a] -
                                    V[idx[i, i], b] * V[idx[j, k], a]
                                )
                            offset += 1
                    t += 1

    # The next homogeneous system is up to scale. To disambiguate it, we
    # can fix the last element to be "1" since it comes from including
    # -rho in the null-space. However this can be ignored since we are
    # going to refine the scale later.
    sol = np.linalg.svd(K.T.dot(K))[2][-1]
    sol /= sol[-1]

    # lambdas of interest (related with the linearized betas).
    lambdas[0] = sqrt(abs(sol[0]))
    lambdas[1] = (-1.0 if (sol[0] > 0) ^ (sol[1] > 0)
                  else 1.0) * sqrt(abs(sol[5]))
    lambdas[2] = (-1.0 if (sol[0] > 0) ^ (sol[2] > 0)
                  else 1.0) * sqrt(abs(sol[9]))
    lambdas[3] = (-1.0 if (sol[0] > 0) ^ (sol[3] > 0)
                  else 1.0) * sqrt(abs(sol[12]))
    lambdas[4] = (-1.0 if (sol[0] > 0) ^ (sol[4] > 0)
                  else 1.0) * sqrt(abs(sol[14]))

    return lambdas


@njit('f8[::1](f8[:,::1], f8[::1])', fastmath=True, cache=True)
def _solve_caseN4(L6_10: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """Case N=4.

    Solving for [beta_{11}, beta_{12}, beta_{13}, beta_{14}, beta_{22},
    beta_{23}, beta_{24}, beta_{33}, beta_{34}].
    The system is solved first with linearization, leading to a parametric
    solution that depends on the nullspace of the system. The right linear
    combination of the nullspace is computed with relinearization as
    explained in the original paper, or in more detail in [Kipnis and
    Shamir, 1999].

    Args:
        L6_10: (6, 10) matrix of constraints.
        rho: (6,) squared pairwise distances between control points in
            world reference.

    Returns:
        betas: (4,) estimates [beta_{1}, beta_{2}, beta_{3}, beta_{4}]
    """
    # form and solve underdetermined homogeneous system with the original
    # constraints (it has a 5-d null space).
    null_space = np.linalg.svd(
        np.concatenate((L6_10, np.expand_dims(-rho, axis=1)), axis=1)
    )[2][-5:].T.copy()

    # get the right linear combination from algebraic constraints.
    lambdas = _get_linear_combination_of_rsv(null_space)
    betas_lin = null_space.dot(lambdas)

    # corresponding solution.
    betas = np.empty((4,))
    betas[0] = sqrt(abs(betas_lin[0]))
    betas[1] = (1.0 if betas_lin[1] > 0 else -1.0) * sqrt(abs(betas_lin[4]))
    betas[2] = (1.0 if betas_lin[2] > 0 else -1.0) * sqrt(abs(betas_lin[7]))
    betas[3] = (1.0 if betas_lin[3] > 0 else -1.0) * sqrt(abs(betas_lin[9]))

    return betas


if __name__ == "__main__":
    from scipy.stats import special_ortho_group as SO

    def generate_pc(n):
        U = np.random.rand(2, n) * 500
        d = np.random.rand(n,) * 100
        Pc = np.linalg.inv(A) @ np.r_[U, np.ones((1, n))] * d
        Pw = R.T @ Pc - R.T.dot(t)[:, None]
        return U, Pc, Pw

    A = np.array([
        [480., 0, 100],
        [0, 350., 200],
        [0., 0, 1],
    ])
    R = SO.rvs(3)
    t = np.random.rand(3,)

    U, Pc, Pw = generate_pc(1000)

    cfg_epnp = {
        "use_pca": True,
        "th_reproj_error": -1,
        "use_median": False,
        "refine_w_cps": True,
        "optimize_best_only": False,
        "gn_iters": 5,
        "verbose": False,
    }

    model = EPnP(cfg_epnp)
    # model(Pw, U, A)
    cProfile.run('for _ in range(1000): model(Pw, U, A)', sort='cumtime')
