from typing import Union
from math import sqrt

import numpy as np
from numba import njit


UmeyamaTransform = Union[
    tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, float]
]


@njit('f8[:,::1](f8[:,::1])', fastmath=True, cache=True)
def _mean_axis1(arr):
    """Compute mean of 2D array through last axis"""
    d, n = arr.shape
    out = np.empty((d, 1))
    for di in range(d):
        out[di] = np.sum(arr[di]) / n
    return out


@njit('f8[:,::1](f8[:,::1], f8[:,::1], f8[:,::1], f8[:,::1])', fastmath=True, cache=True)
def project(
    Pw: np.ndarray, A: np.ndarray, R_cw: np.ndarray, t_cw: np.ndarray
) -> np.ndarray:
    """Project (3,n) matrix of 3D points to image plane -> (2,n)"""
    Urhom = A @ (R_cw @ Pw + t_cw)
    return Urhom[:-1] / Urhom[-1]


def reproj_error(
    Pw: np.ndarray,
    U: np.ndarray,
    R_cw: np.ndarray,
    t_cw: np.ndarray,
    A: np.ndarray,
    reduction_str: str = "mean",
) -> float:
    """Compute the reprojection error from a pose estimate

    Args:
        P: matrix with "n" 3D points in cartesian, (3,n) or (3,),
             or homogeneous, (4,n) or (4,) coordinates.
        U: matrix with "n" 2D image plane points in cartensian, (2,n) or (2,),
             or homogeneous, (3,n) or (3,) coordinates.
        R_cw: rotation matrix, (3,3).
        t_cw: translation vector, (3,1) ir (3,)
        A: calibration matrix, (3,3)
        reduction: reduction applied to the output. It should share the name
                   with a numpy function, e.g. 'mean' will be executed as
                   np.mean(array).

    Returns:
        reduced reprojection errors.
    """
    # ensure 2d matrices as inputs.
    Pshape = Pw.shape
    Ushape = U.shape

    assert Pshape[0] in [3, 4], Pshape
    assert Ushape[0] in [2, 3], Ushape

    if len(Pshape) == 1:
        Pw = Pw[:, None]
    if len(Ushape) == 1:
        U = U[:, None]

    reduction = getattr(np, reduction_str)

    # ensure cartesian coordinates.
    if Pshape[0] == 4:
        Pw = Pw[:-1] / Pw[-1]
    if Ushape[0] == 3:
        U = U[:-1] / U[-1]

    errors = U - project(Pw, A, R_cw, t_cw)
    return reduction(np.sqrt(np.einsum("ij, ij -> j", errors, errors)))


@njit('f8(f8[:,::1], f8[:,::1], f8[:,::1], f8[:,::1], f8[:,::1], b1)', fastmath=True)
def reproj_error_nb(
        Pw: np.ndarray,
        U: np.ndarray,
        R_cw: np.ndarray,
        t_cw: np.ndarray,
        A: np.ndarray,
        use_median: bool):
    """Faster than reproj_error but with stricter input shapes (see decorator)"""
    errors = U - project(Pw, A, R_cw, t_cw)
    if use_median:
        return np.median(np.sqrt(errors[0]**2 + errors[1]**2))
    return np.mean(np.sqrt(errors[0]**2 + errors[1]**2))


def umeyama_alignment(
    Pc: np.ndarray, Pw: np.ndarray, compute_scale: bool = False
) -> UmeyamaTransform:
    """Sim(3) or SE(3) transform minimizing L2 error between two point sets.

    The method followed corresponds to the one presented in [1]. The
    transformation minimizes the following cost:
        sum_i || Pc[:,i] - (s*R_cw*Pw[:,i] + t_cw) ||^2
    Thus, we compute the transformation of coordinate system "w" as seen
    from the coordinate system "c".
    [1] "Least-squares estimation of transformation parameters between two
    point patterns", S. Umeyama.

    Args:
        Pc: (3,n) points expressed in coordinate system c.
        Pw: (3,n) points expressed in coordinate system w.
        compute_scale: compute also the scale i.e. return Sim(3) transform.

    Returns:
        R_cw: (3,3) Rotation matrix.
        t_cw: (3,1) translation vector.
        c: () scale. Only returned if compute_scale is True.
    """
    n = Pw.shape[1]

    # Eqs. 34, 35 of paper.
    mu_c = Pc.mean(axis=1, keepdims=True)
    mu_w = Pw.mean(axis=1, keepdims=True)

    Pc_cent = Pc - mu_c
    Pw_cent = Pw - mu_w

    # Eq. 38.
    Sigma_wc = Pc_cent @ Pw_cent.T / n
    U, d, Vt = np.linalg.svd(Sigma_wc)

    # Eq. 39.
    s = np.ones((3,))
    if np.linalg.det(Sigma_wc) < 0:
        s[2] = -1

    if compute_scale:
        # Eq. 42.
        var_w = np.einsum('ij, ij ->', Pw_cent, Pw_cent) / n
        c = d.dot(s) / var_w
    else:
        c = 1.

    # Eqs. 40, 41.
    R = U * s @ Vt
    t = mu_c - R @ mu_w * c

    if compute_scale:
        return R, t, c
    return R, t


@njit('Tuple((f8[:,::1], f8[:,::1]))(f8[:,::1], f8[:,::1])', fastmath=True, cache=True)
def umeyama_alignment_se3(Pc: np.ndarray, Pw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """SE(3) transform minimizing L2 error between two point sets.

    The method is derived in [1]. Here we fix the scale parameter to 1 to
    compute a SE(3) transform instead of a Sim(3) one.
    [1] "Least-squares estimation of transformation parameters between two
    point patterns", S. Umeyama.

    Args:
        Pc: (3,n) points expressed in coordinate system c.
        Pw: (3,n) points expressed in coordinate system w.

    Returns:
        R_cw: (3,3) Rotation matrix.
        t_cw: (3,1) translation vector.
    """
    n = Pw.shape[1]

    # Eqs. 34, 35 of paper.
    mu_c = _mean_axis1(Pc)
    mu_w = _mean_axis1(Pw)

    # Eq. 38.
    Sigma_wc = (Pc - mu_c) @ (Pw - mu_w).T / n
    U, d, Vt = np.linalg.svd(Sigma_wc)

    # Eq. 39.
    s = np.ones((3,))
    if np.linalg.det(Sigma_wc) < 0:
        s[2] = -1

    # Eqs. 40, 41.
    R = U * s @ Vt
    t = mu_c - R @ mu_w

    return R, t


@njit('Tuple((f8[:,::1], f8[:,::1], f8))(f8[:,::1], f8[:,::1])', fastmath=True, cache=True)
def umeyama_alignment_sim3(
        Pc: np.ndarray, Pw: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Sim(3) transform minimizing L2 error between two point sets.

    Derivation of implemented method in [1].
    [1] "Least-squares estimation of transformation parameters between two
    point patterns", S. Umeyama.

    Args:
        Pc: (3,n) points expressed in coordinate system c.
        Pw: (3,n) points expressed in coordinate system w.
        compute_scale: compute also the scale i.e. return Sim(3) transform.

    Returns:
        R_cw: (3,3) Rotation matrix.
        t_cw: (3,1) translation vector.
        c: () scale. Only returned if compute_scale is True.
    """
    n = Pw.shape[1]

    # Eqs. 34, 35 of paper.
    mu_c = _mean_axis1(Pc)
    mu_w = _mean_axis1(Pw)

    Pc_cent = Pc - mu_c
    Pw_cent = Pw - mu_w

    # Eq. 38.
    Sigma_wc = Pc_cent @ Pw_cent.T / n
    U, d, Vt = np.linalg.svd(Sigma_wc)

    # Eq. 39.
    s = np.ones((3,))
    if np.linalg.det(Sigma_wc) < 0:
        s[2] = -1

    # Eq. 42.
    var_w = np.linalg.norm(Pw_cent)**2 / n
    c = d.dot(s) / var_w

    # Eqs. 40, 41.
    R = U * s @ Vt
    t = mu_c - R @ mu_w * c

    return R, t, c


@njit('f8(f8[:], f8[:])', fastmath=True, cache=True)
def _sqeuclidean_dist(a, b):
    """Squared Euclidean distance between two 1D arrays 'a' and 'b'."""
    n, = a.shape
    out = 0.0
    for i in range(n):
        out += (a[i] - b[i])**2
    return out


@njit('f8[:,::1](f8[:,:], f8[:,:], b1)', fastmath=True, cache=True)
def _euc_cdist(a, b, apply_sqrt):
    """Distance between pairs of the two inputs. Similar to Scipy's cdist."""
    na, da = a.shape
    nb, db = b.shape

    out = np.empty((na, nb))
    for i in range(na):
        for j in range(nb):
            out[i, j] = _sqeuclidean_dist(a[i], b[j])
            if apply_sqrt:
                out[i, j] = sqrt(out[i, j])
    return out


@njit('f8[::1](f8[:,:], b1)', fastmath=True, cache=True)
def _euc_pdist(a, apply_sqrt):
    """Pairwise dist. of 1D arrs whitin a 2D arr. Similar to Scipy's pdist."""
    na, da = a.shape
    out = np.empty((na * (na - 1) // 2))

    idx = 0
    for i in range(na - 1):
        for j in range(i + 1, na):
            out[idx] = _sqeuclidean_dist(a[i], a[j])
            if apply_sqrt:
                out[idx] = sqrt(out[idx])
            idx += 1
    return out
