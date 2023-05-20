from typing import Optional
from functools import partial
import math

import numpy as np
import cv2
from scipy.stats import special_ortho_group as SO
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pio
# pio.renderers.default = 'svg'
pio.renderers.default = 'browser'

from ..epnp import EPnP
from ..epnpu import EPnPU


cfgs_2d_covariances = {
    "iso_hom": {
        "s": 1.0,
        "beta": 0.5,
        "alpha": 0.0
    },
    "iso_inhom": {
        "s": (0.5, 1.5),
        "beta": 0.5,
        "alpha": 0.0
    },
    "ani_hom": {
        "s": 1.0,
        "beta": (0.5, 1.0),
        "alpha": (0., np.pi),
    },
    "ani_inhom": {
        "s": (0.5, 1.5),
        "beta": (0.5, 1.0),
        "alpha": (0.0, np.pi)
    }
}

f = 1e-3
cfgs_3d_covariances = {
    "iso_hom": {
        "s": f,
    },
    "iso_inhom": {
        "s": (0.5 * f, 1.5 * f),
    },
    "ani_hom": {
        "s": f,
    },
    "ani_inhom": {
        "s": (0.5 * f, 1.5 * f),
    }
}


def get_random_generators(seed):
    # random number generator in R.
    rng = np.random.RandomState(seed)
    # random rotation matrix generator in SO(2) and SO(3).
    so3g = SO(3)
    so3g.random_state = np.random.RandomState(seed)
    so2g = SO(2)
    so2g.random_state = np.random.RandomState(seed)
    return rng, so3g, so2g


def get_method_callables(
        methods: list[str], methods_cfg: list[Optional[dict]]):
    available = ["epnp_imp", "epnp_opencv", "epnpu_imp", "epnpu2d_imp"]
    if not all(method in available for method in methods):
        raise ValueError(f"Only the next methods are available: {available}.")

    # create callables to each method.
    callers = []
    for method, cfg in zip(methods, methods_cfg):
        if method == "epnp_imp":
            fun = partial(_imp_epnp_callable, EPnP(cfg))
            fun.name = 'EPnP_imp'
            callers.append(fun)

        elif method == "epnp_opencv":
            fun = _opencv_epnp_callable
            fun.name = 'EPnP_OpenCV'
            callers.append(fun)

        elif method == "epnpu_imp":
            fun = partial(_imp_epnpu_callable, EPnPU(cfg))
            fun.name = 'EPnPU_imp'
            callers.append(fun)

        elif method == "epnpu2d_imp":
            fun = partial(_imp_epnpu2d_callable, EPnPU(cfg))
            fun.name = 'EPnPU_imp'
            callers.append(fun)

        else:
            raise ValueError
    return callers


def _imp_epnp_callable(epnp_instance, p_w, p_im, covs2d, covs3d, only_2d, K):
    return epnp_instance(p_w, p_im, K)


def _imp_epnpu_callable(epnpu_instance, p_w, p_im, covs2d, covs3d, only_2d, K):
    if only_2d:
        return epnpu_instance(p_w, p_im, K, covs2d)
    else:
        return epnpu_instance(p_w, p_im, K, covs2d, covs3d)


def _imp_epnpu2d_callable(
        epnpu_instance, p_w, p_im, covs2d, covs3d, only_2d, K):
    return epnpu_instance(p_w, p_im, K, covs2d)


def _opencv_epnp_callable(p_w, p_im, covs2d, covs3d, only_2d, K):
    _, rvec_tuple, t_tuple, res_arr = cv2.solvePnPGeneric(
        p_w.T, p_im.T, K, np.zeros((4,)), flags=cv2.SOLVEPNP_EPNP)

    # EPnP only returns one solution.
    rvec = rvec_tuple[0]
    t = t_tuple[0]
    res = res_arr[0, 0]

    R, _ = cv2.Rodrigues(rvec)
    return R, t, res


def create_2d_covariances(rng, so2_rng, n, method, std):
    """Synthetic 2D covariances based on [Brooks, 2001] / [Vakhitov, 2021]."""
    if method != "groups":
        # [Brooks, 2001] setup.
        s, beta, alpha = cfgs_2d_covariances[method].values()

        if isinstance(s, tuple):
            s = rng.uniform(s[0], s[1], (n, 1, 1))

        if isinstance(beta, tuple):
            beta = rng.uniform(beta[0], beta[1], (n,))
            beta_m = np.zeros((n, 2, 2))
            beta_m[:, 0, 0] = beta
            beta_m[:, 1, 1] = 1.0 - beta
        else:
            beta_m = np.array([[
                [beta, 0.],
                [0., 1 - beta]
            ]])

        if isinstance(alpha, tuple):
            alpha = rng.uniform(alpha[0], alpha[1], (n,))
            calpha = np.cos(alpha)
            salpha = np.sin(alpha)
            alpha_m = np.empty((n, 2, 2))
            alpha_m[:, 0, 0] = calpha
            alpha_m[:, 0, 1] = -salpha
            alpha_m[:, 1, 0] = salpha
            alpha_m[:, 1, 1] = calpha
        else:
            alpha_m = np.array([[
                [np.cos(alpha), -np.sin(alpha)],
                [np.sin(alpha), np.cos(alpha)]
            ]])

        covs = std * s * alpha_m @ beta_m @ alpha_m.transpose(0, 2, 1)
        if method == "iso_hom":
            covs = np.repeat(covs[None], n, axis=0)

        return np.squeeze(covs)

    else:
        # setup based on [Vakhitov, 2021].
        ngroups = 10
        n_per_group = n // ngroups
        ntrunc = n % ngroups
        nt = ngroups * n_per_group

        sigma_per_group = np.linspace(1.0, 10.0, ngroups)
        sigma1_per_group = rng.uniform(0.0, 1.0, ngroups) * sigma_per_group
        # sigma1_per_group = sigma_per_group  # homogeneous inside group

        covs_per_group = np.zeros((ngroups, 2, 2))
        covs_per_group[:, 0, 0] = sigma_per_group**2
        covs_per_group[:, 1, 1] = sigma1_per_group**2

        Rs = so2_rng.rvs(ngroups)
        covs_per_group = Rs @ covs_per_group @ Rs.transpose(0, 2, 1)

        covs = np.repeat(covs_per_group, n_per_group, 0)

        # add truncated covs.
        if ntrunc > 0:
            tcovs = covs[rng.randint(0, nt, ntrunc)]
            if len(tcovs.shape) == 2:
                tcovs = tcovs[None]
            covs = np.concatenate((covs, tcovs), axis=0)

        return covs


def create_3d_covariances(rng, so3_rng, n, method, std):
    """Create synthetic 3D point covariances."""
    if method != "groups":
        s = cfgs_3d_covariances[method]['s']

        if method == "iso_hom":
            assert isinstance(s, (int, float))
            return np.repeat(std * s * np.eye(3)[None], n, axis=0)

        elif method == "iso_inhom":
            assert isinstance(s, tuple)
            scales = rng.uniform(s[0], s[1], (n, 1, 1))
            return scales * np.repeat(std * np.eye(3)[None], n, axis=0)

        elif method == "ani_hom":
            assert isinstance(s, (int, float))
            rots = so3_rng.rvs(n)
            vars_ = std * s * (vars_ := rng.uniform(0.0, 1.0, 3)) / vars_.sum()
            return rots @ np.diag(vars_) @ rots.transpose(0, 2, 1)

        elif method == "ani_inhom":
            assert isinstance(s, tuple)
            scales = rng.uniform(s[0], s[1], (n, 1))
            rots = so3_rng.rvs(n)

            vars_ = rng.uniform(0.0, 10.0, 3 * n).reshape(n, 3)
            vars_ *= std * scales / vars_.sum(axis=1, keepdims=True)
            diags = np.zeros((n, 3, 3))
            diags[:, 0, 0] = vars_[:, 0]
            diags[:, 1, 1] = vars_[:, 1]
            diags[:, 2, 2] = vars_[:, 2]

            return rots @ diags @ rots.transpose(0, 2, 1)

        else:
            raise ValueError

    else:
        # setup based on [Vakhitov, 2021].
        ngroups = 10
        n_per_group = n // ngroups
        ntrunc = n % ngroups
        nt = ngroups * n_per_group

        sigma_per_group = np.linspace(0.01, 0.1, ngroups)
        sigma1_per_group = rng.uniform(0.0, 1.0, ngroups) * sigma_per_group
        sigma2_per_group = rng.uniform(0.0, 1.0, ngroups) * sigma_per_group
        # sigma1_per_group = sigma_per_group  # homogeneous inside group
        # sigma2_per_group = sigma_per_group  # homogeneous inside group

        covs_per_group = np.zeros((ngroups, 3, 3))
        covs_per_group[:, 0, 0] = sigma_per_group**2
        covs_per_group[:, 1, 1] = sigma1_per_group**2
        covs_per_group[:, 2, 2] = sigma2_per_group**2

        Rs = so3_rng.rvs(ngroups)
        covs_per_group = Rs @ covs_per_group @ Rs.transpose(0, 2, 1)

        covs = np.repeat(covs_per_group, n_per_group, 0)

        # add truncated covs.
        if ntrunc > 0:
            tcovs = covs[rng.randint(0, nt, ntrunc)]
            if len(tcovs.shape) == 2:
                tcovs = tcovs[None]
            covs = np.concatenate((covs, tcovs), axis=0)

        return covs


def multivariate_normal_batch(
        rng,
        covs: np.ndarray,
        mu: Optional[np.ndarray] = None,
        nsamples: int = 1) -> np.ndarray:
    """Given (n, d, d) covariances, generate (n, d, nsamples).

    A Gaussian distribution N(mu, covs) is assumed.

    Args:
        rng: numpy random generator.
        covs: (n, d, d) covariance matrices related to a Gaussian random
            variable of dimension d.
        mu: (n, d) means of the random Gaussian variable. If not given, they
            are set to zeros.
        nsamples: number of samples to draw from each distribution.

    Returns:
        noise: (n, d, nsamples)
    """
    n, d, d1 = covs.shape
    assert d == d1, (d, d1)

    if mu is not None:
        if len(mu.shape) == 2:
            mu = mu[:, :, None]
        nm, dm = mu.shape[:2]
        assert nm == n, (nm, n)
        assert dm == d, (dm, d)

    try:
        # sample from standard Gaussian and transform to desired distribution.
        noise = rng.standard_normal(d * n * nsamples).reshape(n, d, nsamples)
        noise = np.linalg.cholesky(covs) @ noise
        if mu is not None:
            noise += mu
        noise = noise.squeeze()

    except np.linalg.LinAlgError:
        # Cholesky failed. Check non PD matrices and set zero-noise to them.
        mask_pd = np.linalg.eigvals(covs).min(axis=1) > 0.0
        n_pd = sum(mask_pd)
        noise = np.zeros((n, d, nsamples))

        noise_std = rng.standard_normal(
            d * n_pd * nsamples).reshape(n_pd, d, nsamples)

        noise[mask_pd] = np.linalg.cholesky(covs[mask_pd]) @ noise_std
        if mu is not None:
            noise += mu
        noise = noise.squeeze()

    return noise


def rot_and_trans_error(
        Tgt: np.ndarray, T: np.ndarray, degrees: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """ Rotation and translation errors. """

    if Tgt.shape == (4, 4):
        Tgt = Tgt[None]
    if T.shape == (4, 4):
        T = T[None]

    n = len(Tgt)
    assert n == len(T)
    assert Tgt.shape[-2:] == T.shape[-2:] == (4, 4)

    # relative error poses (expressed w.r.t. the reference poses):
    Terr = np.linalg.inv(Tgt) @ T
    t_true_norm = np.linalg.norm(Tgt[:, :-1, -1], axis=1)

    rot_errors = np.zeros((n,))
    t_errors = np.zeros((n,))

    for i, Terri in enumerate(Terr):
        # error angle (geodesic length).
        trace_rot_error_1 = np.trace(Terri[:-1, :-1]) - 1
        so3_error = math.acos((0.5 * trace_rot_error_1).clip(-1, 1))
        if degrees:
            so3_error *= 180 / np.pi

        # translation error.
        t_error = 100 * np.linalg.norm(Terri[:-1, -1]) / t_true_norm[i]

        rot_errors[i] = so3_error
        t_errors[i] = t_error

    return rot_errors, t_errors


def plot_2d3d_correspondences(p_im, p_c, res, K, fig=None):
    """Plot 2D-3D correspondences."""
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "xy"}, {"type": "scene"}]],
        subplot_titles=("Image Plane", "")
    )

    # 2D plot - image plane.
    fig.add_trace(
        go.Scatter(
            x=p_im[0], y=p_im[1],
            marker=dict(
                size=16,
                color=p_im[0],
                colorscale="Hot",
            ),
            mode='markers'
        ),
        row=1, col=1
    )

    # define camera in 3D.
    depth = p_c[2].min() / 5.
    c = (K[0, 2] / K[0, 0], K[1, 2] / K[1, 1])
    frustum = depth * np.array([
        [-c[0], c[0], c[0], -c[0], 0.],
        [-c[1], -c[1], c[1], c[1], -1.3 * c[1]],
        [1., 1., 1., 1., 1.]
    ])

    # add origin
    frustum = np.concatenate((frustum, np.zeros((3, 1))), axis=1)

    # Plot camera in 3D.
    order = [0, 1, 2, 3, 0, 5, 3, 2, 5, 1, 4, 0]
    lines = np.array([frustum[:, i] for i in order])
    fig.add_trace(
        go.Scatter3d(
            x=lines[:, 0], y=lines[:, 1], z=lines[:, 2],
            mode='lines',
            line=dict(
                width=5,
                color="royalblue",
            )
        ))

    # add 2D points to the created image plane.
    p_im_3d = depth * np.linalg.inv(K) @ np.concatenate((
        p_im, np.ones((1, p_im.shape[1]))))
    fig.add_trace(
        go.Scatter3d(
            x=p_im_3d[0], y=p_im_3d[1], z=p_im_3d[2],
            marker=dict(
                size=2,
                color=p_im[0],
                colorscale="Hot",
            ),
            mode="markers"
        ),
        row=1, col=2
    )

    # 3D points.
    fig.add_trace(
        go.Scatter3d(
            x=p_c[0], y=p_c[1], z=p_c[2],
            marker=dict(
                size=5,
                color=p_im[0],
                colorscale="Hot",
            ),
            mode="markers"
        ),
        row=1, col=2
    )

    fig.update_xaxes(showgrid=False, range=[0, res[0]], row=1, col=1)
    fig.update_yaxes(
        showgrid=False, zeroline=False,
        range=[0, res[1]], autorange="reversed",
        scaleanchor="x", scaleratio=1, row=1, col=1,
        constrain='domain'
    )

    fig.update_layout(
        showlegend=False,
        scene=dict(
            xaxis=dict(showbackground=False),
            # yaxis=dict(showbackground=False),
            # zaxis=dict(showbackground=False)
        ),
        scene_camera=dict(
            up=dict(x=0, y=-1, z=0),  # what axis points up
            # center=dict(x=0, y=0, z=0), # default
            eye=dict(x=0.2, y=-0.1, z=-2.5)  # plotly camera position
        )
    )

    fig.show()
