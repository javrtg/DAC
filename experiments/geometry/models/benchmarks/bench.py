import sys
from pathlib import Path
from typing import Optional
from argparse import ArgumentParser

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
# random generator types.
from numpy.random.mtrand import RandomState
from scipy.stats._multivariate import special_ortho_group_frozen as SO_gen

ENV_PATH = str(Path(__file__).parents[4])
if ENV_PATH not in sys.path:
    print(f"inserting {ENV_PATH} to sys.path.")
    sys.path.insert(0, ENV_PATH)
import experiments.geometry.models.benchmarks.utils_bench as ubench


def generate_data(
        n: int,
        rng: RandomState,
        so3g: SO_gen,
        so2g: SO_gen,
        K: Optional[np.ndarray] = None,
        z_lim: tuple[float, float] = (4., 8.),
        res: tuple[int, int] = (640, 480),
        std_noise: float = 0.,
        noise_type: str = "iso_hom",
        only_2d_noise: bool = False,
        only_3d_noise: bool = False,
        seed: int = 0,
        do_viz: bool = False
) -> dict[str, np.ndarray]:
    """Sample rotation, translation and n uniform 2D-3D correspondences.

    Args:
        n: number of samples to draw from uniform distribution.
        K: (3, 3) calibration matrix (pinhole).
        z_lim: depth range of the 3d points w.r.t. synthetic camera.
        res: resolution in pixels of synthetic camera.
        std_noise: Gaussian std value applied to 2D coordinates.
        seed: random seed (to make experiments deterministic).

    Returns:
        dictionary with keys:
            p_im: (2, n) image plane coordinates.
            p_w: (3, n) 3D points in world reference.
            K: (3, 3) input calibration matrix.
            R_cw: (3, 3) rotation matrix.
            t_cw: (3, 1) translation vector.
    """
    if K is None:
        K = np.array([
            [800., 0., 320.],
            [0., 800., 240.],
            [0., 0., 1.]
        ])

    # image plane coordinates.
    p_im = rng.uniform(0., 1., (2, n)) * [[res[0]], [res[1]]]

    if std_noise != 0.0 and not only_3d_noise:
        # 2d covariances that will be considered for sampling Gaussian noise.
        covs2d = ubench.create_2d_covariances(
            rng, so2g, n, noise_type, std_noise)
        noise_2d = ubench.multivariate_normal_batch(rng, covs2d, nsamples=1)

        # add virtual sensor noise.
        p_im_noise = p_im + noise_2d.T
        p_im_noise[0] = p_im_noise[0].clip(0, res[0])
        p_im_noise[1] = p_im_noise[1].clip(0, res[1])
    else:
        covs2d = None
        p_im_noise = p_im

    # 3D points in camera reference (without noise, since its ground-truth).
    rays = np.linalg.inv(K) @ np.concatenate((p_im, np.ones((1, n))))
    p_c = rng.uniform(z_lim[0], z_lim[1], (1, n)) * rays

    # set translation to the centroid of the random point cloud.
    t_cw = p_c.mean(axis=1, keepdims=True)
    # rotation.
    R_cw = so3g.rvs()

    # 3D points in world reference.
    p_w = R_cw.T @ p_c - R_cw.T.dot(t_cw)

    # add virtual noise to 3d points.
    if not only_2d_noise and std_noise != 0.0:
        covs3d = ubench.create_3d_covariances(
            rng, so3g, n, noise_type, std_noise)
        noise_3d = ubench.multivariate_normal_batch(rng, covs3d, nsamples=1)
        p_w_noise = p_w + noise_3d.T
    else:
        covs3d = None
        p_w_noise = p_w

    if do_viz:
        ubench.plot_2d3d_correspondences(p_im, p_c, res, K)

    return {
        'p_im': p_im_noise,
        'p_w': p_w_noise,
        'K': K,
        'R_cw': R_cw,
        't_cw': t_cw,
        'covs2d': covs2d,
        'covs3d': covs3d
    }


def benchmark_accuracy_vs_npoints(
        seed: int,
        methods_str: list[str],
        methods_cfg: list[Optional[dict]],
        npoints_lims: tuple[int, int] = (10, 110),
        npoints_step: int = 10,
        nrepeats: int = 400,
        std: float = 5.,
        noise_type: str = "iso_hom",
        only_2d_noise: bool = False,
        only_3d_noise: bool = False,
        z_lim: tuple[float, float] = (4., 9.),
):
    # number of points to evaluate.
    npoints_eval = np.arange(
        npoints_lims[0], npoints_lims[1] + 1, npoints_step)

    nm = len(methods_str)
    methods = ubench.get_method_callables(methods_str, methods_cfg)
    errors = {
        "rot_mean": np.empty((nm, len(npoints_eval))),
        "t_mean": np.empty((nm, len(npoints_eval))),
        "rot_median": np.empty((nm, len(npoints_eval))),
        "t_median": np.empty((nm, len(npoints_eval))),
    }

    rng, so3g, so2g = ubench.get_random_generators(seed)

    for j, npoints in enumerate(tqdm(npoints_eval)):
        T_gt = np.zeros((nrepeats, 4, 4))
        T_gt[:, -1, -1] = 1.

        T_est = np.zeros((nm, nrepeats, 4, 4))
        T_est[:, :, -1, -1] = 1.

        for i in range(nrepeats):
            data = generate_data(
                npoints, rng, so3g, so2g,
                std_noise=std, noise_type=noise_type,
                only_2d_noise=only_2d_noise,
                only_3d_noise=only_3d_noise,
                z_lim=z_lim)
            T_gt[i, :-1, :-1] = data["R_cw"]
            T_gt[i, :-1, -1:] = data["t_cw"]

            for k, method in enumerate(methods):
                R_cw, t_cw, _ = method(
                    data['p_w'], data['p_im'],
                    data['covs2d'], data['covs3d'], only_2d_noise,
                    data['K'])
                T_est[k, i, :-1, :-1] = R_cw
                T_est[k, i, :-1, -1:] = t_cw

        for k in range(nm):
            e_rot, e_t = ubench.rot_and_trans_error(
                T_gt, T_est[k], degrees=True)
            errors["rot_mean"][k, j] = np.mean(e_rot)
            errors["t_mean"][k, j] = np.mean(e_t)
            errors["rot_median"][k, j] = np.median(e_rot)
            errors["t_median"][k, j] = np.median(e_t)

    fig, axes = plt.subplots(1, 4, sharex=True)
    axes[1].sharey(axes[0])
    axes[3].sharey(axes[2])

    st = ['-', '--', '-.', '-.']
    for k in range(nm):
        axes[0].plot(
            npoints_eval, errors["rot_mean"][k], st[k], label=methods[k].name,
            lw=2.0)
        axes[1].plot(
            npoints_eval, errors["rot_median"][k], st[k], label=methods[k].name,
            lw=2.0)

        axes[2].plot(
            npoints_eval, errors["t_mean"][k], st[k], label=methods[k].name,
            lw=2.0)
        axes[3].plot(
            npoints_eval, errors["t_median"][k], st[k], label=methods[k].name,
            lw=2.0)

    # if noise_type == 'groups':
    #     title = None
    # else:
    #     title = f"$\sigma = {std}$ pix, noise type = {noise_type}"

    for ax in axes:
        ax.set(xlim=npoints_lims, xticks=np.linspace(*npoints_lims, 6))

    axes[0].set(title='mean')
    axes[1].set(title='median')
    axes[2].set(title='mean')
    axes[3].set(title='median')

    axes[0].set(ylabel="rot. error (deg.)")
    axes[2].set(ylabel="transl. error (%)")

    for ax in axes:
        ax.set(xlabel="# points")

    axes[0].legend(framealpha=1.0, edgecolor='white')
    axes[2].legend(framealpha=1.0, edgecolor='white')

    fig.set_size_inches([12.0, 3.0])
    fig.tight_layout()

    # save
    dir_results = Path(__file__).parent / 'results'
    dir_results.mkdir(exist_ok=True)

    exp = '2d_noise' if only_2d_noise else '2d3d_noise'
    fname = dir_results / f'benchmark_{exp}.pdf'
    fig.savefig(fname, bbox_inches='tight')


def benchmark_accuracy_vs_noise(
        seed: int,
        methods_str: list[str],
        methods_cfg: list[Optional[dict]],
        noise_lims: tuple[float, float] = (0., 15.),
        noise_step: float = 1.,
        nrepeats: int = 50,
        npoints: int = 6,
        noise_type="iso_hom",
        only_2d_noise: bool = False,
        z_lim: tuple[float, float] = (4., 9.)
):
    # number of points to evaluate.
    noise_eval = np.arange(noise_lims[0], noise_lims[1] + 1., noise_step)

    nm = len(methods_str)
    methods = ubench.get_method_callables(methods_str, methods_cfg)
    errors = {
        "rot": np.empty((nm, len(noise_eval))),
        "transl": np.empty((nm, len(noise_eval)))
    }

    rng, so3g, so2g = ubench.get_random_generators(seed)

    for j, std in enumerate(tqdm(noise_eval)):
        T_gt = np.zeros((nrepeats, 4, 4))
        T_gt[:, -1, -1] = 1.

        T_est = np.zeros((nm, nrepeats, 4, 4))
        T_est[:, :, -1, -1] = 1.

        for i in range(nrepeats):
            data = generate_data(
                npoints, rng, so3g, so2g,
                std_noise=std, noise_type=noise_type,
                only_2d_noise=only_2d_noise,
                z_lim=z_lim)

            T_gt[i, :-1, :-1] = data["R_cw"]
            T_gt[i, :-1, -1:] = data["t_cw"]

            for k, method in enumerate(methods):
                R_cw, t_cw, _ = method(
                    data['p_w'], data['p_im'],
                    data['covs2d'], data['covs3d'], only_2d_noise,
                    data['K'])
                T_est[k, i, :-1, :-1] = R_cw
                T_est[k, i, :-1, -1:] = t_cw

        for k in range(nm):
            e_rot, e_t = ubench.rot_and_trans_error(
                T_gt, T_est[k], degrees=True)
            errors["rot"][k, j] = np.mean(e_rot)
            errors["transl"][k, j] = np.mean(e_t)

    fig, axes = plt.subplots(2, 1, sharex=True)
    st = ['-', '--', '-.', '-.']
    for k in range(nm):
        axes[0].plot(
            noise_eval, errors["rot"][k], st[k], label=methods_str[k],
            lw=2.0,
        )
        axes[1].plot(
            noise_eval, errors["transl"][k], st[k], label=methods_str[k],
            lw=2.0,
        )

    axes[0].legend()

    axes[0].set(
        title=f"$n = {npoints}$ points, noise type = {noise_type}",
        ylabel="rot. error (deg.)"
    )

    axes[1].set(
        xlabel="# noise (pixels)",
        ylabel="transl. error (%)"
    )


if __name__ == "__main__":
    parser = ArgumentParser(description='EPnPU Validation')
    parser.add_argument('--seed', type=int, default=4,
                        help='seed used when sampling random data.')
    parser.add_argument('--do_acc_vs_npoints', action='store_true')
    parser.add_argument('--only_2d_noise', action='store_true')
    parser.add_argument('--do_acc_vs_noise', action='store_true')
    parser.add_argument('--noise_type', type=str, default="groups",
                        choices=["groups", "iso_hom", "iso_inhom", "ani_hom", "ani_inhom"])

    args = parser.parse_args()

    print(f'\n{args}\n')

    cfg_epnp = {
        "use_pca": True,
        "th_reproj_error": -1,
        "use_median": False,
        "refine_w_cps": True,
        "optimize_best_only": False,
        "gn_iters": 5,
        "verbose": False,
    }

    cfg_epnpu = {
        "use_pca": True,
        "th_reproj_error": -1,
        "use_median": False,
        "refine_w_cps": True,
        "optimize_best_only": False,
        "gn_iters": 5,
        "verbose": False,
        "p3p": {
            'method': 'p3p',
            'reproj_th': 1000.0,
            'confidence': 0.999,
            'max_iter': 10_000
        },
    }

    if args.do_acc_vs_npoints:
        benchmark_accuracy_vs_npoints(
            args.seed,
            methods_str=[
                # "epnp_imp", "epnp_opencv", "epnpu_imp", "epnpu2d_imp"],
                "epnp_opencv", "epnp_imp", "epnpu_imp"],
            methods_cfg=[None, cfg_epnp, cfg_epnpu, cfg_epnpu],
            npoints_lims=(10, 110),
            npoints_step=10,
            nrepeats=400,
            noise_type=args.noise_type,
            std=10.0,  # ignored when noise_type == 'groups'
            only_2d_noise=args.only_2d_noise,
            only_3d_noise=False,
            z_lim=(4., 8.),
        )

    if args.do_acc_vs_noise:
        benchmark_accuracy_vs_noise(
            args.seed,
            methods_str=[
                "epnp_imp", "epnp_opencv", "epnpu_imp", "epnpu2d_imp"],
            methods_cfg=[cfg_epnp, None, cfg_epnpu, cfg_epnpu],
            noise_lims=(0., 20.),
            noise_step=1.,
            nrepeats=50,
            npoints=50,
            noise_type=args.noise_type,
            only_2d_noise=args.only_2d_noise,
            z_lim=(4., 8.)
        )
