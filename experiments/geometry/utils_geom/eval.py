""" Evaluated PnP models """

import sys
import math
from typing import Optional
from pathlib import Path
from abc import ABCMeta, abstractmethod
from importlib import import_module

import numpy as np
from tqdm import tqdm
import yaml

ENV_PATH = str(Path(__file__).parents[3])
if ENV_PATH not in sys.path:
    print(f"inserting {ENV_PATH} to sys.path.")
    sys.path.insert(0, ENV_PATH)

from experiments.geometry.utils_geom.mo_ba_1cam import (
    pose_LM_optimization, jacobian_dprojection_dy)


I_3 = np.eye(3)
RAD2DEG = 180.0 / math.pi


def get_eval_objects(*model_names):
    """ Helper function to gather all models being evaluated """
    assert len(model_names) == len(set(model_names)), (
        'model names are not unique')

    models_to_evaluate = []
    for name in model_names:
        name = name.lower()

        if name == 'epnp':
            models_to_evaluate.append(
                EPnPEval(use_2du=False, use_3du=False, use_scoresu=False))

        elif name == 'epnpu':
            models_to_evaluate += [
                EPnPUEval(use_2du=True, use_3du=False, use_scoresu=False),
                EPnPUEval(use_2du=True, use_3du=False, use_scoresu=True),
                EPnPUEval(use_2du=True, use_3du=True, use_scoresu=False),
                EPnPUEval(use_2du=True, use_3du=True, use_scoresu=True),
            ]

        else:
            raise NotImplementedError(
                f"{name} is not implemented. Implemented models are:\n"
                "\t-EPnP\n\t-EPnPU")

    return models_to_evaluate


def compute_error_metrics(
        Rcw_est: np.ndarray, tcw_est: np.ndarray,
        Rcw_gt: np.ndarray, tcw_gt: np.ndarray
) -> tuple[np.float64, np.float64]:
    """ error metrics """
    # rotation part.
    trace_rot_error_1 = Rcw_gt.ravel().dot(Rcw_est.ravel()) - 1
    e_rot = RAD2DEG * math.acos((0.5 * trace_rot_error_1).clip(-1, 1))
    # translation part
    e_t = np.linalg.norm(tcw_est - tcw_gt)
    return e_rot, e_t


def inliers_refinement(
        Xw, p, Rcw, tcw, K=None, th=None, sigmas2d=None, sigmas3d=None,
        eps=1e-9):
    """ mask for refinement of inliers based on Mahalanobis distance """
    if th is None:
        # no refinement
        return Xw, p

    ignore2du = sigmas2d is None
    ignore3du = sigmas3d is None

    if K is not None:
        # normalize 2d coordinates.
        Kinv = np.linalg.inv(K)
        p = Kinv[:2, :2] @ p + Kinv[:2, 2:]

        if not ignore2du:
            # transform 2d uncertainties to the normalized image plane.
            fxxi = 1 / K[0, 0]**2
            fyyi = 1 / K[1, 1]**2
            fxyi = 1 / (K[0, 0] * K[1, 1])
            norm_factors = np.array([[fxxi, fxyi], [fxyi, fyyi]])
            sigmas2d = norm_factors * sigmas2d

    # project 3d points to normalized image plane.
    p_est = Rcw @ Xw + tcw
    p_est = p_est[:2] / p_est[2:]

    err = p - p_est

    # Mahalanobis distances.
    if ignore2du and ignore3du:
        # assume standard Gaussian distribution for the errors.
        mh_dist = np.einsum('in, in -> n', err, err)

    elif not ignore2du and ignore3du:
        # only 2d uncertainty used.
        mh_dist = np.einsum(
            'in, nij, jn -> n', err, np.linalg.inv(sigmas2d + eps), err)

    elif not ignore2du and not ignore3du:
        # propagate 3d covariance to normalized image plane.
        Jproj = jacobian_dprojection_dy(Rcw @ Xw)[0] @ Rcw
        W = np.linalg.inv(
            sigmas2d + Jproj @ sigmas3d @ Jproj.transpose(0, 2, 1) + eps)

        mh_dist = np.einsum('in, nij, jn -> n', err, W, err)

    else:
        raise ValueError

    # inliers.
    return mh_dist <= th


class _PnPEvalBase(metaclass=ABCMeta):
    """ Base evaluation object for the PnP models """

    PKG_MODELS = 'experiments.geometry.models'

    def __init__(self, use_2du, use_3du, use_scoresu):
        # error metrics history.
        self.e_rot = []
        self.e_t = []
        self.e_rot_opt = []
        self.e_t_opt = []

        # image indexes history.
        self.im_idxes = []

        # if 2d/3d uncertainty should be used if the model allows it.
        self.use_2du = use_2du
        self.use_3du = use_3du

        # name and type of uncertainty info to get access.
        self._init(use_scoresu)

    def __call__(self, matches2d3d, Rcw_gt, tcw_gt, im_idxes, debug=False):
        """ Run PnP, refinement of inliers, optimization anf get metrics. 

        Args:
            matches2d3d: contains geometric information per uncertainty type.
                Expected keywords:
                    'none'  # sub-dict without using uncertainty estimates
                        ├─ 'X' (3, n) 3d points
                        ├─ 'p' (2, n) 2d points in *normalized* image plane
                        ├─ 'Rcw' (3, 3) P3P rot matrix estimate
                        ├─ 'tcw' (3, 1) P3P transl. vector estimate
                        ├─ 'sigmas3d' -> absent, or None.
                        └─ 'sigmas2d' -> absent, or None.

                    'cov'  # sub-dict using 2d uncertainty estimates.
                        ├─ the above items.
                        ├─ 'sigmas3d' (n, 3, 3) 3d covariances.
                        └─ 'sigmas2d' (n, 2, 2) 2d covariances.

                    'cov_s'  # sub-dict using feature scores as uncertainty.
                        └─ all the above items.
            Rcw_gt: (3, 3) ground-truth rotation matrix.
            tcw_gt: (3, 1) ground-truth translation vector.
            im_idxes: (n_ims,) array with image indexes (to explorate results).

        Returns:
            None
        """
        assert Rcw_gt.shape == (3, 3)
        assert tcw_gt.shape == (3, 1)

        self.im_idxes.append(im_idxes.copy())

        # obtain pnp estimate.
        Rcw, tcw = self._run_pnp(matches2d3d[self.u_kw])

        # inliers based on current estimate.
        mask = self._inliers_refinement(
            matches2d3d[self.u_kw], Rcw, tcw, th=6.0)

        if np.sum(mask) < 3:
            if debug:
                tqdm.write(
                    f'{self.name} # inliers not enough. Using P3P instead.')

            e_rot, e_t = compute_error_metrics(
                matches2d3d[self.u_kw]['Rcw'],
                matches2d3d[self.u_kw]['tcw'],
                Rcw_gt, tcw_gt)

            self.e_rot.append(e_rot)
            self.e_t.append(e_t)
            self.e_rot_opt.append(e_rot)
            self.e_t_opt.append(e_t)

            if debug:
                return ((e_rot, e_t),)

        else:
            if debug:
                tqdm.write(f"{self.name} # inliers: {np.sum(mask)}")

            # error metrics.
            e_rot, e_t = compute_error_metrics(Rcw, tcw, Rcw_gt, tcw_gt)
            self.e_rot.append(e_rot)
            self.e_t.append(e_t)

            # motion-only BA.
            Rcw_opt, tcw_opt = self._run_optimization(
                mask, matches2d3d[self.u_kw], Rcw, tcw, niter=10)

            # error metrics.
            e_rot_opt, e_t_opt = compute_error_metrics(
                Rcw_opt, tcw_opt, Rcw_gt, tcw_gt)
            self.e_rot_opt.append(e_rot_opt)
            self.e_t_opt.append(e_t_opt)

            if debug:
                return ((e_rot, e_t, e_rot_opt, e_t_opt),)

    def save_results(self, data_det_path: Path, do_clear: bool = True):
        """ write statistics to disk. """
        save_dir = data_det_path / self.name
        save_dir.mkdir(exist_ok=True, parents=True)

        hist = {
            'e_t': self.e_t,
            'e_rot': self.e_rot,
            'e_t_opt': self.e_t_opt,
            'e_rot_opt': self.e_rot_opt,
            'im_idxes': self.im_idxes
        }

        stats = {
            'e_t_mean': np.mean(self.e_t).item(),
            'e_t_median': np.median(self.e_t).item(),
            'e_rot_mean': np.mean(self.e_rot).item(),
            'e_rot_median': np.median(self.e_rot).item(),
            # after BA.
            'e_t_opt_mean': np.mean(self.e_t_opt).item(),
            'e_t_opt_median': np.median(self.e_t_opt).item(),
            'e_rot_opt_mean': np.mean(self.e_rot_opt).item(),
            'e_rot_opt_median': np.median(self.e_rot_opt).item()
        }

        # npz.
        np.savez(save_dir / 'stats.npz', **hist, **stats)

        # yaml.
        with open(save_dir / 'stats.yaml', 'w') as f:
            yaml.dump(stats, f, default_flow_style=False)

        if do_clear:
            self.e_t = []
            self.e_rot = []
            self.e_t_opt = []
            self.e_rot_opt = []
            self.im_idxes = []

    @staticmethod
    def load_pnp_model(
            pkg_path: str, module_name: str, class_name: str,
            cfg: Optional[dict] = None):
        """ Dynamic load of PnP model """
        if cfg is None:
            cfg = {}
        return getattr(
            import_module(f"{pkg_path}.{module_name}"), class_name)(cfg)

    @abstractmethod
    def _init(self):
        """ TBD by the child class """
        pass

    @abstractmethod
    def _run_pnp(self):
        """ TBD by the child class """
        pass

    @abstractmethod
    def _inliers_refinement(self):
        """ TBD by the child class """
        pass

    @abstractmethod
    def _run_optimization(self):
        """ TBD by the child class """
        pass

    # def _ensure_not_none(*args):
    #     """ Construct decorator to check that certain arguments are given as
    #     kwargs to the decorated methods and that they aren't None """
    #     for arg in args:
    #         assert isinstance(arg, str)

    #     def decorator(method):

    #         def wrapper(self, Xw, p, **kwargs):
    #             for arg in args:
    #                 if arg not in kwargs or kwargs[arg] is None:
    #                     raise ValueError(
    #                         f"Model '{self.name}' needs '{arg}' as input but"
    #                         " it was not given or it was None.")

    #             return method(self, Xw, p, **kwargs)

    #         return wrapper

    #     return decorator


### IMPLEMENTED MODELS ###


class EPnPEval(_PnPEvalBase):
    """ Evaluation object particularized to EPnP """

    BASE_CFG = {
        "use_pca": True,
        "th_reproj_error": -1,
        "use_median": False,
        "refine_w_cps": True,
        "optimize_best_only": False,
        "gn_iters": 5,
        "verbose": False}

    def _init(self, use_scoresu):
        self.name = 'epnp'
        self.u_kw = 'none'

        # load model.
        self.model = self.load_pnp_model(
            self.PKG_MODELS, 'epnp', 'EPnP', self.BASE_CFG)

    def _run_pnp(self, matches2d3d):
        return self.model(
            matches2d3d['X'],
            matches2d3d['p'],
            I_3)[:2]

    def _inliers_refinement(self, matches2d3d, Rcw, tcw, th):
        return inliers_refinement(
            matches2d3d['X'],
            matches2d3d['p'],
            Rcw, tcw,
            th=th)

    def _run_optimization(self, mask, matches2d3d, Rcw, tcw, niter):
        return pose_LM_optimization(
            np.compress(mask, matches2d3d['X'], 1),
            np.compress(mask, matches2d3d['p'], 1),
            I_3, Rcw, tcw, niter=niter
        )


class EPnPUEval(_PnPEvalBase):
    """ Evaluation object particularized to EPnPU """

    BASE_CFG = {
        "use_pca": True,
        "th_reproj_error": -1,
        "use_median": False,
        "refine_w_cps": True,
        "optimize_best_only": False,
        "gn_iters": 5,
        "verbose": False,
        "p3p": {
            'method': 'p3p',
            'reproj_th': 10.0,
            'confidence': 0.999,
            'max_iter': 10_000
        },
    }

    def _init(self, use_scoresu):
        assert self.use_2du

        if use_scoresu:
            self.u_kw = 'cov_s'
            self.name = 'epnpu_2d3u_s' if self.use_3du else 'epnpu_2d_s'

        else:
            self.u_kw = 'cov'
            self.name = 'epnpu_2d3u' if self.use_3du else 'epnpu_2d'

        # load model.
        self.model = self.load_pnp_model(
            self.PKG_MODELS, 'epnpu', 'EPnPU', self.BASE_CFG)

    def _run_pnp(self, matches2d3d):
        sigmas3d = matches2d3d['sigmas3d'] if self.use_3du else None

        return self.model(
            matches2d3d['X'],
            matches2d3d['p'],
            I_3,
            matches2d3d['sigmas2d'],
            sigmas3d,
            matches2d3d['Rcw'],
            matches2d3d['tcw'],
        )[:2]

    def _inliers_refinement(self, matches2d3d, Rcw, tcw, th):
        # uncertainty estimates should ideally be used here. However, since the
        # the uncertainty estimates are scaled by an unknown factor, these do
        # not relate to pixel units. Thus, imposing a threshold in the MH
        # distances would be wrong. Instead, for detection of inliers we use
        # a threshold in the repojection error of ~ 6 pixels^2
        return inliers_refinement(
            matches2d3d['X'],
            matches2d3d['p'],
            Rcw, tcw,
            th=th)

        # sigmas3d = matches2d3d['sigmas3d'] if self.use_3du else None

        # return inliers_refinement(
        #     matches2d3d['X'],
        #     matches2d3d['p'],
        #     Rcw, tcw,
        #     th=th,
        #     sigmas2d=matches2d3d['sigmas2d'],
        #     sigmas3d=sigmas3d)

    def _run_optimization(self, mask, matches2d3d, Rcw, tcw, niter):
        if self.use_3du:
            sigmas3d = np.compress(mask, matches2d3d['sigmas3d'], 0)
        else:
            sigmas3d = None

        return pose_LM_optimization(
            np.compress(mask, matches2d3d['X'], 1),
            np.compress(mask, matches2d3d['p'], 1),
            I_3, Rcw, tcw,
            np.compress(mask, matches2d3d['sigmas2d'], 0),
            sigmas3d,
            niter=niter)
