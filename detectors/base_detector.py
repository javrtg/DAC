import sys
from pathlib import Path
from typing import Union
from abc import ABCMeta, abstractmethod

import torch

env_path = str(Path(__file__).parents[1])
if env_path not in sys.path:
    print(f'inserting {env_path} to sys.path')
    sys.path.insert(0, env_path)
from detectors.matching import MutualNearestNeighbor as MNN


def check_cfg(cfg_base, cfg_to_check):
    if any(k not in cfg_base for k in cfg_to_check):
        raise ValueError(
            f"At least, one key of config dict:\n{cfg_to_check.keys()}\n"
            f"was not found in the default keys:\n{cfg_base.keys()}")


class BaseDetector(metaclass=ABCMeta):
    default_cfg_acorr: dict[str, Union[str, float]] = {
        'method_derivative': 'sobel',
        'sigma_d': 1.0,
        'truncation_d': 3.0,
        'method_weighting': 'gaussian',
        'sigma_w': 1.0,
        'truncation_w': 3.0
        }

    def __init__(self, cfg, cfg_acorr, cfg_matching, disable_grads=True):
        check_cfg(self.default_cfg, cfg)
        check_cfg(self.default_cfg_acorr, cfg_acorr)
        check_cfg(self.default_cfg_matching, cfg_matching)

        cfg = {**self.default_cfg, **cfg}
        cfg.update({'cfg_acorr': {**self.default_cfg_acorr, **cfg_acorr}})
        self._init(cfg)
        self.match_fun = MNN({**self.default_cfg_matching, **cfg_matching})

        if disable_grads:
            torch.set_grad_enabled(False)

    def __call__(self, im):
        return self.run(im)

    def match_descriptors(self, desc1, desc2, return_sim=False):
        return self.match_fun(desc1, desc2, return_sim)

    @abstractmethod
    def run(self, im):
        """To be implemented by the child class."""
        pass

    @abstractmethod
    def _init(self, cfg):
        """To be implemented by the child class."""
        pass

    @property
    @abstractmethod
    def default_cfg(self):
        """Dict with detector parameters. To be defined in the child class."""
        pass

    @property
    @abstractmethod
    def default_cfg_matching(self):
        """Dict with detector parameters. To be defined in the child class."""
        pass