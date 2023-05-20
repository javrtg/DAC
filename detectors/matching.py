from typing import Union, Optional

import numpy as np


def check_cfg(cfg_base, cfg_to_check):
    if any(k not in cfg_base for k in cfg_to_check):
        raise ValueError(
            f"At least, one key of config dict:\n{cfg_to_check.keys()}\n"
            f"was not found in the default keys:\n{cfg_base.keys()}")


class MutualNearestNeighbor():
    default_cfg: dict[str, Union[str, Optional[float]]] = {
        'distance': 'cosine',  # ['cosine', 'euclidean']
        'thr': None
    }

    def __init__(self, cfg: Optional[dict] = None):
        if cfg is None:
            cfg = {}

        check_cfg(self.default_cfg, cfg)
        cfg = {**self.default_cfg, **cfg}
        assert cfg['distance'] in ['cosine', 'euclidean']

        self.distance = cfg['distance']
        self.thr = cfg['thr']

    def __call__(
            self, desc1: np.ndarray, desc2: np.ndarray,
            return_sim: bool = False) -> np.ndarray:
        """Mutual NN matching.

        Args:
            desc1: (n1, d1) feature descriptors from image 1.
            desc2: (n2, d2) feature descriptors from image 2.

        Returns:
            matches: (nmatches, 2) corresponding indexes w.r.t. the input
                descriptors of the features matched.
            sim: (nmatches,) cosine similarities of the matched descriptors.
        """
        if (len(desc1) == 0) or (len(desc2) == 0):
            return np.empty((0, 2), dtype=np.int64)

        # L2 normalization of descriptors
        desc1 = desc1 / \
            np.sqrt(np.einsum('ij, ij -> i', desc1, desc1))[:, None]
        desc2 = desc2 / \
            np.sqrt(np.einsum('ij, ij -> i', desc2, desc2))[:, None]
        # L2 dist is then ||desc1_i - desc2_j|| = 2*(1 - desc1_i @ desc2_j)
        # so similarity can be computed as:
        sim = desc1 @ desc2.T

        # find mutual NN
        nn12 = sim.argmax(axis=1)  # (n1,)
        nn21 = sim.argmax(axis=0)  # (n2,)
        ids1 = np.arange(desc1.shape[0])
        mask = (ids1 == nn21[nn12])
        matches = np.concatenate((ids1[mask, None], nn12[mask, None]), axis=1)

        if self.thr != None:
            # check if thr is satisfied
            if self.distance == 'cosine':
                mask = sim[matches[:, 0], matches[:, 1]] >= self.thr
            else:
                dist_sq = 2. * (1. - sim)
                mask = dist_sq[matches[:, 0], matches[:, 1]] <= (self.thr**2)
            matches = matches[mask]

        if return_sim:
            return matches, sim[matches[:, 0], matches[:, 1]]
        return matches
