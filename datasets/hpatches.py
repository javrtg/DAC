import sys
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image

ENV_PATH = str(Path(__file__).parents[1])
if ENV_PATH not in sys.path:
    print(f'Inserting {ENV_PATH} to sys.path')
    sys.path.insert(0, ENV_PATH)
from datasets.settings import HPATCHES_PATH


class HPatches:

    def __init__(self, use_d2net_subset: bool = True):

        if use_d2net_subset:
            # ignore sequences beyond 1200 x 1600
            ignored_seqs = [
                "i_contruction",
                "i_crownnight",
                "i_dc",
                "i_pencils",
                "i_whitebuilding",
                "v_artisans",
                "v_astronautis",
                "v_talent"
            ]

            self._seq_dirs = sorted(
                f.name for f in HPATCHES_PATH.iterdir()
                if f.name not in ignored_seqs
            )

            # number of illumination / viewpoint type sequences.
            self.n_i, self.n_v = 52, 56

        else:
            self._seq_dirs = sorted(
                f.name for f in HPATCHES_PATH.iterdir())
            self.n_i, self.n_v = 57, 59

    def get(self,
            seq_dir: Union[str, Path, int],
            idx: int,
            return_h: bool = False
            ):

        if return_h and (idx == 1):
            raise ValueError(
                "Can't return H for the reference image (idx must be \in [2,6])")

        if isinstance(seq_dir, int):
            seq_dir = HPATCHES_PATH / self._seq_dirs[seq_dir]
        else:
            seq_dir = HPATCHES_PATH / seq_dir

        # load image
        im_path = seq_dir / (f"{idx}.ppm")
        im = np.array(Image.open(im_path).convert('RGB'))

        if return_h:
            H_gt = np.loadtxt(seq_dir / f"H_1_{idx}")
            return im, H_gt
        else:
            return im
