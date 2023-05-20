import sys
from pathlib import Path

import numpy as np

ENV_PATH = str(Path(__file__).parents[3])
if ENV_PATH not in sys.path:
    print(f"inserting {ENV_PATH} to sys.path.")
    sys.path.insert(0, ENV_PATH)
from datasets.kitti import KITTI_ODOMETRY
from datasets.tum_rgbd import TUM_RGBD


def get_dataset(name: str, seq):
    if name == "kitti":
        return KITTI_PnP(
            seq=seq,
            use_gray=False,
            return_left=True,
            return_right=False,
            return_times=False,
            return_poses=True
        )

    elif name == "tum_rgbd":
        return TUM_RGBD(
            sequence=seq,
            return_pose=True,
            return_depth=True
        )

    else:
        raise NotImplementedError('Dataset not supported.')


class KITTI_PnP(KITTI_ODOMETRY):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sequence = self.seq
        self.distCoeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

        # kitti has variable image spatial dimensions but they are consistent
        # for each sequence.
        self.h, self.w = self[0][0].shape[:2]

    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        im = data['left_im']
        depth = None
        Twc = data['Twc']
        return im, depth, Twc
