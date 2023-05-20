import sys
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R

ENV_PATH = str(Path(__file__).parents[1])
if ENV_PATH not in sys.path:
    print(f'Inserting {ENV_PATH} to sys.path')
    sys.path.insert(0, ENV_PATH)
from datasets.settings import TUM_RGBD_PATH


INTRINSICS = {
    'ros_default': {
        'fx': 525.0,
        'fy': 525.0,
        'cx': 319.5,
        'cy': 239.5,
        'k1': 0.0,
        'k2': 0.0,
        'p1': 0.0,
        'p2': 0.0,
        'k3': 0.0,
    },

    'freiburg1': {
        'fx': 517.3,
        'fy': 516.5,
        'cx': 318.6,
        'cy': 255.3,
        'k1': 0.2624,
        'k2': -0.9531,
        'p1': -0.0054,
        'p2': 0.0026,
        'k3': 1.1633,
    },

    'freiburg2': {
        'fx': 520.9,
        'fy': 521.0,
        'cx': 325.1,
        'cy': 249.7,
        'k1': 0.2312,
        'k2': -0.7849,
        'p1': -0.0033,
        'p2': -0.0001,
        'k3': 0.9172,
    },

    'freiburg3': {
        'fx': 535.4,
        'fy': 539.2,
        'cx': 320.1,
        'cy': 247.6,
        'k1': 0.0,
        'k2': 0.0,
        'p1': 0.0,
        'p2': 0.0,
        'k3': 0.0,
    }

}


class TUM_RGBD:
    """Parser for sequences of TUM dataset [1].

    NOTE: If we want to use both depth info, and the intrinsic distortion of
    the camera, the pipeline to do so, should be:
        1) consider a distorted pixel location, p_d \in R2,
        2) obtain depth using it: d_p = depth_channel[p_d],
        3) undistort pixel location to obtain its normalized coordinates, p_n:
                p_n = undistort(p_d, calib_matrix, distortion_coeffs),
        4) 3d point location, P, is then:
                P = d_p * [p_n^T, 1]^T
    [1] https://vision.in.tum.de/data/datasets/rgbd-dataset/download
    """

    def __init__(
            self,
            sequence: str,
            return_depth: bool = True,
            return_pose: bool = False,
            use_ros_default: bool = False
    ):
        self.sequence = sequence
        self.seq_path = seq_path = TUM_RGBD_PATH / sequence

        if not seq_path.is_dir():
            raise ValueError(
                f"Sequence '{sequence}' was not found in root directory "
                f"'{TUM_RGBD_PATH}'.")

        self.return_depth = return_depth
        self.return_pose = return_pose

        # depth scale. 1 in depth map = depth_scale metres.
        if 'freiburg2' in sequence:
            # as noted in orb-slam2 [1], there is a bias in the scale.
            # [1] https://arxiv.org/pdf/1610.06475.pdf
            self.depth_scale = 1.0 / 5_208.0
        else:
            self.depth_scale = 1.0 / 5_000.0

        # resolution.
        self.h = 480
        self.w = 640

        # intrinsic calibration (it was done with OpenCV).
        if use_ros_default:
            k = 'ros_deafult'
        elif 'freiburg1' in sequence:
            k = 'freiburg1'
        elif 'freiburg2' in sequence:
            k = 'freiburg2'
        elif 'freiburg3' in sequence:
            k = 'freiburg3'
        else:
            raise ValueError
        self.__dict__.update(INTRINSICS[k].copy())

        # calib matrix.
        self.K = np.array([
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0],
        ])

        # order of distortion params in OpenCV's convention.
        self.distCoeffs = np.array(
            [self.k1, self.k2, self.p1, self.p2, self.k3])

        msg = 'associations between {} and {} not found. Saving them...\n'
        if return_depth:
            datafile = 'associations_rgb_depth.txt'

            # association of depth_images w.r.t. rgb_images.
            if not (seq_path / datafile).is_file():
                print(msg.format('rgb', 'depth'))
                associate(seq_path, 'rgb.txt', 'depth.txt', 0.0, 0.02)

            if return_pose:
                datafile = 'associations_rgb_depth_groundtruth.txt'

                # association of the groundtruth w.r.t. the previous result.
                if not (seq_path / datafile).is_file():
                    print(msg.format('rgb, depth', 'groundtruth'))
                    associate(
                        seq_path,
                        'associations_rgb_depth.txt',
                        'groundtruth.txt',
                        0.0, 0.02)

        elif return_pose:
            datafile = 'associations_rgb_groundtruth.txt'

            # association of groundtruth w.r.t. rgb_images.
            if return_pose and not (seq_path / datafile).is_file():
                print(msg.format('rgb', 'groundtruth'))
                associate(seq_path, 'rgb.txt', 'groundtruth.txt', 0.0, 0.02)

        else:
            # store only rgb data.
            datafile = 'rgb.txt'

        # read data.
        with open(seq_path / datafile, 'r') as f:
            self.data = [
                l.rstrip() for l in f if l[0] != '#' and len(l.strip()) > 0]

        # for speed, read all poses first.
        if return_pose:
            self.T_wc_all = _load_poses(seq_path / datafile)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, i: Union[int, slice, list[int]]) -> tuple:
        # lines to extract the data from.
        if isinstance(i, int):
            lines = [self.data[i]]

        elif isinstance(i, slice):
            lines = self.data[i]

        elif isinstance(i, list):
            assert all(isinstance(index, int) for index in i)
            lines = [self.data[idx] for idx in i]

        else:
            raise ValueError(
                f'Index of type {type(i)}, is not accepted. Accepted indexing '
                'types are: Union[int, slice, list[int]].')

        n = len(lines)

        # rgb(s) + depth(s) + pose(s)
        if self.return_depth and self.return_pose:
            if n > 1:
                rgb = [
                    np.asarray(Image.open(self.seq_path / line.split()[1]))
                    for line in lines]

                depth = [
                    np.asarray(Image.open(self.seq_path / line.split()[3]))
                    * self.depth_scale
                    for line in lines]

            else:
                rgb = np.asarray(
                    Image.open(self.seq_path / lines[0].split()[1]))
                depth = self.depth_scale * np.asarray(
                    Image.open(self.seq_path / lines[0].split()[3]))

            return rgb, depth, self.T_wc_all[i]

        # rgb(s) + depth(s)
        elif self.return_depth:
            if n > 1:
                rgb = [
                    np.asarray(Image.open(self.seq_path / line.split()[1]))
                    for line in lines]

                depth = [
                    np.asarray(Image.open(self.seq_path / line.split()[3]))
                    * self.depth_scale
                    for line in lines]

            else:
                rgb = np.asarray(
                    Image.open(self.seq_path / lines[0].split()[1]))
                depth = self.depth_scale * np.asarray(
                    Image.open(self.seq_path / lines[0].split()[3]))

            return rgb, depth

        # rgb(s) + pose(s)
        elif self.return_pose:
            if n > 1:
                rgb = [
                    np.asarray(Image.open(self.seq_path / line.split()[1]))
                    for line in lines]

            else:
                rgb = np.asarray(
                    Image.open(self.seq_path / lines[0].split()[1]))

            return rgb, self.T_wc_all[i]

        else:
            if n > 1:
                rgb = [
                    np.asarray(Image.open(self.seq_path / line.split()[1]))
                    for line in lines]

            else:
                rgb = np.asarray(
                    Image.open(self.seq_path / lines[0].split()[1]))

            return rgb


def _load_poses(datafile):
    """Create array with all the poses from the sequence stored."""
    # get columns of interest as floating point values.
    poses_t_quat = np.genfromtxt(datafile, delimiter=' ', usecols=range(-7, 0))

    # initialize poses.
    T_wc_all = np.zeros((len(poses_t_quat), 4, 4))
    T_wc_all[:, 3, 3] = 1.0

    # translation.
    T_wc_all[:, :3, 3] = poses_t_quat[:, :3]
    # rotation.
    T_wc_all[:, :3, :3] = R.from_quat(poses_t_quat[:, 3:]).as_matrix()

    return T_wc_all


def associate(
        seq_path: Path, first_file: str, second_file: str,
        offset: float = 0.0, max_difference: float = 0.02):
    """Create associations between color and depth channels.

    This function is equivalent to the script provided at
    https://svncvpr.in.tum.de/cvpr-ros-pkg/trunk/rgbd_benchmark/rgbd_benchmark_tools/src/rgbd_benchmark_tools/associate.py

    Args:
        seq_path: path to sequence data (rgb.txt, depth.txt,
                    groundtruth.txt, etc.).
        offset: time offset added to the timestamps of the second file.
        max_difference: maximally allowed time difference for matching entries.

    Returns:

    """
    # equivalent to read_file_list.
    with open(seq_path / first_file, 'r') as f:
        first_list = {
            float(ls[0]): ls[1:]
            for l in f
            if l[0] != '#' and len(ls := l.split()) > 0
        }

    with open(seq_path / second_file, 'r') as f:
        second_list = {
            float(ls[0]): ls[1:]
            for l in f
            if l[0] != '#' and len(ls := l.split()) > 0
        }

    # the same as associate.
    first_keys = list(first_list)
    second_keys = list(second_list)

    potential_matches = [
        (abs(a - (b + offset)), a, b)
        for a in first_keys
        for b in second_keys
        if abs(a - (b + offset)) < max_difference]
    potential_matches.sort()

    matches = []
    for diff, a, b in potential_matches:
        if a in first_keys and b in second_keys:
            first_keys.remove(a)
            second_keys.remove(b)
            matches.append((a, b))
    matches.sort()

    # output filename.
    if "associations_" in first_file:
        first_file = first_file[13:]
    fname = f"associations_{first_file[:-4]}_{second_file[:-4]}.txt"

    # write to disk.
    with open(seq_path / fname, 'w') as f:
        f.writelines([
            f"{ai:.6f} {' '.join(first_list[ai])} {bi:6f} {' '.join(second_list[bi])}\n"
            for ai, bi in matches
        ])


if __name__ == "__main__":
    import cv2
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

    tum = TUM_RGBD('freiburg1_desk', return_depth=True, return_pose=True)

    # define camera frustum:
    frustum_depth = 1.0
    K = tum.K  # calib matrix
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    # bottom-right frustum corner.
    px, py = frustum_depth * (cx / fx), frustum_depth * (cy / fy)

    pz = frustum_depth
    cam_c = np.array([
        [px, py, pz, 1],
        [px, -py, pz, 1],
        [-px, -py, pz, 1],
        [0, -py - py / 2, pz, 1],
        [px, -py, pz, 1],
        [0, 0, 0, 1],
        [-px, py, pz, 1],
        [-px, -py, pz, 1],
        [0, 0, 0, 1],
        [px, py, pz, 1],
        [-px, py, pz, 1],
    ]).T

    # to show the depth.
    max_depth = 3  # meters
    depth_factor = 255 / max_depth

    lims = (np.zeros((3,)), np.zeros((3,)))
    _, _, T_wc0 = tum[0]

    for i in range(len(tum)):
        rgb, depth, T_wc = tum[i]
        T_wc = np.linalg.inv(T_wc0) @ T_wc

        # sample point from first frame and obtain its 3D coord.
        if i == 0:
            h, w = rgb.shape[:2]
            # randomly select points to see if they project correctly across frames.
            n_p = 10
            p2d = np.concatenate((
                np.random.uniform(0, w - 1, (1, n_p)),
                np.random.uniform(0, h - 1, (1, n_p)),
                np.ones((1, n_p))
            ))
            z = depth[p2d[1].astype(int), p2d[0].astype(int)]
            p2d = p2d[:, z > 0]
            z = z[z > 0]
            assert len(z) > 0

            # undistort kp location.
            p2du = cv2.undistortPoints(
                np.ascontiguousarray(p2d[:-1].T),
                tum.K, None,  # tum.distCoeffs,
                R=None, P=None).T[:, 0]
            p2d[:-1] = tum.K[:-1, :-1] @ p2du + tum.K[:-1, -1:]
            # equivalent procedure for undistorting:
            # p2d[:-1] = cv2.undistortPoints(
            #     np.ascontiguousarray(p2d[:-1].T),
            #     tum.K, tum.distCoeffs,
            #     R=None, P=tum.K).T[:, 0]

            # 3d point*s* in cam reference
            print((np.linalg.inv(tum.K) @ p2d
                  * z[None]).shape, np.ones_like(z[None]).shape)
            p3d_c = np.concatenate(
                (np.linalg.inv(tum.K) @ p2d * z[None], np.ones_like(z[None])), axis=0)

            # 3d point in world reference
            p3d_w = T_wc @ p3d_c

        else:
            p3d_c = np.linalg.inv(T_wc) @ p3d_w
            p2d, _ = cv2.projectPoints(
                p3d_c[:-1].T, rvec=np.zeros((3,)), tvec=np.zeros((3,)),
                cameraMatrix=tum.K, distCoeffs=tum.distCoeffs)
            p2d = p2d[:, 0].T
            p2d = np.round(p2d).astype(int)

        for pi in p2d.T:
            cv2.circle(
                rgb,
                (int(round(pi[0])), int(round(pi[1]))),
                radius=5, color=(0, 0, 255), thickness=-1)

        # pose fig
        ax.cla()
        ax.set(
            xlim=(lims[0][0], lims[1][0]),
            ylim=(lims[0][1], lims[1][1]),
            zlim=(lims[0][2], lims[1][2]),
            xlabel='x',
            ylabel='y',
            zlabel='z',
        )

        # plot frustum.
        cam_w = T_wc @ cam_c
        ax.plot(cam_w[0], cam_w[1], cam_w[2], '-m', linewidth=2)
        ax.scatter(T_wc[0, -1], T_wc[1, -1], T_wc[2, -1], 'om', s=10)

        # axes limits.
        lims = (np.min([cam_w[:-1, -1], lims[0]], axis=0),
                np.max([cam_w[:-1, -1], lims[1]], axis=0))

        # aspect ratio:
        limits = [getattr(ax, f'get_{axis}lim')() for axis in 'xyz']
        ax.set_box_aspect(np.ptp(limits, axis=1))

        plt.pause(1e-10)

        # convert depth to np.uint8:
        depth_clipped = np.clip(depth_factor * depth, 0, 255).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_clipped, cv2.COLORMAP_TURBO)

        cv2.imshow('rgb', rgb[:, :, ::-1])
        cv2.imshow('depth', depth_colored)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            plt.close('all')
            break
