""" Parser for KITTI dataset (odometry part) """

import os
import sys
from pathlib import Path
from typing import Union, Optional

import numpy as np
from PIL import Image

ENV_PATH = str(Path(__file__).parents[1])
if ENV_PATH not in sys.path:
    print(f'Inserting {ENV_PATH} to sys.path')
    sys.path.insert(0, ENV_PATH)
from datasets.settings import KITTI_PATH


class KITTI_ODOMETRY:
    """ Parser for KITTI odometry dataset.

    Currently, it parses images, poses and timestamps. It does not parse
    velodyne pointclouds.
    """

    def __init__(
            self,
            seq: Union[int, str],
            use_gray: bool = False,
            return_left: bool = True,
            return_right: bool = False,
            return_times: bool = False,
            return_poses: bool = True
    ):
        seq = str(seq).zfill(2)
        self.seq = seq
        self.return_left = return_left
        self.return_right = return_right
        self.return_times = return_times
        self.return_poses = return_poses

        # paths to image seqs.
        path_im = KITTI_PATH / 'sequences' / seq

        # calib data (see function for details).
        Pi, Tr = _load_calib(path_im / 'calib.txt')
        self.Pi = Pi

        # Images are rectified. Thus, they share the same virtual calib matrix.
        # stored as (3, 3) array with pinhole model.
        self.K = Pi[0, :, :3].copy()

        # setup only the data that will be used.
        cams_lr = (0, 1) if use_gray else (2, 3)

        # length of sequence. TBD within the following setups.
        self.n = None

        if return_left:
            self.path_iml = path_im / f'image_{cams_lr[0]}'
            assert self.path_iml.is_dir()

            self.n = len(os.listdir(self.path_iml))

            # (3, 4) projection matrix.
            # Maps points from velodyne system to cam_left homogeneous pixels.
            self.P_left_velo = Pi[cams_lr[0]] @ Tr

            # (4, 4) transformation matrix from velodyne to cam_left coords.
            # Since cameras are rectified, they have the same virtual image
            # plane. Since `Tr` is the transform velo to cam0, we only need
            # to substract the X-baseline `b_x` w.r.t. cam0. It is stored,
            # w.r.t. cam0 X-coordinates, in the calib matrices as -b_x*f_u.
            self.T_left_velo = Tr.copy()
            self.T_left_velo[0, 3] += Pi[cams_lr[0], 0, 3] / self.K[0, 0]

        if return_right:
            self.path_imr = path_im / f'image_{cams_lr[1]}'
            assert self.path_imr.is_dir()

            if self.n is None:
                self.n = len(os.listdir(self.path_imr))

            # (3, 4) projection matrix.
            # Maps points from velodyne system to cam_right homogeneous pixels.
            self.P_right_velo = Pi[cams_lr[1]] @ Tr

            # (4, 4) transformation matrix from velodyne to cam_right coords.
            self.T_right_velo = Tr.copy()
            self.T_right_velo[0, 3] += Pi[cams_lr[1], 0, 3] / self.K[0, 0]

            # baseline (x-axis) times focal length w.r.t. cam_left system, i.e.
            #                       b_x * f_u
            # computed as:
            #   f_u * b_x(cam0 <-> cam_right) - f_u * b_x(cam0 <-> cam_left).
            # Note that Pi[i, 0, 3] contains `-b_x * f_u` i.e. w. negative sign.
            self.bf = -Pi[cams_lr[1], 0, 3] + Pi[cams_lr[0], 0, 3]

            # baseline in meters.
            self.metric_baseline = self.bf / self.K[0, 0]

        if return_poses:
            try:
                # load cam_left poses. They map a point from cam coordinates to
                # the first cam0 (gray) coordinate system (world).
                T_wc = _load_poses(KITTI_PATH / 'poses' / f'{seq}.txt')

                if not use_gray:
                    # If using cam2, apply transform T_c0_c1 such that T_wc
                    # transforms points form cam1 to world (first pose of cam0)
                    T_wc[:, :3, 3] -= T_wc[:, :3, 0] * \
                        (Pi[2, 0, 3] / self.K[0, 0])

                self.T_wc = T_wc

                # *views* to Rotations and translations.
                self.R_wc = T_wc[:, :3, :3]
                self.t_wc = T_wc[:, :3, 3:4]

                if self.n is None:
                    self.n = T_wc.shape[0]

            except FileNotFoundError:
                print(f"Sequence {seq} doesn't have corresponding {seq}.txt\n")
                raise

        if return_times:
            # timestamps (in seconds) for each pair of synchronized images.
            self.times = np.loadtxt(path_im / 'times.txt')

            if self.n is None:
                self.n = self.times.shape[0]

    def __getitem__(self, idx: Union[int, slice, list[int]]):
        out = {}

        if self.return_poses:
            out['Twc'] = self.T_wc[idx]

        if self.return_times:
            out['t'] = self.times[idx]

        if isinstance(idx, int):
            if self.return_left:
                out['left_im'] = self._get_left(idx)
            if self.return_right:
                out['right_im'] = self._get_right(idx)
            return out

        if isinstance(idx, slice):
            idx_ = range(idx.start, idx.stop, idx.step)
        elif isinstance(idx, list):
            idx_ = idx
        else:
            raise ValueError

        if self.return_left:
            out['left_im'] = [self._get_left(idx_i) for idx_i in idx_]
        if self.return_right:
            out['right_im'] = [self._get_right(idx_i) for idx_i in idx_]

        return out

    def __len__(self):
        return self.n

    def localLeft_to_localRight(self, T_w_cl, inplace=False):
        """ Change of local basis from left camera to right camera.

        In other words, given T_w_cl transform matrix, which transforms points
        from the *left* camera to world coordinates, this function applies
        to it the transform T_cl_cr such that the output matrix transforms 
        points from *right* camera to world coordinates.

        Args:
            T_w_cl: (..., 4, 4) transforms from `cl` to `w` references.
        Returns:
            T_w_cr: (..., 4, 4) transforms from `cr` to `w` references.
        """
        if inplace:
            T_w_cl[..., :3, 3] += self.metric_baseline * T_w_cl[..., :3, 0]
            return

        T_w_cr = T_w_cl.copy()
        T_w_cr[..., :3, 3] += self.metric_baseline * T_w_cl[..., :3, 0]
        return T_w_cr

    def stream(
            self,
            start: int = 0,
            step: Optional[int] = None,
            stop: Optional[int] = None):
        raise NotImplementedError

    def _get_left(self, idx):
        return np.array(Image.open(self.path_iml / f'{str(idx).zfill(6)}.png'))

    def _get_right(self, idx):
        return np.array(Image.open(self.path_imr / f'{str(idx).zfill(6)}.png'))

    def stream_left(self):
        raise NotImplementedError

    def stream_right(self):
        raise NotImplementedError


def _load_calib(p: Path):
    """ Store calib matrices as 3d ndarray

    This data is ordered in the same way within all calib.txt. Thus:
        * returned_val[i] maps to Pi, for i \in {0 .. 3},
        * returned_val[4] maps to Tr
    Each Pi matrix contains [1]:
             | f_u   0     c_u   -f_ub_x |
        Pi = | 0     f_v   c_v   0       |
             | 0     0     1     0       |
    where b_x denotes the baseline in pixels w.r.t. cam0 (gray).
    Lastly, Tr transforms a point from velodyne coordinates into the
    left rectified cam0 (gray) coordinate system.

    Args:
        p: calib.txt path.
    Returns:
        Pi: (4, 3, 4) array containing the projection matrices P(i).
        Tr: (4, 4) trasnformation matrix from velodyne to cam0 coords.
    """
    Pi_Tr = np.loadtxt(
        p, delimiter=' ', usecols=range(1, 13)
    ).reshape(5, 3, 4)

    return Pi_Tr[:4], np.concatenate((Pi_Tr[4], [[0, 0, 0, 1]]), 0)


def _load_poses(p: Path):
    """ Load camera poses for left camera.

    They map a point from current left camera coordinates to the coordinate
    system of the first left camera (defined as world).

    Args:
        p: `seq`.txt path with the poses.
    Returns:
        T_wc: (n, 4, 4) array with the cam0 (gray) poses.
    """
    Twc = np.loadtxt(p, delimiter=' ')  # (n, 3, 4)
    return np.concatenate((
        Twc.reshape(len(Twc), 3, 4),
        np.repeat([[[0, 0, 0, 1]]], len(Twc), 0)
    ), axis=1)


if __name__ == "__main__":
    import cv2
    import open3d as o3d
    import matplotlib.pyplot as plt
    from tqdm import tqdm

    use_o3d = False

    seq = 2
    kitti_gray = KITTI_ODOMETRY(
        seq, return_poses=True, return_left=True, return_right=True, use_gray=True)
    kitti_color = KITTI_ODOMETRY(
        seq, return_poses=True, return_left=True, return_right=True)

    # frustums to draw.
    frustum_depth = 0.6
    K = kitti_gray.K  # it is shared in all virtual cams.
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

    # fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

    for i in tqdm(range(len(kitti_gray))):
        data_gray = kitti_gray[i]
        data_color = kitti_color[i]

        im_lg = data_gray['left_im']
        im_lc = data_color['left_im']

        # left poses (cam0 and cam2)
        Twc_lg = data_gray['Twc']
        Twc_lc = data_color['Twc']

        # right pose (cam1)
        Twc_rg = kitti_gray.localLeft_to_localRight(Twc_lg)
        # right pose (cam3)
        Twc_rc = kitti_color.localLeft_to_localRight(Twc_lc)

        # clear figure.
        # ax.cla()

        # plot frustums.
        # poses = [Twc_lc, Twc_lg, Twc_rc, Twc_rg]
        # color = ['k', 'r', 'g', 'b']
        # for T_wc, ci in zip(poses, color):
        #     cam_w = T_wc @ cam_c
        #     ax.plot(
        #         cam_w[0], cam_w[1], cam_w[2],
        #         '-', color=ci, linewidth=2
        #     )
        #     ax.scatter(
        #         T_wc[0, -1], T_wc[1, -1], T_wc[2, -1],
        #         'o', color=ci, s=10
        #     )

        # # labels.
        # ax.set(xlabel='x', ylabel='y', zlabel='z')

        # # aspect ratio:
        # limits = [getattr(ax, f'get_{axis}lim')() for axis in 'xyz']
        # ax.set_box_aspect(np.ptp(limits, axis=1))

        plt.pause(1e-10)

        # show images.
        cv2.imshow('kitti_gray_left', im_lg)
        cv2.imshow('kitti_color_left', im_lc[:, :, ::-1])
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if use_o3d:
        dataset = KITTI_ODOMETRY(0, return_poses=True, return_left=True)

        vis = o3d.visualization.Visualizer()
        vis.create_window(height=480, width=640)

        CAM_POINTS = np.array([
            [0, 0, 0], [-1, -1, 1], [1, -1, 1], [1, 1, 1],
            [-1, 1, 1], [-0.5, -1, 1], [0.5, -1, 1], [0, -1.2, 1]]).T
        CAM_LINES = np.array([
            [1, 2], [2, 3], [3, 4], [4, 1], [1, 0],
            [0, 2], [3, 0], [0, 4], [5, 7], [7, 6]])

        def add_cam_actor(vis, Twc, scale=0.1):
            cam = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(
                    (scale * Twc[:3, :3] @ CAM_POINTS + Twc[:3, 3:4]).T
                ),
                lines=o3d.utility.Vector2iVector(CAM_LINES)
            )
            cam.paint_uniform_color((0.075, 0.340, 0.830))
            vis.add_geometry(cam, False)

        for i in range(len(dataset)):
            data = dataset[i]

            Twc = data['Twc']
            im = data['left_im']

            add_cam_actor(vis, Twc)

            keep_running = vis.poll_events()
            vis.update_renderer()
            cam = vis.get_view_control().convert_to_pinhole_camera_parameters()
            vis.reset_view_point(True)
            vis.get_view_control().convert_from_pinhole_camera_parameters(cam)

            cv2.imshow('kitti', im)
            if cv2.waitKey(1) & 0xFF == ord('q') or not keep_running:
                break

        cv2.destroyAllWindows()
        vis.destroy_window()
