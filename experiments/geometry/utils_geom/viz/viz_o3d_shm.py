from pathlib import Path
from multiprocessing import Value, Process, Lock, shared_memory, Event
import time

import numpy as np
from cv2 import undistortPoints
from numba import njit
import open3d as o3d
import matplotlib.pyplot as plt


CAM_POINTS = np.array([
    [0, 0, 0],
    [-1, -1, 1],
    [1, -1, 1],
    [1, 1, 1],
    [-1, 1, 1],
    [-0.5, -1, 1],
    [0.5, -1, 1],
    [0, -1.2, 1]])


CAM_LINES = np.array([
    [1, 2], [2, 3], [3, 4], [4, 1], [1, 0], [0, 2], [3, 0], [0, 4], [5, 7], [7, 6]])


@njit('Tuple((f8[:,::1], u1[:,::1]))(f8[:,::1], f8[:,:,::1], f8[:,:,::1], '
      'f8[:,:,::1], u1[:,:,:,::1])', fastmath=False, cache=True)
def iproj(rays, depths, Rwc, twc, images):
    """Given depth maps, get 3D coordinates in world reference. """
    n, h, w = depths.shape
    ni, hi, wi, c = images.shape
    hwr, cr = rays.shape
    assert n == ni
    assert h == hi
    assert w == wi
    assert c == 3
    # assert cr == 3

    hw = h * w
    assert hwr == hw

    depths = depths.reshape(n, hw)
    images = images.reshape(n, hw, 3)

    # mask for valid depths.
    mask = depths > 0
    # valid 3d points.
    npoints = np.sum(mask)
    # preallocate coords and colors.
    out = np.empty((npoints, 3))
    out_c = np.empty((npoints, 3), dtype=np.uint8)

    idx = 0
    for i in range(n):
        r = Rwc[i]
        t = twc[i]
        for j in range(hw):

            if mask[i, j]:
                # world coords. -> p_w = R_wc * p_c + t_wc
                out[idx, 0] = t[0, 0] + depths[i, j] * (
                    r[0, 0] * rays[j, 0] + r[0, 1] * rays[j, 1] + r[0, 2])

                out[idx, 1] = t[1, 0] + depths[i, j] * (
                    r[1, 0] * rays[j, 0] + r[1, 1] * rays[j, 1] + r[1, 2])

                out[idx, 2] = t[2, 0] + depths[i, j] * (
                    r[2, 0] * rays[j, 0] + r[2, 1] * rays[j, 1] + r[2, 2])

                # color.
                out_c[idx] = images[i, j]

                idx += 1

    return out, out_c


def T2Rt(T):
    assert T.shape[-2:] == (4, 4)
    return T[..., :-1, :-1], T[..., :-1, -1:]


class SharableNDArray:
    """ Numpy arrays with shared memory.

    Adapted from Ahmed AEK: https://stackoverflow.com/a/74638164/14559854
    """

    def __init__(self, arr: np.ndarray):
        self.size = arr.nbytes
        self.type = arr.dtype
        self.shape = arr.shape

        self.shm = shared_memory.SharedMemory(create=True, size=self.size)
        self.shm_name = self.shm.name

        self.value = np.ndarray(self.shape, self.type, buffer=self.shm.buf)
        self.value[:] = arr[:]

    def __getstate__(self):
        """ Only pickle cheap references to the preallocated memory block. """
        return (self.shm_name, self.size, self.type, self.shape)

    def __setstate__(self, state):
        """ Unpickle references and use them to access the preallocated shm """
        self.shm_name, self.size, self.type, self.shape = state
        self.shm = shared_memory.SharedMemory(self.shm_name)
        self.value = np.ndarray(self.shape, self.type, buffer=self.shm.buf)


class VizOpen3D:
    """ Class acting as both container and proxy for/with the visualizer """

    def __init__(
            self, nframes, h, w, K, distCoeffs=None, npoints=100, debug=False):
        self.nframes = nframes
        self.h = h
        self.w = w
        self.K = K
        self.distCoeffs = distCoeffs
        self.npoints = npoints
        self.debug = debug

        # shared state across processes.
        self.Rwc_sh = SharableNDArray(
            np.zeros((nframes + 1, 3, 3), dtype=float))
        self.twc_sh = SharableNDArray(
            np.zeros((nframes + 1, 3, 1), dtype=float))
        self.images_sh = SharableNDArray(
            np.zeros((nframes, h, w, 3), dtype=np.uint8))
        self.depths_sh = SharableNDArray(
            np.zeros((nframes, h, w), dtype=float))
        self.points_sh = SharableNDArray(
            np.zeros((3, npoints), dtype=float))
        self.npoints_valid = Value('i', 0)

        # handy references to values.
        self.Rwc = self.Rwc_sh.value
        self.twc = self.twc_sh.value
        self.images = self.images_sh.value
        self.depths = self.depths_sh.value
        self.points = self.points_sh.value

        # colors to visualize triangulated tracks.
        self.colors = plt.get_cmap('hsv')(  # type: ignore
            np.linspace(0.0, 1.0, npoints))[:, :3]
        np.random.RandomState(0).shuffle(self.colors)

        # mutex.
        self.lock = Lock()

        # (boolean) event flag.
        self.do_update = Event()

        # launch visualization. Run as daemonic process.
        self.visualizer = Process(
            target=visualization, args=(self,), daemon=True)
        self.visualizer.start()

    def update_all_state(self, state: dict):
        with self.lock:
            self.Rwc[:] = state['Rwc']
            self.twc[:] = state['twc']
            self.images[:] = state['images']
            self.depths[:] = state['depths']

            if 'points' in state:
                # triangulated points.
                n_tri = state['points'].shape[1]
                if n_tri >= self.npoints:
                    self.npoints_valid.value = self.npoints
                    self.points[:] = state['points'][:, :self.npoints]
                else:
                    self.npoints_valid.value = n_tri
                    self.points[:, :n_tri] = state['points']

            self.do_update.set()

    def __setitem__(self, idx: int, item: dict):
        with self.lock:
            self.Rwc[idx] = item['Rwc']
            self.twc[idx] = item['twc']
            if idx < self.nframes:
                self.images[idx] = state['image']
                self.depths[idx] = state['depth']

    def set_tri_points(self, tri_points):
        n_tri = tri_points.shape[1]
        with self.lock:
            if n_tri >= self.npoints:
                self.npoints_valid.value = self.npoints
                self.points[:] = tri_points[:, :self.npoints]
            else:
                self.npoints_valid.value = n_tri
                self.points[:, :n_tri] = tri_points

    def __getstate__(self):
        state = self.__dict__.copy()
        # delete state that'll be copied since it won't refer anymore
        # to the shared values.
        del (
            state['Rwc'],
            state['twc'],
            state['images'],
            state['depths'],
            state['points'])
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

        with self.lock:
            # set references to shared values.
            self.Rwc = self.Rwc_sh.value
            self.twc = self.twc_sh.value
            self.images = self.images_sh.value
            self.depths = self.depths_sh.value
            self.points = self.points_sh.value


def init_cam_geoms(vis, ncams, scale=0.1):
    cam_geoms = []
    for i in range(ncams):
        cam_geoms.append(
            o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(scale * CAM_POINTS),
                lines=o3d.utility.Vector2iVector(CAM_LINES))
        )

        if i < ncams - 1:
            cam_geoms[i].paint_uniform_color((0.075, 0.340, 0.830))
        else:
            # the last is the query.
            cam_geoms[i].paint_uniform_color((0.705, 0.075, 0.830))

        vis.add_geometry(cam_geoms[i])
    return cam_geoms


def init_tri_map(vis, colors, rad):
    tri_spheres = []
    for i, c in enumerate(colors):
        tri_spheres.append(
            o3d.geometry.TriangleMesh.create_sphere(rad))
        tri_spheres[i].paint_uniform_color(c)
        vis.add_geometry(tri_spheres[i])
    return tri_spheres


def get_rays(h, w, K, distCoeffs):
    X, Y = np.meshgrid(np.arange(float(w)), np.arange(float(h)))
    return undistortPoints(
        np.concatenate((X.reshape(-1, 1), Y.reshape(-1, 1)), axis=1),
        K, distCoeffs, R=None, P=None)[:, 0]


def visualization(video):
    """Visualization. Runs as a daemonic process, launched from "video"."""
    cams_scale = 0.25
    tri_scale = 1.0
    far_th = 1.5

    # initialize visualizer.
    vis = o3d.visualization.VisualizerWithKeyCallback()

    def cams_scale_callback(vis, action, mods):
        """ key S -> increase size. SHIFT + S -> decrease size."""
        nonlocal cams_scale
        if action in (1, 2) and mods == 0:
            # key has just been pressed or keeps being pressed.
            cams_scale *= 1.5
        elif action in (1, 2) and mods == 1:
            cams_scale /= 1.5
        return True

    def tri_scale_callback(vis, action, mods):
        """ key Z -> increase radius. SHIFT + Z -> decrease radius."""
        nonlocal tri_scale
        # check if key has just been pressed or keeps being pressed.
        if action in (1, 2) and mods == 0:
            tri_scale = 1.5
        elif action in (1, 2) and mods == 1:
            tri_scale = 0.7
        elif action == 0:
            tri_scale = 1.0
        return True

    # register callbacks.
    vis.register_key_action_callback(ord('S'), cams_scale_callback)
    vis.register_key_action_callback(ord('Z'), tri_scale_callback)

    # create window display.
    vis.create_window(height=480, width=640)
    # vis.create_window()

    # rendering options.
    vis.get_render_option().load_from_json(
        str(Path(__file__).parent / "renderoption.json"))

    # open3d geometry instances.
    local_map = o3d.geometry.PointCloud()
    vis.add_geometry(local_map)

    triangulated_map = init_tri_map(vis, video.colors, rad=0.01)
    linesets_cams = init_cam_geoms(vis, len(video.Rwc), cams_scale)

    rays = get_rays(video.h, video.w, video.K, video.distCoeffs)

    # leverage do_update event to tell the main process that viz is ready.
    if not video.debug:
        video.do_update.set()
        time.sleep(1)
        video.do_update.clear()

    # run non-blocking visualization.
    keep_running = True
    while keep_running:

        # only update geometries if the parent process requests it.
        if video.do_update.is_set():
            video.do_update.clear()

            with video.lock:
                Rwc_ = video.Rwc.copy()
                twc_ = video.twc.copy()
                images_ = video.images.copy()
                depths_ = video.depths.copy()

                if not video.debug:
                    nvalid_tri = video.npoints_valid.value
                    points_tri_ = video.points[:, :nvalid_tri].copy()

            # coordinates and colors of observed local map.
            map_wc, map_c = iproj(rays, depths_, Rwc_, twc_, images_)
            del images_, depths_

            if video.debug:
                ntri = len(map_wc)
                points_tri_ = map_wc[np.random.randint(
                    0, ntri - 1, min(ntri, video.npoints))].T
            else:
                # if there aren't enough points, just repeat one to avoid
                # removing geometries.
                if nvalid_tri < video.npoints:
                    points_tri_[:, nvalid_tri:] = points_tri_[:, 0:1]

            # new local map.
            local_map.points = o3d.utility.Vector3dVector(map_wc)
            local_map.colors = o3d.utility.Vector3dVector(map_c / 255)
            vis.update_geometry(local_map)

            # new camera poses.
            for lset, Rwc_i, twc_i in zip(linesets_cams, Rwc_, twc_):
                lset.points = o3d.utility.Vector3dVector(
                    (cams_scale * Rwc_i @ CAM_POINTS.T + twc_i).T)
                vis.update_geometry(lset)

            # new triangulated points.
            lmap_ext = np.abs(map_wc.max(axis=0) - map_wc.min(axis=0))
            lmap_cent = np.mean(map_wc, axis=0)

            for point_sph, loc in zip(triangulated_map, points_tri_.T):
                # Ignore too far points (they can hang the visualizer).
                if np.all(np.abs(loc - lmap_cent) / lmap_ext < far_th):
                    point_sph.translate(loc, relative=False)
                    if tri_scale != 1.0:
                        point_sph.scale(tri_scale, point_sph.get_center())
                    vis.update_geometry(point_sph)

            # hacky way to fit the rendered bounding box after the changes.
            cam = vis.get_view_control().convert_to_pinhole_camera_parameters()
            vis.reset_view_point(True)
            vis.get_view_control().convert_from_pinhole_camera_parameters(cam)

        keep_running = vis.poll_events()
        vis.update_renderer()


if __name__ == "__main__":
    import sys
    from tqdm import tqdm

    ENV_PATH = str(Path(__file__).parents[4])
    if ENV_PATH not in sys.path:
        print(f"inserting {ENV_PATH} to sys.path.")
        sys.path.insert(0, ENV_PATH)
    from datasets.load_data import load_dataset

    data = load_dataset(
        'tum_rgbd',
        {'sequence': 'freiburg1_desk',
         'return_depth': True,
         'return_pose': True}
    )

    nframes = 3
    video = VizOpen3D(
        nframes=nframes,
        h=data.h,
        w=data.w,
        K=data.K,
        distCoeffs=data.distCoeffs,
        npoints=100,
        debug=True
    )

    # define state.
    Rwc = np.zeros((nframes + 1, 3, 3))
    twc = np.zeros((nframes + 1, 3, 1))
    images = np.zeros((nframes, data.h, data.w, 3))
    depths = np.zeros((nframes, data.h, data.w))
    state = {
        'Rwc': Rwc,
        'twc': twc,
        'images': images,
        'depths': depths,
    }

    _, _, Twc0 = data[0]
    # Tcw0 = np.linalg.inv(Twc0)
    Tcw0 = np.eye(4)
    time.sleep(10)
    for i in tqdm(range(len(data) - nframes - 1)):
        time.sleep(1e-3)
        for j in range(nframes + 1):
            image, depth, Twc = data[i + j]

            Rwc[j], twc[j] = T2Rt(Tcw0 @ Twc)
            if j < nframes:
                images[j] = image
                depths[j] = depth

        video.update_all_state(state)
