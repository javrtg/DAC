"""PnP experiment"""

import sys
from pathlib import Path
from argparse import ArgumentParser
from typing import Optional, Union

import numpy as np
import cv2
from tqdm import tqdm
from tabulate import tabulate

ENV_PATH = str(Path(__file__).parents[2])
if ENV_PATH not in sys.path:
    print(f"inserting {ENV_PATH} to sys.path.")
    sys.path.insert(0, ENV_PATH)
from experiments.geometry.utils_geom import configs
from experiments.geometry.utils_geom.eval import get_eval_objects
from experiments.geometry.utils_geom.matching_graph import MatchingGraph
from experiments.geometry.utils_geom.data_parsers import get_dataset
from experiments.geometry.utils_geom.viz import VizPlt, VizOpen3D
from detectors.load import load_detector


def cls_name(instance) -> str:
    """ Name of the class to which an instance belongs """
    return instance.__class__.__name__.lower()


def T2Rt(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    assert T.shape[-2:] == (4, 4)
    return T[..., :-1, :-1], T[..., :-1, -1:]


def exhaustive_pairwise_matching(
    det, desc_q: np.ndarray, kps_nq: np.ndarray, frames_data: dict,
    ranscac_f_cfg: Optional[dict] = None, run_ransac: bool = True
) -> list[tuple]:
    """Mutal-NN matching with epipolar geometry verification."""

    # iterate over all the stored frames.
    final_matches = []
    for fd in frames_data:
        # MNN matching and corresponding cosine similarities.
        matches, sim = det.match_descriptors(
            fd['desc'], desc_q, return_sim=True)

        if len(matches) <= 7:
            final_matches.append(None)
            continue

        kps_nr_m = fd['kps_n'][:, matches[:, 0]]
        kps_nq_m = kps_nq[:, matches[:, 1]]

        if run_ransac:
            # epipolar geometry verification.
            _, mask = cv2.findFundamentalMat(
                kps_nr_m.T, kps_nq_m.T, **ranscac_f_cfg)

            # avoid result with number of inliers <= the minimal sample.
            if mask.sum() <= 7:
                final_matches.append(None)
                continue

            idx = mask[:, 0].nonzero()[0]  # (n_verif_matches,)
            matches = matches[idx]
            sim = sim[idx]

        final_matches.append((
            matches,  # keypoint indexes within the image.
            sim  # corresponding similarities.
        ))

    return final_matches


def filter_matches_given_graph(
        graph: MatchingGraph, matching_data: list[tuple],
        kps_nq: np.ndarray, sigmas2d: np.ndarray, sigmas2d_s: np.ndarray,
) -> dict:
    """ Obtain 3d-2d matches given feature tracks and 2d-2d matches of the
    query image w.r.t. the reference images.

    A tentative 2d-2d match between a reference and query keypoints is
    retained only if:
        1) the reference keypoint has been previously triangulated.
        2) all the matches corresponding to a query keypoint are related to
            the same feature track.
    """

    def filter_match(i_q: int, im_idx: int, i_r: int) -> Optional[dict]:
        """ Check if the match is valid and assign dict value to it.

        Args:
            i_q: keypoint index within the query image.
            im_idx: index of the reference image.
            i_r: keypoint index within the reference image.

        Return:
            - None, if the match is invalid
            - {'track': int, 'X': ndarray}, otherwise
        """
        # 1) if ref keypoint not in matching graph -> invalid.
        if (im_idx, i_r) not in graph.kps:
            return None

        # 2) if ref keypoint was not succesfully triangulated -> invalid.
        kp_r = graph.kps[(im_idx, i_r)]
        if kp_r.track.X is None:
            return None

        # 3) if query keypoint is not present -> initialize with the reference.
        if i_q not in matches:
            return {
                'track': kp_r.track.id,
                'X': kp_r.track.X,
                'X_cov': kp_r.track.X_cov,
                'X_s': kp_r.track.X_s,
                'cov': kp_r.track.sigma3d_cov,
                'cov_s': kp_r.track.sigma3d_s}

        kp_q = matches[i_q]
        # 4) if match is already invalid -> mantain invalid.
        if kp_q is None:
            return None

        # 5) if query keypoint presents inconsistent tracks -> invalid.
        if kp_q['track'] != kp_r.track.id:
            return None

        # 6) if any of the above conditions happen, just repeat the current
        # value. Note: 4) check is not needed, it is left just for clarity.
        return kp_q

    # initialize database.
    matches = {}

    for im_idx, md in enumerate(matching_data):
        if md is None:
            continue

        kp_idxes_r = md[0][:, 0]  # (nmatches,)
        kp_idxes_q = md[0][:, 1]  # (nmatches,)

        # update matches if needed.
        matches.update({
            i_q: filter_match(i_q, im_idx, i_r)
            for i_q, i_r in zip(kp_idxes_q, kp_idxes_r)
        })

    # triangulated 3d coordinates (depending in uncertainty used).
    X_wu = []
    X_cov = []
    X_s = []
    # uncertainty measures.
    cov = []
    cov_s = []

    # 2d matches.
    p = []
    covs2d = []
    covs2d_s = []

    for k, v in matches.items():
        if v is not None:
            X_wu.append(v['X'])
            X_cov.append(v['X_cov'])
            X_s.append(v['X_s'])

            cov.append(v['cov'])
            cov_s.append(v['cov_s'])

            p.append(kps_nq[:, k])
            covs2d.append(sigmas2d[k])
            covs2d_s.append(sigmas2d_s[k])

    matches_2d3d = {
        "p": np.array(p).T,
        "X_wu": np.array(X_wu).T,
        "X_cov": np.array(X_cov).T,
        "X_s": np.array(X_s).T,
        "sigmas3d": np.array(cov),
        "sigmas3d_s": np.array(cov_s),
        "sigmas2d": np.array(covs2d),
        "sigmas2d_s": np.array(covs2d_s)
    }

    return matches_2d3d


def filter_matches_p3p(
        matches_2d3d: dict, ransac_cfg: Union[cv2.UsacParams, dict]) -> tuple:
    """Return 2d-3d inliers and a rough estimate of the camera pose"""
    # init filtered version of matches_2d3d, per uncertainty type.
    out = {
        "none": {},
        "cov": {},
        "cov_s": {}
    }

    if isinstance(ransac_cfg, dict):
        # using 3d points without uncertainty.
        ret, qest, test, inliers = cv2.solvePnPRansac(
            objectPoints=matches_2d3d['X_wu'].T,
            imagePoints=matches_2d3d['p'].T,
            cameraMatrix=np.eye(3),
            distCoeffs=None,
            **ransac_cfg)

        if not ret or len(inliers) < 6:
            return False, None

        out['none']['X'] = matches_2d3d['X_wu'][:, inliers[:, 0]]
        out['none']['p'] = matches_2d3d['p'][:, inliers[:, 0]]
        out['none']['Rcw'] = cv2.Rodrigues(qest)[0]
        out['none']['tcw'] = test

        # using 3d points with 2d covariances.
        ret, qest, test, inliers = cv2.solvePnPRansac(
            objectPoints=matches_2d3d['X_cov'].T,
            imagePoints=matches_2d3d['p'].T,
            cameraMatrix=np.eye(3),
            distCoeffs=None,
            **ransac_cfg)

        if not ret or len(inliers) < 6:
            return False, None

        out['cov']['X'] = matches_2d3d['X_cov'][:, inliers[:, 0]]
        out['cov']['p'] = matches_2d3d['p'][:, inliers[:, 0]]

        out['cov']['Rcw'] = cv2.Rodrigues(qest)[0]
        out['cov']['tcw'] = test
        out['cov']['sigmas3d'] = matches_2d3d['sigmas3d'][inliers[:, 0]]
        out['cov']['sigmas2d'] = matches_2d3d['sigmas2d'][inliers[:, 0]]

        # using 3d points with 2d covariances based on scores.
        ret, qest, test, inliers = cv2.solvePnPRansac(
            objectPoints=matches_2d3d['X_s'].T,
            imagePoints=matches_2d3d['p'].T,
            cameraMatrix=np.eye(3),
            distCoeffs=None,
            **ransac_cfg)

        if not ret or len(inliers) < 6:
            return False, None

        out['cov_s']['X'] = matches_2d3d['X_s'][:, inliers[:, 0]]
        out['cov_s']['p'] = matches_2d3d['p'][:, inliers[:, 0]]

        out['cov_s']['Rcw'] = cv2.Rodrigues(qest)[0]
        out['cov_s']['tcw'] = test
        out['cov_s']['sigmas3d'] = matches_2d3d['sigmas3d_s'][inliers[:, 0]]
        out['cov_s']['sigmas2d'] = matches_2d3d['sigmas2d_s'][inliers[:, 0]]

    elif isinstance(ransac_cfg, cv2.UsacParams):
        # using 3d points without uncertainty.
        ret, _, qest, test, inliers = cv2.solvePnPRansac(
            objectPoints=matches_2d3d['X_wu'].T,
            imagePoints=matches_2d3d['p'].T,
            cameraMatrix=np.eye(3),
            distCoeffs=None,
            params=ransac_cfg)

        if not ret or len(inliers) < 6:
            return False, None

        out['none']['X'] = matches_2d3d['X_wu'][:, inliers[:, 0]]
        out['none']['p'] = matches_2d3d['p'][:, inliers[:, 0]]
        out['none']['Rcw'] = cv2.Rodrigues(qest)[0]
        out['none']['tcw'] = test

        # using 3d points with 2d covariances.
        ret, _, qest, test, inliers = cv2.solvePnPRansac(
            objectPoints=matches_2d3d['X_cov'].T,
            imagePoints=matches_2d3d['p'].T,
            cameraMatrix=np.eye(3),
            distCoeffs=None,
            params=ransac_cfg)

        if not ret or len(inliers) < 6:
            return False, None

        out['cov']['X'] = matches_2d3d['X_cov'][:, inliers[:, 0]]
        out['cov']['p'] = matches_2d3d['p'][:, inliers[:, 0]]

        out['cov']['Rcw'] = cv2.Rodrigues(qest)[0]
        out['cov']['tcw'] = test
        out['cov']['sigmas3d'] = matches_2d3d['sigmas3d'][inliers[:, 0]]
        out['cov']['sigmas2d'] = matches_2d3d['sigmas2d'][inliers[:, 0]]

        # using 3d points with 2d covariances based on scores.
        ret, _, qest, test, inliers = cv2.solvePnPRansac(
            objectPoints=matches_2d3d['X_s'].T,
            imagePoints=matches_2d3d['p'].T,
            cameraMatrix=np.eye(3),
            distCoeffs=None,
            params=ransac_cfg)

        if not ret or len(inliers) < 6:
            return False, None

        out['cov_s']['X'] = matches_2d3d['X_s'][:, inliers[:, 0]]
        out['cov_s']['p'] = matches_2d3d['p'][:, inliers[:, 0]]

        out['cov_s']['Rcw'] = cv2.Rodrigues(qest)[0]
        out['cov_s']['tcw'] = test
        out['cov_s']['sigmas3d'] = matches_2d3d['sigmas3d_s'][inliers[:, 0]]
        out['cov_s']['sigmas2d'] = matches_2d3d['sigmas2d_s'][inliers[:, 0]]

    else:
        raise TypeError

    return True, out


def add_matches_to_graph(
        graph: MatchingGraph, matching_data: list[tuple],
        frames_data: list[dict]):
    """ Incorporate new nodes (keypoints) and edges (matches) to the graph.

    Data regarding to the last query image is also added since it is needed
    for triangulating the feature tracks once they have been computed.
    """
    im_idx_q = len(frames_data) - 1

    # add image data to the graph.
    graph.add_image(
        # calib matrix = I_3 since coordinates are normalized.
        np.eye(3),
        # pose: (R, t)
        frames_data[im_idx_q]['pose'][0],
        frames_data[im_idx_q]['pose'][1],
        # identifier
        im_idx_q
    )

    for im_idx, md in enumerate(matching_data):
        if md is None:
            continue

        kp_idxes = md[0]  # (nmatches, 2)
        sim = md[1]  # (nmatches)

        graph.add_keypoints_and_tentative_matches(
            # identifier for the pair of images.
            im1_idx=im_idx,
            im2_idx=im_idx_q,
            # feature indexes in their corresponding image.
            kp1s_idx=kp_idxes[:, 0],
            kp2s_idx=kp_idxes[:, 1],
            # feature coordinates.
            kp1s_coord=frames_data[im_idx]['kps_n'][:, kp_idxes[:, 0]],
            kp2s_coord=frames_data[im_idx_q]['kps_n'][:, kp_idxes[:, 1]],
            # uncertainty measures.
            kp1s_covs=frames_data[im_idx]['covs'][kp_idxes[:, 0]],
            kp2s_covs=frames_data[im_idx_q]['covs'][kp_idxes[:, 1]],
            kp1s_covs_s=frames_data[im_idx]['covs_s'][kp_idxes[:, 0]],
            kp2s_covs_s=frames_data[im_idx_q]['covs_s'][kp_idxes[:, 1]],
            # descriptor similarities (weight of each edge in the graph).
            similarities=sim
        )


def undistort_features(
        kps: np.ndarray, C: np.ndarray, K: np.ndarray, distCoeffs: np.ndarray,
        eps: float = 1e-9) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize the coordinates of the feature and compute their
    corresponding covariances.

    Args:
        kps: (3, n) un-normalized local-features (keypoints) coordinates (
            first two dims.) and their scores (last dimension).
        C: (n, 2, 2) inverse of covariance (information) matrices.
        K: (3, 3) pinhole calibration matrix.
        distCoeffs: OpenCV distortion params.

    Returns:
        kps: (2, n), normalized coordinates.
        covs: (n, 2, 2) propagated covariances based on C.
        covs_s: (n, 2, 2) propagated covariances based on scores.
    """
    # covariance matrices associated with C and s.
    covs = np.linalg.inv(C + eps)
    covs_s = np.zeros_like(C)
    covs_s[:, 0, 0] = (inv_s := 1.0 / kps[2])
    covs_s[:, 1, 1] = inv_s

    # undistort kps.
    kps = np.ascontiguousarray(kps[:2].T)
    kps = cv2.undistortPoints(kps, K, distCoeffs, R=None, P=None).T[:, 0]

    # propagate covariances to the normalized coordinates. As an approximation
    # don't consider the distortion coefficients.
    fx, fy = K[0, 0], K[1, 1]

    covs[:, 0, 0] /= fx * fx
    covs[:, 1, 1] /= fy * fy
    covs[:, 1, 0] /= fx * fy
    covs[:, 0, 1] = covs[:, 1, 0]

    covs_s[:, 0, 0] /= fx * fx
    covs_s[:, 1, 1] /= fy * fy

    return kps.astype(np.float64), covs, covs_s


def min_frame_dist(t_q: np.ndarray, frames_data: list[dict]):
    ts = [frame_i["loc"] for frame_i in frames_data]
    return np.min(np.linalg.norm(ts - t_q, axis=1))


def main(
        dataset, detector, pnp_models, ranscac_f_cfg, ransac_p3p_cfg,
        nframes, min_dist_th,
        do_viz_plt=False, do_viz_o3d=False, verbose=False, save_plt=False):
    """ Evaluation pipeline based on [Vakhitov, 2021] """

    # base dir where results will be saved.
    data_det_path = Path(__file__).parent / (
        f'results/{cls_name(dataset)}/{dataset.sequence}/{cls_name(detector)}'
    )

    if do_viz_plt or do_viz_o3d:
        npoints = 100
    if do_viz_plt:
        viewer_plt = VizPlt(nframes, dataset.h, dataset.w, npoints, save_plt)
    if do_viz_o3d:
        viewer_o3d = VizOpen3D(
            nframes=nframes,
            h=dataset.h,
            w=dataset.w,
            K=dataset.K,
            distCoeffs=dataset.distCoeffs,
            npoints=npoints, debug=False)

        viewer_o3d_state = {
            "Rwc": np.empty((nframes + 1, 3, 3)),
            "twc": np.empty((nframes + 1, 3, 1)),
            "images": np.empty((nframes, dataset.h, dataset.w, 3)),
            "depths": np.empty((nframes, dataset.h, dataset.w)),
            "points": None
        }

        # wait till viz process is launched and ready.
        print("\nWaiting for Open3D viewer...\n")
        viewer_o3d.do_update.wait()
        viewer_o3d.do_update.clear()
        print("\nOpen3D viewer ready!\n")

    if verbose:
        tab_header = ("e_rot", "e_t", "e_rot_opt", "e_t_opt")

    # sequence is assumed to have constant intrinsics through the sequence.
    K = dataset.K
    distCoeffs = dataset.distCoeffs

    # list of image indexes (to further investigate results).
    im_idxes = [None] * (nframes + 1)

    tqdm_desc = (
        f"{cls_name(detector)}, {cls_name(dataset)}.{dataset.sequence}")
    for i in tqdm(range(len(dataset) - nframes), desc=tqdm_desc):
        im_idxes[0] = i

        im, depth, Twc = dataset[i]
        Rwc, twc = T2Rt(Twc)
        Rcw, tcw = Rwc.T, -Rwc.T @ twc

        # extract local features.
        kps, desc, C = detector(im)
        # normalized coordinates and corresponding covariances.
        kps_n, covs, covs_s = undistort_features(kps, C, K, distCoeffs)

        # initialize matches graph.
        graph = MatchingGraph()
        graph.add_image(
            # calib matrix = I_3 since coordinates are normalized.
            np.eye(3),
            # pose and id.
            Rcw, tcw, 0)

        frames_data = [{
            "pose": (Rcw, tcw),
            "loc": twc,
            "kps_n": kps_n,
            "covs": covs,
            "covs_s": covs_s,
            "desc": desc
        }]

        if verbose:
            tqdm.write(f'\nImage {i} as anchor.')

        if do_viz_plt:
            viewer_plt.add_viz_data(im, i, kps)
        if do_viz_o3d:
            viewer_o3d_state['Rwc'][0] = Rwc
            viewer_o3d_state['twc'][0] = twc
            viewer_o3d_state['images'][0] = im
            viewer_o3d_state['depths'][0] = depth

        # -----
        # get new frames / data for triangulation.
        j = i + 1
        while len(frames_data) < nframes and j < len(dataset):
            im_q, depth_q, Twc_q = dataset[j]
            Rwc_q, twc_q = T2Rt(Twc_q)
            Rcw_q, tcw_q = Rwc_q.T, -Rwc_q.T @ twc_q

            # check distance between cameras.
            min_dist = min_frame_dist(twc_q, frames_data)
            if min_dist_th > 0 and min_dist < min_dist_th:
                if verbose:
                    tqdm.write(
                        f"Image {j} is too close. "
                        f"Distance = {min_dist:.2f}")
                # skip to next frame/image.
                j += 1
                continue

            # local-features data of the query image.
            kps_q, desc_q, C_q = detector(im_q)
            kps_nq, covs_q, covs_sq = undistort_features(
                kps_q, C_q, K, distCoeffs)

            # match the query image with the other reference images.
            matching_data = exhaustive_pairwise_matching(
                detector, desc_q, kps_nq, frames_data,
                ranscac_f_cfg, run_ransac=True)

            if all(i is None for i in matching_data):
                if verbose:
                    tqdm.write(f"Image {j}: No matches found.")
                # skip to next frame/image.
                j += 1
                continue

            # update data being used.
            frames_data.append({
                "pose": (Rcw_q, tcw_q),
                "loc": twc_q,
                "kps_n": kps_nq,
                "covs": covs_q,
                "covs_s": covs_sq,
                "desc": desc_q
            })

            # add matches to the bipartite graph.
            add_matches_to_graph(graph, matching_data, frames_data)

            if verbose:
                matches_txt = [
                    len(md[0]) if md is not None else 0 for md in matching_data
                ]
                tqdm.write(f'Image {j}. # matches: {matches_txt}. Total: '
                           f'{sum(matches_txt)}')

            if do_viz_plt:
                # for local-feature tracks.
                viewer_plt.add_viz_data(im_q, j, kps_q)
            if do_viz_o3d:
                # for local map and triangulations.
                idx_cam = len(frames_data) - 1
                viewer_o3d_state['Rwc'][idx_cam] = Rwc_q
                viewer_o3d_state['twc'][idx_cam] = twc_q
                viewer_o3d_state['images'][idx_cam] = im_q
                viewer_o3d_state['depths'][idx_cam] = depth_q

            im_idxes[len(frames_data) - 1] = j

            j += 1

        if len(frames_data) < nframes:
            # not enough frames to triangulate
            if verbose:
                tqdm.write("Sequence length is below the minimum needed. -> "
                           f"{len(frames_data)} < {nframes} (minimum).")
            continue

        # -----
        if verbose:
            tqdm.write("Generating and triangulating feature tracks...")

        graph.generate_tracks()
        graph.triangulate_tracks_nb()

        # -----
        # Use next frame for applying PnP (as in Vakhitov's paper).
        next_not_valid = True
        n_not_valid = 0
        while next_not_valid and j < len(dataset) and n_not_valid < 10:
            im_q, _, Twc_gt = dataset[j]

            kps_q, desc_q, C_q = detector(im_q)
            kps_nq, covs_q, covs_sq = undistort_features(
                kps_q, C_q, K, distCoeffs)

            # match it to images used for triangulation.
            matching_data = exhaustive_pairwise_matching(
                detector, desc_q, kps_nq, frames_data, run_ransac=False)

            if all(i is None for i in matching_data):
                if verbose:
                    tqdm.write(f"Image {j}: No matches found.")
                if do_viz_plt:
                    viewer_plt.clear_regs()
                # skip to next frame/image.
                j += 1
                n_not_valid += 1
                continue

            # prune matches lacking triangulation and track consistency.
            matches_2d3d = filter_matches_given_graph(
                graph, matching_data, kps_nq, covs_q, covs_sq)

            if len(matches_2d3d["p"].shape) == 1 or matches_2d3d["p"].shape[1] < 6:
                # not enough inliers.
                if verbose:
                    tqdm.write(f"Image {j} not enough 3d-2d valid matches.")
                if do_viz_plt:
                    viewer_plt.clear_regs()
                j += 1
                n_not_valid += 1
                continue

            # filter matches_2d3d with p3p ransac and rough pose estimation.
            # Using no-uncertainty and {cov, scores}-based uncertainty.
            ret, matches2d3d = filter_matches_p3p(matches_2d3d, ransac_p3p_cfg)

            if not ret:
                if verbose:
                    tqdm.write(f"Image {j} not enough 3d-2d valid matches.")
                if do_viz_plt:
                    viewer_plt.clear_regs()
                j += 1
                n_not_valid += 1
                continue
            next_not_valid = False

            im_idxes[-1] = j

            # ground-truth pose.
            Rwc_gt, twc_gt = T2Rt(Twc_gt)
            Rcw_gt, tcw_gt = Rwc_gt.T, -Rwc_gt.T @ twc_gt

            for pnp_model in pnp_models:
                metrics = pnp_model(
                    matches2d3d, Rcw_gt, tcw_gt, im_idxes, debug=verbose)

                if verbose:
                    tqdm.write(
                        tabulate(metrics, tab_header, floatfmt=".3f"))

            if do_viz_plt:
                viewer_plt.update_plot(graph.tracks)
            if do_viz_o3d:
                viewer_o3d_state['Rwc'][nframes - 1] = Rwc_gt
                viewer_o3d_state['twc'][nframes - 1] = twc_gt
                viewer_o3d_state['points'] = graph.points3d
                viewer_o3d.update_all_state(viewer_o3d_state)

    # save results to disk.
    for pnp_model in pnp_models:
        pnp_model.save_results(data_det_path, do_clear=True)


if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument(
        '--dataset', required=True, type=str, choices=['tum_rgbd', 'kitti'])
    parser.add_argument('--sequence', type=str, required=True)
    parser.add_argument('--viz_plt', action='store_true')
    parser.add_argument('--viz_o3d', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--save_plt', action='store_true')

    args = parser.parse_args()
    print(f"\n{args}\n")

    dataset = get_dataset(args.dataset, args.sequence)

    if args.dataset == 'tum_rgbd':
        nframes = 3
        min_dist_th = 2.5e-2

    elif args.dataset == 'kitti':
        nframes = 2
        min_dist_th = 0.0

    else:
        raise NotImplementedError('Dataset not supported.')

    pnp_models = get_eval_objects('epnp', 'epnpu')

    # detectors to evaluate.
    detector_names = ('superpoint', 'r2d2', 'd2net', 'keynet',)

    ranscac_f_cfg = {**configs.BASE_FRANSAC_CFG, **{}}
    ransac_p3p_cfg = configs.BASE_USAC_P3P

    # features will be normalized, so, calibrate thresholds (in-place).
    configs.calibrate_thresholds(dataset.K, ranscac_f_cfg, ransac_p3p_cfg)

    for det_name in detector_names:
        detector = load_detector(det_name)

        main(
            dataset,
            detector,
            pnp_models,
            ranscac_f_cfg,
            ransac_p3p_cfg,
            nframes=nframes,
            min_dist_th=min_dist_th,
            verbose=args.verbose,
            do_viz_plt=args.viz_plt,
            do_viz_o3d=args.viz_o3d,
            save_plt=args.save_plt)
