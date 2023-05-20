""" Data structures for the creation of feature-tracks using a greedy strategy
"""

import numpy as np
from numba import njit
from numba.typed import List

from .triangulation import (
    DLT_normalized_nview_point_triangulation,
    triangulation_nview_1point_LM_optimization)


class Track:

    id_prev = 0

    def __init__(self, im_id, kp):
        self.images = {im_id}
        self.kps = [kp]  # each of shape (2,)

        self.id = self.id_prev
        Track.id_prev += 1

        # triangulated coordinates. all as (3,) arrays
        self.X = None  # Triangulated 3D point without uncertainty.
        self.X_cov = None  # Triangulated 3D point with 2d covariances.
        self.X_s = None  # Triangulated 3D point with kp scores.

        # estimated 3d covariances.
        self.sigma3d_cov = None  # 3D point covariance based on 2d covs.
        self.sigma3d_s = None  # 3D point covariance based on scores.

    @property
    def points2d(self):
        """ Return 2d image coordinates as (2, n), ordered based on self.kps
        """
        return np.ascontiguousarray(
            np.array([kp.coord for kp in self.kps]).T
        )

    @property
    def sigmas2d(self):
        """ Return 2d covariances as (n 2, 2), ordered based on self.kps
        """
        return np.array([kp.cov for kp in self.kps])

    @property
    def scores(self):
        """ Return scalar keypoint-scores as (n,), ordered based on self.kps
        """
        return np.array([kp.cov_s for kp in self.kps])


class Keypoint:

    def __init__(self, coord, cov, cov_s, im_id, kp_id):
        self.coord = coord
        self.cov = cov
        self.cov_s = cov_s
        self.id = kp_id
        self.im_id = im_id
        self.track = Track(im_id, self)


class Image:

    def __init__(self, K, Rcw, tcw):
        self.R = Rcw
        self.t = tcw
        self.K = K
        self.P = K @ np.concatenate((Rcw, tcw), axis=1)


class MatchingGraph:

    def __init__(self):
        # use dicts for fast lookups due to hashing of keypoints and images
        # indexes.

        # {(im_idx, kp_idx): Keypoint instance}
        self.kps = {}

        # {(im1_idx, kp1_idx, im2_idx, kp2_idx): (
        # Keypoint instance1, Keypoint instance2, similarity)}
        self.matches = {}

        # (int: {'images': set of im indexes, 'kps': list of Keypoint instances})
        self.tracks = None

        # {im_id: Image}
        self.images = {}

    def add_image(self, K, R, t, im_id):
        assert im_id not in self.images
        self.images.update({im_id: Image(K, R, t)})

    def add_keypoints_and_tentative_matches(
            self,
            im1_idx: int,
            im2_idx: int,
            kp1s_idx: np.ndarray,  # (nmatches,)
            kp2s_idx: np.ndarray,  # (nmatches,)
            kp1s_coord: np.ndarray,  # (2, nmatches)
            kp2s_coord: np.ndarray,  # (2, nmatches)
            kp1s_covs: np.ndarray,  # (nmatches, 2, 2)
            kp2s_covs: np.ndarray,  # (nmatches, 2, 2)
            kp1s_covs_s: np.ndarray,  # (nmatches, 2, 2)
            kp2s_covs_s: np.ndarray,  # (nmatches, 2, 2)
            similarities: np.ndarray,  # (nmatches,)
    ):

        nmatches = len(similarities)

        # Add keypoint nodes only if they have not been previously added.
        new_kps1 = {
            (im1_idx, kp1s_idx[i]):
                Keypoint(kp1s_coord[:, i], kp1s_covs[i],
                         kp1s_covs_s[i], im1_idx, kp1s_idx[i])
            for i in range(nmatches)
            if (im1_idx, kp1s_idx[i]) not in self.kps
        }
        self.kps.update(new_kps1)

        # The same for the keypoints corresponding to the other image.
        new_kps2 = {
            (im2_idx, kp2s_idx[i]):
                Keypoint(kp2s_coord[:, i], kp2s_covs[i],
                         kp2s_covs_s[i], im2_idx, kp2s_idx[i])
            for i in range(nmatches)
            if (im2_idx, kp2s_idx[i]) not in self.kps
        }
        self.kps.update(new_kps2)

        # Add match edges between keypoint nodes.
        # Just in case, check if they have already been added.
        if im1_idx < im2_idx:
            # the previous and next conditions are to create the hashable tuple
            # with fixed ordering. It acts as the key.
            new_matches = {
                (im1_idx, kp1s_idx[i], im2_idx, kp2s_idx[i]):
                    (self.kps[(im1_idx, kp1s_idx[i])],
                     self.kps[(im2_idx, kp2s_idx[i])],
                     similarities[i])

                    for i in range(nmatches)
                    if (im1_idx, kp1s_idx[i], im2_idx, kp2s_idx[i]) not in self.matches
            }

        elif im1_idx > im2_idx:
            new_matches = {
                (im2_idx, kp2s_idx[i], im1_idx, kp1s_idx[i]):
                    (self.kps[(im2_idx, kp2s_idx[i])],
                     self.kps[(im1_idx, kp1s_idx[i])],
                     similarities[i])

                    for i in range(nmatches)
                    if (im2_idx, kp2s_idx[i], im1_idx, kp1s_idx[i]) not in self.matches
            }
        else:
            # matches can't come from the same image.
            raise ValueError

        self.matches.update(new_matches)

    def generate_tracks(self):
        """ greedy track generation based on matching similarities."""
        # sort matches by similarity value. This is the greedy strategy.
        self.matches = {
            k: v for k, v in sorted(
                self.matches.items(), key=lambda item: item[1][2])
        }

        # iterate over matches to form feature-tracks.
        for kp1, kp2, _ in self.matches.values():

            # if the sets of images for each feature track are disjoint. then
            # they can be merged.
            if kp1.track.images.isdisjoint(kp2.track.images):
                # modify set of images and list of keypoints.
                kp1.track.kps.extend(kp2.track.kps)
                kp1.track.images.update(kp2.track.images)

                # merge by pointing to the same track.
                kp2.track = kp1.track

        # final tracks. By using the track's id, we avoid repeated tracks.
        # Avoid also tracks only seen in one image (unmatched)
        self.tracks = {
            kp.track.id: kp.track for kp in self.kps.values()
            if len(kp.track.images) > 1
        }

        # From now on tracks don't need to be accessed by keys nor we don't
        # need to verify if a certain track exists. We will only be iterating
        # over them. Hence turn it from dict to list.
        self.tracks = list(self.tracks.values())

    def triangulate_tracks(self):
        assert self.tracks is not None

        # preallocate arrays for speed reasons.
        max_ims = max(len(track.images) for track in self.tracks)
        P = np.empty((max_ims, 3, 4))
        K = np.empty((max_ims, 3, 3))
        R = np.empty((max_ims, 3, 3))
        t = np.empty((max_ims, 3, 1))

        max_kps = max(len(track.kps) for track in self.tracks)
        p = np.empty((2, max_kps))
        sigmas2d = np.empty((max_kps, 2, 2))

        for track in self.tracks:
            n_kps = len(track.kps)
            n_im = len(track.images)

            # Get 3D and 2d data.
            # don't use track.images since sets don't have order.
            for i, kp in enumerate(track.kps):
                P[i] = self.images[kp.im_id].P
                K[i] = self.images[kp.im_id].K
                R[i] = self.images[kp.im_id].R
                t[i] = self.images[kp.im_id].t

                p[:, i] = kp.coord
                sigmas2d[i] = kp.cov

            # DLT triangulation from n-views and with normalization.
            X = DLT_normalized_nview_point_triangulation(
                np.ascontiguousarray(p[:, :n_kps]),
                P[:n_im],
                (480, 640)
            )

            # Levenberg-Marquardt optimization.
            X, sigma3d = triangulation_nview_1point_LM_optimization(
                X, p[:, :n_kps], K[0], R[:n_im], t[:n_im], sigmas2d[:n_kps], 5)

            track.X = X
            track.sigma3d = sigma3d

    def triangulate_tracks_nb(self):
        assert self.tracks is not None

        # 1) convert variables of interest to either numba or numpy types.

        # transformations.
        im_vals = self.images.values()
        P = np.array([im.P for im in im_vals])
        K = np.array([im.K for im in im_vals])
        R = np.array([im.R for im in im_vals])
        t = np.array([im.t for im in im_vals])

        # image-indexes per track.
        ims_list = List([
            List([kp.im_id for kp in track.kps])
            for track in self.tracks])

        # 2d-kp coordinates per track.
        p_list = List([track.points2d for track in self.tracks])

        # covariances of interest per track.
        sigmas2d_list = List([track.sigmas2d for track in self.tracks])
        scores_list = List([track.scores for track in self.tracks])

        # 2) triangulate.
        X, X_cov, X_s, sigmas3d_cov, sigmas3d_s = triangulate_tracks_v2(
            P, K, R, t, ims_list, p_list, sigmas2d_list, scores_list
        )

        # 3) update graph tracks.
        for i, track in enumerate(self.tracks):
            track.X = X[:, i]
            track.X_cov = X_cov[:, i]
            track.X_s = X_s[:, i]

            track.sigma3d_cov = sigmas3d_cov[i]
            track.sigma3d_s = sigmas3d_s[i]

    @property
    def points3d(self):
        return np.array([track.X for track in self.tracks]).T


@njit('i8(ListType(ListType(i8)))', fastmath=True, cache=True)
def maxlen_list_of_lists(list_):
    m = 0
    for li in list_:
        mt = len(li)
        if mt > m:
            m = mt
    return m


@njit('Tuple((f8[:,::1], f8[:,::1], f8[:,::1], f8[:,:,::1], f8[:,:,::1]))'
      '(f8[:,:,::1], f8[:,:,::1], f8[:,:,::1], '
      'f8[:,:,::1], ListType(ListType(int64)), ListType(f8[:,::1]), '
      'ListType(f8[:,:,::1]), ListType(f8[:,:,::1]))', cache=True)
def triangulate_tracks_v2(
        Ps: List,
        Ks: List,
        Rs: List,
        ts: List,
        ims_lists: List,
        p_list: List,
        sigmas2d_lists: List,
        scores_lists: List
):
    """ Triangulate each feature-track.

    Args:
        Ps: typed-list of (n_kps, 3, 4) of projection matrices per track.
        Ks: typed-list of (n_kps, 3, 3) of calib matrices per track.
        Rs: typed-list of (n_kps, 3, 3) of rotation matrices per track.
        ts: typed-list of (n_kps, 3, 1) of translation vectors per track.
        ims_lists: typed-list of typed-lists containing the indexes to the
            images used.
        p_list: typed-list of (2, n_kps) features-coordinates per track.
        sigmas2d: typed-list of (n_kps, 2, 2) feature-covariances per track.
        scores_lists: typed-list of (n_kps, 2, 2) feature-covariances per
                        track, based on the keypoint scores.

    Return:
        X_wo_unc: (3, n_tracks) triangulated coordinates w/o taking uncertainty
                    into account.
        X_w_covs: (3, n_tracks) triangulated coordinates when using 2d
                    covariances for the local features as their uncertainties.
        X_w_scor: (3, n_tracks) triangulated coordinates when using the
                    inverse of the local feature scores as their uncertainties.
        sigmas3d_covs: (n_tracks, 3, 3) 3d-point covariance when using 2d
                    covariances for the local features as their uncertainties.
        sigmas3d_covs: (n_tracks, 3, 3) 3d-point covariance when using the
                    inverse of the local feature scores as their uncertainties.
    """
    ntracks = len(ims_lists)

    # preallocate memory for per-track variables.
    n_kps_max = maxlen_list_of_lists(ims_lists)
    P = np.empty((n_kps_max, 3, 4))
    K = np.empty((n_kps_max, 3, 3))
    R = np.empty((n_kps_max, 3, 3))
    t = np.empty((n_kps_max, 3, 1))

    # initialize outputs.
    # Triangulated coordinates with, and without uncertainty.
    X_wo_unc = np.empty((3, ntracks))
    X_w_covs = np.empty((3, ntracks))
    X_w_scor = np.empty((3, ntracks))

    # output 3d covariances.
    sigmas3d_covs = np.empty((ntracks, 3, 3))
    sigmas3d_scor = np.empty((ntracks, 3, 3))

    for i in range(ntracks):
        # kp coordinates for each track (2, n).
        p = p_list[i]
        n_kps = p.shape[1]
        assert n_kps == len(ims_lists[i])

        # Besides None. Types of uncertainty used.
        sigmas2d = sigmas2d_lists[i]
        scores2d_inv = scores_lists[i]

        # per-feature 3d transformations.
        for j, k in enumerate(ims_lists[i]):
            P[j] = Ps[k]
            K[j] = Ks[k]
            R[j] = Rs[k]
            t[j] = ts[k]

        # DLT triangulation from n views.
        Xi = DLT_normalized_nview_point_triangulation(
            p, P[:n_kps], (None, None))

        # Levenberg-Marquardt optimization.
        # 1) whithout uncertainty.
        X_wo_unc[:, i:i + 1], _ = triangulation_nview_1point_LM_optimization(
            Xi, p, K[0], R[:n_kps], t[:n_kps], None, 5)

        # 2) with 2d covariance uncertainties.
        X_w_covs[:, i:i + 1], sigmas3d_covs[i] = triangulation_nview_1point_LM_optimization(
            Xi, p, K[0], R[:n_kps], t[:n_kps], sigmas2d, 5)

        # 3) with covariance based on scores.
        X_w_scor[:, i:i + 1], sigmas3d_scor[i] = triangulation_nview_1point_LM_optimization(
            Xi, p, K[0], R[:n_kps], t[:n_kps], scores2d_inv, 5)

    return X_wo_unc, X_w_covs, X_w_scor, sigmas3d_covs, sigmas3d_scor
