import sys
from pathlib import Path
from typing import Dict, Union, Optional
from math import sqrt, ceil

import cv2
import numpy as np
import torch
import torch.nn.functional as F

env_path = str(Path(__file__).parents[2])
if env_path not in sys.path:
    print(f'inserting {env_path} to sys.path')
    sys.path.insert(0, env_path)
from detectors.base_detector import BaseDetector
from detectors.keynet.model.network import KeyNet
from detectors.keynet.model.modules import NonMaxSuppression
from detectors.keynet.model.HyNet.hynet_model import HyNet
from detectors.keynet.model.kornia_tools.utils import (
    custom_pyrdown, laf_from_center_scale_ori as to_laf,
    extract_patches_from_pyramid as extract_patch)
from detectors.structure_tensor import (
    structure_tensor_matrices_at_points as struc_at_kp)


class Keynet(BaseDetector):
    default_cfg_matching: Dict[str, Union[str, Optional[float]]] = {
        'distance': 'cosine',
        'thr': None
    }

    default_cfg = {
        # KeyNet model
        'num_filters': 8,
        'num_levels': 3,
        'kernel_size': 5,
        # trained weights
        'weights_detector': 'model/weights/keynet_pytorch.pth',
        'weights_descriptor': 'model/HyNet/weights/HyNet_LIB.pth',
        # extraction parameters
        'nms_size': 15,
        'pyramid_levels': 4,
        'up_levels': 1,
        'scale_factor_levels': sqrt(2),
        's_mult': 22,
        # extra
        'nms_thr': 1.124,
        'batch_size_dsc': 100,
        'num_keypoints': 5000,
        'return_heatmaps': False
    }

    def _init(self, cfg):
        # structure tensor params
        self.cfg_acorr = cfg['cfg_acorr']
        # KeyNet model variables
        self.num_filters = cfg['num_filters']
        self.num_levels = cfg['num_levels']
        self.kernel_size = cfg['kernel_size']
        # trained weights paths
        self.weights_detector = str(
            Path(__file__).parent / cfg['weights_detector'])
        self.weights_descriptor = str(
            Path(__file__).parent / cfg['weights_descriptor'])
        self.nms_size = cfg['nms_size']
        # extraction parameters
        self.pyramid_levels = cfg['pyramid_levels']
        self.up_levels = cfg['up_levels']
        self.scale_factor_levels = cfg['scale_factor_levels']
        self.s_mult = cfg['s_mult']
        # some extra variables
        self.nms_thr = cfg['nms_thr']
        self.batch_size_dsc = cfg['batch_size_dsc']
        self.num_keypoints = cfg['num_keypoints']
        self.return_heatmaps = cfg['return_heatmaps']

        # use cuda or cpu
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        # load detector system
        keynet_model = KeyNet(
            self.num_filters,
            self.num_levels,
            self.kernel_size
        )
        checkpoint = torch.load(self.weights_detector)
        keynet_model.load_state_dict(checkpoint['state_dict'])
        keynet_model = keynet_model.to(self.device)
        keynet_model.eval()
        self.keynet_model = keynet_model

        # load descriptor model
        desc_model = HyNet()
        checkpoint = torch.load(self.weights_descriptor)
        desc_model.load_state_dict(checkpoint)
        desc_model = desc_model.to(self.device)
        desc_model.eval()
        self.desc_model = desc_model

        # nms module
        self.nms = NonMaxSuppression(nms_size=self.nms_size, thr=self.nms_thr)

    def run(self, im_np: np.array):
        """ 
        Obtain keypoints, descriptors and estimated cov matrices on image im

        args:
            im_np: np.array, containing:
                RGB image of size (H,W,3),
                or gray image of size (H,W) or (H,W,3) (third dimension repeated)
        returns:
            - kps -> array (3,n) with u, v (im coords) and scores as dimensions
            - desc -> array (n, d) "d"-dim descriptors at each of "n" detected kps
            - C -> array (n,2,2) estimated **inverse** covariances of each kp
        """
        # models
        keynet_model = self.keynet_model
        desc_model = self.desc_model
        # extraction configuration
        pyramid_levels = self.pyramid_levels
        up_levels = self.up_levels
        scale_factor_levels = self.scale_factor_levels
        s_mult = self.s_mult
        num_points = self.num_keypoints
        device = self.device
        batch_size_desc = self.batch_size_dsc
        nms = self.nms
        # return heatmaps or not
        return_heatmaps = self.return_heatmaps

        # Compute points per level
        point_level = []
        tmp = 0.0
        factor_points = (scale_factor_levels ** 2)
        levels = pyramid_levels + up_levels + 1
        for idx_level in range(levels):
            tmp += factor_points ** (-1 * (idx_level - up_levels))
            point_level.append(num_points * factor_points **
                               (-1 * (idx_level - up_levels)))
        point_level = np.asarray(
            list(map(lambda x: int(x / tmp), point_level)))

        # convert image to gray if needed
        if len(im_np.shape) == 3:
            im_np = cv2.cvtColor(im_np, cv2.COLOR_RGB2GRAY)
        # normalize it
        im_np = (im_np / 255).astype(np.float32)
        # convert to tensor
        im = torch.from_numpy(im_np).unsqueeze(0).unsqueeze(0)
        im = im.to(device)

        if up_levels:
            im_up = torch.from_numpy(im_np).unsqueeze(0).unsqueeze(0)
            im_up = im_up.to(device)

        src_kp = []
        _, _, h, w = im.shape
        # Extract features from the upper levels
        for idx_level in range(up_levels):

            num_points_level = point_level[len(
                point_level) - pyramid_levels - 1 - (idx_level + 1)]

            # Resize input image
            up_factor = scale_factor_levels ** (1 + idx_level)
            nh, nw = int(h * up_factor), int(w * up_factor)
            up_factor_kpts = (w / nw, h / nh)
            im_up = F.interpolate(
                im_up, (nh, nw), mode='bilinear', align_corners=False)

            src_kp_i, src_dsc_i, im_up, Ci = extract_ms_feats(
                keynet_model, desc_model, im_up, up_factor_kpts,
                s_mult=s_mult, device=device, num_kpts_i=num_points_level,
                nms=nms, down_level=idx_level + 1, up_level=True, im_size=[w, h],
                batch_size_desc=batch_size_desc)

            # this line adds the scale to the kp:
            # src_kp_i = np.asarray(list(map(lambda x: [x[0], x[1], (1 / scale_factor_levels) ** (1 + idx_level), x[2]], src_kp_i)))

            if src_kp == []:
                src_kp = src_kp_i
                src_dsc = src_dsc_i
                src_C = Ci
            else:
                src_kp = np.concatenate([src_kp, src_kp_i], axis=0)
                src_dsc = np.concatenate([src_dsc, src_dsc_i], axis=0)
                src_C = np.concatenate([src_C, Ci], axis=0)

        # Extract features from the downsampling pyramid
        for idx_level in range(pyramid_levels + 1):

            num_points_level = point_level[idx_level]
            if idx_level > 0 or up_levels:
                res_points = int(np.asarray([point_level[a] for a in range(
                    0, idx_level + 1 + up_levels)]).sum() - len(src_kp))
                num_points_level = res_points

            if return_heatmaps and (idx_level == 0):
                src_kp_i, src_dsc_i, im, Ci, heatmap = extract_ms_feats(
                    keynet_model, desc_model, im, scale_factor_levels, s_mult=s_mult,
                    device=device, num_kpts_i=num_points_level, nms=nms,
                    down_level=idx_level, im_size=[w, h],
                    batch_size_desc=batch_size_desc,
                    return_heatmaps=return_heatmaps
                )
                kps_heatmap = src_kp_i[:, 1::-1].T
            else:
                src_kp_i, src_dsc_i, im, Ci = extract_ms_feats(
                    keynet_model, desc_model, im, scale_factor_levels, s_mult=s_mult,
                    device=device, num_kpts_i=num_points_level, nms=nms,
                    down_level=idx_level, im_size=[w, h],
                    batch_size_desc=batch_size_desc
                )

            # this line adds the scale to the kp:
            # src_kp_i = np.asarray(list(map(lambda x: [x[0], x[1], scale_factor_levels ** idx_level, x[2]], src_kp_i)))

            if src_kp == []:
                src_kp = src_kp_i
                src_dsc = src_dsc_i
                src_C = Ci
            else:
                src_kp = np.concatenate([src_kp, src_kp_i], axis=0)
                src_dsc = np.concatenate([src_dsc, src_dsc_i], axis=0)
                src_C = np.concatenate([src_C, Ci], axis=0)

        if return_heatmaps:
            return src_kp.T, src_dsc, src_C, heatmap, kps_heatmap
        else:
            return src_kp.T, src_dsc, src_C


def extract_ms_feats(
        keynet_model, desc_model, image, factor, s_mult, device,
        num_kpts_i=1000, nms=None, down_level=0, up_level=False, im_size=[],
        batch_size_desc=1000, return_heatmaps=False
):
    '''
    Extracts the features for a specific scale level from the pyramid
    :param keynet_model: Key.Net model
    :param desc_model: HyNet model
    :param image: image as a PyTorch tensor
    :param factor: rescaling pyramid factor
    :param s_mult: Descriptor area multiplier
    :param device: GPU or CPU
    :param num_kpts_i: number of desired keypoints in the level
    :param nms: nums size
    :param down_level: Indicates if images needs to go down one pyramid level
    :param up_level: Indicates if image is an upper scale level
    :param im_size: Original image size
    :param batch_size_desc: Max number of patches per descriptor model call
    :return: It returns the local features for a specific image level
    '''

    if down_level and not up_level:
        image = custom_pyrdown(image, factor=factor)
        _, _, nh, nw = image.shape
        factor = (im_size[0] / nw, im_size[1] / nh)
    elif not up_level:
        factor = (1., 1.)

    # score map
    with torch.no_grad():
        det_map = keynet_model(image)
    # get numpy version for extracting the covariances:
    det_map_np = det_map[0, 0].detach().cpu().numpy()
    det_map = remove_borders(det_map, borders=15)

    # src_kps:
    kps = nms(det_map)
    c = det_map[0, 0, kps[0], kps[1]]
    sc, indices = torch.sort(c, descending=True)
    indices = indices[torch.where(sc > 0.)]
    kps = kps[:, indices[:num_kpts_i]]
    kps_np = torch.cat([kps[1].view(-1, 1).float(), kps[0].view(-1, 1).float(), c[indices[:num_kpts_i]].view(-1, 1).float()],
                       dim=1).detach().cpu().numpy()
    num_kpts = len(kps_np)
    kp = torch.cat([kps[1].view(-1, 1).float(),
                   kps[0].view(-1, 1).float()], dim=1).unsqueeze(0).cpu()
    s = s_mult * torch.ones((1, num_kpts, 1, 1))
    src_laf = to_laf(kp, s, torch.zeros((1, num_kpts, 1)))

    # HyNet takes images on the range [0, 255]
    patches = extract_patch(255 * image.cpu(), src_laf,
                            PS=32, normalize_lafs_before_extraction=True)[0]

    if len(patches) > batch_size_desc:
        for i_patches in range(ceil(len(patches) / batch_size_desc)):
            if i_patches == 0:
                sel_patches = patches[:batch_size_desc]
                if len(sel_patches) > 0:
                    descs = desc_model(sel_patches.to(device))
                else:
                    descs = np.empty((0, 128))
            else:
                sel_patches = patches[batch_size_desc
                                      * i_patches:batch_size_desc * (i_patches + 1)]
                if len(sel_patches) > 0:
                    descs_tmp = desc_model(sel_patches.to(device))
                else:
                    descs_tmp = np.empty((0, 128))
                descs = torch.cat([descs, descs_tmp], dim=0)
        descs = descs.cpu().detach().numpy()
    else:
        if len(patches) > 0:
            descs = desc_model(patches.to(device)).cpu().detach().numpy()
        else:
            descs = np.empty((0, 128))

    # get inverse of covariance estimates:
    C = struc_at_kp(det_map_np, kps_np.T[1::-1].astype(int))

    kps_np[:, 0] *= factor[0]
    kps_np[:, 1] *= factor[1]

    if return_heatmaps and (down_level == 0) and (not up_level):
        return kps_np, descs, image.to(device), C, det_map_np
    else:
        return kps_np, descs, image.to(device), C


def remove_borders(score_map, borders):
    '''
    It removes the borders of the image to avoid detections on the corners
    '''
    shape = score_map.shape
    mask = torch.ones_like(score_map)

    mask[:, :, 0:borders, :] = 0
    mask[:, :, :, 0:borders] = 0
    mask[:, :, shape[2] - borders:shape[2], :] = 0
    mask[:, :, :, shape[3] - borders:shape[3]] = 0

    return mask * score_map
