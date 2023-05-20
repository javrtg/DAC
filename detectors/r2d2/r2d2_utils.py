from typing import List, Dict, Union, Optional
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

env_path = str(Path(__file__).parents[2])
if env_path not in sys.path:
    print(f'inserting {env_path} to sys.path')
    sys.path.insert(0, env_path)
from detectors.base_detector import BaseDetector
from detectors.r2d2.tools import common
from detectors.r2d2.tools.dataloader import norm_RGB
from detectors.r2d2.nets.patchnet import *
from detectors.structure_tensor import (
    structure_tensor_matrices_at_points as struc_at_kp)


class NonMaxSuppression(torch.nn.Module):

    def __init__(self, rel_thr=0.7, rep_thr=0.7):
        nn.Module.__init__(self)
        self.max_filter = torch.nn.MaxPool2d(
            kernel_size=3, stride=1, padding=1)
        self.rel_thr = rel_thr
        self.rep_thr = rep_thr

    def forward(self, reliability, repeatability, **kw):
        assert len(reliability) == len(repeatability) == 1
        reliability, repeatability = reliability[0], repeatability[0]
        # local maxima
        maxima = (repeatability == self.max_filter(repeatability))
        # remove low peaks
        maxima *= (repeatability >= self.rep_thr)
        maxima *= (reliability >= self.rel_thr)
        return maxima.nonzero().t()[2:4]  # (2,n)


def load_network(model_fn):
    checkpoint = torch.load(model_fn)
    # print("\n>> Creating net = " + checkpoint['net'])
    net = eval(checkpoint['net'])
    # nb_of_weights = common.model_size(net)
    # print(f" ( Model size: {nb_of_weights/1000:.0f}K parameters )")

    # initialization
    weights = checkpoint['state_dict']
    net.load_state_dict(
        {k.replace('module.', ''): v for k, v in weights.items()})
    return net.eval()


class R2D2(BaseDetector):
    default_cfg_matching: Dict[str, Union[str, Optional[float]]] = {
        'distance': 'cosine',
        'thr': None
    }

    default_cfg: Dict[str, Union[str, int, float, List[int], bool]] = {
        'model_path': 'models/r2d2_WASF_N16.pt',
        'top_k': 5000,
        'scale_f': 2**0.25,
        'min_size': 256,
        'max_size': 1024,
        'min_scale': 0.,
        'max_scale': 1.,
        'reliability_thr': 0.7,
        'repeatability_thr': 0.7,
        'gpu': [0],
        'return_heatmaps': False
    }

    def _init(self, cfg):
        # structure tensor params
        self.cfg_acorr = cfg['cfg_acorr']
        # path to weights
        self.model_path = str(Path(__file__).parent / cfg['model_path'])
        # max number of keypoints that will be return
        self.top_k = cfg['top_k']
        # scaling factor between images in the pyramid
        self.scale_f = cfg['scale_f']
        # extreme sizes and scales that will be used by the system
        self.min_size = cfg['min_size']
        self.max_size = cfg['max_size']
        self.min_scale = cfg['min_scale']
        self.max_scale = cfg['max_scale']
        # device
        self.iscuda = common.torch_set_gpu(cfg['gpu'])
        # return the heatmap at scale 1 or not
        self.return_heatmaps = cfg['return_heatmaps']
        # nms module for detecting keypoints
        self.detector = NonMaxSuppression(
            rel_thr=cfg['reliability_thr'], rep_thr=cfg['repeatability_thr'])
        # load network
        net = load_network(self.model_path)
        if self.iscuda:
            net.cuda()
        self.net = net

    def run(self, img: np.ndarray):
        """
        Obtain keypoints, descriptors and estimated cov matrices on image im

        args:
            im_np: np.ndarray, containing:
                RGB image of size (H,W,3),
                or gray image of size (H,W) or (H,W,3) (third dimension repeated)
        returns:
            - kps -> array (3,n) with u, v (im coords) and scores as dimensions
            - desc -> array (n, d) "d"-dim descriptors at each of "n" detected kps
            - C -> array (n,2,2) estimated **inverse** covariances of each kp
        """
        # convert to rgb format if needed
        if len(img.shape) == 2:
            img = np.repeat(img[:, :, None], 3, axis=2)
        assert len(img.shape) == 3 == img.shape[-1], (
            f"image has shape {img.shape} but RGB format (shape=(H,W,3)) was expected.")

        # preprocessing
        W, H = img.shape[:-1]
        img = norm_RGB(img)[None]
        if self.iscuda:
            img = img.cuda()

        if self.return_heatmaps:
            # extract kps, desc, scores, covs and hm info:
            kps, desc, C, heatmaps, kps_heatmap = self.extract_multiscale(img)
            return kps, desc, C, heatmaps, kps_heatmap
        else:
            # extract kps, desc, scores and covs
            kps, desc, C = self.extract_multiscale(img)
            return kps, desc, C

    def extract_multiscale(self, img):
        # unpack extractor variables
        net = self.net
        detector = self.detector
        scale_f = self.scale_f
        min_scale = self.min_scale
        max_scale = self.max_scale
        min_size = self.min_size
        max_size = self.max_size
        top_k = self.top_k
        return_heatmaps = self.return_heatmaps

        # idk why r2d2 does this next. It is noted that it's for speedup
        # but it is also set to False which I find it to be contradictory(?)
        old_bm = torch.backends.cudnn.benchmark
        torch.backends.cudnn.benchmark = False  # speedup

        # extract keypoints at multiple scales
        B, three, H, W = img.shape
        assert B == 1 and three == 3, "should be a batch with a single RGB image"

        assert max_scale <= 1
        s = 1.0  # current scale factor

        # Note: I deleted the register for the scale since it is not used for matching
        X, Y, C, Q, D, AC = [], [], [], [], [], np.zeros(
            (0, 2, 2), dtype=np.float32)
        while s + 0.001 >= max(min_scale, min_size / max(H, W)):
            if s - 0.001 <= min(max_scale, max_size / max(H, W)):
                nh, nw = img.shape[2:]
                # extract descriptors
                with torch.no_grad():
                    res = net(imgs=[img])

                # get output and reliability map
                descriptors = res['descriptors'][0]
                reliability = res['reliability'][0]
                repeatability = res['repeatability'][0]

                # normalize the reliability for nms
                # extract maxima and descs
                y, x = detector(**res)  # nms
                c = reliability[0, 0, y, x]
                q = repeatability[0, 0, y, x]
                d = descriptors[0, :, y, x].t()  # (n_kp, dim)

                # structure tensors at x,y:
                repeatability_np = repeatability[0, 0].cpu().numpy()
                yx_np = torch.stack([y, x], dim=0).cpu().numpy()
                ac = struc_at_kp(repeatability_np, yx_np,
                                 ** self.cfg_acorr)  # (n_kp,2,2)

                # accumulate multiple scales
                X.append(x.float() * W / nw)
                Y.append(y.float() * H / nh)
                C.append(c)
                Q.append(q)
                D.append(d)

                # structure tensors
                AC = np.concatenate((AC, ac), axis=0)
                # store heatmaps if specified
                if (s == 1) and return_heatmaps:
                    heatmaps = np.concatenate(
                        (repeatability_np[None], reliability[0, 0].cpu().numpy()[None]), axis=0)
                    kps_heatmap = torch.stack([y, x], dim=0).cpu().numpy()
            s /= scale_f

            # down-scale the image for next iteration
            nh, nw = round(H * s), round(W * s)
            img = F.interpolate(
                img, (nh, nw), mode='bilinear', align_corners=False)

        # restore value
        torch.backends.cudnn.benchmark = old_bm

        Y = torch.cat(Y)  # (n,)
        X = torch.cat(X)  # (n,)
        # scores = reliability * repeatability
        scores = torch.cat(C) * torch.cat(Q)
        XYS = torch.stack([X, Y, scores], dim=-1)  # (n,3)
        D = torch.cat(D)  # (n,dim)

        # to cpu numpy
        XYS = XYS.cpu().numpy()
        D = D.cpu().numpy()

        # get data only of top-k features
        idxs = XYS[:, -1].argsort()[-top_k or None:]
        XYS = XYS[idxs]
        D = D[idxs]
        AC = AC[idxs]

        if return_heatmaps:
            # check what kps of scale 1 remain after the purge of the top-k:
            keep = np.in1d(kps_heatmap[0] + kps_heatmap[1]
                           * 1j, XYS[:, 1] + XYS[:, 0] * 1j)
            kps_heatmap = kps_heatmap[:, keep]
            return XYS.T, D, AC, heatmaps, kps_heatmap
        else:
            return XYS.T, D, AC
