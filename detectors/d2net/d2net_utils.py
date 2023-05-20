import sys
from pathlib import Path
from typing import Dict, Union, Optional

import numpy as np
import cv2
import torch

env_path = str(Path(__file__).parents[2])
if env_path not in sys.path:
    print(f'inserting {env_path} to sys.path')
    sys.path.insert(0, env_path)
from detectors.base_detector import BaseDetector
from .lib.model_test import D2Net
from .lib.utils import preprocess_image
from .lib.pyramid import (
    process_multiscale, process_multiscale_acorr, process_multiscale_covs
)


class D2NET(BaseDetector):
    default_cfg_matching: Dict[str, Union[str, Optional[float]]] = {
        'distance': 'cosine',
        'thr': None
    }

    default_cfg = {
        'model_file': "models/d2_tf.pth",
        'use_relu': True,
        'use_cuda': True,
        'max_edge': 1600,
        'max_sum_edges': 2800,
        'preprocessing': 'caffe',
        'multiscale': False,
        'return_heatmaps': False
    }

    def _init(self, cfg):
        use_cuda = torch.cuda.is_available()
        self.model = D2Net(
            model_file=str(Path(__file__).parent / cfg['model_file']),
            use_relu=cfg['use_relu'],
            use_cuda=use_cuda
        )
        self.use_cuda = use_cuda
        self.max_edge = cfg['max_edge']
        self.max_sum_edges = cfg['max_sum_edges']
        self.preprocessing = cfg['preprocessing']
        self.multiscale = cfg['multiscale']
        self.cfg_acorr = cfg['cfg_acorr']
        self.return_heatmaps = cfg['return_heatmaps']

    def run(self, image):
        model = self.model
        use_cuda = self.use_cuda
        max_edge = self.max_edge
        max_sum_edges = self.max_sum_edges
        preprocessing = self.preprocessing
        multiscale = self.multiscale
        cfg_acorr = self.cfg_acorr
        return_heatmaps = self.return_heatmaps

        device = torch.device("cuda:0" if use_cuda else "cpu")

        if len(image.shape) == 2:
            image = image[:, :, np.newaxis]
            image = np.repeat(image, 3, -1)

        resized_image = image
        if max(resized_image.shape) > max_edge:
            factor = max_edge / max(resized_image.shape)
            resized_image = cv2.resize(resized_image,
                                       dsize=None, fx=factor, fy=factor).astype('float')
        if sum(resized_image.shape[: 2]) > max_sum_edges:
            factor = max_sum_edges / sum(resized_image.shape[: 2])
            resized_image = cv2.resize(resized_image,
                                       dsize=None, fx=factor, fy=factor).astype('float')

        fact_i = image.shape[0] / resized_image.shape[0]
        fact_j = image.shape[1] / resized_image.shape[1]

        input_image = preprocess_image(
            resized_image,
            preprocessing=preprocessing
        )

        # D2-NET MS (multi-scale) o D2-NET SS (single-scale):
        scales = ([.5, 1, 2] if multiscale else [1])

        # Needed info to compute inverse covariances:
        with torch.no_grad():
            out = process_multiscale_covs(
                torch.tensor(
                    input_image[np.newaxis, :, :, :].astype(np.float32),
                    device=device
                ),
                model,
                cfg_acorr,
                scales=scales,
                return_heatmaps=return_heatmaps
            )

        if return_heatmaps:
            kps, desc, C, heatmaps, kps_fmap = out
            # adjust the keypoints' positions to the original raw image (before resizing):
            # Input image coordinates
            kps[:, 0] *= fact_i
            kps[:, 1] *= fact_j
            # i, j -> u, v
            kps = kps[:, [1, 0, 2]]
            return kps.T, desc, C, heatmaps, kps_fmap
        else:
            kps, desc, C = out
            # adjust the keypoints' positions to the original raw image (before resizing):
            # Input image coordinates
            kps[:, 0] *= fact_i
            kps[:, 1] *= fact_j
            # i, j -> u, v
            kps = kps[:, [1, 0, 2]]
            return kps.T, desc, C


class D2net_detector_acorr():

    def __init__(self,
                 model_file="detectors/d2net/models/d2_tf.pth",
                 use_relu=True,
                 use_cuda=True,
                 max_edge=1600, max_sum_edges=2800,
                 preprocessing='caffe',
                 multiscale=False
                 ):

        use_cuda = torch.cuda.is_available()

        self.model = D2Net(
            model_file=model_file,
            use_relu=use_relu,
            use_cuda=use_cuda
        )

        self.use_cuda = use_cuda
        self.max_edge = max_edge
        self.max_sum_edges = max_sum_edges
        self.preprocessing = preprocessing
        self.multiscale = multiscale

    def run(self, image):

        model = self.model
        use_cuda = self.use_cuda
        max_edge = self.max_edge
        max_sum_edges = self.max_sum_edges
        preprocessing = self.preprocessing
        multiscale = self.multiscale

        device = torch.device("cuda:0" if use_cuda else "cpu")

        if len(image.shape) == 2:
            image = image[:, :, np.newaxis]
            image = np.repeat(image, 3, -1)

        resized_image = image
        if max(resized_image.shape) > max_edge:
            factor = max_edge / max(resized_image.shape)
            resized_image = cv2.resize(resized_image,
                                       dsize=None, fx=factor, fy=factor).astype('float')
        if sum(resized_image.shape[: 2]) > max_sum_edges:
            factor = max_sum_edges / sum(resized_image.shape[: 2])
            resized_image = cv2.resize(resized_image,
                                       dsize=None, fx=factor, fy=factor).astype('float')

        fact_i = image.shape[0] / resized_image.shape[0]
        fact_j = image.shape[1] / resized_image.shape[1]

        input_image = preprocess_image(
            resized_image,
            preprocessing=preprocessing
        )

        # D2-NET MS (multi-scale) o D2-NET SS (single-scale):
        scales = ([.5, 1, 2] if multiscale else [1])

        # Needed info to compute inverse covs.:
        with torch.no_grad():
            acorr_dict = process_multiscale_acorr(
                torch.tensor(
                    input_image[np.newaxis, :, :, :].astype(np.float32),
                    device=device
                ),
                model,
                scales=scales
            )
        # adjust the keypoints' positions to the original raw image (before resizing):
        # for scale_key in acorr_dict.keys():
        acorr_dict['kp_im_pos'][:, 0] *= fact_i
        acorr_dict['kp_im_pos'][:, 1] *= fact_j
        # i, j -> u, v
        acorr_dict['kp_im_pos'] = acorr_dict['kp_im_pos'][:, [1, 0, 2, 3]]
        return acorr_dict


class D2net_detector_original():

    def __init__(self,
                 model_file="detectors/d2net/models/d2_tf.pth",
                 use_relu=True,
                 use_cuda=True,
                 max_edge=1600, max_sum_edges=2800,
                 preprocessing='caffe',
                 multiscale=False
                 ):

        use_cuda = torch.cuda.is_available()

        self.model = D2Net(
            model_file=model_file,
            use_relu=use_relu,
            use_cuda=use_cuda
        )

        self.use_cuda = use_cuda
        self.max_edge = max_edge
        self.max_sum_edges = max_sum_edges
        self.preprocessing = preprocessing
        self.multiscale = multiscale

    def run(self, image):

        model = self.model
        use_cuda = self.use_cuda
        max_edge = self.max_edge
        max_sum_edges = self.max_sum_edges
        preprocessing = self.preprocessing
        multiscale = self.multiscale

        device = torch.device("cuda:0" if use_cuda else "cpu")

        if len(image.shape) == 2:
            image = image[:, :, np.newaxis]
            image = np.repeat(image, 3, -1)

        resized_image = image
        if max(resized_image.shape) > max_edge:
            factor = max_edge / max(resized_image.shape)
            resized_image = cv2.resize(resized_image,
                                       dsize=None, fx=factor, fy=factor).astype('float')
        if sum(resized_image.shape[: 2]) > max_sum_edges:
            factor = max_sum_edges / sum(resized_image.shape[: 2])
            resized_image = cv2.resize(resized_image,
                                       dsize=None, fx=factor, fy=factor).astype('float')

        fact_i = image.shape[0] / resized_image.shape[0]
        fact_j = image.shape[1] / resized_image.shape[1]

        input_image = preprocess_image(
            resized_image,
            preprocessing=preprocessing
        )

        # D2-NET MS (multi-scale) o D2-NET SS (single-scale):
        scales = ([.5, 1, 2] if multiscale else [1])

        # Needed info to compute inverse covs:
        with torch.no_grad():
            keypoints, scores, descriptors = process_multiscale(
                torch.tensor(
                    input_image[np.newaxis, :, :, :].astype(np.float32),
                    device=device
                ),
                model,
                scales=scales
            )
        # Input image coordinates
        keypoints[:, 0] *= fact_i
        keypoints[:, 1] *= fact_j
        # i, j -> u, v
        keypoints = keypoints[:, [1, 0, 2]]
        return keypoints, scores, descriptors
