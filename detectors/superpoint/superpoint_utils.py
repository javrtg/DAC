import numpy as np
import sys
from pathlib import Path

import cv2
import torch

ENV_PATH = str(Path(__file__).parents[2])
if ENV_PATH not in sys.path:
    print(f'inserting {ENV_PATH} to sys.path')
    sys.path.insert(0, ENV_PATH)
from detectors.base_detector import BaseDetector
from detectors.structure_tensor import (
    structure_tensor_matrices_at_points as struc_at_kp)


class SuperPointNet(torch.nn.Module):
    """ Pytorch definition of SuperPoint Network. """

    def __init__(self):
        super(SuperPointNet, self).__init__()
        self.relu = torch.nn.ReLU(inplace=True)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256
        # Shared Encoder.
        self.conv1a = torch.nn.Conv2d(
            1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = torch.nn.Conv2d(
            c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = torch.nn.Conv2d(
            c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = torch.nn.Conv2d(
            c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = torch.nn.Conv2d(
            c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = torch.nn.Conv2d(
            c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = torch.nn.Conv2d(
            c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = torch.nn.Conv2d(
            c4, c4, kernel_size=3, stride=1, padding=1)
        # Detector Head.
        self.convPa = torch.nn.Conv2d(
            c4, c5, kernel_size=3, stride=1, padding=1)
        self.convPb = torch.nn.Conv2d(
            c5, 65, kernel_size=1, stride=1, padding=0)
        # Descriptor Head.
        self.convDa = torch.nn.Conv2d(
            c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = torch.nn.Conv2d(
            c5, d1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """ Forward pass that jointly computes unprocessed point and descriptor
        tensors.
        Input
          x: Image pytorch tensor shaped N x 1 x H x W.
        Output
          semi: Output point pytorch tensor shaped N x 65 x H/8 x W/8.
          desc: Output descriptor pytorch tensor shaped N x 256 x H/8 x W/8.
        """
        # Shared Encoder.
        x = self.relu(self.conv1a(x))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        # Detector Head.
        cPa = self.relu(self.convPa(x))
        semi = self.convPb(cPa)
        # Descriptor Head.
        cDa = self.relu(self.convDa(x))
        desc = self.convDb(cDa)
        dn = torch.norm(desc, p=2, dim=1)  # Compute the norm.
        desc = desc.div(torch.unsqueeze(dn, 1))  # Divide by norm to normalize.
        return semi, desc


class Superpoint(BaseDetector):
    default_cfg_matching = {
        'distance': 'euclidean',
        'thr': 0.7  # L2 descriptor distance for good match.
    }

    default_cfg = {
        'weights_path': "superpoint_v1.pth",
        'nms_dist': 4,
        'conf_thresh': 0.015,
        'nn_thresh': 0.7,
        'cuda': False,
        'closest_reshape': True,
        'return_heatmaps': False,
        'check_size': False
    }

    def _init(self, cfg):
        self.cuda = cfg['cuda']
        self.nms_dist = cfg['nms_dist']
        self.conf_thresh = cfg['conf_thresh']
        self.cell = 8  # Size of each output cell. Keep this fixed.
        self.border_remove = 4  # Remove points this close to the border.
        self.closest = cfg['closest_reshape']
        self.return_heatmaps = cfg['return_heatmaps']
        self.cfg_acorr = cfg['cfg_acorr']
        self.check_size = cfg['check_size']

        # Load the network in inference mode.
        self.net = SuperPointNet()
        if self.cuda:
            # Train on GPU, deploy on GPU.
            self.net.load_state_dict(torch.load(
                str(Path(__file__).parent / cfg['weights_path'])))
            self.net = self.net.cuda()
        else:
            # Train on GPU, deploy on CPU.
            self.net.load_state_dict(torch.load(
                str(Path(__file__).parent / cfg['weights_path']),
                map_location=lambda storage, loc: storage))
        self.net.eval()

    def nms_fast(self, in_corners, H, W, dist_thresh):
        """
        Run a faster approximate Non-Max-Suppression on numpy corners shaped:
          3xN [x_i,y_i,conf_i]^T

        Algo summary: Create a grid sized HxW. Assign each corner location a 1, rest
        are zeros. Iterate through all the 1's and convert them either to -1 or 0.
        Suppress points by setting nearby values to 0.

        Grid Value Legend:
        -1 : Kept.
         0 : Empty or suppressed.
         1 : To be processed (converted to either kept or supressed).

        NOTE: The NMS first rounds points to integers, so NMS distance might not
        be exactly dist_thresh. It also assumes points are within image boundaries.

        Inputs
          in_corners - 3xN numpy array with corners [x_i, y_i, confidence_i]^T.
          H - Image height.
          W - Image width.
          dist_thresh - Distance to suppress, measured as an infinty norm distance.
        Returns
          nmsed_corners - 3xN numpy matrix with surviving corners.
          nmsed_inds - N length numpy vector with surviving corner indices.
        """
        grid = np.zeros((H, W)).astype(int)  # Track NMS data.
        inds = np.zeros((H, W)).astype(int)  # Store indices of points.
        # Sort by confidence and round to nearest int.
        inds1 = np.argsort(-in_corners[2, :])
        corners = in_corners[:, inds1]
        rcorners = corners[:2, :].round().astype(int)  # Rounded corners.
        # Check for edge case of 0 or 1 corners.
        if rcorners.shape[1] == 0:
            return np.zeros((3, 0)).astype(int), np.zeros(0).astype(int)
        if rcorners.shape[1] == 1:
            out = np.vstack((rcorners, in_corners[2])).reshape(3, 1)
            return out, np.zeros((1)).astype(int)
        # Initialize the grid.
        for i, rc in enumerate(rcorners.T):
            grid[rcorners[1, i], rcorners[0, i]] = 1
            inds[rcorners[1, i], rcorners[0, i]] = i
        # Pad the border of the grid, so that we can NMS points near the border.
        pad = dist_thresh
        grid = np.pad(grid, ((pad, pad), (pad, pad)), mode='constant')
        # Iterate through points, highest to lowest conf, suppress neighborhood.
        count = 0
        for i, rc in enumerate(rcorners.T):
            # Account for top and left padding.
            pt = (rc[0] + pad, rc[1] + pad)
            if grid[pt[1], pt[0]] == 1:  # If not yet suppressed.
                grid[pt[1] - pad:pt[1] + pad + 1,
                     pt[0] - pad:pt[0] + pad + 1] = 0
                grid[pt[1], pt[0]] = -1
                count += 1
        # Get all surviving -1's and return sorted array of remaining corners.
        keepy, keepx = np.where(grid == -1)
        keepy, keepx = keepy - pad, keepx - pad
        inds_keep = inds[keepy, keepx]
        out = corners[:, inds_keep]
        values = out[-1, :]
        inds2 = np.argsort(-values)
        out = out[:, inds2]
        out_inds = inds1[inds_keep[inds2]]
        return out, out_inds

    def normalize(self, img, size=(640, 480), closest=False):
        """
        Resize image to dimensions divisible by 8 (Superpoint requirement).

        Parameters
        ----------
        img : ndarray (H,W,C)
            input image.
        size : tuple of two ints, optional
            DESCRIPTION. Desired output size. The default is (640,480) which
            is used in the paper and recommended in:
            https://github.com/magicleap/SuperPointPretrainedNetwork/issues/5#issuecomment-529297976
        closest : bool, optional
            In case of wanting the input size to be as close as possible to the original,
            if this option is set to True, the image will be resized to the closest
            dimension values divisible by 8. The default is False.

        Returns
        -------
        out : float32 ndarray (H,W,C).
            Resized image.
        """
        # convert to grayscale if needed:
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        interp = cv2.INTER_AREA
        if not closest:
            # check valid resize
            assert (size[1] % 8) == 0, f"h: {size[1]} is not divisible by 8."
            assert (size[0] % 8) == 0, f"w: {size[0]} is not divisible by 8."
            if img.shape[:2] == size[::-1]:
                out = img
            else:
                out = cv2.resize(img, size, interpolation=interp)

        else:
            # retain image w. size divisible by 8 (SP's requirement).
            h, w = img.shape[:2]
            hmod, wmod = h % 8, w % 8
            if (hmod | wmod):
                # h_new = (h - hmod if hmod < 4 else h + hmod)
                # w_new = (w - wmod if wmod < 4 else w + wmod)
                # out = cv2.resize(img, (w_new, h_new), interpolation=interp)
                out = img[:h - hmod, :w - wmod]
            else:
                out = img

        out = out.astype('float32') / 255.
        return out

    def run(self, img):
        """ Process a numpy image to extract points and descriptors.
        Input
          img - HxWx3 np.uint8 image
        Output
          corners - 3xN numpy array with corners [x_i, y_i, confidence_i]^T.
          desc - Nx256 numpy array of corresponding unit normalized descriptors.
          heatmap - HxW numpy heatmap in range [0,1] of point confidences.
          """
        assert (img.dtype == 'uint8')
        if self.check_size:
            h, w = img.shape[:2]
            assert h % 8 == 0 and w % 8 == 0

        # normalization (range -> [0.,1.], gray image)
        img = self.normalize(img, closest=self.closest)
        assert img.ndim == 2, 'Image must be grayscale.'
        assert img.dtype == np.float32, 'Image must be float32.'
        H, W = img.shape[0], img.shape[1]
        inp = img.copy()
        inp = (inp.reshape(1, H, W))
        inp = torch.from_numpy(inp)
        inp = torch.autograd.Variable(inp).view(1, 1, H, W)
        if self.cuda:
            inp = inp.cuda()
        # Forward pass of network.
        outs = self.net.forward(inp)
        semi, coarse_desc = outs[0], outs[1]
        # Convert pytorch -> numpy.
        semi = semi.data.cpu().numpy().squeeze()
        # --- Process points.
        dense = np.exp(semi)  # Softmax.
        dense = dense / (np.sum(dense, axis=0) + .00001)  # Should sum to 1.
        # Remove dustbin.
        nodust = dense[:-1, :, :]
        # Reshape to get full resolution heatmap.
        Hc = int(H / self.cell)
        Wc = int(W / self.cell)
        nodust = nodust.transpose(1, 2, 0)
        heatmap = np.reshape(nodust, [Hc, Wc, self.cell, self.cell])
        heatmap = np.transpose(heatmap, [0, 2, 1, 3])
        heatmap = np.reshape(heatmap, [Hc * self.cell, Wc * self.cell])
        xs, ys = np.where(heatmap >= self.conf_thresh)  # Confidence threshold.
        if len(xs) == 0:
            return np.zeros((3, 0)), None, None
        pts = np.zeros((3, len(xs)))  # Populate point data sized 3xN.
        pts[0, :] = ys
        pts[1, :] = xs
        pts[2, :] = heatmap[xs, ys]
        # Apply NMS.
        pts, _ = self.nms_fast(pts, H, W, dist_thresh=self.nms_dist)
        inds = np.argsort(pts[2, :])
        pts = pts[:, inds[::-1]]  # Sort by confidence.
        # Remove points along border.
        bord = self.border_remove
        toremoveW = np.logical_or(pts[0, :] < bord, pts[0, :] >= (W - bord))
        toremoveH = np.logical_or(pts[1, :] < bord, pts[1, :] >= (H - bord))
        toremove = np.logical_or(toremoveW, toremoveH)
        pts = pts[:, ~toremove]
        # --- Process descriptor.
        D = coarse_desc.shape[1]
        if pts.shape[1] == 0:
            desc = np.zeros((D, 0))
        else:
            # Interpolate into descriptor map using 2D point locations.
            samp_pts = torch.from_numpy(pts[:2, :].copy())
            samp_pts[0, :] = (samp_pts[0, :] / (float(W) / 2.)) - 1.
            samp_pts[1, :] = (samp_pts[1, :] / (float(H) / 2.)) - 1.
            samp_pts = samp_pts.transpose(0, 1).contiguous()
            samp_pts = samp_pts.view(1, 1, -1, 2)
            samp_pts = samp_pts.float()
            if self.cuda:
                samp_pts = samp_pts.cuda()
            desc = torch.nn.functional.grid_sample(
                coarse_desc, samp_pts, align_corners=True)
            desc = desc.data.cpu().numpy().reshape(D, -1)
            desc /= np.linalg.norm(desc, axis=0)[np.newaxis, :]

            # --- structure tensor
            # copy tensor (without dustbin) before applying softmax:
            semi_copy = semi[:-1].copy()
            # heatmap
            nodust_copy = semi_copy.transpose(1, 2, 0)
            heatmap_acorr = np.reshape(
                nodust_copy, [Hc, Wc, self.cell, self.cell])
            heatmap_acorr = np.transpose(heatmap_acorr, [0, 2, 1, 3])
            heatmap_acorr = np.reshape(
                heatmap_acorr, [Hc * self.cell, Wc * self.cell])

        # structure tensor
        C = struc_at_kp(
            heatmap_acorr, pts[1::-1].astype(int), **self.cfg_acorr)
        # conditioned output
        if self.return_heatmaps:
            heatmaps = np.concatenate(
                (heatmap_acorr[None], heatmap[None]), axis=0)
            return pts, desc.T, C, heatmaps
        else:
            return pts, desc.T, C
