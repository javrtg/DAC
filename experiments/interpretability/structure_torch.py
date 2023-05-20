from typing import Dict, Union
from math import sqrt, ceil
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange
import kornia


def check_cfg(cfg_base, cfg_to_check):
    if any(k not in cfg_base for k in cfg_to_check):
        raise ValueError(
            f"At least, one key of config dict:\n{cfg_to_check.keys()}\n"
            f"was not found in the default keys:\n{cfg_base.keys()}")


def gaussian_derivative_filters(wsize: int, sigma: float):
    """ Filters of size (wsize, wsize) to compute Gaussian derivatives

    Principle (with * as convolution): d(f * g)/dx = df/dx * g.
    This function obtains df/dx
    """
    # discrete locations
    x = torch.linspace(-(wsize // 2), wsize // 2, wsize)
    # density values
    f = torch.exp(-0.5 * (x**2) / (sigma**2)) / (sigma * sqrt(2. * torch.pi))
    # normalization to avoid changes in brightness level:
    f /= f.sum()
    # get 1d kernel derivative
    dk1d = (-x / (sigma**2)) * f
    # normalization (so that convolution w. ramp gives its slope)
    dk1d /= (x * dk1d).abs().sum()
    # 2d filters for convolution:
    filter_x = torch.mm(f[:, None], dk1d[None])
    filter_y = filter_x.t()
    # 2d filters for cross-correlation
    return torch.stack([filter_x, filter_y], dim=0).flip([1, 2])


def broadcast_conv_over_featmaps(x: torch.Tensor, filters: torch.Tensor):
    """ Convolve **each** feature map of x with each of the filters. 

    Args:
        - x: input tensor w. shape (B, Cx, H, W)
        - filters: input weights w.shape (Ck, H, W)

    Return:
        - out: tensor with shape (Ck, Cx, H, W)
    """
    assert len(x.shape) == 4, (
        f"Invalid input tensor shape. Expected (B, C, H, W). Got {x.shape}")
    assert len(filters.shape) == 3, (
        f"Invalid filters shape. Expected (C, H, W). Got {filters.shape}")

    if filters.device != x.device:
        filters = filters.to(x)

    b, c, h, w = x.shape
    ck = filters.shape[0]

    # pad input according to filter size
    kh, kw = filters.shape[-2:]
    spatial_pad = [kw // 2, kw // 2, kh // 2, kh // 2]
    x = F.pad(x.view(b * c, 1, h, w), spatial_pad, mode='replicate')

    # cross correlate the filters with each feature map of the input
    x = F.conv2d(x, filters[:, None])
    return x.view(b, c, ck, h, w)


class Structure(nn.Module):
    """ Structure tensor block """
    default_cfg: Dict[str, Union[str, float]] = {
        'method_derivative': 'sobel',  # ['sobel', 'diff', 'gaussian']
        'sigma_d': 1.0,
        'truncation_d': 3.0,
        'method_weighting': 'gaussian',
        'sigma_w': 1.0,
        'truncation_w': 3.0
    }

    eps = 1e-8

    def __init__(self, cfg={}, shortcut=False, return_only=None):
        super().__init__()
        assert return_only in [None, 'm', 'M'], return_only
        self.return_only = return_only

        # update config dict
        check_cfg(self.default_cfg, cfg)
        self.cfg = {**self.default_cfg, **cfg}
        self.shortcut = shortcut

        # function for differentiation
        if self.cfg['method_derivative'] in ['sobel', 'diff']:
            self.dfun = partial(kornia.filters.spatial_gradient,
                                mode=self.cfg['method_derivative'], order=1, normalized=True)

        elif self.cfg['method_derivative'] == 'gaussian':
            dwsize = 2 * ceil(self.cfg['truncation_d'] *
                              self.cfg['sigma_d']) + 1
            kernels = gaussian_derivative_filters(dwsize, self.cfg['sigma_d'])
            # function to compute gaussian derivatives with predefined kernel:
            self.dfun = partial(broadcast_conv_over_featmaps, filters=kernels)

        else:
            raise ValueError(
                f"method {self.cfg['method_derivative']} not supported."
                "Supported methods: {'gaussian', 'sobel', 'diff'}")

        # window size when averaging the structure tensor:
        if self.cfg['method_weighting'] == 'gaussian':
            self.wwsize = 2 * \
                ceil(self.cfg['truncation_w'] * self.cfg['sigma_w']) + 1

        elif self.cfg['method_weighting'] == 'uniform':
            self.wkernel = torch.ones((1, 3, 3)) / 9.

        else:
            raise ValueError

    def forward(self, x: torch.Tensor, kps: torch.Tensor = None):
        # derivatives dx, dy:
        gradients = self.dfun(x)
        dx, dy = gradients[:, :, 0], gradients[:, :, 1]

        # Gaussian-averaged components of the structure tensor ()
        if self.cfg['method_weighting'] == 'gaussian':
            dx2 = kornia.filters.gaussian_blur2d(
                dx * dx, (self.wwsize, self.wwsize), (self.cfg['sigma_w'], self.cfg['sigma_w']))
            dy2 = kornia.filters.gaussian_blur2d(
                dy * dy, (self.wwsize, self.wwsize), (self.cfg['sigma_w'], self.cfg['sigma_w']))
            dxy = kornia.filters.gaussian_blur2d(
                dx * dy, (self.wwsize, self.wwsize), (self.cfg['sigma_w'], self.cfg['sigma_w']))

        elif self.cfg['method_weighting'] == 'uniform':
            dx2 = broadcast_conv_over_featmaps(dx * dx, self.wkernel)[:, :, 0]
            dy2 = broadcast_conv_over_featmaps(dy * dy, self.wkernel)[:, :, 0]
            dxy = broadcast_conv_over_featmaps(dx * dy, self.wkernel)[:, :, 0]

        if kps is None:
            # compute the eigenvalues of the autcorrelation matrix via their
            # characteristic equation
            discriminant = 4 * dxy * dxy + torch.square(dx2 - dy2)
            if self.shortcut:
                # only return the non-constant term(s)
                out = (
                    -discriminant if self.return_only == 'm' else
                    discriminant if self.return_only == 'M' else
                    (discriminant, -discriminant)
                )
                return out

            else:
                # compute the actual eigenvalues:
                const_term = dx2 + dy2
                discriminant_sqrt = torch.sqrt(
                    discriminant.clamp_(min=self.eps))
                max_eig = 0.5 * (const_term + discriminant_sqrt)
                min_eig = 0.5 * (const_term - discriminant_sqrt)

                out = (
                    min_eig if self.return_only == 'm' else
                    max_eig if self.return_only == 'M' else
                    (max_eig, min_eig)
                )
                return out

        else:
            # return structure tensor at kp coordinates
            b, c, h, w = x.shape
            n, p = kps.shape

            assert b == 1, (
                'Only 1 image is expected for structure tensor at kps, '
                f'but {b} images were received')
            assert p in (2, 3), (
                f'kps expected shape: (n, 2) or (n, 3). Got: {(n, p)}'
            )

            # query the tensors on the kps
            x = x[0, 0] if (p == 2) else x[0]
            dx2_q = dx2[(coords := tuple(kps.T))]
            dy2_q = dy2[coords]
            dxy_q = dxy[coords]

            # structure tensors:
            return rearrange(
                [dx2_q, dxy_q, dxy_q, dy2_q], '(h w) l -> l h w', h=2, w=2)
