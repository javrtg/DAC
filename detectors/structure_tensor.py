from math import sqrt, ceil

import numpy as np
from numba import njit
from scipy.signal import convolve
try:
    from scipy.signal import gaussian
except ModuleNotFoundError:  # for SciPy> 1.7.1:
    from scipy.signal.windows import gaussian


@njit('f8[:,:,:](f8[:,:,:])')
def fast2x2inv(C):
    n, h, w = C.shape
    assert h == 2  # help numba
    assert w == 2  # help numba
    out = np.empty((n, h, w))
    for i in range(n):
        ((a, b), (c, d)) = C[i]
        div = (a * d - b * c)
        out[i, 0, 0] = d / div
        out[i, 0, 1] = -b / div
        out[i, 1, 0] = -c / div
        out[i, 1, 1] = a / div
    return out


def _centered_1dgaussian_window(wsize, sigma, normalize=True):
    """ 1D window of length=wsize with gaussian weights"""
    window = gaussian(wsize, sigma)
    if normalize:
        window /= (sqrt(2 * np.pi) * sigma)
    return window


def _centered_2dgaussian_window(wsize, sigma, normalize=True):
    """ 2D window of size=(wsize, wsize) with gaussian weights"""
    window1d = gaussian(wsize, sigma)
    window2d = window1d[:, None] @ window1d[None]
    if normalize:
        window2d /= 2. * np.pi * (sigma**2)
    return window2d


def derivative_tensor(
        fmaps: np.ndarray, method: str = 'sobel',
        sigma: float = 1.0, truncation: float = 3.0
):
    """
    Computes spatial derivative of a tensor of 2d arrays (gray images/ feature maps), 
    i.e. given the 2d array im, this function computes: \partial im / \partial x,
    where x represents the spatial dimensions of im.

    arguments:
        im : 2d np.array
        method : str, method for computing derivatives. Supported:
            * 'gaussian': Gaussian derivatives. Convolve image with the
                          derivative of a gaussian (this is equivalent to 
                          computing the derivative of a convolved image),
            * 'sobel' : use 3x3 sobel filters,
            * 'diff': central difference.
        sigma : std used when method=='gaussian'
        truncation: defines the window size (ws) when method=='gaussian',
                    since ws = ceil(2*truncation)+1        
    """
    # check that input either a tensor or an image/ feature map
    assert (len(fmaps.shape) == 3) | (len(fmaps.shape) == 2), (
        f"input has {len(fmaps.shape)} dimensions, but a 2-d/3-d array was expected")
    # create different filters depending on the method used:
    if method == "sobel":
        # sobel filters:
        filter_x = (1. / 8.) * \
            np.array([[1., 0., -1.], [2., 0., -2.], [1., 0., -1.]])
        filter_y = (1. / 8.) * \
            np.array([[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]])
    elif method == "diff":
        # with scipy, it seems to be faster defining 3x3 kernels rather than 1d equivalent kernels:
        filter_x = 0.5 * np.array([[0., 0., 0.], [1., 0., -1.], [0., 0., 0.]])
        filter_y = 0.5 * np.array([[0., 1., 0.], [0., 0., 0.], [0., -1., 0.]])
    elif method == "gaussian":
        # window size:
        wsize = 2. * ceil(truncation * sigma) + 1.
        # 1d gaussian window:
        win1d = _centered_1dgaussian_window(wsize, sigma)
        # spatial derivative of kernel = -x/sigma * kernel
        x = np.arange(-(wsize - 1) / 2, (wsize - 1) / 2 + 1)
        dwin1d = (-x / (sigma**2)) * win1d
        # final kernels:
        filter_x = win1d[:, None] @ dwin1d[None]
        filter_y = filter_x.T  # = dwin1d[:,None] @ win1d[None]
    else:
        raise ValueError(f"method {method} not supported. Supported methods: "
                         "{'gaussian', 'sobel', 'diff'}")
    # compute x, y derivatives: fx, fy
    if len(fmaps.shape) == 2:
        # in case of just one image/ feature map, output a 2d array:
        fx = convolve(fmaps, filter_x, mode='same')
        fy = convolve(fmaps, filter_y, mode='same')
    else:
        fx = [convolve(fmapi, filter_x, mode='same') for fmapi in fmaps]
        fy = [convolve(fmapi, filter_y, mode='same') for fmapi in fmaps]
    return fx, fy


def weight_tensor(fmaps, method='gaussian', sigma=1.0, truncation=3.0):
    """ Local average weigthing of each image/ feature in fmaps """

    assert (len(fmaps.shape) == 3) | (len(fmaps.shape) == 2), (
        f"input has {len(fmaps.shape)} dimensions, but a 2-d/3-d array was expected")

    # create different filters depending on the method used:
    if method == 'gaussian':
        # window size:
        wsize = 2. * ceil(truncation * sigma) + 1.
        filter_w = _centered_2dgaussian_window(wsize, sigma, normalize=True)
    elif method == 'uniform':
        filter_w = (1. / 9.) * np.ones((3, 3))
    else:
        raise ValueError(f"method {method} not supported. Supported methods: "
                         "{'gaussian', 'uniform'}")
    # compute weighting:
    if len(fmaps.shape) == 2:
        # in case of just one image/ feature map, output a 2d array:
        fw = convolve(fmaps, filter_w, mode='same')
    else:
        fw = [convolve(fmapi, filter_w, mode='same') for fmapi in fmaps]
    return fw


def structure_tensor_matrices(
        fmaps: np.ndarray,
        method_derivative: str = 'sobel', sigma_d: float = 1.0, truncation_d: float = 3.0,
        method_weighting: str = 'gaussian', sigma_w: float = 1.0, truncation_w: float = 3.0
):
    """ Compute autocorrelation matrices. """

    # check that input either a tensor or an image/ feature map
    assert (len(fmaps.shape) == 3) | (len(fmaps.shape) == 2), (
        f"input has {len(fmaps.shape)} dimensions, but a 2-d/3-d array was expected")
    # spatial derivatives:
    fx, fy = derivative_tensor(fmaps, method_derivative, sigma_d, truncation_d)
    # autocorrelation matrices components -> C = [[Cxx, Cxy], [Cxy, Cyy]]
    Cxx = np.square(fx)
    Cyy = np.square(fy)
    Cxy = np.multiply(fx, fy)
    # del fx, fy
    # average surrounding gradients:
    C = np.empty((Cxx.shape + (2, 2)))
    C[..., 0, 0] = weight_tensor(Cxx, method_weighting, sigma_w, truncation_w)
    # del Cxx
    C[..., 1, 1] = weight_tensor(Cyy, method_weighting, sigma_w, truncation_w)
    # del Cyy
    C_diag = weight_tensor(Cxy, method_weighting, sigma_w, truncation_w)
    C[..., 0, 1], C[..., 1, 0] = C_diag, C_diag
    # del Cxy
    return C


def structure_tensor_matrices_at_points(
        fmaps: np.ndarray, kps: np.ndarray,
        method_derivative: str = 'sobel', sigma_d: float = 1.0, truncation_d: float = 3.0,
        method_weighting: str = 'gaussian', sigma_w: float = 1.0, truncation_w: float = 3.0
):
    """ Autocorrelation matrices at specific coord of the feature maps """
    assert (len(kps) == 2) | (len(kps) == 3)
    return structure_tensor_matrices(
        fmaps,
        method_derivative=method_derivative,
        sigma_d=sigma_d,
        truncation_d=truncation_d,
        method_weighting=method_weighting,
        sigma_w=sigma_w,
        truncation_w=truncation_w
    )[tuple(kps)]
