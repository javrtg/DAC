import sys
import inspect
import subprocess
from argparse import ArgumentParser
from pathlib import Path
from importlib import import_module
from datetime import datetime

from tqdm import tqdm
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib
from kornia.filters import gaussian_blur2d
from omegaconf import DictConfig, OmegaConf as oc

ENV_PATH = str(Path(__file__).parents[2])
if ENV_PATH not in sys.path:
    print(f'Inserting {ENV_PATH} to sys.path')
    sys.path.insert(0, ENV_PATH)
from experiments.interpretability.structure_torch import Structure
from experiments.interpretability import detectors


matplotlib.use('agg')


torch.autograd.set_detect_anomaly(True)


class Uncertainty(torch.nn.Module):
    """DeepDream experiment based on structure tensor/scores"""

    def __init__(
            self, Detector, cfg_acorr={}, return_scores=False, shortcut=False):
        super().__init__()

        # special processing depending on the model
        if Detector.__name__.lower() in ["superpoint", "r2d2"]:
            self.detector = Detector(return_scores)
        else:
            self.detector = Detector()
        # disable their gradients since we don't need them (for memory savings)
        for param in self.parameters():
            param.requires_grad = False

        # choose between structure tensor and scores
        if return_scores:
            self.confidence = lambda x: x  # identity (don't process scores)
        else:
            self.confidence = Structure(
                cfg_acorr, shortcut, return_only='m')

    def forward(self, x):
        # keypoint scores heatmap:
        detection_heatmap = self.detector(x)
        return self.confidence(detection_heatmap)


def create_timelapse_video(out_dir, fps=30):
    """Convert the sequence of stored images to a video/GIF w. ffmpeg"""
    cmd = (f"ffmpeg -framerate {fps} "
           f"-i {str(out_dir / 'images' / '%d.png')} "
           "-pix_fmt yuv420p "
           # resolution must be even number
           "-vf crop=trunc(iw/2)*2:trunc(ih/2)*2 "
           # "-c:v libx264 -strict -2 -preset slow "
           f"{str(out_dir / 'video.mp4')} "
           f"{str(out_dir / 'video.gif')}").split()
    proc = subprocess.run(
        cmd,
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE,
        universal_newlines=True
    )
    # Note: proc = subprocess.run(cmd, capture_output=True, text=True) is equivalent
    print(proc.stdout, '\n\n', proc.stderr)


def dynamic_load(root, model):
    module_path = f'{root.__name__}.{model}.{model}'
    module = import_module(module_path)
    # classes available in module scope
    classes = inspect.getmembers(module, inspect.isclass)
    # retain only the class defined in the module whose name matches model
    classes = [c for c in classes
               if (c[1].__module__ == module_path) and (c[0].lower() == model)]
    assert len(classes) == 1, classes
    return classes[0][1]


def img2tensor(img, device):
    """
    Conversions:
        - (H, W[, C]) numpy image -> (1, C, H, W) tensor w. requires_grad=True.
        - (C, H, W) tensor -> (1, C, H, W) tensor w. requires_grad=True.
    """
    if isinstance(img, np.ndarray):
        if len(img.shape) == 2:
            return torch.from_numpy(
                img).float()[None, None].float().to(device).requires_grad_()
        elif len(img.shape) == 3:
            return torch.from_numpy(
                img).permute(2, 0, 1).float()[None].float().to(device).requires_grad_()
        else:
            raise ValueError

    elif isinstance(img, torch.Tensor):
        if len(img.shape) == 3:
            return img[None].to(device).requires_grad_()
        else:
            raise NotImplementedError
    else:
        raise ValueError


def update_plot(img_a, img_s, fig, axes, coord, i, out_dir, lim=10):
    for axi in axes:
        axi.cla()
    # convert to gray images if needed
    if len(img_a.shape) == 3:
        img_a = cv2.cvtColor(img_a, cv2.COLOR_RGB2GRAY)
        img_s = cv2.cvtColor(img_s, cv2.COLOR_RGB2GRAY)

    cmap = 'jet'
    axes[0].imshow(
        img_a[
            coord[0] - lim: coord[0] + lim,
            coord[1] - lim: coord[1] + lim
        ],
        cmap=cmap, vmin=0, vmax=255
    )

    im_h = axes[1].imshow(
        img_s[
            coord[0] - lim: coord[0] + lim,
            coord[1] - lim: coord[1] + lim
        ],
        cmap=cmap, vmin=0, vmax=255
    )

    axes[0].set(
        title='Struc. Tensor',
        xticks=[0, 2 * lim - 1],
        yticks=[0, 2 * lim - 1]
    )

    axes[1].set(
        title='scores',
        xticks=[0, 2 * lim - 1],
        yticks=[0, 2 * lim - 1]
    )

    if len(fig.axes) == 2:
        fig.subplots_adjust(bottom=0.1)
        cbar_ax = fig.add_axes([0.12, .1, 0.78, 0.05])
        cbar = fig.colorbar(im_h, cax=cbar_ax, orientation='horizontal')
        cbar.set_ticks([0, 125, 255])

    if out_dir is not None:
        fig.savefig(out_dir / 'images' / f'{i}.png', bbox_inches='tight')

    # plt.pause(1e-6)


def gradient_ascent(
        tensor, uncertainty,
        coord, coord0,
        lr=1e-1, lr_pbias=0.,
        deb=None, debug=False):
    """Update input tensor in-place, amplifying structure tensor/scores."""

    def gaussian2dmask(tensor, mean, sigma=1.):
        x = torch.arange(0, tensor.shape[-1])
        y = torch.arange(0, tensor.shape[-2])
        dx = torch.exp(-0.5 * ((x - mean[1]) / sigma)**2)
        dy = torch.exp(-0.5 * ((y - mean[0]) / sigma)**2)
        return torch.mm(dy[:, None], dx[None]).to(tensor)

    confidence = uncertainty(tensor)  # high values -> better keypoint
    # fig, ax = plt.subplots()
    # ax.imshow(confidence.detach().cpu().numpy()[0,0])
    # print(confidence.mean(), confidence.std())
    # plt.show()

    # loss at a small local window of the input tensor. We directly use the
    # confidence values in order to amplify them by modifying the input image
    to_amplify = confidence[0, 0, coord[0], coord[1]].abs().sum()

    # small loss component to bias gradients at coord to positive values.
    # Without it, both versions (scores and structure tensor) may be optimized
    # with gradients in opposite orientations.
    # positive_bias = - lr_pbias * torch.exp(-confidence[0, 0, coord[0], coord[1]])
    positive_bias = - lr_pbias * \
        (torch.exp(-confidence) * gaussian2dmask(confidence,
         coord, sigma=1.)[None, None]).sum()
    # positive_bias = 0

    loss = to_amplify + positive_bias
    # loss = torch.norm(to_amplify, p='fro')
    # loss = torch.nn.MSELoss(reduction='mean')(to_amplify, torch.zeros_like(to_amplify))

    loss.backward()

    with torch.no_grad():
        grad = tensor.grad
        # drecrease the influence of the gradients if far from coord:
        grad = grad * gaussian2dmask(grad, coord0, sigma=3.)[None, None]
        # smoothing
        grad = gaussian_blur2d(grad, (7, 7), (1., 1.))

        # normalization of the gradients
        # g_n0 = grad[grad>0]
        # g_mean = torch.mean(g_n0)
        # g_std = torch.std(g_n0)
        # grad = grad - g_mean
        # grad = grad / g_std

        # gradient ascent
        tensor += lr * grad
        # clear gradients so that they dont accumulate over the iterations
        tensor.grad.zero_()  # tensor.grad = None can be used too

        # clamp the image values according to the detector
        lb, ub = uncertainty.detector.lower_bound, uncertainty.detector.upper_bound
        if isinstance(lb, (float, int)):
            tensor.clamp_(min=lb, max=ub)

        elif isinstance(lb, list):
            for i in range(len(lb)):
                tensor[0, i].clamp_(min=lb[i], max=ub[i])

        else:
            raise ValueError

        if debug:
            tqdm.write(
                f"\n{deb} update\t {lr * grad[0,:, coord0[0], coord0[1]]}")
            tqdm.write(
                f"{deb} image value\t {tensor[0,:, coord0[0], coord0[1]]}")
            tqdm.write(
                f"{deb} conf value\t {confidence[0,:, coord[0], coord[1]]}\n")


def optimizeSGD(
        cfg_optim, device,
        img_a, img_s, uncertainty_a, uncertainty_s,
        coord, coord0,
        fig, axes, debug, out_dir):
    """SGD optimization of the input image"""
    # initial input values
    tensor_a = img2tensor(uncertainty_a.detector.preprocess(img_a), device)
    tensor_s = img2tensor(uncertainty_a.detector.preprocess(img_s), device)

    # update tensor in-place, amplifying the structure tensor
    for i in tqdm(range(cfg_optim.iters), leave=False):
        gradient_ascent(
            tensor_a, uncertainty_a, coord, coord0,
            lr=cfg_optim.lr_a, lr_pbias=cfg_optim.lr_pbias,
            deb='a', debug=debug)
        gradient_ascent(
            tensor_s, uncertainty_s, coord, coord0,
            lr=cfg_optim.lr_s, lr_pbias=cfg_optim.lr_pbias,
            deb='s', debug=debug)

        # undo the processing: detach tensor and convert to numpy:
        img_a = uncertainty_a.detector.postprocess(tensor_a)
        img_s = uncertainty_s.detector.postprocess(tensor_s)

        update_plot(img_a, img_s, fig, axes, coord0, i, out_dir)


def gradient_ascent_Adam(
        tensor, uncertainty,
        moments, t,
        coord, coord0,
        lr, lr_pbias,
        deb='a', debug=False,
        beta1=0.9, beta2=0.999, eps=1e-8):

    def gaussian2dmask(tensor, mean, sigma=1.):
        x = torch.arange(0, tensor.shape[-1])
        y = torch.arange(0, tensor.shape[-2])
        dx = torch.exp(-0.5 * ((x - mean[1]) / sigma)**2)
        dy = torch.exp(-0.5 * ((y - mean[0]) / sigma)**2)
        return torch.mm(dy[:, None], dx[None]).to(tensor)

    confidence = uncertainty(tensor)  # high values -> better keypoint
    # fig, ax = plt.subplots()
    # ax.imshow(confidence.detach().cpu().numpy()[0,0])
    # print(confidence.mean(), confidence.std())
    # plt.show()

    # loss at a small local window of the input tensor. We directly use the
    # confidence values in order to amplify them by modifying the input image
    to_amplify = confidence[0, 0, coord[0], coord[1]].abs().sum()

    # small loss component to bias gradients at coord to positive values.
    # Without it, both versions (scores and struc. tensor) may be optimized
    # with gradients in opposite orientations.
    # positive_bias = - lr_pbias * torch.exp(-confidence[0, 0, coord[0], coord[1]])
    positive_bias = - lr_pbias * \
        (torch.exp(-confidence) * gaussian2dmask(confidence,
         coord, sigma=1.)[None, None]).sum()
    # positive_bias = 0

    loss = to_amplify + positive_bias
    # loss = torch.norm(to_amplify, p='fro')
    # loss = torch.nn.MSELoss(reduction='mean')(to_amplify, torch.zeros_like(to_amplify))

    loss.backward()

    with torch.no_grad():
        grad = tensor.grad
        # drecrease the influence of the gradients if far from coord:
        grad = grad * gaussian2dmask(grad, coord0, sigma=3.)[None, None]
        # smoothing
        grad = gaussian_blur2d(grad, (7, 7), (1., 1.))

        # normalization of the gradients
        # g_n0 = grad[grad>0]
        # g_mean = torch.mean(g_n0)
        # g_std = torch.std(g_n0)
        # grad = grad - g_mean
        # grad = grad / g_std

        # update moments
        moments['m'] = beta1 * moments['m'] + (1 - beta1) * grad
        moments['v'] = beta2 * moments['v'] + (1 - beta2) * torch.square(grad)
        # bias correction
        mb = moments['m'] / (1 - beta1**t)
        vb = moments['v'] / (1 - beta2**t)

        # update
        upd = mb / (torch.sqrt(vb) + eps) * \
            gaussian2dmask(grad, coord0, sigma=3.)[None, None]
        upd = gaussian_blur2d(upd, (7, 7), (1., 1.))

        # gradient ascent
        # tensor += lr * mb / ( torch.sqrt(vb) + eps )
        tensor += lr * upd

        # clear gradients so that they dont accumulate over the iterations
        tensor.grad.zero_()

        # clamp the image values according to the detector
        lb, ub = uncertainty.detector.lower_bound, uncertainty.detector.upper_bound
        if isinstance(lb, (float, int)):
            tensor.clamp_(min=lb, max=ub)

        elif isinstance(lb, list):
            for i in range(len(lb)):
                tensor[0, i].clamp_(min=lb[i], max=ub[i])

        else:
            raise ValueError

        if debug:
            tqdm.write(
                f"\n{deb} update\t {(lr * mb / ( torch.sqrt(vb) + eps ))[0,:, coord0[0], coord0[1]]}")
            tqdm.write(
                f"{deb} image value\t {tensor[0,:, coord0[0], coord0[1]]}")
            tqdm.write(
                f"{deb} conf value\t {confidence[0,:, coord[0], coord[1]]}\n")


def optimizeAdam(
        cfg_optim, device,
        img_a, img_s, uncertainty_a, uncertainty_s,
        coord, coord0,
        fig, axes, debug, out_dir):
    """Adam optimization of the input image"""

    # initial input values
    tensor_a = img2tensor(uncertainty_a.detector.preprocess(img_a), device)
    tensor_s = img2tensor(uncertainty_a.detector.preprocess(img_s), device)

    # initialize 1st and 2nd moments
    moments_a = dict(
        m=torch.zeros_like(tensor_a),  # 1st
        v=torch.zeros_like(tensor_a)  # 2nd
    )
    moments_s = dict(
        m=torch.zeros_like(tensor_s),
        v=torch.zeros_like(tensor_s)
    )

    # update tensor in-place, amplifying the autocorrelation
    for i in tqdm(range(cfg_optim.iters), leave=False):
        gradient_ascent_Adam(
            tensor_a, uncertainty_a,
            moments_a, i + 1,
            coord, coord0,
            lr=cfg_optim.lr_a, lr_pbias=cfg_optim.lr_pbias,
            deb='a', debug=debug)
        gradient_ascent_Adam(
            tensor_s, uncertainty_s,
            moments_s, i + 1,
            coord, coord0,
            lr=cfg_optim.lr_s, lr_pbias=cfg_optim.lr_pbias,
            deb='s', debug=debug)

        # undo the processing: detach tensor and convert to numpy:
        img_a = uncertainty_a.detector.postprocess(tensor_a)
        img_s = uncertainty_s.detector.postprocess(tensor_s)

        update_plot(img_a, img_s, fig, axes, coord0, i, out_dir)


def main(model: str, conf: DictConfig, out_dir: Path, debug: bool = False, img0=None):
    """ DeepDream-like optimization w. structure tensor and scores

    We optimize a dummy input image at a local region, seeking to amplify the
    structure tensor and scores estimates.    
    """
    device = ('cuda' if torch.cuda.is_available() else 'cpu')

    fig, axes = plt.subplots(ncols=2, sharey=True)
    if out_dir is not None:
        (out_dir / 'images').mkdir(exist_ok=True)

    # create input image
    if img0 is None:
        shape0 = (240, 240, 3)
        np.random.seed(0)
        img0 = (np.random.random(shape0) * 10).astype(np.uint8)

    # dirty modification to generate strong gradient in the optimized region
    if model.lower() == 'keynet':
        h0, w0 = img0.shape[:2]
        coord0 = (int(h0 / 2), int(w0 / 2))
        img0 = (np.random.random(shape0) * 10).astype(np.uint8)
        img0[coord0[0] - 3:coord0[0] + 3, coord0[1] - 3:coord0[1] + 3] = 100

    img_a = img0.copy()  # structure tensor
    img_s = img0.copy()  # scores

    # coordinates where the structure tensor/scores will be amplified:
    h0, w0 = img0.shape[:2]
    coord0 = (int(h0 / 2), int(w0 / 2))
    coord = coord0
    if model.lower() == 'd2net':
        # d2net uses low-resolution (x0.25 feature maps):
        coord = (int(round(coord[0] / 4)), int(round(coord[1] / 4)))

    # pipeline to extract the confidence/uncertainty estimates
    cfg_acorr = conf.cfg_acorr
    Detector = dynamic_load(detectors, model)
    uncertainty_a = Uncertainty(
        Detector, cfg_acorr, return_scores=False).eval().to(device)
    uncertainty_s = Uncertainty(
        Detector, cfg_acorr, return_scores=True).eval().to(device)

    # optimization
    op_method = conf.cfg_optim.method.lower()
    if op_method == 'sgd':
        optimizeSGD(
            conf.cfg_optim, device,
            img_a, img_s, uncertainty_a, uncertainty_s,
            coord, coord0,
            fig, axes, debug, out_dir)

    elif op_method == 'adam':
        optimizeAdam(
            conf.cfg_optim, device,
            img_a, img_s, uncertainty_a, uncertainty_s,
            coord, coord0,
            fig, axes, debug, out_dir)

    else:
        raise ValueError

    # video
    if out_dir is not None:
        create_timelapse_video(out_dir, fps=30)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        '--det', type=str, choices=['d2net', 'superpoint', 'r2d2', 'keynet'],
        default='superpoint')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('other', nargs='*')

    # args = parser.parse_args(['--save'])
    args = parser.parse_args()
    print('\n', args, '\n')

    conf = oc.load(Path(__file__).parent / 'configs' / f'{args.det}.yaml')
    conf = oc.merge(conf, oc.from_dotlist(args.other))
    print('Configuration:\n', oc.to_yaml(conf))

    if args.save:
        date = datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
        out_dir = Path(__file__).parent / 'results' / args.det / date
        out_dir.mkdir(parents=True, exist_ok=True)

        with open(out_dir / 'config.yaml', 'w') as f:
            oc.save(config=conf, f=f.name)
    else:
        out_dir = None

    main(args.det, conf, out_dir, debug=args.debug)
    # plt.show()
