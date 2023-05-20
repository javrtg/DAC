import torch
import cv2
import numpy as np
from os import mkdir, path
import torch.nn.functional as F

from .network import KeyNet
from .modules import NonMaxSuppression
from .HyNet.hynet_model import HyNet
from .kornia_tools.utils import custom_pyrdown
from .kornia_tools.utils import laf_from_center_scale_ori as to_laf
from .kornia_tools.utils import extract_patches_from_pyramid as extract_patch

from math import ceil


def create_result_dir(result_path):
    '''
    It creates the directory where features will be stored
    '''
    directories = result_path.split('/')
    tmp = ''
    for idx, dir in enumerate(directories):
        tmp += (dir + '/')
        if idx == len(directories)-1:
            continue
        if not path.isdir(tmp):
            mkdir(tmp)


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

    return mask*score_map


def extract_ms_feats(keynet_model, desc_model, image, factor, s_mult, device,
                     num_kpts_i=1000, nms=None, down_level=0, up_level=False, im_size=[]):
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
    :return: It returns the local features for a specific image level
    '''

    if down_level and not up_level:
        image = custom_pyrdown(image, factor=factor)
        _, _, nh, nw = image.shape
        factor = (im_size[0]/nw, im_size[1]/nh)
    elif not up_level:
        factor = (1., 1.)

    # src kpts:
    with torch.no_grad():
        det_map = keynet_model(image)
    # print(det_map.numel(), det_map[(det_map>1.123) & (det_map<1.124)].numel())
    # relu scores
    det_map = remove_borders(det_map, borders=15)
    # kps obtained after NMS with default window size of 15 and threshold=0
    kps = nms(det_map)
    # relu scores of the keypoints
    c = det_map[0, 0, kps[0], kps[1]]
    # sort scores from high to low value
    sc, indices = torch.sort(c, descending=True)
    # another threshold for the scores (it is the same as the default one = 0)
    indices = indices[torch.where(sc > 0.)]
    # select num_kpts (or less) based on their score
    kps = kps[:, indices[:num_kpts_i]]
    # kps_np = array (n,3) with 3 = (u,v,score=c) i.e. keypoint coors and their score
    kps_np = torch.cat([kps[1].view(-1, 1).float(), kps[0].view(-1, 1).float(), c[indices[:num_kpts_i]].view(-1, 1).float()],
        dim=1).detach().cpu().numpy()
    num_kpts = len(kps_np)
    # same but as torch tensors in order to pass them to the desc model (Hynet)
    kp = torch.cat([kps[1].view(-1, 1).float(), kps[0].view(-1, 1).float()],dim=1).unsqueeze(0).cpu()
    # create local affine versions (LAFs) of the kps (to improve later the patch extraction with kornia)
    s = s_mult * torch.ones((1, num_kpts, 1, 1))
    src_laf = to_laf(kp, s, torch.zeros((1, num_kpts, 1)))

    # comment from paper author: HyNet takes images on the range [0, 255]
    # (Objective) comment from me:
    # This type of patch extraction is not a direct sampling
    # from the image. Instead, gaussian noise is applied to remove details
    # that may not be present at different scales. Thereby the patches come from
    # a smoothed image so that the descriptors vary less for different scales.
    # a related and useful blog: https://ducha-aiki.github.io/wide-baseline-stereo-blog/2020/07/22/patch-extraction.html
    # The amount of blur added depends on the LAF scale, s -> pyramid level = round(log2(s*2))
    # (Subjective) comment from me:
    # Since the author associates the same fix LAF scale to **every** keypoint (variable s_mult)
    # then I think this procedure may not be benefitial. All patches are
    # blurred the same amount irrespective of their true scale -> this may incur
    # in lost information from the original patch without being useful.
    patches = extract_patch(255*image.cpu(), src_laf, PS=32, normalize_lafs_before_extraction=True)[0]
    
    # due to memory resoruces, divide the descriptor extraction into batches
    # of 1e3 kps at maximum.
    limit = 10 # default = 1000
    if len(patches) > limit:
        for i_patches in range(ceil(len(patches)/limit)): # range(len(patches)//limit+1):
            if i_patches == 0:
                descs = desc_model(patches[:limit].to(device))
            else:
                descs_tmp = desc_model(patches[limit*i_patches:limit*(i_patches+1)].to(device))
                descs = torch.cat([descs, descs_tmp], dim=0)
        descs = descs.cpu().detach().numpy()
    else:
        descs = desc_model(patches.to(device)).cpu().detach().numpy()

    kps_np[:, 0] *= factor[0]
    kps_np[:, 1] *= factor[1]

    return kps_np, descs, image.to(device)


def compute_kpts_desc(im_path, keynet_model, desc_model, conf, device, num_points):
    '''
    The script computes Multi-scale kpts and desc of an image.

    :param im_path: path to image
    :param keynet_model: Detector model
    :param desc_model: Descriptor model
    :param conf: Configuration file to load extraction settings
    :param device: GPU or CPU
    :param num_points: Number of total local features
    :return: Keypoints and descriptors associated with the image
    '''

    # Load extraction configuration
    pyramid_levels = conf['pyramid_levels']
    up_levels = conf['up_levels']
    scale_factor_levels = conf['scale_factor_levels']
    s_mult = conf['s_mult']
    nms_size = conf['nms_size']
    th = conf['threshold']
    nms = NonMaxSuppression(nms_size=nms_size, thr=th)

    # Compute points per level
    point_level = []
    tmp = 0.0
    factor_points = (scale_factor_levels ** 2)
    levels = pyramid_levels + up_levels + 1
    for idx_level in range(levels):
        tmp += factor_points ** (-1 * (idx_level - up_levels))
        point_level.append(num_points * factor_points ** (-1 * (idx_level - up_levels)))

    point_level = np.asarray(list(map(lambda x: int(x / tmp), point_level)))

    im_np = np.asarray(cv2.imread(im_path, 0) / 255., np.float32)
    # im_np = np.ones_like(im_np) * 150
    im = torch.from_numpy(im_np).unsqueeze(0).unsqueeze(0)
    im = im.to(device)

    if up_levels:
        im_up = torch.from_numpy(im_np).unsqueeze(0).unsqueeze(0)
        im_up = im_up.to(device)

    src_kp = []
    _, _, h, w = im.shape
    # Extract features from the upper levels
    for idx_level in range(up_levels):

        num_points_level = point_level[len(point_level) - pyramid_levels - 1 - (idx_level+1)]

        # Resize input image
        up_factor = scale_factor_levels ** (1 + idx_level)
        nh, nw = int(h * up_factor), int(w * up_factor)
        up_factor_kpts = (w/nw, h/nh)
        im_up = F.interpolate(im_up, (nh, nw), mode='bilinear', align_corners=False)

        src_kp_i, src_dsc_i, im_up = extract_ms_feats(keynet_model, desc_model, im_up, up_factor_kpts,
                                                      s_mult=s_mult, device=device, num_kpts_i=num_points_level,
                                                      nms=nms, down_level=idx_level+1, up_level=True, im_size=[w, h])

        src_kp_i = np.asarray(list(map(lambda x: [x[0], x[1], (1 / scale_factor_levels) ** (1 + idx_level), x[2]], src_kp_i)))

        if src_kp == []:
            src_kp = src_kp_i
            src_dsc = src_dsc_i
        else:
            src_kp = np.concatenate([src_kp, src_kp_i], axis=0)
            src_dsc = np.concatenate([src_dsc, src_dsc_i], axis=0)

    # Extract features from the downsampling pyramid
    for idx_level in range(pyramid_levels + 1):

        num_points_level = point_level[idx_level]
        if idx_level > 0 or up_levels:
            res_points = int(np.asarray([point_level[a] for a in range(0, idx_level + 1 + up_levels)]).sum() - len(src_kp))
            num_points_level = res_points

        src_kp_i, src_dsc_i, im = extract_ms_feats(keynet_model, desc_model, im, scale_factor_levels, s_mult=s_mult,
                                                   device=device, num_kpts_i=num_points_level, nms=nms,
                                                   down_level=idx_level, im_size=[w, h])

        src_kp_i = np.asarray(list(map(lambda x: [x[0], x[1], scale_factor_levels ** idx_level, x[2]], src_kp_i)))

        if src_kp == []:
            src_kp = src_kp_i
            src_dsc = src_dsc_i
        else:
            src_kp = np.concatenate([src_kp, src_kp_i], axis=0)
            src_dsc = np.concatenate([src_dsc, src_dsc_i], axis=0)

    return src_kp, src_dsc


def initialize_networks(conf, device):
    '''
    It loads the detector and descriptor models
    :param conf: It contains the configuration and weights path of the models
    :return: Key.Net and HyNet models
    '''
    # use_cuda = torch.cuda.is_available()
    # device = torch.device("cuda:0" if use_cuda else "cpu")
    detector_path = conf['weights_detector']
    descriptor_path = conf['weights_descriptor']

    # Define keynet_model model
    keynet_model = KeyNet(conf)
    checkpoint = torch.load(detector_path)
    keynet_model.load_state_dict(checkpoint['state_dict'])
    keynet_model = keynet_model.to(device)
    keynet_model.eval()

    desc_model = HyNet()
    checkpoint = torch.load(descriptor_path)
    desc_model.load_state_dict(checkpoint)

    desc_model = desc_model.to(device)
    desc_model.eval()

    return keynet_model, desc_model
