import sys
from typing import Optional
from pathlib import Path
from importlib import import_module

ENV_PATH = str(Path(__file__).parents[1])
if ENV_PATH not in sys.path:
    print(f'inserting {ENV_PATH} to sys.path')
    sys.path.insert(0, ENV_PATH)


# dict w. notation -> "model": (relative path, main class name).
Detectors = {
    "superpoint": ("superpoint.superpoint_utils", "Superpoint"),
    "d2net": ("d2net.d2net_utils", "D2NET"),
    "keynet": ("keynet.keynet_utils", "Keynet"),
    "r2d2": ("r2d2.r2d2_utils", "R2D2"),
    }


def dynamic_load(pkg_name, pkg_dict, model_name):
    """Dynamic load of model main class."""
    assert model_name in pkg_dict.keys()
    module_path = f"{pkg_name}.{pkg_dict[model_name][0]}"
    module = import_module(module_path)
    return getattr(module, pkg_dict[model_name][1])


def load_detector(
        det: str,
        cfg_det: Optional[dict] = None,
        cfg_acorr: Optional[dict] = None,
        cfg_matching: Optional[dict] = None):
    """
    Instanciate a detector model with custom configuration.

    If cfg_det keyword return_heatmaps is set to False (
    cfg['return_heatmaps']=False), which is the default, the general output is:
        1) kps -> array (3,n) with u, v (im coords) and scores as dimensions
        2) desc -> array (n, d) "d"-dim descriptors at each of "n" detected kps
        3) C -> array (n,2,2) estimated **inverse** covariances of each kp
    otherwise, if retun_heatmaps is set to True (cfg['return_heatmaps']=True),
    then the output depends on the detector:
        a) det == superpoint:
            if heatmaps are returned (cfg['return_heatmaps']=True) then the
            output is:
                kps, desc, C, heatmaps
            with hetamps a (2,H,W) array:
                heatmaps[0] = pre-softmaxed heatmap
                heatmaps[1] = softmaxed heatmap
        b) det == d2net:
            if heatmaps are returned (cfg['return_heatmaps']=True) then the
            output is:
                kps, desc, C, heatmaps, kps_fmap
            with heatmaps a (512, H/4, W/4) array (only the ones from scale 1
            are returned), and kps_fmap, the coordinates (c, h, w) of the kps
            in the heatmaps.
        c) det == keynet:
            if heatmaps are returned (cfg['return_heatmaps']=True) then the
            output is:
                kps, desc, C, heatmap, kps_heatmap
            with heatmap a (H,W) array (only the one from scale 1 is returned),
            and kps_heatmap, the kps that were found in that heatmap
        d) det == r2d2:
            if heatmaps are returned (cfg['return_heatmaps']=True) then the output is:
                kps, desc, C, heatmaps, kps_heatmap
            with heatmaps a (2,H,W) array (only the ones from scale 1 are
            returned):
                heatmaps[0] = repeatability map
                heatmaps[1] = reliability map
            and kps_heatmap, the kps that were found in the repeatability heatmap
    """
    det = det.lower()

    if cfg_det is None: cfg_det = {}
    if cfg_acorr is None: cfg_acorr = {}
    if cfg_matching is None: cfg_matching = {}

    model_class = dynamic_load("detectors", Detectors, det)
    return model_class(cfg_det, cfg_acorr, cfg_matching)
