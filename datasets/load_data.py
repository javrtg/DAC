import sys
from typing import Optional
from pathlib import Path
from importlib import import_module

ENV_PATH = str(Path(__file__).parents[1])
if ENV_PATH not in sys.path:
    print(f'inserting {ENV_PATH} to sys.path')
    sys.path.insert(0, ENV_PATH)


AVAILABLE_DATASETS = {
    "tum_mono": ("tum_mono", "TUM_MONO"),
    "tum_rgbd": ("tum_rgbd", "TUM_RGBD"),
    "hpatches": ("hpatches", "HPatches"),
    "icl": ("icl", "ICL"),
    "kitti": ("kitti", "KITTI_ODOMETRY")
}


def dynamic_load_class(pkg_name, module_name, class_name):
    return getattr(import_module(f"{pkg_name}.{module_name}"), class_name)


def load_dataset(name: str, cfg: Optional[dict] = None):
    """ Load an instance of an implemented dataset-parser. """
    name = name.lower()
    assert name in AVAILABLE_DATASETS, AVAILABLE_DATASETS
    loc = AVAILABLE_DATASETS[name]

    if cfg is None:
        cfg = {}

    return dynamic_load_class('datasets', loc[0], loc[1])(**cfg)
