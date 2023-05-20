# DAC: Detector-Agnostic Spatial Covariances for Deep Local Features

> DAC: Detector-Agnostic Spatial Covariances for Deep Local Features <br>
> Javier Tirado-Garín, Frederik Warburg, Javier Civera

## Setup

Dependencies can be installed via [conda](https://docs.conda.io/en/latest/)

```shell
conda create -n dac python=3.10 --yes
conda activate dac
conda install pytorch torchvision pytorch-cuda=11.6 -c pytorch -c nvidia
conda install numpy scipy ffmpeg scikit-learn matplotlib tqdm plotly pillow pyyaml numba
pip install opencv-python einops kornia==0.6.9 open3d==0.16.0 tabulate omegaconf
```

or more directly via [`environment.yml`](environment.yml) (it may be very slow):

```shell
conda env create -f environment.yml
conda activate dac
```

## Datasets
For running the experiments, please download
- [hpatches-sequences-release](http://icvl.ee.ic.ac.uk/vbalnt/hpatches/hpatches-sequences-release.tar.gz) (direct link)

    <details>
    <summary> [expected directory structure] </summary>

    ```shell
        HPATCHES
        └── hpatches-sequences-release
            ├── i_ajuntament
            ├── i_autannes
            .
            .
            .
    ```

    </details>

- [TUM-RGBD freiburg1 (fr1) sequences](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download)

    <details>
    <summary>[expected directory structure]</summary>

    ```shell
    TUM_RGBD
    ├── freiburg1_<name> # e.g. freiburg1_360
    │   ├── rgb
    │   ├── groundtruth.txt
    │   └── rgb.txt
    ├── ...
    .
    .
    .
    ```
    </details>

- [KITTI odometry sequences](https://www.cvlibs.net/datasets/kitti/eval_odometry.php) (color images, calibration files and  ground-truth poses).

    <details>
    <summary> [expected directory structure] </summary>

    ```shell
    KITTI
    └── odometry
        └── dataset
            ├── poses
            │   ├── 00.txt
            │   ├── 01.txt
            .   .
            .   .
            .   .
            └── sequences
                ├── 00
                │   ├── image_2
                │   ├── calib.txt
                │   └── times.txt
                .
                .
                .
    ```

    </details>

Absolute paths for each dataset can be modified in [`datasets/settings.py`](datasets/settings.py).

## Reproducibility

- To reproduce the first experiment (uncertainty vs. matching accuracy):
    ```shell
    python run_matching.py --det <model_name>
    ````
    where `<model_name>` corresponds to one of the evaluated systems:<br/> `superpoint` / `d2net` / `r2d2` / `keynet`

    Results will be saved in `experiments/matching/results`.

- To reproduce the second experiment (geometry estimation):
    ```shell
    python run_geometry.py --dataset <dataset> --sequence <sequence> --detectors <model_names>
    ```
    `<dataset>` can be either: `kitti` / `tum_rgbd` <br/>
    `<sequence>` depends on the selected `<dataset>`:

    - when `kitti`, `<sequence>` can be: <br/> `00` / `01` / `02`
    - when `tum_rgbd`, `<sequence>` can be: <br/> `freiburg1_xyz` /  `freiburg1_rpy` / `freiburg1_360`.

    `<model_names>` correspond to the names of the systems to be evaluated. They can be supplied in batch e.g.: <br/>
    `--detectors superpoint d2net r2d2 keynet`

    Results will be saved in `experiments/geometry/results`.

    > **Note**
    > Since [numba](https://numba.readthedocs.io/en/stable/) is used to speed up some calculations, some warnings will appear while compiling the first time. Once executed, compilation results are cached and no more warnings will appear in succesive executions.

- To reproduce the interpretability experiment (supp. material):
    ```shell
    python run_interpretability.py --save --det <model_name>
    ```
    where `<model_name>` corresponds to one of the evaluated systems:<br/> `superpoint` / `d2net` / `r2d2` / `keynet`

    Results will be saved in `experiments/interpretability/results`

- To reproduce the validation of the EPnPU implementation (supp. material):

    ```shell
    # to use 2D and 3D synthetic noise
    python experiments/geometry/models/benchmarks/bench.py --do_acc_vs_npoints

    # to use 2D noise only
    python experiments/geometry/models/benchmarks/bench.py --do_acc_vs_npoints --only_2d_noise
    ```

    Results will be saved in `experiments/geometry/models/benchmarks/results`


## License and Acknowledgements
The folders [`detectors/d2net`](detectors/d2net), [`detectors/keynet`](detectors/keynet), [`detectors/r2d2`](detectors/r2d2) and [`detectors/superpoint`](detectors/superpoint) contain code based on the original  system repositories ([D2Net](https://github.com/mihaidusmanu/d2-net), [Key.Net](https://github.com/axelBarroso/Key.Net-Pytorch), [R2D2](https://github.com/naver/r2d2), [SuperPoint](https://github.com/magicleap/SuperPointPretrainedNetwork)), each of them having different licenses. The rest of the repository is MIT licensed.