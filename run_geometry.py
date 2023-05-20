from argparse import ArgumentParser

from detectors.load import load_detector
from experiments.geometry.eval_geom import main
from experiments.geometry.utils_geom import configs
from experiments.geometry.utils_geom.eval import get_eval_objects
from experiments.geometry.utils_geom.data_parsers import get_dataset


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('--dataset', required=True, type=str,
                        choices=['tum_rgbd', 'kitti'])
    parser.add_argument('--sequence', type=str, required=True)
    parser.add_argument('--viz_plt', action='store_true')
    parser.add_argument('--viz_o3d', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--save_plt', action='store_true')
    parser.add_argument('--detectors', type=str, nargs='+', required=True,
                        choices=['superpoint', 'r2d2', 'keynet', 'd2net'])

    args = parser.parse_args()
    print(f"\n{args}\n")

    dataset = get_dataset(args.dataset, args.sequence)

    if args.dataset == 'tum_rgbd':
        nframes = 3
        min_dist_th = 2.5e-2

    elif args.dataset == 'kitti':
        nframes = 2
        min_dist_th = 0.0

    else:
        raise NotImplementedError('Dataset not supported.')

    pnp_models = get_eval_objects('epnp', 'epnpu')

    ranscac_f_cfg = {**configs.BASE_FRANSAC_CFG, **{}}
    ransac_p3p_cfg = configs.BASE_USAC_P3P

    # features will be normalized, so, calibrate thresholds (in-place).
    configs.calibrate_thresholds(dataset.K, ranscac_f_cfg, ransac_p3p_cfg)

    for det_name in args.detectors:
        detector = load_detector(det_name)

        main(
            dataset,
            detector,
            pnp_models,
            ranscac_f_cfg,
            ransac_p3p_cfg,
            nframes=nframes,
            min_dist_th=min_dist_th,
            verbose=args.verbose,
            do_viz_plt=args.viz_plt,
            do_viz_o3d=args.viz_o3d,
            save_plt=args.save_plt)
