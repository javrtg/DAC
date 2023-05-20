from argparse import ArgumentParser

import numpy as np

from detectors.load import load_detector
from datasets.hpatches import HPatches
from experiments.matching.eval_hpatches import eval_hpatches

if __name__ == "__main__":

    parser = ArgumentParser(description='HPatches experiment')
    parser.add_argument('--det', type=str, required=True,
                        choices=['d2net', 'keynet', 'superpoint', 'r2d2'])
    parser.add_argument('--thresholds', type=np.array,
                        default=np.linspace(0.1, 10., 10),
                        help="thresholds considered when measuring the matching accuracy.")
    parser.add_argument('--uncertainty_levels', type=int, default=10)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--do_viz', action='store_true')
    parser.add_argument('--show', action='store_true')

    # sys.argv = ['features_uncertainty/experiments/hpatches/eval_hpatches.py', '--det', 'd2net']
    args = parser.parse_args()
    print('\n', args, '\n')

    assert args.uncertainty_levels > 1, "uncertainty_levels must be > 1."

    # instanciate dataset and model
    use_d2net_subset = args.det == 'd2net'
    hpatches = HPatches(use_d2net_subset)

    model = load_detector(args.det)

    eval_hpatches(
        hpatches, model, ths=args.thresholds,
        levels=args.uncertainty_levels,
        debug=args.debug, verbose=args.verbose, do_viz=args.do_viz,
        show=args.show)
