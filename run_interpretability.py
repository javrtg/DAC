from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

from omegaconf import OmegaConf as oc

from experiments.interpretability.interpretability_exp import main

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--det', type=str, required=True,
                        choices=['d2net', 'superpoint', 'r2d2', 'keynet'])
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('other', nargs='*')

    # args = parser.parse_args(['--save'])
    args = parser.parse_args()
    print('\n', args, '\n')

    conf = oc.load(Path(__file__).parent
                   / f'experiments/interpretability/configs/{args.det}.yaml')
    conf = oc.merge(conf, oc.from_dotlist(args.other))
    print('Configuration:\n', oc.to_yaml(conf))

    if args.save:
        date = datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
        out_dir = Path(__file__).parent / \
            f'experiments/interpretability/results/{args.det}/{date}'
        out_dir.mkdir(parents=True, exist_ok=True)

        with open(out_dir / 'config.yaml', 'w') as f:
            oc.save(config=conf, f=f.name)
    else:
        out_dir = None

    main(args.det, conf, out_dir, debug=args.debug)
