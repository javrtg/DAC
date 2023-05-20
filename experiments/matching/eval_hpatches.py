import sys
from pathlib import Path
from argparse import ArgumentParser
from datetime import datetime
from typing import Dict, Tuple, Optional, Union
import pickle

import numpy as np
from tqdm import tqdm
from tabulate import tabulate
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')

env_path = str(Path(__file__).parents[2])
if env_path not in sys.path:
    print(f'inserting {env_path} to sys.path')
    sys.path.insert(0, env_path)
from datasets.hpatches import HPatches
from detectors.load import load_detector
from utils.plotting import plot_matches, plot_ellipses
from experiments.matching.alternative_plots import plot_mma_vs_unc


def print_summary(results: Dict, latex: bool = False,
                  out_dir: Optional[Union[str, Path]] = None):
    np.set_printoptions(precision=2)

    print('Per-sequence summary')
    rps = results['per_seq']
    headers = ['Method', 'Overall', 'Illumination', 'Viewpoint']
    table = [
        ['raw', rps['a_err'][0], rps['i_err'][0], rps['v_err'][0]],
        ['w. covs.', rps['a_err'][1], rps['i_err'][1], rps['v_err'][1]],
        ['w. scores', rps['a_err'][2], rps['i_err'][2], rps['v_err'][2]]
    ]
    tb_obj_ps = tabulate(table, headers, tablefmt="simple")
    print(tb_obj_ps)

    print('\nAll-sequences summary')
    ras = results['all_seq']
    headers = ['Method', 'Overall', 'Illumination', 'Viewpoint']
    table = [
        ['raw', ras['a_err'][0], ras['i_err'][0], ras['v_err'][0]],
        ['w. covs.', ras['a_err'][1], ras['i_err'][1], ras['v_err'][1]],
        ['w. scores', ras['a_err'][2], ras['i_err'][2], ras['v_err'][2]]
    ]
    tb_obj_as = tabulate(table, headers, tablefmt="simple")
    print(tb_obj_as)

    if out_dir is not None:
        with open(Path(out_dir) / 'summary.txt', 'w') as f:
            f.write('Per-sequence summary:\n')
            f.write(tb_obj_ps)
            f.write('\n\nAll-sequences summary:\n')
            f.write(tb_obj_as)


def mma_plot(results: Dict, out_dir: Optional[Union[str, Path]] = None,
             fsize: Tuple = (0.39 * 25, 0.39 * 16)):
    """MMA plot."""
    fig, axes = plt.subplots(nrows=2, ncols=3)
    x = results['thresholds']
    # plot configs
    titles = [
        ["Overall (per-seq)", "Illumination (per-seq)", "Viewpoint (per-seq)"],
        ["Overall (all-seqs)", "Illumination (all-seqs)",
         "Viewpoint (all-seqs)"]
    ]
    # line styles
    ls = ['-', '--', ':']
    # color per uncertainty level
    nlevels = len(results['all_seq']['i_err'][1])
    unc_c = plt.get_cmap('Blues')(np.linspace(0.3, 1.0, nlevels))
    # color per score level
    sc_c = plt.get_cmap('Greens')(np.linspace(0.3, 1.0, nlevels))
    # all line colors
    lc = ['k', unc_c, sc_c]
    # labels
    lb_cov = [f"Cov. level:{i}" for i in range(nlevels)]
    lb_sc = [f"Score level:{i}" for i in range(nlevels)]
    lb = ['Raw', lb_cov, lb_sc]

    for i in range(2):
        result_i = (results['per_seq'] if (i == 0) else results['all_seq'])

        for j, result_ij in enumerate([result_i['a_err'], result_i['i_err'], result_i['v_err']]):
            ax = axes[i, j]
            title = titles[i][j]

            for y, lsi, lci, lbi in zip(result_ij, ls, lc, lb):
                if len(y.shape) > 1:
                    for yk, lbik, lcik in zip(y, lbi, lci):
                        ax.plot(x, yk, c=lcik, ls=lsi, lw=3, label=lbik)
                else:
                    ax.plot(x, y, c=lci, ls=lsi, lw=3, label=lbi)

            ax.set(
                title=title,
                # xticks=np.round(x, decimals=1),
                yticks=np.round(np.linspace(0.0, 1.0, 6), decimals=1),
                xlim=[x.min(), x.max()],
                ylim=[0., 1.],
                ylabel=("" if (j > 0) else "MMA"),
                xlabel=("" if (j != 1) else "Threshold [px]")
            )
            if i == 0:
                ax.set(xticklabels=[])
            if j > 0:
                ax.set(yticklabels=[])
            ax.grid()

    # legend
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='center left', bbox_to_anchor=(0.8, 0.5))
    fig.tight_layout(rect=[0, 0, 0.8, 1])

    fig.set_size_inches(fsize[0], fsize[1])
    # plt.show()

    if out_dir is not None:
        fname = Path(out_dir) / 'mma_plot.pdf'
        fig.savefig(fname, bbox_inches='tight')
    return fig, ax


def _update_viz_figure(
        im1, im2, kps1, kps2, C1, C2, fig, axes, dist, max_eig,
        max_fsize=None, save_path=None, show=False):
    """Visualize reprojection error and estimated uncertainties."""
    for axi in axes:
        axi.cla()
    # matches with color as a function of the error
    plot_matches(
        im1, im2, kps1, kps2, fig=fig, axi=axes[0],
        plot_kps=False, max_fsize=None, orient='h', scores_color=dist
    )
    # matches with color as a function of the max eigenvalue of the cov.
    plot_matches(
        im1, im2, kps1, kps2, fig=fig, axi=axes[1],
        plot_kps=False, max_fsize=None, orient='h', cmap='inferno',
        scores_color=max_eig
    )
    # plot cov ellipses wit reproj error as color
    plot_ellipses(
        im1, kps1, C1, lib='plt', fig=fig, axi=axes[2], max_fsize=None, n=25,
        scores_color=dist
    )
    plot_ellipses(
        im2, kps2, C2, lib='plt', fig=fig, axi=axes[3], max_fsize=None, n=25,
        scores_color=dist
    )

    if max_fsize is not None:
        # keep the same figure size and aspect ratio
        ar = (fsize := fig.get_size_inches())[1] / fsize[0]
        max_fsize *= 2.54  # cm -> inches
        if fsize[1] > fsize[0]:
            fig.set_size_inches(w=ar * max_fsize, h=max_fsize)
        else:
            fig.set_size_inches(w=max_fsize, h=ar * max_fsize)

    # fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path)
    if show:
        plt.pause(1e-6)
        # plt.show()


def reproj_err_and_match_cov(
        p1: np.array, p2: np.array, S1_inv: np.array, S2_inv: np.array,
        H21: np.array) -> Tuple[np.array, np.array]:
    """Compute:
        1) Euclidean distance between matched keypoints.
        2) Covariance matrix of the match: S1_propagated + S2 (i.e. assuming independence)

    The kps in the reference image, p1, are reprojected as:
        [p1_proj, 1] = y / y[3];    y = H @ [p1, 1]
    Thereby the Jacobian needed to lineraly propagate S1 is:
        J = d(p1_proj) / d(p1) = ( d(p1_proj) / d(y) ) * ( d(y) / d(p1) )
          = J1 * J2

    Args:                                                       Shape
        - p1: keypoints in reference image.                     (2,n)
        - p2: keypoints in target image.                        (2,n)
        - S1_inv: autocorr. matrix (inverse cov.) of p1.        (n,2,2)
        - S2_inv: autocorr. matrix (inverse cov.) of p2.        (n,2,2)
        - H21: ground-truth homography matrix.                  (3,3)

    Return:
        - dist: reprojection errors.                            (n,)
        - S: covariance matrices of the matches.                (n,2,2)
    """
    # get input covariances:
    # S1 = fast2x2inv(S1_inv)
    # S2 = fast2x2inv(S2_inv)
    S1 = np.linalg.inv(S1_inv)
    S2 = np.linalg.inv(S2_inv)

    error2d = (p1_proj_hom := (
        y := H21[:, :2] @ p1 + H21[:, -1:])[:-1] / y[-1:]
    ) - p2
    # euclidean distance error
    dist = np.sqrt(np.einsum('ij, ij -> j', error2d, error2d))

    # J1 (1st term of the chain-rule)
    J1 = np.repeat([[[1., 0., 0.], [0., 1., 0.]]], len(S1_inv), axis=0)
    J1[:, :, -1] = -p1_proj_hom.T
    J1 /= y[2, :, None, None]
    # complete the chain-rule:
    J = J1 @ H21[:, :2]
    # covariance of the match with linear propagation of S1:
    S = (J @ S1 @ J.transpose(0, 2, 1)) + S2
    return dist, S


def eval_hpatches(
        hpatches: HPatches, model,
        ths: np.array = np.linspace(0.1, 10., 10), levels: int = 3,
        debug: bool = False, verbose: bool = False,
        do_viz: bool = False, show: bool = False) -> Dict:
    """Evaluate "model" on "hpatches" subset of sequences. The code is adapted
    from [1]. The matching metric  is adopted from D2Net paper, i.e., the 
    precentage of correctly matched keypoints at the given re-projection error 
    "thresholds". A number of "levels" is considered in order to study the 
    influence of the uncertainty estimates. A difference w.r.t. [1] is that 
    the previous metric is computed not only per sequence (averaging all of 
    them), but also over all the matches of all sequences.
    [1]: https://github.com/GrumpyZhou/image-matching-toolbox

    Args:
        - hpatches: instance of HPatches class
        - model: instance of Detector class type
        - ths: error thresholds in pixels to compute the metric.
        - levels: levels of uncertainty considered.

    Returns:
        - results: Dict with the following structure:
            results = {
                'per_seq':
                    'i_err':
                        List[np.array, np.array, np.array]
                        (the arrays' shapes = [(n_thr,), (levels, n_thr), (levels, n_thr)],
                         containing -> [base_results, cov_results, score_results])
                    'v_err':
                        .
                'all_seq':
                    .
                    .
                'thresholds': thresholds (just to save them)
                }
    """
    # create output folder.
    det = model.__class__.__name__.lower()
    out_dir = (Path(__file__).parent / 'results' / 'matching' / det
               / datetime.today().strftime('%Y-%m-%d_%H-%M-%S'))
    out_dir.mkdir(parents=True, exist_ok=True)

    if do_viz:
        (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    n_thr = len(ths)
    # initialize output
    results = {
        'per_seq': {
            'i_err': [np.zeros((n_thr,)), np.zeros((levels, n_thr)), np.zeros((levels, n_thr))],
            'v_err': [np.zeros((n_thr,)), np.zeros((levels, n_thr)), np.zeros((levels, n_thr))]
        },
        'all_seq': {
            'i_err': [np.zeros((n_thr,)), np.zeros((levels, n_thr)), np.zeros((levels, n_thr))],
            'v_err': [np.zeros((n_thr,)), np.zeros((levels, n_thr)), np.zeros((levels, n_thr))]
        },
        'thresholds': ths
    }

    # registers for all_seq statistics
    max_eigs_hist = {'i': np.empty((0,)), 'v': np.empty((0,))}
    scores_hist = {'i': np.empty((0,)), 'v': np.empty((0,))}
    dist_hist = {'i': np.empty((0,)), 'v': np.empty((0,))}

    if debug:
        cnt = 0

    if do_viz:
        fig = plt.figure(constrained_layout=True)
        axes = [axi for axi in fig.subplot_mosaic("""AA\nBB\nCD""").values()]

    for seq_dir in tqdm(hpatches._seq_dirs):
        if debug:
            if cnt > 10:
                break
            cnt += 1
        sname = seq_dir

        # load reference image and extract kps, desc and estimated_covs
        im1 = hpatches.get(seq_dir, idx=1, return_h=False)
        kps1, desc1, C1 = model(im1)

        # traverse sequence
        for i in range(2, 7):
            im_i, H_i_1 = hpatches.get(seq_dir, idx=i, return_h=True)
            kpsi, desci, Ci = model(im_i)
            # match descriptors between both images
            matches = model.match_descriptors(desc1, desci)

            if len(matches) > 0:
                # matched data
                kps1_m, kpsi_m, C1_m, Ci_m = (
                    kps1[:, matches[:, 0]], kpsi[:, matches[:, 1]],
                    C1[matches[:, 0]], Ci[matches[:, 1]]
                )
                # errors and estimated covariances:
                dist, S = reproj_err_and_match_cov(
                    kps1_m[:-1], kpsi_m[:-1], C1_m, Ci_m, H_i_1)
                # as a measure of estimated uncertainty, retain max eigenvalue:
                max_eig = np.linalg.eigvals(S).max(axis=1)

                # alternative measure -> sum of kp scores:
                scores = kps1_m[-1] + kpsi_m[-1]
                # errors and estimated covariances:
                Cs1_m = np.eye(2) * kps1_m[-1:, None].T
                Csi_m = np.eye(2) * kpsi_m[-1:, None].T
                dist, S = reproj_err_and_match_cov(
                    kps1_m[:-1], kpsi_m[:-1], Cs1_m, Csi_m, H_i_1)
                # as a measure of estimated uncertainty, retain max eigenvalue:
                scores = np.linalg.eigvals(S).max(axis=1)

                if do_viz:
                    fig_path = out_dir / "figures" / f"{sname}_1_{i}.png"
                    _update_viz_figure(
                        im1, im_i, kps1_m[:-1], kpsi_m[:-1], C1_m, Ci_m,
                        fig, axes, dist, max_eig, save_path=fig_path, show=show)

                # --> compute statistics per sequence
                t_err = f'{sname[0]}_err'
                results['per_seq'][t_err][0] += (dist <
                                                 ths[:, None]).mean(axis=1)
                # partition of reproj error in nlevels of [uncertainty, scores]
                nmod = (len(dist) % levels)
                for k, unc_measure in enumerate([max_eig, scores], 1):
                    if len(dist) >= levels:
                        # split error based on uncertainty
                        dist_part = dist[
                            unc_measure.argsort()[:(-nmod or None)].reshape(levels, -1)]
                        # add the statistic for each unc. level
                        results['per_seq'][t_err][k] += (
                            dist_part[:, None] <= ths[:, None]).mean(axis=-1)
                    # # add the leftovers
                    # if nmod > 0:
                    #     results['per_seq'][t_err][k][-1] += (
                    #         dist[idxs[-nmod:]] <= ths[:,None]).mean(axis=1)

                # store data to compute all_seq statistics after:
                max_eigs_hist[f'{sname[0]}'] = np.concatenate(
                    (max_eigs_hist[f'{sname[0]}'], max_eig))
                scores_hist[f'{sname[0]}'] = np.concatenate(
                    (scores_hist[f'{sname[0]}'], scores))
                dist_hist[f'{sname[0]}'] = np.concatenate(
                    (dist_hist[f'{sname[0]}'], dist))

            if verbose:
                if len(matches) == 0:
                    dist = float('inf')
                tqdm.write(
                    f'Scene {sname}, pair 1-{i}, matches: {len(matches)}\n'
                    f'Median matching dist: {np.median(dist):.2f}, <1 pix: '
                    f'{np.mean(dist<=1):.2f}')

    n_i, n_v = hpatches.n_i, hpatches.n_v
    n_a = n_i + n_v

    # average per-seq statistics
    results['per_seq']['a_err'] = []
    for result_i, result_v in zip(
            results['per_seq']['i_err'], results['per_seq']['v_err']):
        results['per_seq']['a_err'].append(
            (result_i + result_v) / (n_a * 5))
        result_i /= (n_i * 5)
        result_v /= (n_v * 5)

    # compute all-seq statistics
    results['all_seq']['a_err'] = []
    results['all_seq']['i_err'][0] = (
        dist_hist['i'] < ths[:, None]).mean(axis=1)
    results['all_seq']['v_err'][0] = (
        dist_hist['v'] < ths[:, None]).mean(axis=1)
    results['all_seq']['a_err'].append(
        (n_i / n_a) * results['all_seq']['i_err'][0]
        + (n_v / n_a) * results['all_seq']['v_err'][0])

    # partition of reproj error in nlevels of [uncertainty, scores]:
    for k, unc_measure in enumerate([max_eigs_hist, scores_hist], 1):
        for t in ['i', 'v']:
            nmod = (len(dist_hist[t]) % levels)
            dist_part = dist_hist[t][
                unc_measure[t].argsort()[:(-nmod or None)].reshape(levels, -1)]
            results['all_seq'][f'{t}_err'][k] = (
                dist_part[:, None] <= ths[:, None]).mean(axis=-1)

        #     if nmod > 0:
        #         results['all_seq'][f'{t}_err'][k][-1] += (
        #             dist_hist[t][idxs[-nmod:]] <= ths[:,None]).mean(axis=1)

        results['all_seq']['a_err'].append(
            (n_i / n_a) * results['all_seq']['i_err'][k]
            + (n_v / n_a) * results['all_seq']['v_err'][k])

    # save results
    with open(out_dir / 'results.pkl', 'wb') as p:
        pickle.dump(results, p)

    # print_summary(results, out_dir=out_dir)
    # mma_plot(results, out_dir)
    plot_mma_vs_unc(results, out_dir)


if __name__ == "__main__":
    parser = ArgumentParser(description='HPatches experiment')
    parser.add_argument(
        '--det', type=str, choices=['d2net', 'keynet', 'superpoint', 'r2d2'], required=True)
    parser.add_argument('--use_full_dataset', action='store_true',
                        help="If False, only the cut of D2Net is used, otherwise all scenes are considered.")
    parser.add_argument('--thresholds', type=np.array, default=np.linspace(0.1, 10., 10),
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
    hpatches = HPatches(not args.use_full_dataset)
    model = load_detector(args.det)

    eval_hpatches(
        hpatches, model, ths=args.thresholds,
        levels=args.uncertainty_levels,
        debug=args.debug, verbose=args.verbose, do_viz=args.do_viz,
        show=args.show)
