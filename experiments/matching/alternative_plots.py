import pickle
from argparse import ArgumentParser
from pathlib import Path
from math import floor, ceil

import matplotlib.pyplot as plt
import numpy as np


def plot_mma_vs_unc(results, out_dir=None):
    fig, axes = plt.subplots(nrows=2, ncols=3)
    titles = [
        ["Overall (per-seq)", "Illumination (per-seq)", "Viewpoint (per-seq)"],
        ["Overall (all-seqs)", "Illumination (all-seqs)",
         "Viewpoint (all-seqs)"]
    ]

    nlevels = len(results['all_seq']['a_err'][1])
    x = np.arange(1, nlevels + 1)

    # lines configs
    ls = ['--', ':']
    lc = ['b', 'g']
    lb = ['Full', 'Isotropic']

    for i, ti in enumerate(['per_seq', 'all_seq']):
        result_i = results[ti]
        ymin, ymax = 1., 0.

        for j, result_ij in enumerate([result_i['a_err'], result_i['i_err'], result_i['v_err']]):
            ax = axes[i, j]
            title = titles[i][j]

            for k, (ypre, lsi, lci, lbi) in enumerate(zip(result_ij[1:], ls, lc, lb)):
                # per uncertainty-level mean (if scores, reverse it to match the tendency)
                # y = (ypre.mean(axis=1) if k == 0 else ypre[::-1].mean(axis=1))
                y = ypre.mean(axis=1)
                ax.plot(x, y, lw=3, c=lci, ls=lsi, label=lbi)
                ymin, ymax = min(ymin, y.min()), max(ymax, y.max())

            # ymin = floor(ymin * 10) / 10
            # ymax = ceil(ymax * 10) / 10
            ax.set(
                title=title,
                xticks=x,
                # yticks=np.linspace(ymin, ymax, int((ymax - ymin) * 10)),
                # yticks=np.round(np.linspace(ymin, ymax, 4), decimals=1),
                xlim=[x.min(), x.max()],
                # ylim=[ymin, ymax],
                ylabel=("" if (j > 0) else "MMA"),
                xlabel=("" if (j != 1) | (i == 0) else "Uncertainty level")
            )
            if i == 0:
                ax.set(xticklabels=[])
            if j > 0:
                ax.set(yticklabels=[])
            ax.grid()

        # ymin = floor(ymin * 10) / 10
        # ymax = ceil(ymax * 10) / 10
        ymin = 0.50
        ymax = 0.85
        for axi in axes[i, :]:
            axi.set(
                ylim=[ymin, ymax],
                # yticks=np.arange(round(ymin - .05, 1), round(ymax, 1)),
                # yticks=np.linspace(ymin, ymax, int((ymax - ymin) * 10 + 1))
            )
            # axi.yaxis.set_major_formatter('{x:.1f}')

    # locate legend in the first axis:
    axes[0, 0].legend()
    fig.tight_layout()
    plt.show()

    # figure with only overall results

    if out_dir is not None:
        fname = Path(out_dir) / 'mma_vs_unc.pdf'
        fig.savefig(fname, bbox_inches='tight')
    return fig, ax


def final_plot(do_save=False):
    dets = ('d2net', 'r2d2', 'superpoint', 'keynet')

    figsize = (5.0, 3.5)
    # figsize = (5.0, 5.0)

    ls = ['--', ':']
    lc = ['b', 'g']
    lb = ['Full', 'Isotropic']
    lw = 5

    res_dir = base_dir / 'results'

    for i, det in enumerate(dets):
        fig, ax = plt.subplots()

        file_path = base_dir / \
            f'results/matching/{det}/{folds[det]}/results.pkl'
        with open(file_path, 'rb') as f:
            results = pickle.load(f)

        base, res_cov, res_s = results['all_seq']['a_err']

        # #uncertainty levels used.
        nlevels = len(res_cov)
        x = np.arange(1, nlevels + 1)

        # aggregate results over the thresholds.
        base = base.mean()
        res_cov = res_cov.mean(axis=1)
        res_s = res_s.mean(axis=1)

        ax.plot(x, res_cov, lw=lw, c=lc[0], ls=ls[0], label=lb[0])
        ax.plot(x, res_s, lw=lw, c=lc[1], ls=ls[1], label=lb[1])

        ax.legend(
            loc='lower left', title=names[det],
            frameon=True, edgecolor='white', framealpha=1.0,
            title_fontsize=18, fontsize=18)

        # ymin = floor(100 * min(res_cov.min(), res_s.min())) / 100
        # ymax = round(min(res_cov.max(), res_s.max()), 2)
        ymin = min(res_cov.min(), res_s.min())
        ymax = max(res_cov.max(), res_s.max())

        ax.set(
            # xlabel='Uncertainty level',
            # ylabel='mean MMA' if i == 0 else None,
            xticks=x,
            yticks=[
                round(ymin, 2),
                round(ymax, 2)
            ]
        )
        ax.tick_params(axis='both', which='major', labelsize=14)

        fig.set_size_inches(*figsize)
        fig.tight_layout()
        if do_save:
            fig.savefig(res_dir / f'{det}_mean_MMA.pdf', bbox_inches='tight')


if __name__ == "__main__":
    det = 'r2d2'
    base_dir = Path(__file__).parent
    folds = {
        'keynet': '2023-03-07_05-53-01',
        'r2d2': '2023-03-07_00-42-17',
        'superpoint': '2023-03-07_00-41-43',
        'd2net': '2023-03-07_09-13-41'
    }
    names = {
        'superpoint': 'SuperPoint',
        'keynet': 'Key.Net',
        'r2d2': 'R2D2',
        'd2net': 'D2Net',
    }

    file_path = base_dir / f'results/matching/{det}/{folds[det]}/results.pkl'
    assert file_path.is_file()

    with open(file_path, 'rb') as f:
        results = pickle.load(f)
    # raise

    # plot_mma_vs_unc(
    #     results,
    #     # out_dir=file_path.parent
    # )

    final_plot()
