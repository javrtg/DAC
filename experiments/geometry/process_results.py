import sys
from pathlib import Path
from typing import Optional, Union
from collections.abc import Sequence
from pprint import pprint

import numpy as np
import numpy.typing as npt
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import rankdata


EFFECTS = {
    "b": r"\textbf",
    "i": r"\textit",
    "u": r"\underline",
    "ub": r"\fontseries{b}\selectfont",
}


def add_effect(value, effect):
    if effect is None:
        return value
    return f"{EFFECTS[effect]}{{{value}}}"


def add_color(value, color):
    if color is None:
        return value
    return f"\\textcolor{{{color}}}{{{value}}}"


def ensure_list(optional_val, maybe_type):
    if optional_val is not None and isinstance(optional_val, maybe_type):
        return [optional_val]
    return optional_val


def numpy2latex(
    data: np.ndarray,
    axis: Optional[int] = None,
    sections_np_split: Optional[npt.ArrayLike] = None,
    sections_slices: Optional[list[slice]] = None,
    best: Optional[Sequence[str]] = None,
    effects: Optional[Sequence[str]] = None,
    colors: Optional[Sequence[str]] = None,
    fmt: Sequence[str] = ".2f",
    out_file: Optional[Union[Path, str]] = None,
):
    """Convert 2d ndarray to a customized latex table based on the values.

    Args:
        data: 2d numpy array.
        axis: axis along comparisons between values will be done.
            Default = None.
        sections_np_split: used to split input data into independent arrays,
            according to np.split(). This argument is equivalent to
            indices_or_sections arg in np.split(). Must be None if
            sections_slices is given (and viceversa). Default = None.
        sections_slices: list of slices is given, 'data' will be splitted
            according to them. Must be None if 'sections_np_split' is given
            (and viceversa). Default = None.
        best: a sequence of str, indicating what values are best:
            'h' for higher and 'l' for lower. If a unique str is given as
            input, then it is used for all the data. Default = None.
        effects: what to do with the best values. Currently implemented:
            'b' (bold), 'ub' (non-extended bold), 'i' (italic) and 'u'
            (underlined). When a list of n of them is given, then they are
            applied in corresponding order to the n-'best' values.
            Default = None.
        colors: Similar to 'effects' i.e. colors to apply to the (n-)best
            values. Default = None.
        fmt: format for the values. Default = '.2f'.
        out_file: file in which the result is written. When None, it is written
            to sys.stdout. Default = None.
    """
    assert len(data.shape) == 2, "'data' must be a 2d-ndarray."
    assert axis in (0, 1), "'axis' can be only '0' (rows) or '1' (cols)"
    assert all(
        i in ("h", "l") for i in best
    ), "'best' can only be Sequence[str] with the values: {'h', 'l'}"
    if not isinstance(best, str):
        assert len(best) == data.shape[axis ^ 1]  # axis ^ 1 swaps 0 <-> 1
    if effects is not None:
        assert all(
            e in EFFECTS or e is None for e in effects
        ), f"'effects' {effects} not valid. Valid ones are: {EFFECTS}"
    assert (
        sections_np_split is None or sections_slices is None
    ), "Only one partition instruction is allowed"

    # fast processing if there is no need for scanning the values.
    if axis is None or effects is None and colors is None:
        print(
            "\nPrinting/saving as is. To scan and apply 'effects' and "
            "'colors', specify, at least, one of them, as well as the "
            "'axis'.\n"
        )
        np.savetxt(
            out_file if out_file else sys.stdout,
            data,
            delimiter=" & ",
            fmt=f"%{fmt}",
        )
        return

    # invert sign of the values where higher means best.
    if "h" in best:
        data_ = data.copy()

        if isinstance(best, str):
            data_ *= -1

        else:
            where_h = [i for i, h_or_l in enumerate(best) if h_or_l == "h"]
            # indexing depends on the axis.
            if axis == 0:
                data_[:, where_h] *= -1
            else:
                data_[where_h] *= -1
    else:
        data_ = data

    effects = ensure_list(effects, str)
    colors = ensure_list(colors, str)
    e_len = len(effects) if effects else 0
    c_len = len(colors) if colors else 0

    # rank values.
    if sections_np_split is not None:
        # split input in multiple arrays to scan in each of them individually.
        data_split = np.array_split(data_, sections_np_split, axis=axis)
        ranking = -1 + np.concatenate(
            [rankdata(arr, "min", axis=axis) for arr in data_split], axis=axis
        )

    elif sections_slices is not None:
        sections_slices = ensure_list(sections_slices, slice)

        # Initialization s.t. values not within slices are ignored.
        ranking = np.ones(data.shape, dtype=np.int64) * max(e_len, c_len)

        # treat each slice of input array individually.
        slc_tmp = [slice(None)] * 2  # equivalent to numpy's [:, :]
        for slice_ in sections_slices:
            slc_tmp[axis] = slice_
            ranking[tuple(slc_tmp)] = -1 + rankdata(
                data[tuple(slc_tmp)], "min", axis=axis
            )

    else:
        ranking = -1 + rankdata(data_, "min", axis=axis)

    out = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val_str = f"{data[i, j]:{fmt}}"
            rank_ij = ranking[i, j]

            if rank_ij < e_len:
                val_str = add_effect(val_str, effects[rank_ij])

            if rank_ij < c_len:
                val_str = add_color(val_str, colors[rank_ij])

            out.append(f"{val_str} & ")

        out[-1] = out[-1][:-2] + "\n"

    out = "".join(out)

    if out_file:
        with open(out_file, "w") as f:
            f.write(out)
    else:
        print(out)


def gather_results(p_data, seqs, dets, pnps, do_tabs=False):
    """Collect results in a dict and write table to disk if requested"""
    l_dets = len(dets)
    l_pnps = len(pnps)
    l_seqs = len(seqs)

    if do_tabs:
        mean_tab = np.empty(((l_seqs + 1) * l_dets, l_pnps * 2))
        med_tab = np.empty(((l_seqs + 1) * l_dets, l_pnps * 2))

    ntotal = np.zeros(l_dets, dtype=np.int32)
    results = {}

    for i, seq in enumerate(seqs):
        p_seq = p_data / seq
        results[seq] = {}

        for j, det in enumerate(dets):
            p_det = p_seq / det
            results[seq][det] = {}

            for k, pnp in enumerate(pnps):
                p_pnp = p_det / pnp
                with np.load(p_pnp / "stats.npz") as res:
                    results[seq][det][pnp] = {**res}

                    if do_tabs:
                        mean_tab[l_dets * i + j, 2 * k] = res["e_rot_opt_mean"]
                        mean_tab[l_dets * i + j, 2 * k + 1] = res[
                            "e_t_opt_mean"
                        ]

                        med_tab[l_dets * i + j, 2 * k] = res[
                            "e_rot_opt_median"
                        ]
                        med_tab[l_dets * i + j, 2 * k + 1] = res[
                            "e_t_opt_median"
                        ]

            ntotal[j] += len(results[seq][det][pnp]["e_rot"])

    # gather overall errors.
    results["all"] = {}

    for j, det in enumerate(dets):
        results["all"][det] = {}

        erot_per_det = np.empty((ntotal[j], l_pnps))
        et_per_det = np.empty((ntotal[j], l_pnps))

        erot_per_det_noopt = np.empty((ntotal[j], l_pnps))
        et_per_det_noopt = np.empty((ntotal[j], l_pnps))

        for k, pnp in enumerate(pnps):
            n = 0
            for seq in seqs:
                lseq = len(results[seq][det][pnp]["e_rot"])

                erot_per_det[n: n + lseq, k] = results[seq][det][pnp][
                    "e_rot_opt"
                ]
                et_per_det[n: n + lseq, k] = results[seq][det][pnp]["e_t_opt"]

                erot_per_det_noopt[n: n + lseq, k] = results[seq][det][pnp][
                    "e_rot"
                ]
                et_per_det_noopt[n: n + lseq, k] = results[seq][det][pnp][
                    "e_t"
                ]

                n += lseq

            results["all"][det][pnp] = {
                "e_rot": erot_per_det_noopt[:, k],
                "e_t": et_per_det_noopt[:, k],
                "e_rot_opt": erot_per_det[:, k],
                "e_t_opt": et_per_det[:, k],
            }

        if do_tabs:
            mean_det_erot = np.mean(erot_per_det, axis=0)
            med_det_erot = np.median(erot_per_det, axis=0)

            mean_det_et = np.mean(et_per_det, axis=0)
            med_det_et = np.median(et_per_det, axis=0)

            mean_tab[-l_dets + j, ::2] = mean_det_erot
            mean_tab[-l_dets + j, 1::2] = mean_det_et
            med_tab[-l_dets + j, ::2] = med_det_erot
            med_tab[-l_dets + j, 1::2] = med_det_et

    if do_tabs:
        # change units to 10*deg and cm.
        mean_tab[:, ::2] *= 1e1
        mean_tab[:, 1::2] *= 1e2

        med_tab[:, ::2] *= 1e1
        med_tab[:, 1::2] *= 1e2

        # write mean and median tables.
        numpy2latex(
            mean_tab,
            axis=1,
            sections_slices=[slice(0, None, 2), slice(1, None, 2)],
            best="l",
            effects="ub",
            colors=None,
            fmt=".2f",
            out_file=p_data / "mean.txt",
        )

        numpy2latex(
            med_tab,
            axis=1,
            sections_slices=[slice(0, None, 2), slice(1, None, 2)],
            best="l",
            effects="ub",
            colors=None,
            fmt=".2f",
            out_file=p_data / "median.txt",
        )

    return results


def cumulative_plots(ax, err, labels, ls, lc, lw=2):
    err_sorted = np.sort(err, axis=1)

    # recall: [1/ne, 2/ne ... 1]
    ne = err.shape[1]
    y = np.linspace(1, ne, ne) / ne

    for e, label, st, c in zip(err_sorted, labels, ls, lc):
        ax.plot(e, y, c=c, ls=st, lw=lw, label=label)


def cumulative_figs_per_seq_det(results, seq, det, pnps, kw_rot, kw_t, ls, lc):
    l_pnps = len(pnps)

    # initialize empirical cumulative curves.
    l_seq = len(results[seq][det]["epnp"]["e_rot"])
    e_rot = np.empty((l_pnps, l_seq))
    e_t = np.empty((l_pnps, l_seq))

    for k, pnp in enumerate(pnps):
        e_rot[k] = results[seq][det][pnp][kw_rot]  # * 1e1 if 0.1deg
        e_t[k] = results[seq][det][pnp][kw_t] * 1e2  # cm

    fig, axes = plt.subplots(nrows=1, ncols=2, sharey=True)
    cumulative_plots(axes[0], e_rot, labels, ls, lc)
    cumulative_plots(axes[1], e_t, labels, ls, lc)

    axes[0].set(
        # xlabel="Rotation error [0.1$\\times$degrees]",
        xlabel="Rotation error [degrees]",
        xlim=xlim_rot,
        ylim=ylim,
        # xticks=np.linspace(0, 10, 6, dtype=int),
        yticks=np.linspace(0.0, 1.0, 11),
    )
    axes[1].set(
        xlabel="Translation error [cm]",
        xlim=xlim_t,
        ylim=ylim,
        # xticks=np.linspace(0, 10, 6, dtype=int)
    )
    axes[0].set_xlabel("Rotation error [degrees]", fontsize=14)
    axes[1].set_xlabel("Translation error [cm]", fontsize=14)

    axes[0].text(
        # 0.99, 0.01, f'Recall\n$_{{\\regular{{{det}}}}}$',
        0.95,
        0.05,
        f"{det}",
        ha="right",
        va="bottom",
        c="k",
        size="xx-large",
        transform=axes[0].transAxes,
        backgroundcolor="white",
    )

    axes[1].legend(
        loc="lower right",
        frameon=True,
        edgecolor="white",
        framealpha=1.0,
        fontsize=12,
    )

    for ax in axes:
        ax.yaxis.set_major_formatter("{x:.2f}")

    axes[0].grid(True)
    axes[1].grid(True)

    axes[0].tick_params(axis="both", which="major", labelsize=14)
    axes[1].tick_params(axis="both", which="major", labelsize=14)

    return fig, axes


def cumulative_figs(
    results,
    seqs,
    dets,
    pnps,
    labels,
    base_dir,
    xlim_rot=(4.0, 20.0),
    xlim_t=(0.0, 10.0),
    ylim=(0.6, 1.0),
    do_opt=True,
    do_noopt=False,
    do_save=True,
    interactive=False,
    figsize=None,
):

    if interactive:
        matplotlib.use("qtagg")
    else:
        matplotlib.use("agg")

    seqs = seqs + ["all"]

    ls = ["-", "-", "--", "-", "--"]
    lc = ["k", "royalblue", "royalblue", "limegreen", "limegreen"]

    base_dir = base_dir / "figs"
    base_dir.mkdir(exist_ok=True)

    for det in dets:
        for seq in seqs:

            # if seq == 'all':

            if do_noopt:
                fig, axes = cumulative_figs_per_seq_det(
                    results, seq, det, pnps, "e_rot", "e_t", ls, lc
                )

                if figsize is not None:
                    fig.set_size_inches(*figsize)

                fig.tight_layout()

                if do_save:
                    fname = base_dir / f"ecdf_{seq}_{det}"
                    fig.savefig(fname.with_suffix(".png"), bbox_inches="tight")
                    fig.savefig(fname.with_suffix(".pdf"), bbox_inches="tight")

            if do_opt:
                fig, axes = cumulative_figs_per_seq_det(
                    results, seq, det, pnps, "e_rot_opt", "e_t_opt", ls, lc
                )

                if figsize is not None:
                    fig.set_size_inches(*figsize)

                fig.tight_layout()

                if do_save:
                    fname = base_dir / f"ecdf_{seq}_{det}_opt"
                    fig.savefig(fname.with_suffix(".png"), bbox_inches="tight")
                    fig.savefig(fname.with_suffix(".pdf"), bbox_inches="tight")

            if not interactive:
                plt.close("all")


def percent_at_t_rot_ths(err_rot, err_t, th_rot, th_t):
    nt = len(err_t)
    assert nt == len(err_rot)
    return np.sum((err_rot < th_rot) & (err_t < th_t)) / nt


def auc_at_th(err, th, is_sorted=False):
    """AUC@th given array of estimation errors"""
    assert err.ndim == 1
    err_sorted = err if is_sorted else np.sort(err)

    # recall: [1/ne, 2/ne ... 1]
    n = len(err)
    recall = np.linspace(1, n, n) / n

    idx = np.searchsorted(err_sorted, th)

    recall_for_auc = np.concatenate((recall[:idx], [recall[idx - 1]]))
    error_for_auc = np.concatenate((err_sorted[:idx], [th]))

    return np.trapz(recall_for_auc, error_for_auc) / th


def compute_aucs_pnp_exp(
    results,
    seqs,
    dets,
    pnps,
    labels,
    ths_rot=[1, 5.0],
    ths_t=[5.0, 10.0],
    save=False,
):
    """Compute AUC given errors and thresholds"""

    if "all" not in seqs:
        seqs = seqs + ["all"]

    aucs = {"ths_rot": ths_rot, "ths_t": ths_t}

    percents = {"ths_rot": ths_rot, "ths_t": ths_t}
    updt_percents = None not in (ths_rot, ths_t) and len(ths_rot) == len(ths_t)

    # compute aucs independently for translation and rotation.
    for seq in seqs:
        aucs[seq] = {}
        aucs_s = aucs[seq]

        percents[seq] = {}
        ps = percents[seq]

        res_s = results[seq]

        for det in dets:
            aucs_s[det] = {}
            aucs_sd = aucs_s[det]

            ps[det] = {}
            psd = ps[det]

            res_sd = res_s[det]

            for label, pnp in zip(labels, pnps):
                aucs_sd[label] = {}
                aucs_sdp = aucs_sd[label]

                psd[label] = {}
                psdl = psd[label]

                res_sdp = res_sd[pnp]

                if ths_rot is not None:
                    e_sorted = np.sort(res_sdp["e_rot"])
                    aucs_sdp["rot"] = [
                        auc_at_th(e_sorted, th, True) for th in ths_rot
                    ]

                    e_sorted = np.sort(res_sdp["e_rot_opt"])
                    aucs_sdp["rot_opt"] = [
                        auc_at_th(e_sorted, th, True) for th in ths_rot
                    ]

                if ths_t is not None:
                    e_sorted = np.sort(res_sdp["e_t"])
                    aucs_sdp["t"] = [
                        auc_at_th(e_sorted, th, True) for th in ths_t
                    ]

                    e_sorted = np.sort(res_sdp["e_t_opt"])
                    aucs_sdp["t_opt"] = [
                        auc_at_th(e_sorted, th, True) for th in ths_t
                    ]

                if updt_percents:
                    psdl["noopt"] = [
                        percent_at_t_rot_ths(
                            res_sdp["e_rot"], res_sdp["e_t"], th_rot, th_t
                        )
                        for th_rot, th_t in zip(ths_rot, ths_t)
                    ]

                    psdl["opt"] = [
                        percent_at_t_rot_ths(
                            res_sdp["e_rot_opt"],
                            res_sdp["e_t_opt"],
                            th_rot,
                            th_t,
                        )
                        for th_rot, th_t in zip(ths_rot, ths_t)
                    ]

    return aucs, percents


if __name__ == "__main__":

    # dataset and sequences.
    data = "tum_rgbd"
    seqs = ["360", "desk", "desk2", "floor", "xyz"]
    seqs = ["360", "rpy", "xyz"]
    seqs = ["freiburg1_" + i for i in seqs]

    # data = 'kitti_pnp'
    # seqs = ['00', '01', '02']
    # seqs = ['00']

    # detectors.
    dets = ["d2net", "keynet", "r2d2", "superpoint"]
    # dets = ['sp2']

    # pnp models.
    pnps = ["epnp", "epnpu_2d", "epnpu_2d_s", "epnpu_2d3u", "epnpu_2d3u_s"]
    labels = ["none", "2D-full", "2D-iso", "2D-3D-full", "2D-3D-iso"]

    # gather results.
    p_data = Path(__file__).parent / "results" / data
    results = gather_results(p_data, seqs, dets, pnps, do_tabs=True)

    # --- cumulative error plots.
    if data == "kitti_pnp":
        xlim_rot = (0.0, 5.0)
        xlim_t = (100.0, 1000.0)
        ylim = (0.8, 1.0)

        # for auc.
        ths_rot = [1.0, 5.0]  # degrees
        ths_t = [100.0, 500.0]  # cm

    elif data == "tum_rgbd":
        xlim_rot = (0.0, 5.0)
        xlim_t = (0.0, 20.0)
        ylim = (0.8, 1.0)

        # for auc.
        ths_rot = [1.0, 5.0]  # degrees
        ths_t = [10.0, 50.0]  # cm

    else:
        raise ValueError

    cumulative_figs(
        results,
        seqs,
        dets,
        pnps,
        labels,
        p_data,
        xlim_rot=xlim_rot,
        xlim_t=xlim_t,
        ylim=ylim,
        do_opt=True,
        do_noopt=True,
        do_save=True,
        interactive=False,
        figsize=(9.0, 2.9),
    )

    aucs, percents = compute_aucs_pnp_exp(
        results,
        seqs,
        dets,
        pnps,
        labels,
        ths_rot=ths_rot,
        ths_t=ths_t,
    )
    pprint(aucs)
    pprint(percents)
