from typing import Optional, Union
from pathlib import Path

import numpy as np
import cv2
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import LineCollection, EllipseCollection
import plotly.graph_objects as go
import plotly.io as pio
# pio.renderers.default = 'pdf'
pio.renderers.default = 'browser'


def plt3d_real_scale(ax: plt.Axes):
    """ Set a real scale in a matplotlib plot """
    limits = [ax.get_xlim(), ax.get_ylim(), ax.get_zlim()]
    ax.set_box_aspect(np.ptp(limits, axis=1))


def savefig_nomargin(
        fig, ax, fname,
        top=1, bottom=0, right=1, left=0, wspace=0, hspace=0,
        margins=(0, 0),
        pad_inches=0,
        axis_off=True,
        no_ticks=True
):
    """ save figure deleting all margins """
    # remove spines (if ticks and frame are not needed, this should be True).
    if axis_off:
        ax.set_axis_off()

    fig.subplots_adjust(
        top=top, bottom=bottom, right=right, left=left,
        hspace=hspace, wspace=wspace)

    # set axes limits according to margins. If (0, 0) they'll fit the data.
    if margins is not None:
        ax.margins(*margins)

    # remove ticks of the axes.
    if no_ticks:
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())

    fig.savefig(fname, bbox_inches='tight', pad_inches=pad_inches)


def plotly_img_to_T(
        fig, im, Twc, K,
        downsamp_factor=1.0,
        depth=1.0,
        n_colors=32, n_training_pixels=5000):
    """Show RGB image in 3D-plotly figure according to transformation Twc."""
    fx, cx = K[0, 0], K[0, 2]
    fy, cy = K[1, 1], K[1, 2]

    rows, cols = im.shape[:2]

    im_aux = im
    if downsamp_factor != 1.0:
        # downsample the number of pixels to plot.
        im_aux = cv2.resize(
            im, dsize=None,
            fx=downsamp_factor, fy=downsamp_factor,
            interpolation=cv2.INTER_NEAREST)

    if len(im_aux.shape) == 2:
        im_aux = cv2.cvtColor(im_aux, cv2.COLOR_GRAY2RGB)

    rows_aux, cols_aux = im_aux.shape[:2]
    n = rows_aux * cols_aux

    # pixel rays.
    x, y = np.meshgrid(
        np.linspace(0, cols - 1, cols_aux),
        np.linspace(0, rows - 1, rows_aux))
    x = (x - cx) / fx
    y = (y - cy) / fy

    # given a certain depth, express the image w.r.t. world reference.
    coords = Twc[:-1, :-1] @ (depth * np.concatenate((
        x.reshape(1, n), y.reshape(1, n), np.ones((1, n))
    ))) + Twc[:-1, -1:]

    I, J, K, tri_color_intensity, pl_colorscale = _mesh_data(
        im_aux, n_colors, n_training_pixels)

    fig.add_mesh3d(
        x=coords[0], y=coords[1], z=coords[2],
        i=I, j=J, k=K,
        intensity=tri_color_intensity,
        intensitymode="cell",
        colorscale=pl_colorscale,
        showscale=False)


def plotly_heatmap_to_T(
        fig, hm, Twc, K,
        cmap_im='gray', cmin=0.0, cmax=255.0,
        downsamp_factor=1.0,
        depth=1.0,
        show_colorbar=False):
    """Show heatmap in 3D-plotly figure according to transform. Twc."""
    fx, cx = K[0, 0], K[0, 2]
    fy, cy = K[1, 1], K[1, 2]

    rows, cols = hm.shape[:2]

    hm_aux = hm
    if downsamp_factor != 1.0:
        # downsample the number of pixels to plot.
        hm_aux = cv2.resize(
            hm, dsize=None,
            fx=downsamp_factor, fy=downsamp_factor,
            interpolation=cv2.INTER_NEAREST)

    rows_aux, cols_aux = hm_aux.shape[:2]
    n = rows_aux * cols_aux

    # pixel rays.
    x, y = np.meshgrid(
        np.linspace(0, cols - 1, cols_aux),
        np.linspace(0, rows - 1, rows_aux))
    x = (x - cx) / fx
    y = (y - cy) / fy

    # given a certain depth, express the image w.r.t. world reference.
    coords = Twc[:-1, :-1] @ (depth * np.concatenate((
        x.reshape(1, n), y.reshape(1, n), np.ones((1, n))
    ))) + Twc[:-1, -1:]

    x = coords[0].reshape(x.shape)
    y = coords[1].reshape(x.shape)
    z = coords[2].reshape(x.shape)

    fig.add_surface(
        x=x, y=y, z=z,
        surfacecolor=hm_aux,
        colorscale=cmap_im,
        cmin=cmin, cmax=cmax,
        showscale=show_colorbar
    )


def plotly_add_frustum(
        fig: go.Figure,
        Twc: np.ndarray,
        K: np.ndarray = np.array(
            [[500., 0., 500.], [0., 500., 500.], [0., 0., 1.]]),
        depth: float = 1.0,
        color: str = "royalblue",
        im: Optional[np.ndarray] = None,
        downsamp_factor: float = 1.0,
        make_coordsystem: bool = True,
        cmap_im: Optional[str] = None,
        cmin: Optional[float] = None,
        cmax: Optional[float] = None,
        show_colorbar: bool = False,
        add_peak: bool = True
):
    """ add a camera frustum to a plotly figure"""
    fx, cx = K[0, 0], K[0, 2]
    fy, cy = K[1, 1], K[1, 2]

    cnx = cx / fx
    cny = cy / fy

    # frustum in cam reference.
    if im is None:
        points = np.array([
            [-cnx, cnx, cnx, -cnx],
            [-cny, -cny, cny, cny],
            [1.0, 1.0, 1.0, 1.0]])
    else:
        # adapt frustum to the resolution of the input image.
        rows, cols = im.shape[:2]
        points = np.array([
            [-cnx, (cols - cx) / fx, (cols - cx) / fx, -cnx],
            [-cny, -cny, (rows - cy) / fy, (rows - cy) / fy],
            [1.0, 1.0, 1.0, 1.0]])

    if add_peak:
        points = np.concatenate((
            points, np.array([[0, 0], [-1.3 * cny, 0], [1, 0]])),
            axis=1
        )
        order = [0, 1, 2, 3, 0, 5, 3, 2, 5, 1, 4, 0]

    else:
        # add optical center only.
        points = np.concatenate((
            points, np.zeros((3, 1))),
            axis=1
        )
        order = [0, 1, 2, 3, 0, 4, 3, 2, 4, 1]

    points *= depth

    # add image to frustum
    if im is not None:

        if cmap_im is not None:
            # image to plot is assumed to be a heatmap.
            if len(im.shape) > 2:
                raise ValueError(
                    "cmap_im has been especified. Thereby a 2D heatmap (H, W) "
                    f"was expected. However, a {im.shape} array was received.")

            plotly_heatmap_to_T(
                fig, im, Twc, K,
                cmap_im=cmap_im, cmin=cmin, cmax=cmax,
                downsamp_factor=downsamp_factor,
                depth=depth,
                show_colorbar=show_colorbar)

        else:
            if len(im.shape) == 2:
                im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)

            plotly_img_to_T(
                fig, im, Twc, K,
                downsamp_factor=downsamp_factor,
                depth=depth,
                n_colors=32, n_training_pixels=5000)

    # add camera frustum to the figure.
    points = Twc[:-1, :-1] @ points + Twc[:-1, -1:]
    lines = np.array([points[:, i] for i in order])

    fig.add_trace(
        go.Scatter3d(
            x=lines[:, 0],
            y=lines[:, 1],
            z=lines[:, 2],
            mode='lines',
            line=dict(
                width=5,
                color=color))
    )

    if make_coordsystem:
        # if specified, add coordinate system.
        plotly_3dcoord(
            fig,
            Twc,
            length=0.5 * depth,
            lw=5.0,
            colors=['red', 'green', 'blue'],
            make_arrows=True
        )


def plotly_3dcoord(
        fig: go.Figure, Twc: np.ndarray,
        length: float = 0.1, lw: float = 10.0,
        colors=['red', 'green', 'blue'],
        make_arrows=True
):
    # decompose.
    R, t = Twc[:-1, :-1], Twc[:-1, -1]

    # extreme points of the axes.
    axes = np.repeat(t[None], 6, axis=0)
    axes[1::2] += length * R.T
    axes = axes.reshape(3, 2, 3)  # reshape by axis

    # add axes to figure.
    for axis, color in zip(axes, colors):
        fig.add_trace(
            go.Scatter3d(
                x=axis[:, 0],
                y=axis[:, 1],
                z=axis[:, 2],
                mode='lines',
                line=dict(
                    width=lw,
                    color=color,
                ),
                showlegend=False)
        )

    if make_arrows:
        # origin and direction.
        xyz_cone = axes[:, 1]
        uvw_cone = (0.6 * length) * R.T

        # color for each cone.
        colors_cones = [
            [[0, 'rgb(255,0,0)'], [1, 'rgb(255,0,0)']],
            [[0, 'rgb(0,255,0)'], [1, 'rgb(0,255,0)']],
            [[0, 'rgb(0,0,255)'], [1, 'rgb(0,0,255)']]
        ]

        # add each cone.
        for xyz, uvw, color in zip(xyz_cone, uvw_cone, colors_cones):
            fig.add_trace(
                go.Cone(
                    x=[xyz[0]],
                    y=[xyz[1]],
                    z=[xyz[2]],
                    u=[uvw[0]],
                    v=[uvw[1]],
                    w=[uvw[2]],
                    sizemode="absolute",
                    showlegend=False,
                    showscale=False,
                    colorscale=color)
            )


def plotly_add_3d_points(
        fig: go.Figure,
        p: np.ndarray,
        color: Union[str, list[str]] = 'royalblue',
        colorscale=None,
        s=5
):
    fig.add_trace(
        go.Scatter3d(
            x=p[0], y=p[1], z=p[2],
            marker=dict(
                size=s,
                color=color,
                colorscale=colorscale,
            ),
            mode="markers"
        ),
    )


def plotly_add_ellipsoids(
        fig: go.Figure,
        covs: np.ndarray,
        centers: Optional[np.ndarray] = None,
        color: Union[str, list[str]] = 'royalblue',
        scale: float = 1,
        set_scale_w: str = 'median',  # median, max, mean, etc.
        npoints: int = 100
):
    """ add 3D ellipsoids to plotly figure 

    Args:
        fig: go.Figure instance
        covs: (n, 3, 3) or (3, 3) array with cov matrices encoding the axes.
        centers: (n, 3, 1) or (3, 1) location for each ellipsoid
        color: colors for each ellipsoid.
        scale: size along the axis of highest variance according to 
            `set_scale_w`.
        set_scale_w: ellipse used to set the scale, e.g. 'median' -> 
            median ellipse will have the size corresponding to `scale`.
    """
    assert covs.shape[-2:] == (3, 3)
    if centers is not None:
        assert centers.shape[-2:] == (3, 1)

    if len(covs.shape) == 2:
        covs = covs[None]
    if len(centers.shape) == 2:
        centers = centers[None]

    # create base unit sphere.
    u = np.linspace(0, 2 * np.pi, npoints)
    v = np.linspace(0, np.pi, npoints)
    sv = np.sin(v)
    sphere = np.concatenate((
        (np.cos(u)[:, None] * sv[None]).reshape(1, -1),
        (np.sin(u)[:, None] * sv[None]).reshape(1, -1),
        np.repeat(np.cos(v)[None], npoints, 0).reshape(1, -1),
    ))

    # use largest eigenvalue of the covariances to set the scales.
    f = scale / getattr(np, set_scale_w)(np.linalg.eigvals(covs).max(axis=1))
    covs = f * covs

    # transform points of the unit sphere with the covs to form ellipsoids.
    ellipsoids = covs @ sphere
    if centers is not None:
        ellipsoids += centers

    # add them to plotly figure.
    if isinstance(color, str):
        data = [go.Mesh3d(
            x=ell[0], y=ell[1], z=ell[2], alphahull=0, opacity=0.5, color=color
        ) for ell in ellipsoids]

    else:
        data = [go.Mesh3d(
            x=ell[0], y=ell[1], z=ell[2], alphahull=0, opacity=0.5, color=ci
        ) for ell, ci in zip(ellipsoids, color)]

    fig._data += data


def plot_ellipses(
        im: np.ndarray,
        kps: np.ndarray,
        C: np.ndarray,
        lib: str = 'cv2',
        c: Union[tuple, np.ndarray] = (122, 21, 21),
        set_scale_w: str = 'median',
        scale: float = 50.,
        do_filtering: bool = True,
        n: Optional[int] = None,
        lw: Union[float, int] = 3,
        radius: Union[float, int] = 2,
        fig: Optional[plt.Figure] = None,
        axi: Optional[plt.Axes] = None,
        plot_extremes: bool = False,
        max_fsize: Optional[Union[float, int]] = 25.0,
        scores_color: Optional[np.ndarray] = None,  # e.g. reprojection error
        scores_lims: list[float] = [0.0, 10.0],
        cmap: Optional[str] = None,
        cmap_im: Optional[str] = None,
        cmap_im_max: Optional[float] = 255.0,
        cmap_im_min: Optional[float] = 0.0,
        return_selected: bool = False,
        eps: float = 1e-10,
        seed: int = 0
):
    """ Make figure with up-to-scale covariance ellipses

    Args:
        im: background image.
        c: color of the ellipses (array or just a RGB tuple).
        set_scale_w: ellipse used to set the scale, e.g. 'median' -> 
            median ellipse will have the size corresponding to `scale`.
        scale: size along the axis of highest variance according to 
            `set_scale_w`.
        do_filtering: remove ellipses w. highest 
            variance > 3 times the one of set_scale_w.
        n: number of ellipses to plot. Selected randomly according to `seed`.
        lw: thickness of ellipses.
        radius: size of ellipses centers.

    Returns:
        if method == 'cv2': BGR image with inpainted ellipse.
        if method == 'plt': figure and axes instance of matplotlib.
    """
    kps = kps.T  # input expected to have shape (2,n)
    assert kps.shape[1] == 2
    assert C.shape[-2:] == (2, 2)
    lib = lib.lower()
    set_scale_w = set_scale_w.lower()
    assert lib in ['cv2', 'plt']
    assert set_scale_w in ['max', 'median', 'mean']

    if lib == 'cv2':
        if len(im.shape) == 3:
            im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

        if isinstance(c, tuple) or isinstance(c, np.ndarray) and len(c.shape) == 1:
            c = c[::-1]
        else:
            c = c[:, ::-1]

    else:
        if isinstance(c, tuple):
            c = tuple([ci / 255 for ci in c])
        elif isinstance(c, np.ndarray):
            c = c / 255
        else:
            raise ValueError

    # eigen decomposition
    eigv, eigV = np.linalg.eig(C)
    eigv = np.clip(eigv, eps, None)
    eigv = 1.0 / eigv  # C is autocorr matrix (inverse of cov matrix)

    if return_selected:
        # initialize mask.
        keep = np.ones(len(kps)).astype(np.bool_)

    if do_filtering:
        # apply median absolute deviation filtering of too extreme ellipses
        thr_mad = 10.  # 1.48 for gaussian
        mad = np.median(
            np.abs((vals := np.max(eigv, axis=1)) - (median := np.median(vals))))
        keep_f = (vals <= (median + thr_mad * mad))
        # keep_f &= (vals >= (median - 1.48*mad))
        eigv = eigv[keep_f]
        eigV = eigV[keep_f]
        kps = kps[keep_f]

        if not isinstance(c, tuple):
            c = [ci for ci, ki in zip(c, keep_f) if ki]

        if scores_color is not None:
            scores_color = scores_color[keep_f]

        if return_selected:
            keep[keep] = keep_f

    if plot_extremes:
        ext_c = [230, 230, 18]  # yellow
        ext_c = (tuple(ext_c[::-1]) if lib
                 == 'cv2' else tuple([ci / 255 for ci in ext_c]))
        ext_idx = [
            np.unravel_index(eigv.argmax(), eigv.shape)[0],
            eigv.max(axis=1).argmin()
        ]
        ext_eigv = eigv[ext_idx]
        ext_eigV = eigV[ext_idx]
        ext_kps = kps[ext_idx]

    if n is not None:
        if n < len(kps):
            # plot only n ellipses
            np.random.seed(seed)
            keep_n = np.random.choice(len(kps), (n,), replace=False)

            if return_selected:
                keep[keep] = [i in keep_n for i in range(len(kps))]

            eigv = eigv[keep_n]
            eigV = eigV[keep_n]
            kps = kps[keep_n]

            if not isinstance(c, tuple):
                c = [c[ki] for ki in keep_n]

            if scores_color is not None:
                scores_color = scores_color[keep_n]

    # scale factor
    if lib == 'plt':
        scale *= 2  # To match openCV -> half of the axes size equals scale
    if set_scale_w == 'median':
        f = scale / np.median(eigv.max(axis=1))
    elif set_scale_w == 'mean':
        f = scale / np.mean(eigv.max(axis=1))
    else:
        f = scale / np.max(eigv)
    eigv = f * eigv

    if plot_extremes:
        ext_eigv *= f
        ext_angles = np.arctan2(
            ext_eigV[:, 1, 0], ext_eigV[:, 0, 0]) * (180 / np.pi)

    # angles of the ellipses
    angles = np.arctan2(eigV[:, 1, 0], eigV[:, 0, 0]) * (180 / np.pi)

    if lib == 'cv2':
        c_isiter = isinstance(c, np.ndarray) and len(c.shape) > 1
        if c_isiter:
            c = c.tolist()

        for i, (kp, eigvi, angle) in enumerate(zip(kps, eigv, angles)):
            ci = c[i] if c_isiter else c

            center = np.round(kp).astype(int)
            axes = eigvi.astype(int)

            cv2.circle(im, center, radius, ci, -1)
            cv2.ellipse(
                im,
                center,
                axes,
                angle=angle,
                startAngle=0,
                endAngle=360,
                color=ci,
                thickness=lw
            )

        if plot_extremes:
            for kp, eigvi, angle in zip(ext_kps, ext_eigv, ext_angles):
                center = np.round(kp).astype(int)
                axes = eigvi.astype(int)
                cv2.circle(im, center, int(radius * 1.5), ext_c, -1)
                cv2.ellipse(
                    im,
                    center,
                    axes,
                    angle=angle,
                    startAngle=0,
                    endAngle=360,
                    color=ext_c,
                    thickness=lw
                )

        if return_selected:
            return cv2.cvtColor(im, cv2.COLOR_BGR2RGB), keep
        return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    elif lib == 'plt':
        if axi is None:
            fig, ax = plt.subplots()
        else:
            ax = axi

        if cmap_im is None and len(im.shape) == 2:
            cmap_im = 'gray'

        ax.imshow(
            im,
            cmap_im,
            vmin=cmap_im_min,
            vmax=cmap_im_max,
        )

        # get colors for each ellipse:
        if scores_color is not None:
            c = _color_based_on_scores(scores_color, scores_lims, cmap)

        ax.add_collection(EllipseCollection(
            widths=eigv[:, 0],
            heights=eigv[:, 1],
            angles=angles,
            color=c,
            offsets=kps,
            facecolor="none",
            lw=lw,
            units='xy',
            transOffset=ax.transData
        ))
        ax.scatter(kps[:, 0], kps[:, 1], s=radius * 2, color=c)

        if plot_extremes:
            ax.add_collection(EllipseCollection(
                widths=ext_eigv[:, 0],
                heights=ext_eigv[:, 1],
                angles=ext_angles,
                color=ext_c,
                offsets=ext_kps,
                facecolor="none",
                lw=lw,
                units='xy',
                transOffset=ax.transData
            ))
            ax.scatter(ext_kps[:, 0], ext_kps[:, 1], s=radius * 2, color=ext_c)

        # set figure to size in cm specified in fsize
        if max_fsize is not None:
            if fig is None:
                fig = ax.get_figure()
            fig = _fig_maxsize_wrt_im(fig, max_fsize, im.shape[:2])

        ax.set(
            xlim=(0, im.shape[1]),
            ylim=(im.shape[0], 0),
            xticks=[],
            yticks=[],
            # frame_on=False
        )
        # fig.tight_layout()
        if axi is None:
            if return_selected:
                return fig, ax, keep
            return fig, ax

        if return_selected:
            return keep


def plot_matches(
        im1: np.ndarray, im2: np.ndarray,
        kps1: np.ndarray, kps2: np.ndarray,
        fig: Optional[plt.Figure] = None,
        axi: Optional[plt.Axes] = None,
        orient: str = 'v',  # ['v', 'h']
        space: int = 10,  # pixels
        kp_s: Union[float, int] = 10,
        lw: Union[float, int] = 1,
        alpha_kp: float = 1.0,
        alpha_match: float = 1.0,
        plot_kps: bool = True,
        plot_matches: bool = True,
        color: Union[str, list[tuple]] = [(10 / 255, 209 / 255, 240 / 255)],
        scores_color: Optional[np.ndarray] = None,  # e.g. reprojection error
        scores_lims: list[float] = [0., 10.],
        cmap: Optional[str] = None,
        max_fsize: Optional[Union[int, float]] = 25,  # in centimetres
        frame_images: bool = True,
        n: Optional[int] = None,
        seed: int = 0
):
    """Plot matches between two images.

    Some aspects about Args:
        - kpsi shape is expected to be (2,n)
        - scores_color high/low values are considered as bad/good respectively.
    """
    # some checks
    assert (kps1.shape[1] == kps2.shape[1] > 0) & (2 == len(kps1) == len(kps2))
    if (scores_color is not None) and (len(scores_color) != kps1.shape[1]):
        raise ValueError(
            "If not None, scores_color must have length = number of kps, but"
            "len(scores_color)={len(scores_color)} and #kps = {kps.shape[1]}")

    if axi is None:
        fig, ax = plt.subplots()
    else:
        ax = axi

    # adjust output to the desired orientation:
    h1, w1 = im1.shape[:-1]
    h2, w2 = im2.shape[:-1]
    if orient == 'v':
        im_out = np.ones((h1 + space + h2, max(w1, w2), 3),
                         dtype=np.uint8) * 255
        im_out[:h1, :w1] = im1
        im_out[h1 + space:, :w2] = im2
        kps2 = kps2.copy() + [[0], [h1 + space]]
    elif orient == 'h':
        im_out = np.ones((max(h1, h2), w1 + space + w2, 3),
                         dtype=np.uint8) * 255
        im_out[:h1, :w1] = im1
        im_out[:h2, w1 + space:] = im2
        kps2 = kps2.copy() + [[w1 + space], [0]]
    else:
        raise ValueError(
            "orient is expected to be 'h' (horizontal) or 'v' (vertical), but "
            f"'{orient}' was received.")

    if n is not None:
        if n < kps1.shape[1]:
            # plot only "n" uniformly sampled matches.
            np.random.seed(seed)
            keep = np.random.choice(kps1.shape[1], (n,), replace=False)
            kps1 = kps1[:, keep]
            kps2 = kps2[:, keep]
            if scores_color is not None:
                scores_color = scores_color[keep]
            if isinstance(color, list) and len(color) > 1:
                color = [color[ki] for ki in keep]

    # get colors for each match
    if scores_color is not None:
        color = _color_based_on_scores(scores_color, scores_lims, cmap)

    # plot image and kps
    ax.imshow(im_out)
    if plot_kps:
        cfg_common = {
            'c': color, 'cmap': cmap, 'vmin': scores_lims[0], 'vmax': scores_lims[1]
        }
        ax.scatter(kps1[0], kps1[1], alpha=alpha_kp, s=kp_s, **cfg_common)
        ax.scatter(kps2[0], kps2[1], alpha=alpha_kp, s=kp_s, **cfg_common)

    # plot matches (lines)
    if plot_matches:
        ax.add_collection(LineCollection(
            np.concatenate((kps1.T, kps2.T), axis=1).reshape(-1, 2, 2),
            colors=color,
            alpha=alpha_match,
            lw=lw
        ))

    # draw black rectangle around image
    if frame_images:
        ax.add_patch(Rectangle(
            (-0.5, -0.5),
            w1, h1, linewidth=1, edgecolor='k', facecolor='none'))
        ax.add_patch(Rectangle((
            (-0.5, h1 + space - 0.5) if orient == 'v' else
            (w1 + space - 0.5, -0.5)),
            w2, h2, linewidth=1, edgecolor='k', facecolor='none'))

    # set figure to size in cm specified in fsize
    if max_fsize is not None:
        if fig is None:
            fig = ax.get_figure()
        fig = _fig_maxsize_wrt_im(fig, max_fsize, im_out.shape[:2])

    ax.set(
        xticks=[],
        yticks=[],
        frame_on=False
    )
    # fig.tight_layout()
    if axi == None:
        return fig, ax


def _fig_maxsize_wrt_im(
        fig: plt.Figure, max_fsize: Union[int, float], im_hw: np.ndarray):
    """Set figure max size (max_fsize in cm) maintaining the aspect ratio."""
    max_fsize *= 0.393701  # cm -> inches
    h, w = im_hw
    if h > w:
        fig.set_size_inches(w=(w / h) * max_fsize, h=max_fsize)
    else:
        fig.set_size_inches(w=max_fsize, h=(h / w) * max_fsize)
    return fig


def _color_based_on_scores(scores, scores_lims, cmap):
    """Get colors based on scores values which are mapped to RGB or cmap."""
    assert scores is not None

    # get normalized (\in [0,1]) scores.
    if scores_lims[0] is None:
        scores_lims[0] = scores.min()
    if scores_lims[1] is None:
        scores_lims[1] = scores.max()

    scores = scores.clip(
        min=scores_lims[0], max=scores_lims[1]) / scores_lims[1]

    if cmap is None:
        # create RGB colors ranging from green (low) to red (high).
        color = np.concatenate((
            scores[:, None], 1 - scores[:, None], np.zeros((len(scores), 1))
        ), axis=1)
    else:
        color = plt.get_cmap(cmap)(scores)
    return color


def _image2zvals(
        img: np.ndarray,
        n_colors: int = 64,
        n_training_pixels: int = 800,
        rngs=123):
    """ Image color quantization. Author: empet [1].
    [1] https://chart-studio.plotly.com/~empet/16132/color-image-quantization-and-tex/#/

    Args:
        img: np.ndarray of shape (m, n, 3) or (m, n, 4)
        n_colors: int,  number of colors for color quantization
        n_training_pixels: number of image pixels to fit a KMeans instance to them

    Returns:
        array of z_values for the heatmap representation, and a plotly colorscale
    """

    if img.ndim != 3:
        raise ValueError(
            f"Your image does not appear to  be a color image. It's shape is  {img.shape}")
    rows, cols, d = img.shape
    if d < 3:
        raise ValueError(
            f"A color image should have the shape (m, n, d), d=3 or 4. Your  d = {d}")

    range0 = img[:, :, 0].max() - img[:, :, 0].min()
    if range0 > 1:  # normalize the img values
        img = np.clip(img.astype(float) / 255, 0, 1)

    observations = img[:, :, :3].reshape(rows * cols, 3)
    training_pixels = shuffle(observations, random_state=rngs)[
        :n_training_pixels]
    model = KMeans(n_clusters=n_colors, random_state=rngs).fit(training_pixels)

    codebook = model.cluster_centers_
    indices = model.predict(observations)
    # normalization (i.e. map indices to  [0,1])
    z_vals = indices.astype(float) / (n_colors - 1)
    z_vals = z_vals.reshape(rows, cols)
    # define the Plotly colorscale with n_colors entries
    scale = np.linspace(0, 1, n_colors)
    colors = (codebook * 255).astype(np.uint8)
    pl_colorscale = [[sv, f'rgb{tuple(color)}']
                     for sv, color in zip(scale, colors)]

    # Reshape z_vals  to  img.shape[:2]
    return z_vals.reshape(rows, cols), pl_colorscale


def _regular_tri(rows, cols):
    """  Define triangles for a np.meshgrid(
    np.linspace(a, b, cols), np.linspace(c,d, rows)). Author: empet [1].
    [1] https://chart-studio.plotly.com/~empet/16132/color-image-quantization-and-tex/#/
    """
    triangles = []
    for i in range(rows - 1):
        for j in range(cols - 1):
            k = j + i * cols
            triangles.extend(
                [[k, k + cols, k + 1 + cols], [k, k + 1 + cols, k + 1]])
    return np.array(triangles)


def _mesh_data(img, n_colors=32, n_training_pixels=800):
    """ Get meah arguments for plotly Mesh3d. Author: empet [1].
    [1] https://chart-studio.plotly.com/~empet/16132/color-image-quantization-and-tex/#/
    """
    rows, cols, _ = img.shape
    z_data, pl_colorscale = _image2zvals(
        img, n_colors=n_colors, n_training_pixels=n_training_pixels)
    triangles = _regular_tri(rows, cols)
    I, J, K = triangles.T
    zc = z_data.ravel()[triangles]
    tri_color_intensity = [zc[k, 2] if k % 2 else zc[k, 1]
                           for k in range(len(zc))]
    return I, J, K, tri_color_intensity, pl_colorscale


def boxplot_grouped(
        do_scatter=False,
        scatter_points=10
):
    """ Do grouped boxplot plot"""
    raise NotImplementedError()
    # useful auxiliar functions

    def set_box_color(bp, color):
        plt.setp(bp['boxes'], color=color)
        plt.setp(bp['whiskers'], color=color, linestyle='--')
        plt.setp(bp['caps'], color=color)
        plt.setp(bp['medians'], color=color)
