from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


COLOR_KPS_DET = [[0.204, 0.922, 0.922]]


class VizPlt:
    """ Visualization of feature-{detections, tracks} using blitting. """

    def __init__(self, nframes, h, w, n_colors=100, do_save=False):
        self.nframes = nframes

        self.fig, self.axes = plt.subplots(nrows=2, ncols=nframes)
        self.fig.set_size_inches(15.0, 7.7)

        # show window without blocking code execution.
        plt.show(block=False)

        # colors to visualize tracks.
        self.colors = plt.get_cmap('hsv')(
            np.linspace(0.0, 1.0, n_colors))[:, :3]
        np.random.RandomState(0).shuffle(self.colors)

        # set background lims and initialize artists.
        self.imshows = [[], []]
        self.scatters = [[], []]
        self.txts = [[], []]
        self.bgs = [[], []]
        for i, axes_row in enumerate(self.axes):
            for j, ax_col in enumerate(axes_row):
                self.imshows[i].append(
                    ax_col.imshow(
                        np.zeros((h, w, 3), dtype=int), animated=True))

                self.scatters[i].append(ax_col.scatter([], [], s=10.0))

                if i != 1 or j == 0:
                    self.txts[i].append(ax_col.text(
                        0.05, 0.95, "", fontsize=15, va='top', ha='left'))

                ax_col.set(
                    xlim=[0, w - 1],
                    ylim=[h - 1, 0],
                    xticks=[],
                    yticks=[]
                )

                # copy of the figure.
                self.bgs[i].append(self.fig.canvas.copy_from_bbox(ax_col.bbox))

        self.fig.tight_layout()

        # cache the renderer so that ax.draw_artist(...) works.
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        # default txt for upper and bottom row.
        self.txt_u = "Image {}\n#keypoints: {}"
        # self.txt_b = "#tracks: {}\n#matched_kps: {}"
        self.txt_b = "#tracks: {}"

        # data for upper row.
        self.ims = []
        self.im_ids = []
        self.kps_det = []

        if do_save:
            self.out_dir = Path(__file__).parent / 'data'
            self.out_dir.mkdir(exist_ok=True)
            self.n_saved = 0
        else:
            self.out_dir = None

    def add_viz_data(self, im, im_id, kps):
        self.ims.append(im)
        self.im_ids.append(im_id)
        self.kps_det.append(kps)

    def clear_regs(self):
        self.ims = []
        self.im_ids = []
        self.kps_det = []

    def update_plot(self, tracks):
        # get keypoints and colors per track and image.
        lc = len(self.colors)
        kps_tracks = [[] for _ in range(self.nframes)]
        kps_tracks_c = [[] for _ in range(self.nframes)]

        for i, track in enumerate(tracks):
            for kp in track.kps:
                kps_tracks[kp.im_id].append(self.kps_det[kp.im_id][:, kp.id])
                kps_tracks_c[kp.im_id].append(self.colors[i % lc])

        for j in range(self.nframes):
            # update artists.
            self.imshows[0][j].set(data=self.ims[j])
            self.imshows[1][j].set(data=self.ims[j])

            self.scatters[0][j].set(
                color=COLOR_KPS_DET,
                offsets=self.kps_det[j].T,
                # sizes=[10.0]
            )
            self.scatters[1][j].set(
                color=kps_tracks_c[j],
                offsets=kps_tracks[j],
                # sizes=[10.0]
            )

            self.txts[0][j].set(
                text=self.txt_u.format(
                    self.im_ids[j], self.kps_det[j].shape[1]),
                color='k' if self.ims[j][:100, :150].mean() > 150 else 'w'
            )
            if j == 0:
                self.txts[1][j].set(
                    text=self.txt_b.format(len(tracks)),
                    color='k' if self.ims[j][:100, :150].mean() > 150 else 'w'
                )

            # reset the background.
            self.fig.canvas.restore_region(self.bgs[0][j])
            self.fig.canvas.restore_region(self.bgs[1][j])

            # re-render artists.
            self.axes[0, j].draw_artist(self.imshows[0][j])
            self.axes[1, j].draw_artist(self.imshows[1][j])

            self.axes[0, j].draw_artist(self.scatters[0][j])
            self.axes[1, j].draw_artist(self.scatters[1][j])

            self.axes[0, j].draw_artist(self.txts[0][j])
            if j == 0:
                self.axes[1, j].draw_artist(self.txts[1][j])

            # copy image to the matplotlib window state.
            self.fig.canvas.blit(self.axes[0, j].bbox)
            self.fig.canvas.blit(self.axes[1, j].bbox)

        # flush (do) any pending GUI events.
        self.fig.canvas.flush_events()

        # re-initialize viz data.
        self.clear_regs()

        # save.
        if self.out_dir is not None:
            self.fig.savefig(
                self.out_dir / f"{str(self.n_saved).zfill(6)}.png",
                bbox_inches='tight')
            self.n_saved += 1
