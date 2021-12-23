import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mpl_colors


from mcu.scenes import Scene
from mcu.harmonics.structures import Harmonic, Space3D


class QuantumCompute3D(Scene):
    SCENE_ID = 'QC3D'

    def __init__(self, h1: Harmonic, h2: Harmonic, space: Space3D) -> None:
        super().__init__()

        self.h1 = h1
        self.h2 = h2
        self.space = space

        grid = gridspec.GridSpec(
            nrows=2,
            ncols=2,
            width_ratios=[1, 2],
            # hspace=0,
            # wspace=0
        )

        self.figsize = [7.4, 4.8]
        fig = plt.figure(
            figsize=self.figsize,
            facecolor='silver'
        )
        ax1 = fig.add_subplot(grid[0, 0], projection='3d')
        ax2 = fig.add_subplot(grid[1, 0], projection='3d')
        ax3 = fig.add_subplot(grid[:, 1], projection='3d')

        self.figure = fig
        self.axes = [ax1, ax2, ax3]

    def draw_3d(self, Y: np.array, ax: plt.Axes, angle: int) -> None:
        Yx, Yy, Yz = np.abs(Y) * self.space.xyz
        cmap = plt.cm.ScalarMappable(cmap=plt.get_cmap('viridis'))
        cmap.set_clim(-1, 1)

        ax.plot_surface(
            Yx, Yy, Yz,
            facecolors=cmap.to_rgba(Y.real),
            rstride=2,
            cstride=2,
            alpha=0.666,
        )

        # TODO relative to the Y amplitude?
        lim = 0.3
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
        ax.view_init(30, angle)

        # title = r'$Y^2_2(\theta\phi)$'
        # ax.text2D(0.01, 0.9, title, transform=ax.transAxes, fontsize=16)
        # ax.set_title(title, loc='left')
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

        floor_color = mpl_colors.to_rgb('silver')
        alpha = 0.6
        floor_color = list(floor_color) + [alpha]
        ax.zaxis.set_pane_color(floor_color)
        # ax.zaxis.set_pane_color((0.29, 0.0, 0.5, 0.6))
        ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
        ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
        ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)

        # Get rid of the vertical axis line
        ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax.set_zticks([])
        # ax.set_axis_off()

        ax.tick_params(
            axis='both',
            which='both',
            bottom=True,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labelleft=False
        )

    def draw(self, angle: int, alpha: float) -> None:
        self.clean_figure()
        self.figure.tight_layout()

        self.draw_all(angle, alpha)

    def draw_all(self, angle: int, alpha: float) -> None:
        # combination = self.h1 * (1 - alpha) + self.h2 * alpha
        combination = self.h1 * (1 - alpha) + self.h2 * alpha

        Y1 = np.abs(self.h1.Y)
        # Y1 = self.h1.Y.real
        el1 = self.h1.el
        em1 = self.h1.em
        ax = self.axes[0]
        self.draw_3d(Y1, ax, angle)
        title = f'|{el1}{em1}>'
        ax.text2D(0.01, 0.89, title, transform=ax.transAxes, fontsize=14)

        Y2 = np.abs(self.h2.Y)
        # Y2 = self.h2.Y.real
        el2 = self.h2.el
        em2 = self.h2.em
        ax = self.axes[1]
        self.draw_3d(Y2, ax, angle)
        title = f'|{el2}{em2}>'
        ax.text2D(0.01, 0.89, title, transform=ax.transAxes, fontsize=14)

        Y3 = np.abs(combination.Y) * 0.8
        # Y3 = combination.Y.real
        ax = self.axes[2]
        self.draw_3d(Y3, ax, angle)
        title = f'|{1-alpha:.2f}|{el1}{em1}> + {alpha:.2f}|{el2}{em2}>|^2'
        ax.text2D(0.01, 0.94, title, transform=ax.transAxes, fontsize=16)

        self.figure.suptitle('Linear Combination of Spherical Harmonics', fontsize=16, y=0.95)

    def render(self) -> None:
        print('Rendering to:', self.content_dir)
        angles = range(360)
        angles = list(range(0, 360, 2)) * 2

        # Go up and down in a sine
        x = np.linspace(0, 1, 360) * np.pi * 2
        alpha = (1 - np.cos(x)) / 2

        howmany_frames = len(angles)
        df = pd.DataFrame({
            'angle': angles,
            'alpha': alpha,
            'counter': range(howmany_frames)
        })

        # WTF 3D Matplotlib
        # For yet unknown reasons, the layout shifts a little
        # (but noticably) between first and the second frame
        # To solve this I draw one frame without saving it
        for it, row in df[:4].iterrows():
            angle = row.angle
            alpha = row.alpha
            savepath = f'tmp/discard_{angle}.png'
            self.draw(alpha=alpha, angle=angle)
            self.save_frame(savepath=savepath)

        # Back to the program
        for it, row in tqdm(df.iterrows(), total=len(df)):
            angle = row.angle
            alpha = row.alpha
            frame_counter = row.counter
            self.draw(alpha=alpha, angle=angle)
            savepath = self.content_dir / f'{10000000 + frame_counter}.png'
            self.save_frame(savepath)
            self.frame_paths.append(savepath)

        return self.content_dir


class QuantumPreview3D(Scene):
    SCENE_ID = 'QP3D'

    def __init__(self, h1: Harmonic, space: Space3D) -> None:
        super().__init__()

        self.h1 = h1
        self.space = space

        self.figsize = [12, 4]
        f, axes = plt.subplots(
            nrows=1,
            ncols=3,
            subplot_kw={'projection': '3d'},
            figsize=self.figsize
        )

        self.figure = f
        self.axes = axes

    def draw_3d(self, Y: np.array, ax: plt.Axes, angle: int, title: str) -> None:
        Yx, Yy, Yz = np.abs(Y) * self.space.xyz
        cmap = plt.cm.ScalarMappable(cmap=plt.get_cmap('viridis'))
        cmap.set_clim(-1, 1)

        ax.plot_surface(
            Yx, Yy, Yz,
            facecolors=cmap.to_rgba(Y.real),
            rstride=2,
            cstride=2
        )
        lim = 0.3
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
        ax.view_init(30, angle)
        ax.set_title(title)
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
        ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
        ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
        ax.set_axis_off()
        ax.tick_params(
            axis='both',
            which='both',
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labelleft=False
        )

    def draw_all(self, angle: int, alpha: float) -> None:
        Y1 = np.abs(self.h1.Y)
        title = f'Angle: {angle}, abs(Y)'
        self.draw_3d(Y1, self.axes[0], angle, title)
        Y2 = self.h1.Y.real
        title = f'Angle: {angle}, Y.real'
        self.draw_3d(Y2, self.axes[1], angle, title)
        Y3 = self.h1.Y.imag
        title = f'Angle: {angle}, Y.imag'
        self.draw_3d(Y3, self.axes[2], angle, title)

    def draw(self, angle: int) -> None:
        self.clean_figure()
        self.figure.tight_layout()

        self.draw_all(angle)

    def render_mp(self) -> None:
        print('Rendering to:', self.content_dir)
        howmany_frames = 360
        angles = range(360)

        df = pd.DataFrame({'angle': angles, 'counter': range(howmany_frames)})

        # WTF
        # For yet unknown reasons, the layout shifts a little
        # (but noticably) between first and the second frame
        # To solve this I draw one frame without saving it
        for it, row in df[:4].iterrows():
            angle = row.angle
            savepath = f'tmp/discard_{angle}.png'
            self.draw(angle)
            self.save_frame(savepath=savepath)

        # Back to the program
        for it, row in tqdm(df.iterrows(), total=len(df)):
            angle = row.angle
            frame_counter = row.counter
            self.draw(angle)
            savepath = self.content_dir / f'{10000000 + frame_counter}.png'
            self.save_frame(savepath)
            self.frame_paths.append(savepath)

        return self.content_dir
