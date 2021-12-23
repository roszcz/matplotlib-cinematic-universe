import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt

from mcu.scenes import Scene
from mcu.pendulum.structures import PendulumSwing


class PendulumBase(Scene):
    SCENE_ID = 'SWINGBASE'

    def draw(self, idx: int) -> None:
        self.clean_figure()
        self.figure.tight_layout()

        self.draw_all(idx)

    def render(self, framerate: int = 30) -> None:
        ends = self.temporal_arrangement(framerate)
        howmany_frames = len(ends)

        df = pd.DataFrame({'end_idx': ends, 'counter': range(howmany_frames)})

        for it, row in tqdm(df.iterrows(), total=len(df)):
            end = row.end_idx
            frame_counter = row.counter
            self.draw(end)
            savepath = self.content_dir / f'{10000000 + frame_counter}.png'
            self.save_frame(savepath)
            self.frame_paths.append(savepath)

        return self.content_dir

    def temporal_arrangement(self, framerate: int) -> pd.DataFrame:
        dt_scene = 1 / framerate
        dt_pendulum = 1e-3
        # `ds` says how many samples of signal we need
        # for one frame of animation
        step = dt_scene / dt_pendulum

        howmany_steps = len(self.swing.pendulum_states) / step
        howmany_steps = np.floor(howmany_steps)

        ends = (np.arange(0, howmany_steps) + 1) * step
        ends = np.floor(ends).astype(int)
        return ends


class SwingingPendulum(PendulumBase):
    SCENE_ID = 'DPSWING'

    def __init__(self, swing: PendulumSwing) -> None:
        super().__init__()

        # Simulation data container
        self.swing = swing
        self.xy = swing.get_cartesian_dataframe()
        self.pq = swing.get_canonical_dataframe()

        pendulum = swing.hardware
        view_limit = pendulum.L1 + pendulum.L2
        self.view_limit = view_limit * 1.06

        self.figsize = [7, 7]
        f, ax = plt.subplots(
            nrows=1,
            ncols=1,
            figsize=self.figsize
        )

        self.figure = f
        self.ax = ax
        self.axes = [self.ax]

    def draw_all(self, idx: int) -> None:
        afterglow = 700
        start = idx - afterglow
        start = max(0, start)
        part_xy = self.xy[start: idx]
        self.ax.plot(
            part_xy.x2,
            part_xy.y2,
            color='lightslategray',
            lw=4,
            zorder=1
        )

        latest = part_xy.iloc[-1]
        self.ax.scatter(
            latest.x1,
            latest.y1,
            color='slateblue',
            s=400,
            zorder=2
        )
        self.ax.scatter(
            latest.x2,
            latest.y2,
            color='teal',
            s=900,
            zorder=2
        )
        self.ax.plot(
            [0, latest.x1, latest.x2],
            [0, latest.y1, latest.y2],
            color='black',
            lw=4,
            zorder=1
        )
        self.ax.plot(
            [-0.06, 0.06],
            [0, 0],
            color='black',
            lw=12
        )
        self.ax.set_xlim(-self.view_limit, self.view_limit)
        self.ax.set_ylim(-self.view_limit, self.view_limit)

        self.ax.tick_params(
            axis='both',
            which='both',
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labelleft=False
        )


class SwingWithStats(PendulumBase):
    SCENE_ID = 'STATSWING'

    def __init__(self, swing: PendulumSwing, afterglow_length: int = 500) -> None:
        super().__init__()

        # Aesthetic choices
        self.afterglow_length = afterglow_length

        # Simulation data container
        self.swing = swing
        self.xy = swing.get_cartesian_dataframe()
        self.pq = swing.get_canonical_dataframe()

        pendulum = swing.hardware
        view_limit = pendulum.L1 + pendulum.L2
        self.view_limit = view_limit * 1.1

        self.figsize = [14, 7]
        f, axes = plt.subplots(
            nrows=8,
            ncols=2,
            figsize=self.figsize,
            gridspec_kw={
                'width_ratios': [1, 1],
                'hspace': 0
            }
        )
        gs = axes[0, 1].get_gridspec()
        for ax in axes[:, 1]:
            ax.remove()

        self.ax = f.add_subplot(gs[:, 1])

        self.figure = f
        self.metric_axes = axes[:, 0]
        self.axes = [self.ax, *self.metric_axes]

    def draw_pendulum(self, idx: int) -> None:
        start = max(0, idx - self.afterglow_length)
        part_xy = self.xy[start: idx]
        self.ax.plot(
            part_xy.x2,
            part_xy.y2,
            color='lightslategray',
            zorder=1
        )

        latest = part_xy.iloc[-1]
        self.ax.scatter(
            latest.x1,
            latest.y1,
            color='slateblue',
            s=100,
            zorder=2
        )
        self.ax.scatter(
            latest.x2,
            latest.y2,
            color='teal',
            s=400,
            zorder=2
        )
        self.ax.plot(
            [0, latest.x1, latest.x2],
            [0, latest.y1, latest.y2],
            color='black',
            zorder=1
        )
        self.ax.plot(
            [-0.03, 0.03],
            [0, 0],
            color='black',
            lw=6
        )
        self.ax.set_xlim(-self.view_limit, self.view_limit)
        self.ax.set_ylim(-self.view_limit, self.view_limit)

    def draw_stat(self, ax, values, name):
        ax.plot(values, label=name, color='teal')
        ax.set_xlim(0, 5200)
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
        ax.set_ylabel(name)
        # ax.legend(loc='lower left')

    def draw_stats(self, idx: int) -> None:
        howmany = 5200
        start = max(0, idx - howmany)
        part_xy = self.xy[start: idx]
        self.draw_stat(self.metric_axes[0], part_xy.x1.values, 'x1')
        self.draw_stat(self.metric_axes[1], part_xy.y1.values, 'y1')
        self.draw_stat(self.metric_axes[2], part_xy.x2.values, 'x2')
        self.draw_stat(self.metric_axes[3], part_xy.y2.values, 'y2')

        part_pq = self.pq[start: idx]
        self.draw_stat(self.metric_axes[4], part_pq.p1.values, 'p1')
        self.draw_stat(self.metric_axes[5], part_pq.t1.values, 't1')
        self.draw_stat(self.metric_axes[6], part_pq.p2.values, 'p2')
        self.draw_stat(self.metric_axes[7], part_pq.t2.values, 't2')

    def draw_all(self, idx: int) -> None:
        self.draw_pendulum(idx)
        self.draw_stats(idx)
        self.figure.subplots_adjust(hspace=0)


class SwingingWithEnergyBars(PendulumBase):
    SCENE_ID = 'ENSWING'

    def __init__(self, swing: PendulumSwing) -> None:
        super().__init__()

        # Simulation data container
        self.swing = swing
        self.xy = swing.get_cartesian_dataframe()
        self.pq = swing.get_canonical_dataframe()
        self.en = swing.get_energy_dataframe()

        pendulum = swing.hardware
        view_limit = pendulum.L1 + pendulum.L2
        self.view_limit = view_limit + .15
        self.y_max = self.xy.y2.max() + .15

        self.figsize = [8, 4]
        f, axes = plt.subplots(
            nrows=1,
            ncols=2,
            figsize=self.figsize,
            gridspec_kw={
                'width_ratios': [3, 1],
                'hspace': 0,
                'wspace': 0
            },
        )

        self.figure = f
        self.ax = axes[0]
        self.energy_ax = axes[1]
        self.axes = axes.tolist()

    def draw_all(self, idx: int) -> None:
        self.clean_figure()
        self.draw_pendulum(idx)
        self.draw_energies(idx)

    def draw_energies(self, idx: int) -> None:
        idx = min(idx, len(self.en) - 1)
        energies = self.en.iloc[idx]
        # 0-based
        values = [
            energies.kinetic - self.en.kinetic.min(),
            energies.potential - self.en.potential.min()
        ]
        labels = ['kinetic', 'potential']
        self.energy_ax.bar(labels, values, color=['skyblue', 'cornflowerblue'])
        max_y = 1.2 * sum(values)
        self.energy_ax.set_ylim(0, max_y)
        self.energy_ax.set_title('Total Energy')
        self.energy_ax.tick_params(
            axis='both',
            top=False,
            left=False,
            right=False,
            labelleft=False
        )

    def draw_pendulum(self, idx: int) -> None:
        afterglow = 700
        start = idx - afterglow
        start = max(0, start)
        part_xy = self.xy[start: idx]
        self.ax.plot(
            part_xy.x2,
            part_xy.y2,
            color='lightslategray',
            lw=2,
            zorder=1
        )
        self.ax.set_title('The Double Pendulum')

        latest = part_xy.iloc[-1]
        self.ax.scatter(
            latest.x1,
            latest.y1,
            color='teal',
            s=400,
            zorder=2
        )
        self.ax.scatter(
            latest.x2,
            latest.y2,
            color='teal',
            s=900,
            zorder=2
        )
        self.ax.plot(
            [0, latest.x1, latest.x2],
            [0, latest.y1, latest.y2],
            color='black',
            lw=4,
            zorder=1
        )
        self.ax.plot(
            [-0.06, 0.06],
            [0, 0],
            color='black',
            lw=12
        )
        self.ax.set_xlim(-self.view_limit, self.view_limit)
        self.ax.set_ylim(-self.view_limit, self.y_max)

        self.ax.tick_params(
            axis='both',
            which='both',
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labelleft=False
        )
