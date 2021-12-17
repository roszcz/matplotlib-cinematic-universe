from mcu.scenes import Scene, SceneDimensions

import numpy as np
import pandas as pd
from pathlib import Path
import multiprocessing as mp
from matplotlib import pyplot as plt

from typing import List, Tuple


plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')


def psi(x: np.array, phi: float, m: int = 4) -> np.array:
    frequencies = range(1, m + 1)
    signals = []
    for frequency in frequencies:
        amp = np.cos(phi * frequency)
        signal = amp * np.sin(x * frequency)
        signals.append(signal)

    psi_values = np.sum(signals, axis=0) / m
    return psi_values


def psi_spectrum(phi: float, m: int) -> Tuple[List[int], List[float]]:
    amps = []
    frequencies = range(1, m + 1)
    for frequency in frequencies:
        amp = np.cos(phi * frequency)
        amps.append(amp)

    return frequencies, amps


class WavyBase(Scene):
    def draw(self, phase_shift: float) -> None:
        self.clean_figure()
        self.figure.tight_layout()

        self.draw_all(phase_shift)

    def render_mp(self) -> Path:
        print('Rendering to:', self.content_dir)

        howmany_frames = 300
        phase_shifts = np.linspace(0, 2 * np.pi, howmany_frames, endpoint=False)
        df = pd.DataFrame({'phase_shift': phase_shifts, 'counter': range(howmany_frames)})

        mp_step = 30
        parts = [df[sta: sta + mp_step] for sta in range(0, howmany_frames, mp_step)]
        with mp.Pool(20) as pool:
            pool.map(self.animate_part, parts)

        return self.content_dir

    def animate_part(self, steps: pd.DataFrame) -> None:
        for it, row in steps.iterrows():
            phase_shift = row.phase_shift
            frame_counter = row.counter

            self.draw(phase_shift)
            savepath = self.content_dir / f'{10000000 + frame_counter}.png'
            self.save_frame(savepath)
            self.frame_paths.append(savepath)


class WavyScene(WavyBase):
    SCENE_ID = 'WAVY'

    def __init__(self, m: int = 4, howmany_pi: int = 6) -> None:
        super().__init__()

        self.figsize = SceneDimensions(width=10, height=3)
        f, ax = plt.subplots(
            nrows=1,
            ncols=1,
            figsize=self.figsize
        )

        self.x_range = howmany_pi * np.pi
        self.x = np.linspace(0, self.x_range, 500)
        self.m = m
        self.freqs = range(1, m + 1)

        self.figure = f
        self.axes = [ax]

    def draw_all(self, phase_shift: float) -> None:
        ax = self.axes[0]
        signals = []
        for freq in self.freqs:
            amp = np.cos(freq * phase_shift)
            signal = amp * np.sin(self.x * freq + phase_shift)
            signals.append(signal)

        signal = np.sum(signals, axis=0) / self.m
        ax.plot(self.x, signal, lw=5)
        ax.set_xlim(0, self.x_range)
        ax.set_ylim(-1, 1)

        shift_pi = phase_shift / np.pi
        title = fr'Phase shift: $\pmb{{\phi}} = {shift_pi:.2f}\pi$'
        ax.set_title(title, loc='left', fontsize=16, usetex=True)

        ax.tick_params(
            axis='both',
            which='both',
            right=False,
            top=False
        )

        ax.set_xlabel(r'$x$', usetex=True, fontsize=16, loc='left')
        y_label = fr'$\Psi^{{ {self.m} }}(\phi, x)$'
        ax.set_ylabel(y_label, usetex=True, fontsize=16)


class WaveList(WavyBase):
    SCENE_ID = 'WAVYLIST'

    def __init__(self, m_values: List[int], howmany_pi: int = 6) -> None:
        super().__init__()

        howmany_m = len(m_values)
        self.figsize = SceneDimensions(width=10, height=howmany_m)
        f, axes = plt.subplots(
            nrows=howmany_m,
            ncols=1,
            gridspec_kw={
                'hspace': 0
            },
            figsize=self.figsize
        )

        self.m_values = m_values
        self.x_range = howmany_pi * np.pi
        self.x = np.linspace(0, self.x_range, 500)

        self.figure = f
        self.axes = axes

    def draw_all(self, phase_shift: float) -> None:
        shift_pi = phase_shift / np.pi
        title = fr'Phase shift: $\pmb{{\phi}} = {shift_pi:.2f}\pi$'
        self.axes[0].set_title(title, loc='left', fontsize=16, usetex=True)
        self.axes[-1].set_xlabel(r'$x$', usetex=True, fontsize=16)

        for it, m in enumerate(self.m_values):
            ax = self.axes[it]
            self.draw_psi_m(ax, phase_shift, m)

    def draw_psi_m(self, ax: plt.Axes, phase_shift: float, m: int) -> None:
        signal = psi(self.x, phi=phase_shift, m=m)
        ax.plot(self.x, signal, lw=5)
        ax.set_xlim(0, self.x_range)
        ax.set_ylim(-1.1, 1.1)

        ax.tick_params(
            axis='both',
            which='both',
            right=False,
            top=False,
            bottom=False,
            left=False,
            labelleft=False,
            labelbottom=False
        )

        y_label = fr'$\Psi^{{ {m} }}(\phi, x)$'
        ax.set_ylabel(y_label, usetex=True, fontsize=16)


class WaveListSpectrums(WavyBase):
    SCENE_ID = 'WAVYLIST'

    def __init__(self, m_values: List[int], howmany_pi: int = 6) -> None:
        super().__init__()

        self.howmany_m = len(m_values)
        fig_height = max(2.5, self.howmany_m)
        self.figsize = SceneDimensions(width=10, height=fig_height)
        f, axes = plt.subplots(
            nrows=self.howmany_m,
            ncols=2,
            gridspec_kw={
                'width_ratios': [2, 1],
                'hspace': 0,
                'wspace': 0
            },
            figsize=self.figsize
        )

        self.m_values = m_values
        self.x_range = howmany_pi * np.pi
        self.x = np.linspace(0, self.x_range, 500)

        self.figure = f
        self.psi_axes = axes[:, 0].tolist()
        self.spectral_axes = axes[:, 1].tolist()
        self.axes = self.psi_axes + self.spectral_axes

    def draw_all(self, phase_shift: float) -> None:
        shift_pi = phase_shift / np.pi
        psi_title = fr'Phase shift: $\pmb{{\phi}} = {shift_pi:.2f}\pi$'
        self.psi_axes[0].set_title(psi_title, loc='left', fontsize=16, usetex=True)
        self.psi_axes[-1].set_xlabel(r'$x$', usetex=True, fontsize=16)

        spectral_title = 'Power Spectrum'
        self.spectral_axes[0].set_title(spectral_title, loc='center', fontsize=16, usetex=True)
        self.spectral_axes[-1].set_xlabel('Frequency', usetex=True, fontsize=12)

        for it, m in enumerate(self.m_values):
            psi_ax = self.psi_axes[it]
            self.draw_psi_m(psi_ax, phase_shift, m)
            spectral_ax = self.spectral_axes[it]
            self.draw_spectrum_m(spectral_ax, phase_shift, m)

    def draw_spectrum_m(self, ax: plt.Axes, phase_shift: float, m: int) -> None:
        frequencies, amps = psi_spectrum(phase_shift, m)

        ax.bar(frequencies, amps)
        ax.set_xlim(0.5, self.m_values[-1] + 0.5)
        ax.set_ylim(-1.1, 1.1)

        ax.set_xticks(frequencies)
        ax.tick_params(
            axis='both',
            which='both',
            right=False,
            top=False,
            bottom=False,
            left=False,
            labelleft=False,
            labelbottom=False
        )

    def draw_psi_m(self, ax: plt.Axes, phase_shift: float, m: int) -> None:
        signal = psi(self.x, phi=phase_shift, m=m)
        ax.plot(self.x, signal, lw=5)
        ax.set_xlim(0, self.x_range)
        ax.set_ylim(-1.1, 1.1)

        ax.tick_params(
            axis='both',
            which='both',
            right=False,
            top=False,
            bottom=False,
            left=False,
            labelleft=False,
            labelbottom=False
        )

        y_label = fr'$\Psi^{{ {m} }}(\phi, x)$'
        ax.set_ylabel(y_label, usetex=True, fontsize=16)
