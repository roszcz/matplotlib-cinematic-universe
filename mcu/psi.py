import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from matplotlib import pyplot as plt

from typing import List, Tuple

from mcu.scenes import Scene


def psi_values(x: np.array, phi: float, m: int = 4) -> np.array:
    frequencies = range(1, m + 1)
    signals = []
    for frequency in frequencies:
        amp = np.cos(phi * frequency)
        signal = amp * np.sin(x * frequency)
        signals.append(signal)

    values = np.sum(signals, axis=0) / m
    return values


def psi_spectrum(phi: float, m: int) -> Tuple[List[int], List[float]]:
    amps = []
    frequencies = range(1, m + 1)
    for frequency in frequencies:
        amp = np.cos(phi * frequency)
        amps.append(amp)

    return frequencies, amps


class PsiAnimationBase(Scene):
    def draw(self, phi: float) -> None:
        self.clean_figure()
        self.figure.tight_layout()

        self.draw_all(phi)

    def render(self) -> Path:
        print('Rendering to:', self.content_dir)

        howmany_frames = 300
        phase_shifts = np.linspace(0, 2 * np.pi, howmany_frames, endpoint=False)
        df = pd.DataFrame({'phi': phase_shifts, 'counter': range(howmany_frames)})

        for it, row in tqdm(df.iterrows(), total=len(df)):
            phi = row.phi
            frame_counter = row.counter

            self.draw(phi)
            savepath = self.content_dir / f'{10000000 + frame_counter}.png'
            self.save_frame(savepath)
            self.frame_paths.append(savepath)

        return self.content_dir


class SinglePsi(PsiAnimationBase):
    SCENE_ID = 'PSI'

    def __init__(self, m: int = 4, howmany_pi: int = 6) -> None:
        super().__init__()

        self.figsize = [10, 3]
        f, ax = plt.subplots(
            nrows=1,
            ncols=1,
            figsize=self.figsize
        )

        self.x_range = howmany_pi * np.pi
        self.x = np.linspace(0, self.x_range, 500)
        self.m = m

        self.figure = f
        self.axes = [ax]

    def draw_all(self, phi: float) -> None:
        ax = self.axes[0]

        signal = psi_values(self.x, phi, self.m)
        ax.plot(self.x, signal, lw=5)
        ax.set_xlim(0, self.x_range)
        ax.set_ylim(-1, 1)

        shift_pi = phi / np.pi
        title = f'Phase shift: {shift_pi:.2f}'
        ax.set_title(title, loc='left', fontsize=16)

        ax.tick_params(
            axis='both',
            which='both',
            right=False,
            top=False
        )

        ax.set_xlabel('x', fontsize=16)
        y_label = f'Psi, m = {self.m}'
        ax.set_ylabel(y_label, fontsize=16)


class PsiList(PsiAnimationBase):
    SCENE_ID = 'PSILIST'

    def __init__(self, m_values: List[int], howmany_pi: int = 6) -> None:
        super().__init__()

        howmany_m = len(m_values)
        self.figsize = [10, howmany_m]
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

    def draw_all(self, phi: float) -> None:
        shift_pi = phi / np.pi
        title = f'Phase shift: {shift_pi:.2f}'
        self.axes[0].set_title(title, loc='left', fontsize=16)
        self.axes[-1].set_xlabel('x', fontsize=16)

        for it, m in enumerate(self.m_values):
            ax = self.axes[it]
            self.draw_psi_m(ax, phi, m)

    def draw_psi_m(self, ax: plt.Axes, phi: float, m: int) -> None:
        signal = psi_values(self.x, phi=phi, m=m)
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

        y_label = f'm = {m}'
        ax.set_ylabel(y_label, fontsize=16)


class PsiListSpectrums(PsiAnimationBase):
    SCENE_ID = 'PSISPEC'

    def __init__(self, m_values: List[int], howmany_pi: int = 6) -> None:
        super().__init__()

        self.howmany_m = len(m_values)
        fig_height = max(2.5, self.howmany_m)
        self.figsize = [10, fig_height]
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

    def draw_all(self, phi: float) -> None:
        # Express `phi` in PI units
        shift_pi = phi / np.pi
        psi_title = f'Phase shift: {shift_pi:.2f}'
        self.psi_axes[0].set_title(psi_title, loc='left', fontsize=16, )
        self.psi_axes[-1].set_xlabel('x', fontsize=16)

        spectral_title = 'Power Spectrum'
        self.spectral_axes[0].set_title(spectral_title, loc='center', fontsize=16, )
        self.spectral_axes[-1].set_xlabel('Frequency', fontsize=12)

        for it, m in enumerate(self.m_values):
            psi_ax = self.psi_axes[it]
            self.draw_psi_m(psi_ax, phi, m)
            spectral_ax = self.spectral_axes[it]
            self.draw_spectrum_m(spectral_ax, phi, m)

    def draw_spectrum_m(self, ax: plt.Axes, phi: float, m: int) -> None:
        frequencies, amps = psi_spectrum(phi, m)

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

    def draw_psi_m(self, ax: plt.Axes, phi: float, m: int) -> None:
        signal = psi_values(self.x, phi=phi, m=m)
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

        y_label = f'm = {m}'
        ax.set_ylabel(y_label, fontsize=16)
