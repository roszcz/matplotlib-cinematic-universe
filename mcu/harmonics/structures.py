import numpy as np
from dataclasses import dataclass
from scipy.special import sph_harm


@dataclass
class WaveFunction:
    Y: np.array

    def __add__(self, other):
        Y_new = self.Y + other.Y
        new_wf = WaveFunction(Y=Y_new)
        return new_wf

    def __rich_repr__(self):
        yield 'Arbitrary Wave Function'
        yield 'Y shape', self.Y.shape


@dataclass
class Space3D:
    theta: np.array
    phi: np.array
    xyz: np.array

    def __rich_repr__(self):
        yield '3d space coordinates'
        yield 'xyz shape', self.xyz.shape
        yield 'phi shape', self.phi.shape
        yield 'theta shape', self.theta.shape


@dataclass
class Harmonic:
    el: int
    em: int
    space: Space3D
    Y: np.array = None

    def __add__(self, other: 'Harmonic') -> WaveFunction:
        Y_out = self.Y + other.Y
        wf = WaveFunction(Y=Y_out)
        return wf

    def __mul__(self, factor: float) -> WaveFunction:
        Y_out = self.Y * factor
        wf = WaveFunction(Y=Y_out)
        return wf

    def __rmul__(self, factor: float) -> WaveFunction:
        return self.__mul__(factor)

    def __post_init__(self):
        if self.Y is None:
            self.Y = sph_harm(
                abs(self.em),
                self.el,
                self.space.phi,
                self.space.theta
            )

    def real_form(self):
        if self.em < 0:
            Y = np.sqrt(2) * (-1)**self.em * self.Y.imag
        elif self.em >= 0:
            Y = np.sqrt(2) * (-1)**self.em * self.Y.real

        return Y

    def __rich_repr__(self):
        yield 'Spherical Harmonic'
        yield 'el', self.el
        yield 'em', self.em
        yield 'space', self.space
        yield 'Y shape', self.Y.shape


def prepare_coordinates(size: int = 1000) -> Space3D:
    # Grids of polar and azimuthal angles
    theta = np.linspace(0, np.pi, size)
    phi = np.linspace(0, 2 * np.pi, size)

    # Create a 2-D meshgrid of (theta, phi) angles.
    theta, phi = np.meshgrid(theta, phi)

    # Calculate the Cartesian coordinates of each point in the mesh.
    xyz = np.array([
        np.sin(theta) * np.sin(phi),
        np.sin(theta) * np.cos(phi),
        np.cos(theta)
    ])

    space = Space3D(
        theta=theta,
        phi=phi,
        xyz=xyz
    )

    return space
