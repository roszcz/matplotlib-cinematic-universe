import copy
import uuid
import numpy as np
import pandas as pd
from dataclasses import dataclass, field

from typing import List


@dataclass
class Pendulum:
    m1: float = 1.0
    m2: float = 1.0
    L1: float = 1.0
    L2: float = 1.0


@dataclass
class PendulumState:
    hardware: Pendulum
    t1: float = 0.0
    t2: float = 0.0
    w1: float = 0.0
    w2: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    g: float = 9.82

    @property
    def cartesian(self):
        return self.get_cartesian()

    def copy(self) -> "PendulumState":
        return copy.deepcopy(self)

    def __post_init__(self):
        m1 = self.hardware.m1
        m2 = self.hardware.m2
        L1 = self.hardware.L1
        L2 = self.hardware.L2
        t1 = self.t1
        t2 = self.t2
        w1 = self.w1
        w2 = self.w2

        # compute the initial canonical momenta
        self.p1 = (m1 + m2) * (L1**2) * w1 + m2 * L1 * L2 * w2 * np.cos(t1 - t2)
        self.p2 = m2 * (L2**2) * w2 + m2 * L1 * L2 * w1 * np.cos(t1 - t2)

    def get_cartesian(self):
        t1 = self.t1
        t2 = self.t2
        L1 = self.hardware.L1
        L2 = self.hardware.L2

        x1 = L1 * np.sin(t1)
        y1 = -L1 * np.cos(t1)
        x2 = x1 + L2 * np.sin(t2)
        y2 = y1 - L2 * np.cos(t2)
        result = {
            'x1': x1,
            'y1': y1,
            'x2': x2,
            'y2': y2
        }
        return result

    @property
    def canonical(self) -> dict:
        return {
            't1': self.t1,
            't2': self.t2,
            'p1': self.p1,
            'p2': self.p2
        }

    @property
    def energies(self) -> dict:
        return {
            'potential': self.potential_energy,
            'kinetic': self.kinetic_energy,
            'total': self.kinetic_energy + self.potential_energy
        }

    @property
    def potential_energy(self) -> float:
        """Computes the potential energy of the system."""
        m1 = self.hardware.m1
        m2 = self.hardware.m2
        L1 = self.hardware.L1
        L2 = self.hardware.L2
        t1 = self.t1
        t2 = self.t2

        g = self.g

        # compute the height of each bob
        y1 = -L1 * np.cos(t1)
        y2 = y1 - L2 * np.cos(t2)

        mgh1 = m1 * g * y1
        mgh2 = m2 * g * y2

        return mgh1 + mgh2

    @property
    def kinetic_energy(self) -> float:
        """Computes the kinetic energy of the system."""
        m1 = self.hardware.m1
        m2 = self.hardware.m2
        L1 = self.hardware.L1
        L2 = self.hardware.L2
        t1 = self.t1
        t2 = self.t2

        # compute the angular velocity of each bob
        (w1, w2) = self.omega()

        # compute the kinetic energy of each bob
        K1 = 0.5 * m1 * (L1 * w1)**2
        K2 = 0.5 * m2 * (
            (L1 * w1)**2 + (L2 * w2)**2 + 2 * L1 * L2 * w1 * w2 * np.cos(t1 - t2)
        )

        return K1 + K2

    @property
    def mechanical_energy(self) -> float:
        """
        Computes the mechanical energy (total energy) of the
        system.
        """
        return self.kinetic_energy + self.potential_energy

    def omega(self):
        """
        Computes the angular velocities of the bobs and returns them
        as a tuple.
        """
        m1 = self.hardware.m1
        m2 = self.hardware.m2
        L1 = self.hardware.L1
        L2 = self.hardware.L2
        t1 = self.t1
        t2 = self.t2
        p1 = self.p1
        p2 = self.p2

        C0 = L1 * L2 * (m1 + m2 * np.sin(t1 - t2)**2)

        w1 = (L2 * p1 - L1 * p2 * np.cos(t1 - t2)) / (L1 * C0)
        w2 = (L1 * (m1 + m2) * p2 - L2 *
              m2 * p1 * np.cos(t1 - t2)) / (L2 * m2 * C0)

        return (w1, w2)

    def update_canonical_coordinates(self, dt1, dt2, dp1, dp2) -> None:
        self.t1 += dt1
        self.t2 += dt2
        self.p1 += dp1
        self.p2 += dp2

        # Update non-canonical coordinates as well
        self.w1, self.w2 = self.omega()


@dataclass
class PendulumSwing:
    pendulum_states: List[PendulumState]
    swing_id: uuid.UUID = field(default_factory=uuid.uuid4)

    @property
    def name(self):
        return self.swing_id.hex

    def partial_swing(self, start: int, end: int) -> "PendulumSwing":
        part_states = self.pendulum_states[start: end]
        new_swing = PendulumSwing(pendulum_states=part_states)
        return new_swing

    def __rich_repr__(self):
        yield 'double pendulum swinging simulation'
        # yield 'pendulum', self.initial_state.hardware
        yield 'initial_state', self.initial_state
        yield "states_count", self.states_count

    @property
    def states_count(self) -> int:
        return len(self.pendulum_states)

    @property
    def initial_state(self) -> PendulumState:
        return self.pendulum_states[0]

    @property
    def hardware(self) -> Pendulum:
        return self.initial_state.hardware

    def get_coordinates_dataframe(self) -> pd.DataFrame:
        pq = self.get_canonical_dataframe()
        xy = self.get_cartesian_dataframe()
        df = pd.concat([xy, pq], axis=1)
        return df

    def get_cartesian_dataframe(self) -> pd.DataFrame:
        cartesians = [state.cartesian for state in self.pendulum_states]
        df = pd.DataFrame(cartesians)
        return df

    def get_canonical_dataframe(self) -> pd.DataFrame:
        canonical = [state.canonical for state in self.pendulum_states]
        df = pd.DataFrame(canonical)
        return df

    def get_energy_dataframe(self) -> pd.DataFrame:
        energies = [state.energies for state in self.pendulum_states]
        df = pd.DataFrame(energies)
        return df
