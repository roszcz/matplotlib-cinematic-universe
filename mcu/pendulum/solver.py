import math
import numpy

from mcu.pendulum.structures import PendulumState, PendulumSwing


class DoublePendulumHamiltonianSolver:
    """
        Based on: https://github.com/dassencio/double-pendulum
    """
    def __init__(self, state: PendulumState) -> None:
        self.state = state

    def hamilton_rhs(self, t1, t2, p1, p2):
        """
        Computes the right-hand side of the Hamilton's equations for
        the double pendulum and returns it as an array.
        t1 - The angle of bob #1.
        t2 - The angle of bob #2.
        p1 - The canonical momentum of bob #1.
        p2 - The canonical momentum of bob #2.
        """

        m1 = self.state.hardware.m1
        m2 = self.state.hardware.m2
        L1 = self.state.hardware.L1
        L2 = self.state.hardware.L2

        g = self.state.g

        C0 = L1 * L2 * (m1 + m2 * math.sin(t1 - t2)**2)
        C1 = (p1 * p2 * math.sin(t1 - t2)) / C0
        C2 = (m2 * (L2 * p1)**2 + (m1 + m2) * (L1 * p2)**2 -
              2 * L1 * L2 * m2 * p1 * p2 * math.cos(t1 - t2)) * \
            math.sin(2 * (t1 - t2)) / (2 * C0**2)

        # F is the right-hand side of the Hamilton's equations
        F_t1 = (L2 * p1 - L1 * p2 * math.cos(t1 - t2)) / (L1 * C0)
        F_t2 = (L1 * (m1 + m2) * p2 - L2 *
                m2 * p1 * math.cos(t1 - t2)) / (L2 * m2 * C0)
        F_p1 = -(m1 + m2) * g * L1 * math.sin(t1) - C1 + C2
        F_p2 = -m2 * g * L2 * math.sin(t2) + C1 - C2

        return numpy.array([F_t1, F_t2, F_p1, F_p2])

    def time_step(self, dt):
        """
        Advances one time step using RK4 (classical Runge-Kutta
        method).
        """
        t1 = self.state.t1
        t2 = self.state.t2
        p1 = self.state.p1
        p2 = self.state.p2

        # y is an array with the canonical variables (angles + momenta)
        y = numpy.array([t1, t2, p1, p2])

        # compute the RK4 constants
        k1 = self.hamilton_rhs(*y)
        k2 = self.hamilton_rhs(*(y + dt * k1 / 2))
        k3 = self.hamilton_rhs(*(y + dt * k2 / 2))
        k4 = self.hamilton_rhs(*(y + dt * k3))

        # compute the RK4 right-hand side
        R = 1.0 / 6.0 * dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        # update the angles and momenta
        dt1, dt2, dp1, dp2 = R
        self.state.update_canonical_coordinates(dt1, dt2, dp1, dp2)

    def solve(self, time: float, dt: float) -> PendulumSwing:
        states = []
        number_of_steps = int(time / dt)
        for it in range(number_of_steps):
            self.time_step(dt)
            state = self.state.copy()
            states.append(state)

        pendulum_swing = PendulumSwing(pendulum_states=states)
        return pendulum_swing
