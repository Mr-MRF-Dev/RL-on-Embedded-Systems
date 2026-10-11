"""Cart-Pole balancing environment (standard CartPole-v1 physics).

Continuous 4-D state, discretized into `num_bins` uniform bins per dimension:
    x (cart position), x_dot (cart velocity),
    theta (pole angle), theta_dot (pole angular velocity)

State  : single index in [0, num_bins^4)
Actions: 0 = push left, 1 = push right
Reward : +1 per step. The episode terminates if the pole leaves +-12 degrees
         or the cart leaves +-2.4 m, and is truncated after MAX_STEPS.
"""

import math
import random


class CartPoleEnv:
    # Physics
    GRAVITY = 9.8
    MASSCART = 1.0
    MASSPOLE = 0.1
    TOTAL_MASS = MASSCART + MASSPOLE
    LENGTH = 0.5  # half the pole length
    POLEMASS_LENGTH = MASSPOLE * LENGTH
    FORCE_MAG = 10.0
    TAU = 0.02  # seconds per step

    # Termination
    THETA_THRESHOLD = 12 * 2 * math.pi / 360  # 12 degrees in rad
    X_THRESHOLD = 2.4
    MAX_STEPS = 200

    # Discretization
    DEFAULT_NUM_BINS = 6
    X_BOUNDS = (-2.4, 2.4)
    XDOT_BOUNDS = (-3.0, 3.0)
    THETA_BOUNDS = (-THETA_THRESHOLD, THETA_THRESHOLD)
    THETADOT_BOUNDS = (-3.5, 3.5)

    N_ACTIONS = 2

    def __init__(self, num_bins=None, seed=None):
        self.num_bins = num_bins if num_bins is not None else self.DEFAULT_NUM_BINS
        self.n_states = self.num_bins**4
        self.rng = random.Random(seed)
        self.reset()

    def reset(self):
        self.x = self.rng.uniform(-0.05, 0.05)
        self.x_dot = self.rng.uniform(-0.05, 0.05)
        self.theta = self.rng.uniform(-0.05, 0.05)
        self.theta_dot = self.rng.uniform(-0.05, 0.05)
        self.t = 0
        return self._discretize()

    def step(self, action):
        assert action in (0, 1)
        self.t += 1

        force = self.FORCE_MAG if action == 1 else -self.FORCE_MAG
        costheta = math.cos(self.theta)
        sintheta = math.sin(self.theta)

        temp = (
            force + self.POLEMASS_LENGTH * self.theta_dot**2 * sintheta
        ) / self.TOTAL_MASS
        thetaacc = (self.GRAVITY * sintheta - costheta * temp) / (
            self.LENGTH * (4.0 / 3.0 - self.MASSPOLE * costheta**2 / self.TOTAL_MASS)
        )
        xacc = temp - self.POLEMASS_LENGTH * thetaacc * costheta / self.TOTAL_MASS

        self.x += self.TAU * self.x_dot
        self.x_dot += self.TAU * xacc
        self.theta += self.TAU * self.theta_dot
        self.theta_dot += self.TAU * thetaacc

        terminated = bool(
            abs(self.x) > self.X_THRESHOLD or abs(self.theta) > self.THETA_THRESHOLD
        )
        truncated = self.t >= self.MAX_STEPS

        info = {"x": self.x, "theta": self.theta}
        return self._discretize(), 1.0, terminated, truncated, info

    def _bin_index(self, value, bounds):
        lo, hi = bounds
        clipped = max(lo, min(hi, value))
        idx = int((clipped - lo) / (hi - lo) * self.num_bins)
        return min(idx, self.num_bins - 1)

    def _discretize(self):
        n = self.num_bins
        bx = self._bin_index(self.x, self.X_BOUNDS)
        bxd = self._bin_index(self.x_dot, self.XDOT_BOUNDS)
        bth = self._bin_index(self.theta, self.THETA_BOUNDS)
        bthd = self._bin_index(self.theta_dot, self.THETADOT_BOUNDS)
        return ((bx * n + bxd) * n + bth) * n + bthd
