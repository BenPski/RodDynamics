import numpy as np
from scipy.optimize import fsolve
from scipy.linalg import expm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import copy
import csv

from utils import skew, unskew, adjoint, flatten, unflatten, se, unse, toMatrix, toQuaternion


class Series():
    """
    A series of rods
    integrates and manages multiple rods
    """

    def __init__(self, rods):
        self.rods = rods
        self._initRods()

    @property
    def g(self):
        return np.concatenate([rod.g[:-1] for rod in self.rods[:-1]] + [self.rods[-1].g])

    @property
    def xi(self):
        return np.concatenate([rod.xi[:-1] for rod in self.rods[:-1]] + [self.rods[-1].xi])

    @property
    def eta(self):
        return np.concatenate([rod.eta[:-1] for rod in self.rods[:-1]] + [self.rods[-1].eta])

    def _initRods(self):
        g0 = np.eye(4)
        for rod in self.rods:
            rod._initRod(g0)
            g0 = unflatten(rod.g[-1, :])

    def plot(self, ax=None):
        for rod in self.rods:
            ax = rod.plot(ax)
        return ax

    def energy(self):
        return sum([rod.energy() for rod in self.rods])

    def step(self, dt, q):
        prev = copy.deepcopy(self)
        xi0 = fsolve(lambda x: self._condition(prev, dt, x, q), self.xi[0, :])

    def _condition(self, prev, dt, xi0, q):
        # same as before except just final rod
        self._integrate(prev, dt, xi0, q)

        # all tip loads
        W = 0
        # data
        g = unflatten(self.g[-1, :])
        xi = self.xi[-1, :]
        eta = self.eta[-1, :]
        xi_dot = (self.xi[-1, :] - prev.xi[-1, :]) / dt
        eta_dot = (self.eta[-1, :] - prev.eta[-1, :]) / dt
        for load in self.rods[-1].loads:
            W += load.tip_load(g, xi, eta, xi_dot, eta_dot, self.rods[-1], q)

        return self.rods[-1].body.Psi(xi, self.rods[-1].body.L) - W

    def _integrate(self, prev, dt, xi0, q):
        g0 = np.eye(4)
        eta0 = np.array([0, 0, 0, 0, 0, 0])
        for (i, rod) in enumerate(self.rods):
            g0 = rod._integrate(prev.rods[i], dt, xi0, q, g0=g0, eta0=eta0, intermediate=i != (len(self.rods) - 1))
            xi0 = rod.xi[-1, :]
            eta0 = rod.eta[-1, :]


class Rod():
    """
    Rod stores the material properties and geometric properties of the cylindrical rod
    Need to specify:
        D: diameter
        L: length
        E: Young's Modulus
        rho: density
        mu: shear viscosity
        N: number of discretizations
        xi_init: function of s that specifies the initial value of xi (defaults to straight)
        eta_init: function os s that specifies the initial value of eta (defaults to stationary)
        loads: list of Loads to use in the simulation
    """

    def __init__(self, body, N, xi_init=lambda s: np.array([0, 0, 0, 0, 0, 1]),
                 eta_init=lambda s: np.array([0, 0, 0, 0, 0, 0]), loads=None):

        # definition of the body, holds geometry and material properties
        self.body = body

        self.ds = self.body.L / (N - 1)
        self.N = N
        self.xi_init = xi_init
        self.eta_init = eta_init

        # initialize state
        self.g = None
        self.xi = None
        self.eta = None
        self._initRod()

        # the different kinds of loads, viscosity always assumed to occur
        if loads is None:
            self.loads = []
        else:
            self.loads = loads

    def _initRod(self, g0=np.eye(4)):
        # setup g, xi, and eta for the initial configuration
        g = np.zeros((self.N, 7))
        xi = np.zeros((self.N, 6))
        eta = np.zeros((self.N, 6))

        # set xi and eta
        for i in range(self.N):
            s = self.ds * i
            xi[i, :] = self.xi_init(s)
            eta[i, :] = self.eta_init(s)

        # integrate G
        G = g0
        g[0, :] = flatten(G)
        for i in range(1, self.N):
            G = G @ expm(se(self.ds * xi[i - 1, :]))
            g[i, :] = flatten(G)

        # set state
        self.g = g
        self.xi = xi
        self.eta = eta

    def plot(self, ax=None):
        # not sure if this is the best way, but if an axis isn't specified generate it, if it is then modify it
        if ax is None:
            fig, ax = plt.subplot(111, projection='3d')
        # ax.plot(self.g[:, 9], self.g[:, 10], self.g[:, 11])
        ax.plot(self.g[:, 4], self.g[:, 5], self.g[:, 6])
        return ax

    def energy(self):
        H = 0  # total energy (aka Hamiltonian)
        for i in range(self.N):
            s = i * self.ds
            T = self.eta[i, :].T @ self.body.M(s) @ self.eta[i, :]
            U = self.body.strain_energy(self.xi[i, :], s)
            H += 1 / 2 * (T + U)
        return self.ds * H

    def step(self, dt, q):
        # since we are modifying the state want to keep track of the previous state for the integration process
        prev = copy.deepcopy(self)
        # just need to solve for xi0 and the state should be updated
        xi0 = fsolve(lambda x: self._condition(prev, dt, x, q), self.xi[0, :])

    def _condition(self, prev, dt, xi0, q):
        # integrate and see if the tip condition is satisfied
        self._integrate(prev, dt, xi0, q)

        # all tip loads
        W = 0
        # data
        g = unflatten(self.g[-1, :])
        xi = self.xi[-1, :]
        eta = self.eta[-1, :]
        xi_dot = (self.xi[-1, :] - prev.xi[-1, :]) / dt
        eta_dot = (self.eta[-1, :] - prev.eta[-1, :]) / dt
        for load in self.loads:
            W += load.tip_load(g, xi, eta, xi_dot, eta_dot, self, q)

        return self.body.Psi(self.xi[-1, :], self.body.L) - W

    def _integrate(self, prev, dt, xi0, q, g0=np.eye(4), eta0=np.array([0, 0, 0, 0, 0, 0]), intermediate=False):
        self.xi[0, :] = xi0
        self.eta[0, :] = eta0

        # integration over the body (don't need the initial point as the initial values are determined already)
        g_half = g0  # known initial condition
        for i in range(self.N - 1):
            s = i * self.ds
            # averaging over steps to get half step values
            xi_half = (self.xi[i, :] + prev.xi[i, :]) / 2
            eta_half = (self.eta[i, :] + prev.eta[i, :]) / 2

            # implicit midpoint approximation
            xi_dot = (self.xi[i, :] - prev.xi[i, :]) / dt
            eta_dot = (self.eta[i, :] - prev.eta[i, :]) / dt

            # external loads
            A_bar = 0
            B_bar = 0
            # viscosity
            B_bar += self.body.viscosity(xi_dot, s)

            # other loads
            for load in self.loads:
                A, B = load.dist_load(g_half, xi_half, eta_half, xi_dot, eta_dot, self, q, s)
                A_bar += A
                B_bar += B

            if intermediate and i == self.N - 2:
                for load in self.loads:
                    W = load.tip_load(g_half, xi_half, eta_half, xi_dot, eta_dot, self, q)
                    B_bar += W

            # spatial derivatives
            xi_der = np.linalg.inv(self.body.Psi_der(xi_half, s) - A_bar) @ (
                    (self.body.M(s) @ eta_dot) - (adjoint(eta_half).T @ self.body.M(s) @ eta_half) + (
                    adjoint(xi_half).T @ self.body.Psi(xi_half, s)) + B_bar - self.body.Psi_prime(xi_half, s))
            eta_der = xi_dot - (adjoint(xi_half) @ eta_half)

            # explicit Euler step
            xi_half_next = xi_half + self.ds * xi_der
            eta_half_next = eta_half + self.ds * eta_der
            R = g_half[:3, :3]
            p = g_half[:3, 3]
            p = p + self.ds * R @ xi_half[3:]
            q = toQuaternion(R)
            q = q + self.ds / 2 * np.array(
                [[0, -xi_half[0], -xi_half[1], -xi_half[2]], [xi_half[0], 0, xi_half[2], -xi_half[1]],
                 [xi_half[1], -xi_half[2], 0, xi_half[0]], [xi_half[2], xi_half[1], -xi_half[0], 0]]) @ q
            g_half = unflatten(np.concatenate([q, p]))

            # determine next step from half step value
            self.xi[i + 1, :] = 2 * xi_half_next - prev.xi[i + 1, :]
            self.eta[i + 1, :] = 2 * eta_half_next - prev.eta[i + 1, :]

        # midpoint RKMK to step the g values
        for i in range(self.N):
            eta_half = (self.eta[i, :] + prev.eta[i, :]) / 2
            q = prev.g[i, :4]
            p = prev.g[i, 4:]
            q = q + dt / 2 * np.array(
                [[0, -eta_half[0], -eta_half[1], -eta_half[2]], [eta_half[0], 0, eta_half[2], -eta_half[1]],
                 [eta_half[1], -eta_half[2], 0, eta_half[0]], [eta_half[2], eta_half[1], -eta_half[0], 0]]) @ q
            p = p + dt * toMatrix((q + prev.g[i, :4]) / 2) @ eta_half[3:]
            self.g[i, :] = np.concatenate([q, p])

        return g_half


def grav_energy(sys):
    H = sys.energy()
    for i in range(sys.N):
        H += -sys.ds * sys.rho * sys.A * sys.g[i, 9] * 9.81  # negative sign because it drops below the original height
    return H
