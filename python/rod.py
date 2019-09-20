import numpy as np
from scipy.optimize import fsolve
from scipy.linalg import expm, logm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import copy

from utils import skew, unskew, adjoint, flatten, unflatten, se, unse


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

        return self.rods[-1].K @ (self.xi[-1, :] - self.rods[-1].xi_ref) - W

    def _integrate(self, prev, dt, xi0, q):
        g0 = np.eye(4)
        eta0 = np.array([0, 0, 0, 0, 0, 0])
        for (i, rod) in enumerate(self.rods):
            g0 = rod._integrate(prev.rods[i], dt, xi0, q, g0=g0, eta0=eta0, intermediate=i != (len(self.rods) - 1))
            xi0 = rod.xi[-1, :]
            eta0 = rod.eta[-1, :]


class RodFixedFree():
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

    def __init__(self, D, L, E, rho, mu, N, xi_init=lambda s: np.array([0, 0, 0, 0, 0, 1]),
                 eta_init=lambda s: np.array([0, 0, 0, 0, 0, 0]), loads=None):
        # setup properties
        A = np.pi / 4 * D ** 2
        I = np.pi / 64 * D ** 4
        J = 2 * I
        G = E / 3  # assuming incompressible material

        # store values important to simulation
        self.K = np.diag([E * I, E * I, G * J, G * A, G * A, E * A])
        self.M = rho * np.diag([I, I, J, A, A, A])
        self.V = mu * np.diag([3 * I, 3 * I, J, A, A, 3 * A])
        self.L = L
        self.D = D
        self.rho = rho
        self.A = A
        self.xi_ref = np.array([0, 0, 0, 0, 0, 1])
        self.ds = L / (N - 1)
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
        g = np.zeros((self.N, 12))
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
        ax.plot(self.g[:, 9], self.g[:, 10], self.g[:, 11])
        return ax

    def energy(self):
        H = 0  # total energy (aka Hamiltonian)
        for i in range(self.N):
            T = self.eta[i, :].T @ self.M @ self.eta[i, :]
            U = (self.xi[i, :] - self.xi_ref).T @ self.K @ (self.xi[i, :] - self.xi_ref)
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

        return self.K @ (self.xi[-1, :] - self.xi_ref) - W

    def _integrate(self, prev, dt, xi0, q, g0=np.eye(4), eta0=np.array([0, 0, 0, 0, 0, 0]), intermediate=False):
        self.xi[0, :] = xi0
        self.eta[0, :] = eta0

        # integration over the body (don't need the initial point as the initial values are determined already)
        g_half = g0  # known initial condition
        for i in range(self.N - 1):
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
            B_bar += self.V @ xi_dot

            # other loads
            for load in self.loads:
                A, B = load.dist_load(g_half, xi_half, eta_half, xi_dot, eta_dot, self, q)
                A_bar += A
                B_bar += B

            if intermediate and i == self.N-2:
                for load in self.loads:
                    W = load.tip_load(g_half, xi_half, eta_half, xi_dot, eta_dot, self, q)
                    B_bar += W

            # spatial derivatives
            xi_der = np.linalg.inv(self.K - A_bar) @ (
                    (self.M @ eta_dot) - (adjoint(eta_half).T @ self.M @ eta_half) + (
                    adjoint(xi_half).T @ self.K @ (xi_half - self.xi_ref)) + B_bar)
            eta_der = xi_dot - (adjoint(xi_half) @ eta_half)

            # explicit Euler step
            xi_half_next = xi_half + self.ds * xi_der
            eta_half_next = eta_half + self.ds * eta_der
            g_half = g_half @ expm(se(self.ds * xi_half))

            # determine next step from half step value
            self.xi[i + 1, :] = 2 * xi_half_next - prev.xi[i + 1, :]
            self.eta[i + 1, :] = 2 * eta_half_next - prev.eta[i + 1, :]

        # midpoint RKMK to step the g values
        for i in range(self.N):
            self.g[i, :] = flatten(unflatten(prev.g[i, :]) @ expm(se(dt * (self.eta[i, :] + prev.eta[i, :]) / 2)))

        return g_half

class RodFixedFixed():
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

    def __init__(self, D, L, E, rho, mu, N, g_des, xi_init=lambda s: np.array([0, 0, 0, 0, 0, 1]),
                 eta_init=lambda s: np.array([0, 0, 0, 0, 0, 0]), loads=None):
        # setup properties
        A = np.pi / 4 * D ** 2
        I = np.pi / 64 * D ** 4
        J = 2 * I
        G = E / 3  # assuming incompressible material

        # store values important to simulation
        self.K = np.diag([E * I, E * I, G * J, G * A, G * A, E * A])
        self.M = rho * np.diag([I, I, J, A, A, A])
        self.V = mu * np.diag([3 * I, 3 * I, J, A, A, 3 * A])
        self.L = L
        self.D = D
        self.rho = rho
        self.A = A
        self.xi_ref = np.array([0, 0, 0, 0, 0, 1])
        self.ds = L / (N - 1)
        self.N = N
        self.xi_init = xi_init
        self.eta_init = eta_init
        self.g_des = g_des

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
        g = np.zeros((self.N, 12))
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
        ax.plot(self.g[:, 9], self.g[:, 10], self.g[:, 11])
        return ax

    def energy(self):
        H = 0  # total energy (aka Hamiltonian)
        for i in range(self.N):
            T = self.eta[i, :].T @ self.M @ self.eta[i, :]
            U = (self.xi[i, :] - self.xi_ref).T @ self.K @ (self.xi[i, :] - self.xi_ref)
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

        # data
        g = unflatten(self.g[-1, :])
        R = g[:3,:3]
        R_des = self.g_des[:3,:3]
        p = g[:3,3]
        p_des = self.g_des[:3,3]

        angle_err = unskew(logm(R.T @ R_des))
        pos_err = p-p_des

        return np.concatenate([angle_err**2,pos_err**2])

    def _integrate(self, prev, dt, xi0, q, g0=np.eye(4), eta0=np.array([0, 0, 0, 0, 0, 0]), intermediate=False):
        self.xi[0, :] = xi0
        self.eta[0, :] = eta0

        # integration over the body (don't need the initial point as the initial values are determined already)
        g_half = g0  # known initial condition
        for i in range(self.N - 1):
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
            B_bar += self.V @ xi_dot

            # other loads
            for load in self.loads:
                A, B = load.dist_load(g_half, xi_half, eta_half, xi_dot, eta_dot, self, q)
                A_bar += A
                B_bar += B

            if intermediate and i == self.N-2:
                for load in self.loads:
                    W = load.tip_load(g_half, xi_half, eta_half, xi_dot, eta_dot, self, q)
                    B_bar += W

            # spatial derivatives
            xi_der = np.linalg.inv(self.K - A_bar) @ (
                    (self.M @ eta_dot) - (adjoint(eta_half).T @ self.M @ eta_half) + (
                    adjoint(xi_half).T @ self.K @ (xi_half - self.xi_ref)) + B_bar)
            eta_der = xi_dot - (adjoint(xi_half) @ eta_half)

            # explicit Euler step
            xi_half_next = xi_half + self.ds * xi_der
            eta_half_next = eta_half + self.ds * eta_der
            g_half = g_half @ expm(se(self.ds * xi_half))

            # determine next step from half step value
            self.xi[i + 1, :] = 2 * xi_half_next - prev.xi[i + 1, :]
            self.eta[i + 1, :] = 2 * eta_half_next - prev.eta[i + 1, :]

        # midpoint RKMK to step the g values
        for i in range(self.N):
            self.g[i, :] = flatten(unflatten(prev.g[i, :]) @ expm(se(dt * (self.eta[i, :] + prev.eta[i, :]) / 2)))

        return g_half


class RodFreeFree():
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

    def __init__(self, D, L, E, rho, mu, N, xi_init=lambda s: np.array([0, 0, 0, 0, 0, 1]),
                 eta_init=lambda s: np.array([0, 0, 0, 0, 0, 0]), loads=None):
        # setup properties
        A = np.pi / 4 * D ** 2
        I = np.pi / 64 * D ** 4
        J = 2 * I
        G = E / 3  # assuming incompressible material

        # store values important to simulation
        self.K = np.diag([E * I, E * I, G * J, G * A, G * A, E * A])
        self.M = rho * np.diag([I, I, J, A, A, A])
        self.V = mu * np.diag([3 * I, 3 * I, J, A, A, 3 * A])
        self.L = L
        self.D = D
        self.rho = rho
        self.A = A
        self.xi_ref = np.array([0, 0, 0, 0, 0, 1])
        self.ds = L / (N - 1)
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
        g = np.zeros((self.N, 12))
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
        ax.plot(self.g[:, 9], self.g[:, 10], self.g[:, 11])
        return ax

    def energy(self):
        H = 0  # total energy (aka Hamiltonian)
        for i in range(self.N):
            T = self.eta[i, :].T @ self.M @ self.eta[i, :]
            U = (self.xi[i, :] - self.xi_ref).T @ self.K @ (self.xi[i, :] - self.xi_ref)
            H += 1 / 2 * (T + U)
        return self.ds * H

    def step(self, dt, q):
        # since we are modifying the state want to keep track of the previous state for the integration process
        prev = copy.deepcopy(self)
        # just need to solve for xi0 and the state should be updated
        eta0 = fsolve(lambda x: self._condition(prev, dt, x, q), self.eta[0, :])

    def _condition(self, prev, dt, eta0, q):
        # integrate and see if the tip condition is satisfied
        self._integrate(prev, dt, eta0, q)

        # # all tip loads
        # W = 0
        # # data
        # g = unflatten(self.g[-1, :])
        # xi = self.xi[-1, :]
        # eta = self.eta[-1, :]
        # xi_dot = (self.xi[-1, :] - prev.xi[-1, :]) / dt
        # eta_dot = (self.eta[-1, :] - prev.eta[-1, :]) / dt
        # for load in self.loads:
        #     W += load.tip_load(g, xi, eta, xi_dot, eta_dot, self, q)

        # return self.K @ (self.xi[-1, :] - self.xi_ref) - W
        return self.xi[-1, :] - self.xi_ref

    def _integrate(self, prev, dt, eta0, q, g0=np.eye(4), intermediate=False):
        self.xi[0, :] = np.array([0,0,0,0,0,1])
        self.eta[0, :] = eta0

        # integration over the body (don't need the initial point as the initial values are determined already)
        # g_half = g0  # known initial condition
        g_half = expm(dt/2*se(prev.eta[0,:]))
        for i in range(self.N - 1):
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
            B_bar += self.V @ xi_dot

            # other loads
            for load in self.loads:
                A, B = load.dist_load(g_half, xi_half, eta_half, xi_dot, eta_dot, self, q)
                A_bar += A
                B_bar += B

            if intermediate and i == self.N-2:
                for load in self.loads:
                    W = load.tip_load(g_half, xi_half, eta_half, xi_dot, eta_dot, self, q)
                    B_bar += W

            # spatial derivatives
            xi_der = np.linalg.inv(self.K - A_bar) @ (
                    (self.M @ eta_dot) - (adjoint(eta_half).T @ self.M @ eta_half) + (
                    adjoint(xi_half).T @ self.K @ (xi_half - self.xi_ref)) + B_bar)
            eta_der = xi_dot - (adjoint(xi_half) @ eta_half)

            # explicit Euler step
            xi_half_next = xi_half + self.ds * xi_der
            eta_half_next = eta_half + self.ds * eta_der
            g_half = g_half @ expm(se(self.ds * xi_half))

            # determine next step from half step value
            self.xi[i + 1, :] = 2 * xi_half_next - prev.xi[i + 1, :]
            self.eta[i + 1, :] = 2 * eta_half_next - prev.eta[i + 1, :]

        # midpoint RKMK to step the g values
        for i in range(self.N):
            self.g[i, :] = flatten(unflatten(prev.g[i, :]) @ expm(se(dt * (self.eta[i, :] + prev.eta[i, :]) / 2)))

        return g_half

def grav_energy(sys):
    H = sys.energy()
    for i in range(sys.N):
        H += -sys.ds * sys.rho * sys.A * sys.g[i, 9] * 9.81  # negative sign because it drops below the original height
    return H


