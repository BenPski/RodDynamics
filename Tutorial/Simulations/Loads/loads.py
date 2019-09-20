import numpy as np
from scipy.optimize import fsolve
from scipy.linalg import expm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import copy

from utils import *


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
    """

    def __init__(self, D, L, E, rho, mu, N, xi_init=lambda s: np.array([0, 0, 0, 0, 0, 1]),
                 eta_init=lambda s: np.array([0, 0, 0, 0, 0, 0])):
        # setup properties
        A = np.pi / 4 * D ** 2
        I = np.pi / 64 * D ** 4
        J = 2 * I
        G = E / 3  # assuming incompressible material

        # store values important to simulation
        self.K = np.diag([E * I, E * I, G * J, G * A, G * A, E * A])
        self.M = rho * np.diag([I, I, J, A, A, A])
        self.V = mu * np.diag([3*I,3*I,J,A,A,3*A])
        self.L = L
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

    def _initRod(self):
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
        G = np.eye(4)
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

    def step(self, dt):
        # since we are modifying the state want to keep track of the previous state for the integration process
        prev = copy.deepcopy(self)
        # just need to solve for xi0 and the state should be updated
        xi0 = fsolve(lambda x: self._condition(prev, dt, x), self.xi[0, :])

    def _condition(self, prev, dt, xi0):
        # integrate and see if the tip condition is satisfied
        self._integrate(prev, dt, xi0)
        return self.xi[-1, :] - self.xi_ref

    def _integrate(self, prev, dt, xi0):
        self.xi[0, :] = xi0
        grav = np.array([-9.81,0,0])

        # integration over the body (don't need the initial point as the initial values are determined already)
        g_half = np.eye(4) # known initial condition
        for i in range(self.N - 1):
            # averaging over steps to get half step values
            xi_half = (self.xi[i, :] + prev.xi[i, :]) / 2
            eta_half = (self.eta[i, :] + prev.eta[i, :]) / 2

            # implicit midpoint approximation
            xi_dot = (self.xi[i, :] - prev.xi[i, :]) / dt
            eta_dot = (self.eta[i, :] - prev.eta[i, :]) / dt

            # external loads
            W_bar = 0
            W_bar += self.V @ xi_dot
            R = g_half[:3,:3]
            W_bar += self.rho *self.A * np.concatenate([np.array([0,0,0]), R.T @ grav])

            # spatial derivatives
            xi_der = np.linalg.inv(self.K) @ (
                    (self.M @ eta_dot) - (adjoint(eta_half).T @ self.M @ eta_half) + (
                        adjoint(xi_half).T @ self.K @ (xi_half - self.xi_ref)) + W_bar)
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


def grav_energy(sys):
    H = sys.energy()
    for i in range(sys.N):
        H += -sys.ds*sys.rho*sys.A*sys.g[i,9]*9.81 # negative sign because it drops below the original height
    return H

# Call the script as python conservative.py
if __name__ == "__main__":
    fig = plt.figure()
    ax = fig.add_subplot(111)#, projection='3d')
    # ax.set_aspect('equal')
    # ax.view_init(elev=0, azim=90)
    E = []
    # tip = []
    sys = Rod(1e-2, 10e-2, 1e6, 1e3, 30000, 100, xi_init=lambda s: np.array([0, 0, 0, 0, 0, 1]))
    # tip.append(sys.g[-1,9])
    for i in range(100):
        sys.step(0.01)
        # tip.append(sys.g[-1,9])
        E.append(grav_energy(sys))
    ax.plot(E)
    # ax.plot(tip)
    plt.show()
