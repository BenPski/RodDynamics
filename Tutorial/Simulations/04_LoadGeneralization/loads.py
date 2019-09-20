import numpy as np
from scipy.optimize import fsolve
from scipy.linalg import expm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import copy
from abc import ABCMeta, abstractmethod
import csv

from utils import skew, adjoint, flatten, unflatten, se


class Load(metaclass=ABCMeta):
    """
    The general class for dealing with loads
    need to implement distributed load that gives both A_bar and B_bar
    need to implement tip load
    takes current g, xi, eta, rod properties, and inputs
    """

    @abstractmethod
    def dist_load(self, g, xi, eta, xi_dot, eta_dot, rod, q):
        return 0, 0  # if no distributed load

    @abstractmethod
    def tip_load(self, g, xi, eta, xi_dot, eta_dot, rod, q):
        return 0  # no tip load


class Gravity(Load):
    """
    Gravity load
    """

    def __init__(self, grav):
        self.grav = grav  # acceleration vector

    def dist_load(self, g, xi, eta, xi_dot, eta_dot, rod, q):
        R = g[:3, :3]
        A_bar = 0
        B_bar = rod.rho * rod.A * np.concatenate([np.array([0, 0, 0]), R.T @ self.grav])

        return (A_bar, B_bar)

    def tip_load(self, g, xi, eta, xi_dot, eta_dot, rod, q):
        return 0


class Cables(Load):
    """
    Cables load
    """

    def __init__(self, r, N):
        self.r = r  # displacements
        self.N = N  # number of actuators

    def dist_load(self, g, xi, eta, xi_dot, eta_dot, rod, q):
        omega = xi[:3]
        nu = xi[3:]
        R = g[:3, :3]
        A_bar = 0
        B_bar = 0
        for i in range(self.N):
            r_i = self.r(i)
            pa_der = R @ (nu - skew(r_i) @ omega)
            P = R.T @ -skew(pa_der) * skew(pa_der) / np.linalg.norm(pa_der) ** 3 @ R
            b = P @ skew(omega) @ (nu - skew(r_i) @ omega)
            B_bar += q[i] * np.concatenate([skew(r_i) @ b, b])
            A_bar += q[i] * np.concatenate([np.concatenate([-skew(r_i) @ P @ skew(r_i), skew(r_i) @ P], 1),
                                            np.concatenate([-P @ skew(r_i), P], 1)])

        return A_bar, B_bar

    def tip_load(self, g, xi, eta, xi_dot, eta_dot, rod, q):
        W = 0
        z = np.array([0, 0, 1])
        for i in range(self.N):
            W += q[i] * np.concatenate([-skew(self.r(i)) @ z, z])
        return W


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

    def __init__(self, D, L, E, rho, mu, N, xi_init=lambda s: np.array([0, 0, 0, 0, 0, 1]),
                 eta_init=lambda s: np.array([0, 0, 0, 0, 0, 0]), loads=[]):
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
        self.loads = loads

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
        eta = self.xi[-1, :]
        xi_dot = (self.xi[-1, :] - prev.xi[-1, :]) / dt
        eta_dot = (self.eta[-1, :] - prev.eta[-1, :]) / dt
        for load in self.loads:
            W += load.tip_load(g, xi, eta, xi_dot, eta_dot, self, q)

        return self.K @ (self.xi[-1, :] - self.xi_ref) - W

    def _integrate(self, prev, dt, xi0, q):
        self.xi[0, :] = xi0

        # integration over the body (don't need the initial point as the initial values are determined already)
        g_half = np.eye(4)  # known initial condition
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


def grav_energy(sys):
    H = sys.energy()
    for i in range(sys.N):
        H += -sys.ds * sys.rho * sys.A * sys.g[i, 9] * 9.81  # negative sign because it drops below the original height
    return H


def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


# Call the script as python conservative.py
if __name__ == "__main__":

    # # fea results
    # time = []
    # disp = []
    # with open("/home/ben/School/research/rodSimulation/Simulations/fea_results_grav.csv") as f:
    #     reader = csv.reader(f)
    #     for row in reader:
    #         time.append(float(row[0]))
    #         disp.append(float(row[1]))

    E_sim = [0.00054501, 0.000354287, 0.000293463, 0.000244743, 0.00020238, 0.000167151, 0.000138178, 0.000114085, 9.43419e-05, 7.80093e-05, 6.45031e-05, 5.33374e-05, 4.41078e-05, 3.65003e-05, 3.01801e-05, 2.49619e-05, 2.06665e-05, 1.7086e-05, 1.41347e-05, 1.17051e-05, 9.67604e-06, 8.00652e-06, 6.6307e-06, 5.48249e-06, 4.53572e-06, 3.75576e-06, 3.1034e-06, 2.57066e-06, 2.12877e-06, 1.75871e-06, 1.45875e-06, 1.20797e-06, 9.98368e-07, 8.29222e-07, 6.86594e-07, 5.68002e-07, 4.72411e-07, 3.91084e-07, 3.24019e-07, 2.69836e-07, 2.25508e-07, 1.8661e-07, 1.5532e-07, 1.29478e-07, 1.07395e-07, 9.19264e-08, 7.67912e-08, 6.36929e-08, 5.45449e-08, 4.54214e-08, 3.77849e-08, 3.23601e-08, 2.72821e-08, 2.24623e-08, 1.91585e-08, 1.6014e-08, 1.3277e-08, 1.13315e-08, 9.47353e-09, 7.82886e-09, 6.66627e-09, 5.47976e-09, 4.66731e-09, 3.98774e-09, 3.28724e-09, 2.9449e-09, 2.85995e-09, 2.36212e-09, 2.13012e-09, 2.02359e-09, 1.64018e-09, 1.71847e-09, 1.66555e-09, 1.3333e-09, 1.40519e-09, 1.34821e-09, 1.08317e-09, 1.1495e-09, 1.09113e-09, 8.80205e-10, 9.40192e-10, 8.82811e-10, 7.15645e-10, 7.68814e-10, 7.14077e-10, 5.82151e-10, 6.2851e-10, 5.77456e-10, 4.73803e-10, 5.13666e-10, 4.66874e-10, 4.99581e-10, 6.01596e-10, 6.63038e-10, 5.54989e-10, 6.31204e-10, 5.6673e-10, 4.73589e-10, 5.17458e-10, 5.66682e-10, 4.70174e-10, 5.32124e-10, 4.81419e-10, 4.00118e-10, 4.36624e-10, 4.91168e-10, 3.94005e-10, 4.40448e-10, 4.03562e-10, 4.41538e-10, 5.33067e-10, 5.82266e-10, 4.93164e-10, 5.62533e-10, 5.02967e-10, 4.21564e-10, 4.6081e-10, 4.98023e-10, 4.20787e-10, 4.78763e-10, 4.30422e-10, 3.59282e-10, 3.92421e-10, 4.3261e-10, 3.55872e-10, 4.01636e-10, 5.04408e-10, 3.70179e-10, 3.95061e-10, 3.74553e-10, 3.66108e-10, 4.4022e-10, 5.39311e-10, 4.01894e-10, 4.33847e-10, 4.08663e-10, 4.08209e-10, 4.91937e-10, 5.88372e-10, 4.47924e-10, 4.90664e-10, 4.57198e-10, 4.73607e-10, 5.71935e-10, 5.06355e-10, 4.25372e-10, 4.64907e-10, 4.94104e-10, 4.28138e-10, 4.90103e-10, 5.80146e-10, 4.48797e-10, 4.96189e-10, 4.58248e-10, 4.89231e-10, 5.91043e-10, 6.60637e-10, 5.42475e-10, 6.13818e-10, 5.54617e-10, 4.61448e-10, 5.03769e-10, 5.62918e-10, 4.55149e-10, 5.1044e-10, 4.66252e-10, 3.85103e-10, 4.19454e-10, 4.86803e-10, 3.77334e-10, 4.14662e-10, 3.85557e-10, 4.01096e-10, 4.84327e-10, 5.55277e-10, 4.42074e-10, 4.9495e-10, 4.52385e-10, 3.7358e-10, 4.06961e-10, 4.71991e-10, 3.66051e-10, 4.02378e-10, 3.74054e-10, 3.89407e-10, 4.70221e-10, 5.38718e-10, 4.29249e-10, 4.80747e-10, 4.39259e-10, 3.62815e-10, 3.95264e-10, 4.57935e-10, 3.55536e-10, 3.91067e-10, 3.63357e-10, 3.78915e-10, 4.5757e-10, 5.23334e-10, 4.1782e-10, 4.68305e-10, 4.2756e-10, 3.53324e-10, 3.84989e-10, 4.44904e-10, 3.46319e-10, 3.81493e-10, 3.54042e-10, 3.70693e-10, 4.47679e-10, 5.09971e-10, 4.09087e-10, 4.5933e-10, 4.18601e-10, 3.46322e-10, 3.77506e-10, 4.33666e-10, 3.39692e-10, 3.75473e-10, 3.47483e-10, 3.67313e-10, 4.43649e-10, 5.00662e-10, 4.06223e-10, 4.57926e-10, 4.15543e-10, 3.44746e-10, 3.76096e-10, 4.26106e-10, 3.38928e-10, 3.77463e-10, 3.47052e-10, 3.75178e-10, 4.53085e-10, 5.00458e-10, 4.1745e-10, 4.74414e-10, 4.26329e-10, 3.56045e-10, 3.88988e-10, 4.27168e-10, 3.5312e-10, 3.99176e-10, 4.98382e-10, 3.6739e-10, 3.93622e-10, 3.72232e-10, 3.66918e-10, 4.41621e-10, 5.35932e-10, 4.02612e-10, 4.37217e-10, 4.10088e-10, 4.15488e-10, 5.01231e-10, 5.90474e-10, 4.56304e-10, 5.04057e-10, 4.6646e-10, 4.94646e-10, 5.97578e-10, 6.72318e-10, 5.47522e-10, 6.17951e-10, 5.60008e-10, 4.65009e-10, 5.07413e-10, 5.72412e-10, 4.57581e-10, 5.10745e-10, 4.68639e-10, 3.85996e-10, 4.19974e-10, 4.94697e-10, 3.77962e-10, 4.11583e-10, 3.85349e-10, 3.91626e-10, 4.72433e-10, 5.54752e-10, 4.30168e-10, 4.76021e-10, 4.39846e-10, 4.68864e-10, 5.66432e-10, 6.34076e-10, 5.19675e-10, 5.87684e-10, 5.31361e-10, 4.41898e-10, 4.82376e-10, 5.40177e-10, 4.35615e-10, 4.88022e-10, 4.46231e-10, 3.6833e-10, 4.01092e-10, 4.67058e-10, 3.60811e-10, 3.9571e-10, 3.68512e-10, 3.81329e-10, 4.60387e-10, 5.30667e-10, 4.1989e-10, 4.68948e-10, 4.29667e-10, 3.54282e-10, 3.8572e-10, 4.50951e-10, 3.46965e-10, 3.79562e-10, 3.54164e-10, 3.64063e-10, 4.39431e-10, 5.09938e-10, 4.0046e-10, 4.45805e-10, 4.09717e-10, 4.44899e-10, 5.37349e-10, 5.91087e-10, 4.95749e-10, 5.64177e-10, 5.06071e-10, 4.23184e-10, 4.62434e-10, 5.04905e-10, 4.20616e-10, 4.76633e-10, 4.30599e-10, 3.58226e-10, 3.90999e-10, 4.37821e-10, 3.53161e-10, 3.95692e-10, 3.61762e-10, 3.98816e-10, 4.81364e-10, 5.22155e-10, 4.46604e-10, 5.10502e-10, 4.55016e-10, 3.82247e-10, 4.17937e-10, 4.47273e-10, 3.83297e-10, 4.3768e-10, 5.24423e-10, 4.00759e-10, 4.40209e-10, 4.08768e-10, 4.28351e-10, 5.17436e-10, 5.88948e-10, 4.72855e-10, 5.31099e-10, 4.83853e-10, 4.00388e-10, 4.36467e-10, 5.00887e-10, 3.92777e-10, 4.34398e-10, 4.01822e-10, 4.25452e-10, 5.13874e-10, 5.78986e-10, 4.70707e-10, 5.30961e-10, 4.81469e-10, 3.99632e-10, 4.36029e-10, 4.9285e-10, 3.93079e-10, 4.38306e-10, 4.02546e-10, 4.36841e-10, 5.27511e-10, 5.80578e-10, 4.86634e-10, 5.53715e-10, 4.96784e-10, 4.15361e-10, 4.53876e-10, 4.95863e-10, 4.12743e-10, 4.67593e-10, 4.22555e-10, 3.51465e-10, 3.83602e-10, 4.29941e-10, 3.46412e-10, 3.87952e-10, 3.54843e-10, 3.90587e-10, 4.71458e-10, 5.12128e-10, 4.37155e-10, 4.99492e-10, 4.45484e-10, 3.74066e-10, 4.08973e-10, 4.38548e-10, 3.74737e-10, 4.27602e-10, 5.13972e-10, 3.91583e-10, 4.29371e-10, 3.99265e-10, 4.16402e-10, 5.02945e-10, 5.75187e-10, 4.59243e-10, 5.14712e-10, 4.69937e-10, 3.88344e-10, 4.23138e-10, 4.89038e-10, 3.80703e-10, 4.1932e-10, 3.89152e-10, 4.07475e-10, 4.92052e-10, 5.60566e-10, 4.4981e-10, 5.04952e-10, 4.60166e-10, 3.80743e-10, 4.14985e-10, 4.76793e-10, 3.73784e-10, 4.12982e-10, 3.82152e-10, 4.04323e-10, 4.88095e-10, 5.50745e-10, 4.4792e-10, 5.04502e-10, 4.57624e-10, 3.79899e-10, 4.14251e-10, 4.69307e-10, 3.75313e-10, 4.17233e-10, 3.83235e-10, 4.16791e-10, 5.01969e-10, 5.53367e-10, 4.68029e-10, 5.29781e-10, 4.7487e-10, 3.98014e-10, 4.33824e-10, 4.75506e-10, 4.04725e-10, 4.53613e-10, 4.08683e-10, 3.42031e-10, 3.71253e-10, 4.17775e-10, 3.55147e-10, 3.89142e-10, 4.95906e-10, 4.0273e-10, 4.08827e-10, 3.82055e-10, 4.18228e-10, 4.71346e-10, 4.18258e-10, 3.59499e-10, 3.79998e-10, 4.36997e-10, 4.61199e-10, 4.65102e-10, 4.0771e-10, 3.6226e-10, 3.71746e-10, 4.40813e-10, 3.70305e-10, 3.48617e-10, 3.22329e-10, 2.89291e-10, 2.84484e-10, 3.93999e-10, 3.38872e-10, 3.00068e-10, 2.88736e-10, 3.89359e-10, 3.56826e-10, 3.13327e-10, 2.95642e-10, 2.7866e-10, 3.91286e-10, 3.67856e-10, 4.48503e-10, 4.14039e-10, 3.65931e-10, 3.4063e-10, 3.2495e-10, 2.97221e-10, 2.77448e-10, 2.6434e-10, 3.68157e-10, 3.39064e-10, 4.28342e-10, 3.98373e-10, 3.47181e-10, 3.26311e-10, 3.12133e-10, 4.18787e-10, 3.90684e-10, 3.41637e-10, 3.20503e-10, 3.05795e-10, 4.15767e-10, 3.87442e-10, 3.37504e-10, 3.17809e-10, 3.02964e-10, 4.0879e-10, 3.8334e-10, 3.34148e-10, 3.13644e-10, 3.00103e-10, 4.02626e-10, 3.75605e-10, 3.28463e-10, 3.08171e-10, 2.93997e-10, 3.99792e-10, 3.72613e-10, 3.24556e-10, 3.0562e-10, 2.9137e-10, 3.92989e-10, 3.68533e-10, 4.67094e-10, 4.25125e-10, 3.73026e-10, 3.516e-10, 3.32317e-10, 3.03157e-10, 2.86444e-10, 2.70204e-10, 3.77727e-10, 3.5487e-10, 4.34602e-10, 4.00865e-10, 3.53904e-10, 3.29914e-10, 3.14493e-10, 2.87461e-10, 2.68729e-10, 2.55819e-10, 3.55947e-10, 3.28607e-10, 4.14091e-10, 3.84535e-10, 4.84302e-10, 4.43544e-10, 3.89581e-10, 3.66982e-10, 3.47149e-10, 3.16568e-10, 2.98965e-10, 2.82267e-10, 3.93806e-10, 3.69669e-10, 4.54041e-10, 4.18648e-10, 3.69305e-10, 3.44572e-10, 3.28368e-10, 2.99973e-10, 2.80677e-10, 2.67096e-10, 3.71203e-10, 3.43194e-10, 4.32029e-10, 4.0073e-10, 3.49964e-10, 3.28677e-10, 3.13972e-10, 4.24579e-10, 3.95546e-10, 3.45168e-10, 3.24558e-10, 3.09398e-10, 4.19281e-10, 3.92214e-10, 3.41653e-10, 3.212e-10, 3.06884e-10, 4.12237e-10, 3.8558e-10, 3.36809e-10, 3.1594e-10, 3.01863e-10, 4.08194e-10, 3.80352e-10, 3.319e-10, 3.12101e-10, 2.97522e-10, 4.03107e-10, 3.77127e-10, 3.28521e-10, 3.08831e-10, 2.95087e-10, 3.96367e-10, 3.70692e-10, 3.23821e-10, 3.03759e-10, 2.90205e-10, 3.9253e-10, 3.6576e-10, 3.1914e-10, 3.00122e-10, 2.86104e-10, 3.87557e-10, 3.62618e-10, 4.59302e-10, 4.1919e-10, 3.67521e-10, 3.46258e-10, 3.27781e-10, 2.98643e-10, 2.82104e-10, 2.66519e-10, 3.70651e-10, 3.48073e-10, 4.28468e-10, 3.94505e-10, 3.47941e-10, 3.2489e-10, 3.09343e-10, 2.82638e-10, 2.64645e-10, 2.51614e-10, 3.50189e-10, 3.24149e-10, 4.06873e-10, 3.7739e-10, 4.77805e-10, 4.37006e-10, 3.8337e-10, 3.61737e-10, 3.41866e-10, 3.11536e-10, 2.94703e-10, 2.77953e-10, 3.87504e-10, 3.6474e-10, 4.46572e-10, 4.11088e-10, 3.63227e-10, 3.38638e-10, 3.22459e-10, 2.95063e-10, 2.75824e-10, 2.62299e-10, 3.66488e-10, 3.38333e-10, 4.24817e-10, 3.95135e-10, 3.45011e-10, 3.23705e-10, 3.09724e-10, 4.17288e-10, 3.88131e-10, 3.3924e-10, 3.18782e-10, 3.03619e-10, 4.13748e-10, 3.86598e-10, 3.36283e-10, 3.16673e-10, 3.02345e-10, 4.05315e-10, 3.80157e-10, 3.32029e-10, 3.11124e-10, 2.97743e-10, 4.01228e-10, 3.73218e-10, 3.26185e-10, 3.06558e-10, 2.91956e-10, 3.97791e-10, 3.7178e-10, 4.69238e-10, 4.29447e-10, 3.76557e-10, 3.5432e-10, 3.35973e-10, 3.05947e-10, 2.88673e-10, 2.73192e-10, 3.78664e-10, 3.54934e-10, 4.39338e-10, 4.04369e-10, 3.56053e-10, 3.3301e-10, 3.16947e-10, 2.89227e-10, 2.71275e-10, 2.57782e-10, 3.57751e-10, 3.32068e-10, 4.16237e-10, 3.85134e-10, 3.37103e-10, 3.16281e-10, 3.01768e-10, 4.11328e-10, 3.82531e-10, 3.33145e-10, 3.14e-10, 2.99007e-10, 4.0413e-10, 3.79545e-10, 3.30529e-10, 3.10287e-10, 2.97146e-10, 3.97085e-10, 3.70537e-10, 3.2441e-10, 3.04042e-10, 2.90114e-10, 3.95462e-10, 3.67913e-10, 4.67141e-10, 4.28565e-10, 3.7474e-10, 3.53135e-10, 3.35229e-10, 3.04429e-10, 2.8774e-10, 2.72567e-10, 3.74514e-10, 3.52047e-10, 4.37357e-10, 4.00631e-10, 3.52969e-10, 3.30614e-10, 3.13796e-10, 2.86785e-10, 2.69313e-10, 2.55207e-10, 3.56759e-10, 3.31829e-10, 4.12132e-10, 3.82024e-10, 3.3514e-10, 3.13559e-10, 2.99567e-10, 4.09544e-10, 3.79097e-10, 3.30272e-10, 3.1182e-10, 2.96118e-10, 4.02722e-10, 3.79225e-10, 4.74417e-10, 4.32263e-10, 3.8072e-10, 3.57469e-10, 3.38231e-10, 3.09406e-10, 2.91186e-10, 2.75057e-10, 3.86852e-10, 3.61104e-10, 4.43905e-10, 4.11724e-10, 3.62306e-10, 3.37954e-10, 3.23099e-10, 2.94202e-10, 2.75317e-10, 2.62807e-10, 3.60711e-10, 3.33525e-10, 4.24383e-10, 3.91732e-10, 3.41552e-10, 3.21901e-10, 3.06561e-10, 4.15411e-10, 3.89252e-10, 3.38916e-10, 3.18483e-10, 3.04618e-10, 4.07923e-10, 3.81276e-10, 3.33441e-10, 3.12573e-10, 2.98532e-10, 4.05123e-10, 3.77058e-10, 3.28765e-10, 3.09505e-10, 2.94842e-10, 3.99264e-10, 3.74234e-10, 4.72983e-10, 4.31245e-10, 3.78491e-10, 3.56403e-10, 3.37224e-10, 3.07575e-10, 2.90357e-10, 2.74203e-10, 3.82639e-10, 3.58961e-10, 4.41195e-10, 4.06965e-10, 3.58866e-10, 3.34888e-10, 3.19201e-10, 2.91487e-10, 2.72793e-10, 2.59637e-10, 3.60386e-10, 3.33301e-10, 4.19843e-10, 3.89178e-10, 3.39886e-10, 3.19288e-10, 3.0489e-10, 4.12634e-10, 3.84568e-10, 3.35466e-10, 3.15477e-10, 3.00807e-10, 4.07111e-10, 3.80921e-10, 3.31928e-10, 3.11939e-10, 2.98081e-10, 4.00618e-10, 3.74474e-10, 3.27113e-10, 3.06924e-10, 2.93139e-10, 3.96701e-10, 3.69796e-10, 3.22574e-10, 3.03366e-10, 2.89263e-10, 3.91407e-10, 3.66257e-10, 4.64292e-10, 4.23546e-10, 3.71307e-10, 3.49923e-10, 3.31154e-10, 3.01726e-10, 2.8509e-10, 2.69259e-10, 3.74621e-10, 3.5195e-10, 4.32822e-10, 3.98495e-10, 3.51577e-10, 3.28194e-10, 3.12491e-10, 2.85592e-10, 2.67333e-10, 2.54177e-10, 3.54011e-10, 3.27531e-10, 4.11135e-10, 3.81537e-10, 4.82637e-10, 4.41365e-10, 3.87323e-10, 3.65379e-10, 3.45291e-10, 3.14752e-10, 2.97667e-10, 2.80741e-10, 3.91725e-10, 3.68556e-10, 4.51177e-10, 4.15553e-10, 3.67116e-10, 3.42232e-10, 3.25981e-10, 2.98214e-10, 2.78755e-10, 2.65165e-10, 3.70123e-10, 3.41653e-10, 4.29419e-10, 3.99282e-10, 3.48562e-10, 3.27142e-10, 3.12943e-10, 4.21603e-10, 3.92353e-10, 3.42886e-10, 3.2217e-10, 3.06941e-10, 4.17892e-10, 3.904e-10, 3.39703e-10, 3.19831e-10, 3.05331e-10, 4.09724e-10, 3.84159e-10, 3.35453e-10, 3.14434e-10, 3.00847e-10, 4.05367e-10, 3.77271e-10, 3.29689e-10, 3.09809e-10, 2.95145e-10, 4.01782e-10, 3.75432e-10, 3.26672e-10, 3.07539e-10, 2.93634e-10, 3.93908e-10, 3.69286e-10, 3.22507e-10, 3.02282e-10, 2.89202e-10, 3.89845e-10, 3.6279e-10, 3.17e-10, 2.97924e-10, 2.83805e-10, 3.8629e-10, 3.61035e-10, 4.56078e-10, 4.17237e-10, 3.65801e-10, 3.44299e-10, 3.26388e-10, 2.97214e-10, 2.8051e-10, 2.65395e-10, 3.67947e-10, 3.45045e-10, 4.26737e-10, 3.92719e-10, 3.45906e-10, 3.23444e-10, 3.0783e-10, 2.80987e-10, 2.63479e-10, 2.50369e-10, 3.47744e-10, 3.22646e-10, 4.04376e-10, 3.74351e-10, 4.7606e-10, 4.35383e-10, 3.81402e-10, 3.6035e-10, 3.40493e-10, 3.09934e-10, 2.93589e-10, 2.76822e-10, 3.84848e-10, 3.63049e-10, 4.44198e-10, 4.08006e-10, 3.60878e-10, 3.36444e-10, 3.19989e-10, 2.93186e-10, 2.74025e-10, 2.60292e-10, 3.65453e-10, 3.37306e-10, 4.21874e-10, 3.93175e-10, 3.43477e-10, 3.21849e-10, 3.08325e-10, 4.14928e-10, 3.85103e-10, 3.36918e-10, 3.1664e-10, 3.01205e-10, 4.12388e-10, 3.8538e-10, 3.34723e-10, 3.15572e-10, 3.01305e-10, 4.02441e-10, 3.78221e-10, 3.30535e-10, 3.0931e-10, 2.96359e-10, 3.99003e-10, 3.70328e-10]


    fig = plt.figure()
    ax = fig.add_subplot(111)  # , projection='3d')
    # ax.set_aspect('equal')
    # ax.view_init(elev=0, azim=90)
    E = []
    tip = []
    dt = 0.05
    steps = 1000
    N = 100
    n = 50
    loads = [Gravity(np.array([0, 0, 0]))]
    sys = Rod(1e-2, 10e-2, 1e6, 1e3, 0, N, xi_init=lambda s: np.array([0, np.pi/(4*10e-2), 0, 0, 0, 1]), loads=loads)
    # print(sys.g[-1,9])
    # print(sys.K[0,0]*np.pi/(4*10e-2))
    # sys.plot(ax)
    # plt.pause(0.01)
    # tip.append(sys.g[-1, 9])
    for i in range(steps):
        print(i, '/', steps)
        sys.step(dt, np.array([]))
        # sys.plot(ax)
        # plt.pause(0.01)
        E.append(sys.energy())
        # tip.append(sys.g[-1, 9])
        # E.append(grav_energy(sys))
    t = np.linspace(0, steps * dt, len(E))
    t_sim = np.linspace(0,50,len(E_sim))
    # ax.plot(t, -1000 * np.array(tip))
    # print("RMSE:", np.sqrt(np.mean(((-1000 * np.array(tip) - np.interp(t, time, disp)) ** 2))))
    # ax.plot(time, disp)
    ax.plot(t,E)
    ax.plot(t_sim,E_sim)
    # mv = moving_average(np.array(E),n)
    # ax.plot(t[n-1:],mv)
    # ax.plot(t,np.linspace(np.mean(E),np.mean(E), len(E)))
    print("mean: ", np.mean(E))
    print("std: ", np.std(E))

    # ax.plot(tip)
    plt.show()
