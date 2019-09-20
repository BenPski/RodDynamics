"""
The generalized load defintions
"""
from abc import ABCMeta, abstractmethod
import numpy as np
from utils import skew, Adjoint


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
        return super().tip_load(g, xi, eta, xi_dot, eta_dot, rod, q)


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


class PointLoadBody(Load):
    """
    A constant point load defined in the body frame
    """

    def __init__(self, W):
        self.W = W

    def dist_load(self, g, xi, eta, xi_dot, eta_dot, rod, q):
        return super().dist_load(g, xi, eta, xi_dot, eta_dot, rod, q)

    def tip_load(self, g, xi, eta, xi_dot, eta_dot, rod, q):
        return self.W

class PointLoadFixed(Load):
    """
    A constant point load defined in the fixed frame
    """
    def __init__(self, W):
        self.W = W

    def dist_load(self, g, xi, eta, xi_dot, eta_dot, rod, q):
        return super().dist_load(g, xi, eta, xi_dot, eta_dot, rod, q)

    def tip_load(self, g, xi, eta, xi_dot, eta_dot, rod, q):
        return Adjoint(g).T @ self.W