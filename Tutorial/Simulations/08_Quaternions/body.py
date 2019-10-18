"""
General body definition
"""
import numpy as np
from abc import ABCMeta, abstractmethod


class Body(metaclass=ABCMeta):
    """
    General interface for the body definition

    has to implement the values associated with the material and geometric parameters
    Psi, Psi', and dPsi/dxi
    """

    @abstractmethod
    def Psi(self, xi, s):
        return np.array([0, 0, 0, 0, 0, 0])

    @abstractmethod
    def Psi_prime(self, xi, s):
        return np.array([0, 0, 0, 0, 0, 0])

    @abstractmethod
    def Psi_der(self, xi, s):
        return np.zeros((6, 6))

    @abstractmethod
    def M(self, s):
        pass

    @abstractmethod
    def viscosity(self, xi_dot, s):
        return np.array([0, 0, 0, 0, 0, 0])

    @property
    @abstractmethod
    def L(self):
        pass

    def grav_data(self, s):
        raise NotImplementedError("Tried to get gravity data for the rod, but doesn't know how.")


class DefaultBody(Body):

    def __init__(self, D, E, L, rho, mu, xi_ref=np.array([0, 0, 0, 0, 0, 1])):
        self.A = np.pi / 4 * D ** 2
        self.I = np.pi / 64 * D ** 4
        self.J = 2 * self.I
        self.E = E
        self.G = E / 3
        self.L_int = L
        self.rho = rho
        self.mu = mu
        self.xi_ref = xi_ref

    @property
    def L(self):
        return self.L_int

    def M(self, s):
        return self.rho * np.diag([self.I, self.I, self.J, self.A, self.A, self.A])

    def viscosity(self, xi_dot, s):
        V = self.mu * np.diag([3 * self.I, 3 * self.I, self.J, self.A, self.A, 3 * self.A])
        return V @ xi_dot

    def Psi(self, xi, s):
        return self.Psi_der(xi, s) @ (xi - self.xi_ref)

    def Psi_der(self, xi, s):
        K = np.diag(
            [self.E * self.I, self.E * self.I, self.G * self.J, self.A * self.G, self.A * self.G, self.A * self.E])
        return K

    def Psi_prime(self, xi, s):
        return super().Psi_prime(xi, s)

    def grav_data(self, s):
        return self.rho * self.A


class SimpleBody(Body):
    """
    The interface for defining a body with a selected material model and the geometry
    """

    def __init__(self, material, geometry, xi_ref=np.array([0, 0, 0, 0, 0, 1])):
        self.material = material
        self.geometry = geometry
        self.xi_ref = xi_ref  # ?

    @property
    def L(self):
        return self.geometry.L

    def Psi(self, xi, s):
        return self.K(s) @ (xi - self.xi_ref)

    def Psi_der(self, xi, s):
        return self.K(s)

    def Psi_prime(self, xi, s):
        return self.K_prime(s) @ (xi - self.xi_ref)

    def M(self, s):
        A = self.geometry.A(s)
        Ix = self.geometry.Ix(s)
        Iy = self.geometry.Iy(s)
        J = self.geometry.J(s)  # J= Ix + Iy ?
        return self.material.rho * np.diag([Ix, Iy, J, A, A, A])

    def viscosity(self, xi_dot, s):
        A = self.geometry.A(s)
        Ix = self.geometry.Ix(s)
        Iy = self.geometry.Iy(s)
        J = self.geometry.J(s)  # J= Ix + Iy ?
        return self.material.mu * np.diag([3 * Ix, 3 * Iy, J, A, A, 3 * A]) @ xi_dot

    def K(self, s):
        E = self.material.E
        G = self.material.G
        A = self.geometry.A(s)
        Ix = self.geometry.Ix(s)
        Iy = self.geometry.Iy(s)
        J = self.geometry.J(s)  # J= Ix + Iy ?
        return np.diag([E * Ix, E * Iy, G * J, G * A, G * A, E * A])

    def K_prime(self, s):
        E = self.material.E
        G = self.material.G
        A = self.geometry.A_prime(s)
        Ix = self.geometry.Ix_prime(s)
        Iy = self.geometry.Iy_prime(s)
        J = self.geometry.J_prime(s)  # J= Ix + Iy ?
        return np.diag([E * Ix, E * Iy, G * J, G * A, G * A, E * A])

    def grav_data(self, s):
        return self.geometry.A(s)*self.material.rho


class FirstOrderMaterial():
    """
    First order incompressible hyperelastic model
    """

    def __init__(self, E, G, rho, mu):
        self.E = E
        self.G = G
        self.rho = rho
        self.mu = mu


class Geometry(metaclass=ABCMeta):
    """
    Interface for defining the geometry of the rod
    """

    @property
    @abstractmethod
    def L(self):
        pass

    @abstractmethod
    def A(self, s):
        pass

    @abstractmethod
    def A_prime(self, s):
        pass

    @abstractmethod
    def Ix(self, s):
        pass

    @abstractmethod
    def Ix_prime(self, s):
        pass

    @abstractmethod
    def Iy(self, s):
        pass

    @abstractmethod
    def Iy_prime(self, s):
        pass

    @abstractmethod
    def J(self, s):
        pass

    @abstractmethod
    def J_prime(self, s):
        pass


class Cylinder(Geometry):
    """
    Cylinder with constant cross section
    """

    def __init__(self, D, L):
        self.D = D
        self.L_int = L

    @property
    def L(self):
        return self.L_int

    def A(self, s):
        return np.pi / 4 * self.D ** 2

    def A_prime(self, s):
        return 0

    def Ix(self, s):
        return np.pi / 64 * self.D ** 4

    def Ix_prime(self, s):
        return 0

    def Iy(self, s):
        return np.pi / 64 * self.D ** 4

    def Iy_prime(self, s):
        return 0

    def J(self, s):
        return np.pi / 32 * self.D ** 4

    def J_prime(self, s):
        return 0
