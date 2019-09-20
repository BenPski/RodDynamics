import numpy as np
from scipy.optimize import fsolve
from scipy.linalg import expm
import matplotlib.pyplot as plt

# Some utilities
# map a vector to a skew symmetric matrix
def skew(x):
    return np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])

# map a twist to its adjoint form
def adjoint(x):
    return np.concatenate(
        [np.concatenate([skew(x[:3]), np.zeros((3, 3))], 1), np.concatenate([skew(x[3:]), skew(x[:3])], 1)])

# flatten a homogeneous transformation matrix to a vector
def flatten(g):
    return np.concatenate([np.reshape(g[:3, :3], (9,)), g[:3, 3]])

# unflatten a homogeneous transformation
def unflatten(g):
    return np.row_stack((np.column_stack((np.reshape(g[:9], (3, 3)), g[9:])), np.array([0, 0, 0, 1])))

# the matrix representation of a twist vector
def se(x):
    return np.row_stack((np.column_stack((skew(x[:3]), x[3:])), np.array([0, 0, 0, 0])))


# Initialization
def initRod(N):
    L = 10e-2 # length of the rod

    g = np.zeros((N, 12))
    xi = np.repeat(np.array([[0, np.pi/4/L, 0, 0, 0, 1]]), N, 0)
    eta = np.zeros((N, 6))

    #explicit Euler RKMK
    G = np.eye(4)
    ds = L / (N - 1)
    g[0, :] = flatten(G)
    for i in range(1, N):
        G = G @ expm(se(ds * xi[i - 1, :]))
        g[i, :] = flatten(G)

    return g, xi, eta

#Integration
def step(g, xi, eta):
    # determine xi0 by solving tip condition
    xi0 = fsolve(lambda x: condition(g, xi, eta, x), xi[0, :])
    # integrate the system with the solved xi0
    return integrate(g, xi, eta, xi0)

def condition(g, xi, eta, xi0):
    g_next, xi_next, eta_next = integrate(g, xi, eta, xi0)
    return xi_next[-1, :] - np.array([0, 0, 0, 0, 0, 1])

def integrate(g, xi, eta, xi0):
    # initialize empty matrices for storage
    g_next = np.zeros_like(g)
    xi_next = np.zeros_like(xi)
    eta_next = np.zeros_like(eta)

    # determine number of spatial points, just believe everything is the right size
    (N, _) = xi.shape

    # set the guessed value
    xi_next[0, :] = xi0

    # material and geometric properties
    xi_ref = np.array([0, 0, 0, 0, 0, 1])
    L = 10e-2
    D = 1e-2
    E = 1e6
    rho = 1e3
    ds = L / (N - 1)
    dt = 0.01
    A = np.pi / 4 * D ** 2
    I = np.pi / 64 * D ** 4
    J = 2 * I
    G = E / 3
    K = np.diag([E * I, E * I, G * J, G * A, G * A, E * A])
    M = rho * np.diag([I, I, J, A, A, A])

    # integration over the body (don't need the initial point as the initial values are determined already)
    for i in range(N - 1):
        # averaging over steps to get half step values
        xi_half = (xi_next[i, :] + xi[i, :]) / 2
        eta_half = (eta_next[i, :] + eta[i, :]) / 2

        # implicit midpoint approximation
        xi_dot = (xi_next[i, :] - xi[i, :]) / dt
        eta_dot = (eta_next[i, :] - eta[i, :]) / dt

        # spatial derivatives
        xi_der = np.linalg.inv(K) @ (
                (M @ eta_dot) - (adjoint(eta_half).T @ M @ eta_half) + (adjoint(xi_half).T @ K @ (xi_half - xi_ref)))
        eta_der = xi_dot - (adjoint(xi_half) @ eta_half)

        # explicit Euler step
        xi_half_next = xi_half + ds * xi_der
        eta_half_next = eta_half + ds * eta_der

        # determine next step from half step value
        xi_next[i + 1, :] = 2 * xi_half_next - xi[i+1, :]
        eta_next[i + 1, :] = 2 * eta_half_next - eta[i+1, :]

    # midpoint RKMK to step the g values
    for i in range(N):
        g_next[i, :] = flatten(unflatten(g[i,:]) @ expm(se(dt * (eta_next[i,:] + eta[i,:])/2)))

    return g_next, xi_next, eta_next


# Testing functions
def plotDynamics(N, steps):
    # start figure
    fig, ax = plt.subplots()
    g, xi, eta = initRod(N)
    ax.plot(g[:,9], g[:,11])
    ax.set_aspect('equal')
    plt.pause(0.01) # make the plots show up as they're updated

    for i in range(steps):
        g, xi, eta = step(g, xi, eta)
        ax.plot(g[:,9], g[:,11])
        plt.pause(0.01) # make the plots show up as they're updated

    #make sure it stays open for looking at and saving
    plt.show()

def energy(xi,eta):
    # similar to the setup for the integrator
    (N, _) = xi.shape
    xi_ref = np.array([0, 0, 0, 0, 0, 1])
    L = 10e-2
    D = 1e-2
    E = 1e6
    rho = 1e3
    ds = L / (N - 1)
    dt = 0.01
    A = np.pi / 4 * D ** 2
    I = np.pi / 64 * D ** 4
    J = 2 * I
    G = E / 3
    K = np.diag([E * I, E * I, G * J, G * A, G * A, E * A])
    M = rho * np.diag([I, I, J, A, A, A])

    H = 0 # total energy

    # integrate over the rod
    for i in range(N):
        T = eta[i,:].T @ M @ eta[i,:]
        U = (xi[i,:]-xi_ref).T @ K @ (xi[i,:]-xi_ref)
        H += 1/2*(T + U)
    return ds*H #multiply by discrete step size to scale

def plotEnergy(N, steps):
    fig, ax = plt.subplots()
    g, xi, eta = initRod(N)
    E = []

    for i in range(steps):
        g, xi, eta = step(g, xi, eta)
        E.append(energy(xi,eta))

    ax.plot(E)
    plt.show()

# Call the script as python conservative.py
if __name__ == "__main__":
    # plotDynamics(100, 20)
    plotEnergy(100,100)