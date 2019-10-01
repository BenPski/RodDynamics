import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Simulations.Loads import Gravity, PointLoadBody, PointLoadFixed
import numpy as np
from Simulations.Rod import RodFixedFree, Series

if __name__ == "__main__":

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=0, azim=-90)
    plt.axis('equal')
    dt = 0.01
    steps = 100
    N = 100
    l1 = [PointLoadFixed(np.array([0, 0, 0, 10e-3 * 9.81, 0, 0])), Gravity(np.array([9.81, 0, 0]))]
    l2 = [Gravity(np.array([9.81, 0, 0]))]

    r1 = RodFixedFree(1e-2, 10e-2 / 2, 1e6, 1e3, 0, N // 2, xi_init=lambda s: np.array([0, 0, 0, 0, 0, 1]), loads=l1)
    r2 = RodFixedFree(1e-2, 10e-2 / 2, 1e6, 1e3, 0, N // 2, xi_init=lambda s: np.array([0, 0, 0, 0, 0, 1]), loads=l2)

    rod = Series([r1, r2])
    ax = rod.plot(ax)
    plt.pause(0.01)
    for i in range(steps):
        print(i, "/", steps)
        rod.step(dt, np.array([]))
        ax = rod.plot(ax)
        plt.pause(0.01)
    plt.show()
