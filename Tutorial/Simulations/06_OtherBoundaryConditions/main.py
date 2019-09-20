import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from loads import Gravity, PointLoadBody, PointLoadFixed
import numpy as np
from rod import RodFixedFree, Series, RodFixedFixed, RodFreeFree

if __name__ == "__main__":

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=0, azim=-90)
    plt.axis('equal')
    dt = 0.01
    steps = 50
    N = 100

    loads = [Gravity(np.array([9.81,0,0]))]

    rod = RodFreeFree(1e-2, 10e-2, 1e6, 1e3, 0, N, xi_init=lambda s: np.array([0,0,0,0,0,1]))
    ax = rod.plot(ax)
    plt.pause(0.01)
    for i in range(steps):
        print(i, "/", steps)
        rod.step(dt,np.array([]))
        ax = rod.plot(ax)
        plt.pause(0.01)
    plt.show()



