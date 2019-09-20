import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from loads import Gravity, PointLoadBody, PointLoadFixed
import numpy as np
from rod import RodFixedFree, Series, RodFixedFixed, RodFreeFree

if __name__ == "__main__":

    fig = plt.figure()
    ax = fig.add_subplot(111)#, projection='3d')
    # ax.view_init(elev=0, azim=-90)
    # plt.axis('equal')
    dt = 0.05
    steps = 1000
    N = 100
    # loads = [TipLoad(np.array([0,1e6*np.pi/64*1e-2**4*np.pi/(4*10e-2),0,0,0,0]))]
    # loads = []
    loads = [Gravity(np.array([0,0,0]))]

    # rod = RodFixedFixed(1e-2, 10e-2, 1e7, 1e3, 30000, N, np.array([[1,0,0,0],[0,1,0,0],[0,0,1,10e-2],[0,0,0,1]]), xi_init=lambda s: np.array([0, 0, 0, 0, 0, 1]), loads=loads)
    # ax = rod.plot(ax)
    # plt.pause(0.01)
    # for i in range(steps):
    #     print(i, "/", steps)
    #     rod.step(dt,np.array([]))
    #     ax = rod.plot(ax)
    #     plt.pause(0.01)
    # plt.show()
    E = []

    rod = RodFreeFree(1e-2, 10e-2, 1e6, 1e3, 0, N, xi_init=lambda s: np.array([0,np.pi/(6*10e-2),0,0,0,1]))
    # ax = rod.plot(ax)
    # plt.pause(0.01)
    for i in range(steps):
        print(i, "/", steps)
        rod.step(dt,np.array([]))
        # ax = rod.plot(ax)
        # plt.pause(0.01)
        E.append(rod.energy())
    t = np.linspace(0,steps*dt,len(E))
    ax.plot(t,E)
    plt.show()



