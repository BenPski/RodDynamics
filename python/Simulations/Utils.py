"""
Just some utility functions
"""
import numpy as np


# Some utilities
# map a vector to a skew symmetric matrix
def skew(x):
    return np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])


# inverse of skew
def unskew(x):
    return np.array([x[2, 1], x[0, 2], x[1, 0]])


# map a twist to its adjoint form
def adjoint(x):
    return np.concatenate(
        [np.concatenate([skew(x[:3]), np.zeros((3, 3))], 1), np.concatenate([skew(x[3:]), skew(x[:3])], 1)])


# adjoint form of the group
def Adjoint(g):
    return np.concatenate(
        [np.concatenate([g[:3, :3], np.zeros((3, 3))], 1), np.concatenate([skew(g[:3, 3]) @ g[:3, :3], g[:3, :3]], 1)])


# flatten the configuration
def flatten(g):
    return np.concatenate([toQuaternion(g[:3, :3]), g[:3, 3]])


# unflatten the configuration
def unflatten(g):
    return np.row_stack((np.column_stack((toMatrix(g[:4]), g[4:])), np.array([0, 0, 0, 1])))


# the matrix representation of a twist vector
def se(x):
    return np.row_stack((np.column_stack((skew(x[:3]), x[3:])), np.array([0, 0, 0, 0])))


# inverse of se
def unse(x):
    return np.array([x[2, 1], x[0, 2], x[1, 0], x[0, 3], x[1, 3], x[2, 3]])


# quaternion form to matrix form
def toMatrix(q):
    return np.eye(3) + 2 / (q @ q) * (np.array(
        [[-q[2] ** 2 - q[3] ** 2, q[1] * q[2] - q[3] * q[0], q[1] * q[3] + q[2] * q[0]],
         [q[1] * q[2] + q[3] * q[0], -q[1] ** 2 - q[3] ** 2, q[2] * q[3] - q[1] * q[0]],
         [q[1] * q[3] - q[2] * q[0], q[2] * q[3] + q[1] * q[0], -q[1] ** 2 - q[2] ** 2]]))


# matrix form to quaternion
def toQuaternion(R):
    w = 1 / 2 * np.sqrt(1 + R[0, 0] + R[1, 1] + R[2, 2])
    x = 1 / (4 * w) * (R[2, 1] - R[1, 2])
    y = 1 / (4 * w) * (R[0, 2] - R[2, 0])
    z = 1 / (4 * w) * (R[1, 0] - R[0, 1])
    return np.array([w, x, y, z])
