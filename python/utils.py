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


# flatten a homogeneous transformation matrix to a vector
def flatten(g):
    return np.concatenate([np.reshape(g[:3, :3], (9,)), g[:3, 3]])


# unflatten a homogeneous transformation
def unflatten(g):
    return np.row_stack((np.column_stack((np.reshape(g[:9], (3, 3)), g[9:])), np.array([0, 0, 0, 1])))


# the matrix representation of a twist vector
def se(x):
    return np.row_stack((np.column_stack((skew(x[:3]), x[3:])), np.array([0, 0, 0, 0])))


# inverse of se
def unse(x):
    return np.array([x[2, 1], x[0, 2], x[1, 0], x[0, 3], x[1, 3], x[2, 3]])
