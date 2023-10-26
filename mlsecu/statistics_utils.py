import numpy as np


def get_euclidean_dist(a, b):
    return np.linalg.norm(a-b)


def get_l0(a, b):
    return np.linalg.norm(a-b, ord=0)


def get_l1(a, b):
    return np.linalg.norm(a-b, ord=1)


def get_l2(a, b):
    return np.linalg.norm(a-b, ord=2)


def get_linf(a, b):
    return np.linalg.norm(a-b, ord=np.inf)
