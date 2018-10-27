import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import cvxopt
import cvxopt.solvers


def linear_kernel(x1, x2):
    return np.dot(x1, x2)


def polynomial_kernel(x1, x2, p):
    return (1 + np.dot(x1, x2))**p


def gaussian_kernel(x1, x2, sigma):
    return np.exp(- np.linalg.norm(x1 - x2)**2/(2 * (sigma**2)))



