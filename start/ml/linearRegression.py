import numpy as np
from statistics import mean
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use('fivethirtyeight')

xs = np.array([1, 2, 3, 4, 5, 6])
ys = np.array([5, 6, 7, 7, 9, 10])


def create_dataset(num_points, variance, step=2, correlation='none'):
    y_result = []
    x_result = []
    init = 1

    for i in range(num_points):
        x_result.append(i)

        y_plus = init + random.randrange(-variance, variance)
        if correlation == 'positive':
            init += step
        elif correlation == 'negative':
            init -= step
        y_result.append(y_plus)

    return np.array(y_result, dtype=np.float64), np.array(x_result, dtype=np.float64)


def best_fit_line(x, y):

    m = (mean(x)*mean(y) - mean(x * y)) / (mean(x) * mean(x) - mean(x * x))
    b = mean(y) - m*mean(x)

    return m, b


def squared_error(y_orig, y_fit):

    er = sum((y_fit - y_orig)**2)

    return er


def coefficient_of_determination(y_orig, y_fit):

    squared_error_fit = squared_error(y_orig, y_fit)

    y_mean = []
    for y in y_orig:
         y_mean.append(mean(y_orig))

    squared_error_mean = squared_error(y_orig, y_mean)
    coefficient = 1 - (squared_error_fit / squared_error_mean)
    return coefficient


def main1():

    m, b = best_fit_line(xs, ys)

    regression_line = []
    for x in xs:
        regression_line.append((m*x) + b)

    r_square = coefficient_of_determination(ys, regression_line)

    print(m, b, r_square)

    plt.scatter(xs, ys)
    plt.plot(xs, regression_line)
    plt.show()


def main2():
    x_t, y_t = create_dataset(10, 2, 2, 'positive')
    print(x_t, y_t)

    plt.plot(x_t, y_t)
    plt.show()


def main():
    x_m, y_m = create_dataset(40, 30, 3, 'positive')
    m, b = best_fit_line(x_m, y_m)

    y_fit = []
    for x in x_m:
        y_fit.append(m*x + b)

    r_square = coefficient_of_determination(y_m, y_fit)

    print(r_square)

    plt.plot(x_m, y_m)
    plt.show()


if __name__ == "__main__":
    main()



