import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style

style.use('ggplot')


def data_generate():
    data_dict = {-1: np.array([[2,2], [2,3], [3,2]]),

                 1: np.array([[10,9], [9,11], [11,12]])
                 }


class support_vector_machine:

    def __init__(self, visualization=True):
        self.visualization = visualization
        self.colors = {1: 'r', -1: 'b'}
        if self.visualization:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1, 1, 1)


    def fit(self, data):
        self.data = data
        # dic = {||w||: [w, b]}
        pipline_dic = {}

        transforms = [
            [1, 1],
            [-1, 1],
            [-1, -1],
            [1, -1]
        ]

        all_points = []
        for yi in data:
            for xi in [data[yi]]:
                for coordinate in xi:
                    all_points.append(coordinate)

        self.max_coordinate = max(all_points)
        self.min_coordinate = min(all_points)
        all_data = None

        step_sizes = [[self.max_coordinate * 0.1],
                      [self.max_coordinate] * 0.01,
                      [self.max_coordinate * 0.001]
                      ]

        self.brange_multiple = 5
        self.b_mutiple = 5

        latest_optimum = self.max_coordinate * 10

        for step in step_sizes:
            pass







    def predict(self, features):
        # sign(w.x + b): >0 --> +, <0 --> -
        classification = np.sign(np.dot(np.array(features), self.w) + self.b)

        return classification



def test():
    a = [[1,2], [3,4], [4,1]]
    print(a[0])





if __name__ == '__main__':
    test()
