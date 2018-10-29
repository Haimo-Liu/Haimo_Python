import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.cluster import KMeans


class target_data():

    def __init__(self):
        self.x_features = np.array([[1, 2],
                                    [1.5, 1.8],
                                    [10, 12],
                                    [9, 13],
                                    [2.1, 2.5],
                                    [11, 9]])

    def show_data(self):
        plt.scatter(x=self.x_features[:, 0], y=self.x_features[:, 1], s=100)
        plt.show()

class k_means():
    def __init__(self, k=2, tol=0.001, max_iteration=500):
        self.k = k
        self.tol = tol
        self.max_ite = max_iteration

    def fit(self, data):

        self.centroids = {}

        for i in range(self.k):
            self.centroids[i] = data[i]

        for i in range(self.max_ite):

            self.classifications = {}

            for i in range(self.k):
                self.classifications[i] = []

            for featureset in data:
                for centroid in self.centroids:
                    distances = np.linalg.norm(data[featureset] - self.centroids[centroid])

                selection = distances.index(min(distances))
                self.classifications[selection].append(data[featureset])

            prev_centroids = dict(self.centroids)

            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification], axis=0)

            optimized = True

            for c in self.centroids:
                if np.sum((self.centroids[c] - prev_centroids[c]) * 1.0 /prev_centroids[c]) > self.tol:
                    optimized = False

            if optimized == True:
                break


    def predict(self, data):
        for centroid in self.centroids:
            distances = np.linalg.norm(data[data] - self.centroids[centroid])
        classification = distances.index(min(distances))
        return classification


def main():
    dataset = target_data()
    #dataset.show_data()

    clf = k_means()
    clf.fit(dataset.x_features)

    for centroid in clf.centroids:
        plt.scatter(clf.centroids[centroid][0], clf.centroids[centroid][1], s=120, color='b')
        plt.show()



if __name__ == '__main__':
    main()