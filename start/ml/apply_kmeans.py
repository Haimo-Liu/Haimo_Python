import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random
import pandas as pd

style.use('ggplot')

from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 100)

class k_means:
    def __init__(self):
        url = '/Users/haimo.liu/Documents/python_files/titanic.xls'
        self.df = pd.read_excel(url)
        self.df = self.df.drop(['body', 'name', 'ticket', 'pclass'], axis=1)
        self.df = self.df.fillna(0)
        #print(self.df.head())

        self.df_processed = pd.get_dummies(self.df)
        #print(self.df_processed.head())
        self.x_features = np.array(self.df_processed.drop(['survived'], axis=1))
        self.x_features = preprocessing.scale(self.x_features)
        self.y_label = np.array(self.df_processed['survived'])

    def fit(self):
        self.clf = KMeans(n_clusters=2)
        self.clf.fit(self.x_features)
        # print(self.x_features)

    def confidence(self):
        self.correct = 0
        self.clf_label = self.clf.labels_
        # for i in range(5):
        #     print(self.y_label[i])
        #     print(self.clf_label[i])

        for i in range(len(self.x_features)):
            if self.y_label[i] == self.clf_label[i]:
                self.correct += 1

            accuracy = self.correct * 1.0/len(self.x_features)

        return accuracy


def main():
    haimo_k_means = k_means()
    haimo_k_means.fit()
    print(haimo_k_means.confidence())


if __name__ == '__main__':
    main()


