import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import random
import statistics

from sklearn import svm
from sklearn.model_selection import train_test_split


def main():

    url = '/Users/haimo.liu/Documents/python_files/breast_cancer/breast-cancer-wisconsin.data.txt'

    cancer_df = pd.read_csv(url)
    cancer_df.replace('?', -99999, inplace=True)
    cancer_df.drop(['id'], axis=1, inplace=True)

    x_features = np.array(cancer_df.drop(['class'], axis=1))
    y_label = np.array(cancer_df['class'])

    x_train, x_test, y_train, y_test = train_test_split(x_features, y_label, test_size=0.2)


    #soft margin classifier...
    clf = svm.SVC()
    clf.fit(x_train, y_train)

    accuracy = clf.score(x_test, y_test)

    print(accuracy)


if __name__ == '__main__':
    main()
