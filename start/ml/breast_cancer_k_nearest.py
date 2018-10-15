import numpy as np
from sklearn import preprocessing, neighbors
from sklearn.model_selection import train_test_split
import pandas as pd


def main():

    url = '/Users/haimo.liu/Documents/python_files/breast_cancer/breast-cancer-wisconsin.data.txt'

    cancer_df = pd.read_csv(url)
    cancer_df.replace('?', -99999, inplace=True)
    # - 99999 is going to be an outlier, and be handled by the algorithm itself

    cancer_df.drop(['id'], axis=1, inplace=True)

    #print(cancer_df.head())

    x_f = np.array(cancer_df.drop(['class'], axis=1))
    y_l = np.array(cancer_df['class'])

    x_train, x_test, y_train, y_test = train_test_split(x_f, y_l, test_size=0.2)

    clf = neighbors.KNeighborsClassifier()
    clf.fit(x_train, y_train)

    accuracy = clf.score(x_test, y_test)

    print(accuracy)

    example_point = np.array([[4,2,1,1,1,2,3,4,1]])

    #example_point = example_point.reshape(2,-1)

    res = clf.predict(example_point)

    print(res)



if __name__ == '__main__':
    main()

