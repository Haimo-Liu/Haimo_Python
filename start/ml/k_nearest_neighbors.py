from matplotlib import style
from math import sqrt
import random
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

from sklearn import neighbors
from sklearn.model_selection import train_test_split

style.use('fivethirtyeight')

dataset = {'k': [[1, 2], [2, 1], [3, 1]], 'r': [[5, 6], [6, 7], [7, 7]]}
new_point = [8, 7]


def visualize():
    for i in dataset:
        for ii in dataset[i]:
            plt.scatter(ii[0], ii[1], s=50, color=i)

    plt.scatter(new_point[0], new_point[1], s=80)
    plt.show()


def k_nearest_neighbors(data, target, k=3):
    distance_list = []

    for group in data:
        for points in data[group]:
            euclidean_distance = np.linalg.norm(np.array(points) - np.array(target))
            distance_list.append([euclidean_distance, group])

    votes = []
    for i in range(k):
        votes.append(sorted(distance_list)[i][1])

    vote_result = Counter(votes).most_common(1)[0][0]
    confidence = Counter(votes).most_common(1)[0][1] * 1.0 / k

    return vote_result, confidence


def test():
    test_1 = ['a', 'a', 'b']
    vote = Counter(test_1).most_common(1)[0][0]

    print(vote)

    result = k_nearest_neighbors(dataset, new_point, k=3)
    print(result)


def apply_breast_cancer():
    url = '/Users/haimo.liu/Documents/python_files/breast_cancer/breast-cancer-wisconsin.data.txt'

    cancer_df = pd.read_csv(url)
    cancer_df.replace('?', -99999, inplace=True)
    cancer_df.drop(['id'], axis=1, inplace=True)
    full_data = cancer_df.astype(float).values.tolist()

    #print(full_data[:10])

    accuracies = []

    for i in range(25):

        random.shuffle(full_data)

        #print(20 * '#')

        test_size = 0.2
        shift = int(test_size * len(full_data))
        train_data = full_data[:-shift]
        test_data = full_data[-shift:]
        train_dic = {2:[], 4:[]}
        test_dic = {2:[], 4:[]}

        for i in train_data:
            train_dic[i[-1]].append(i[:-1])
        for i in test_data:
            test_dic[i[-1]].append(i[:-1])

        correct = 0
        total = 0

        for group in test_dic:
            for point in test_dic[group]:
                vote, confidence = k_nearest_neighbors(train_dic, point, k=5)
                if vote == group:
                    correct += 1
                #else:
                    #print(confidence)
                total += 1
                #print(correct, total)

        accuracy = correct * 1.0 / total
        accuracies.append(accuracy)

    print(sum(accuracies) * 1.0 / len(accuracies))


# def test2():
#     test = {2:[], 4:[]}
#     print(test)



if __name__ =='__main__':
    apply_breast_cancer()

