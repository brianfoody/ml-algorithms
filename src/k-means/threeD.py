import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans
from typing import List, Tuple

# https://towardsdatascience.com/machine-learning-algorithms-part-9-k-means-example-in-python-f2ad05ed5203

ThreeDArray = List[Tuple[float, float, float]]


def displayData(data: ThreeDArray):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(data[:, 0], data[:, 1], data[:, 2])
    plt.show()


def checkClusters(data: ThreeDArray):
    wcss: List[float] = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++',
                        max_iter=300, n_init=10, random_state=0)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)

    plt.plot(range(1, 11), wcss)
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()


def categoriseData(data: ThreeDArray):
    kmeans = KMeans(n_clusters=5, init='k-means++',
                    max_iter=300, n_init=10, random_state=0)
    kmeans.fit_predict(data)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(data[:, 0], data[:, 1], data[:, 2])

    ax.scatter(kmeans.cluster_centers_[:, 0],
               kmeans.cluster_centers_[:, 1],
               kmeans.cluster_centers_[:, 2],
               s=300,
               c='red')
    plt.show()


if __name__ == "__main__":
    randomData: ThreeDArray
    y: ThreeDArray
    randomData, y = make_blobs(n_samples=300, centers=5, n_features=3,
                               cluster_std=0.60, random_state=0)

    # Uncomment to first visualise your data
    # displayData(randomData)

    # Uncomment to calculate the number of features
    # checkClusters(randomData)

    # Finally uncomment to run your k-means and categorise the data
    # categoriseData(randomData)
    print("Done")
