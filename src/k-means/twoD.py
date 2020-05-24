import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans
from typing import List, Tuple

# https://towardsdatascience.com/machine-learning-algorithms-part-9-k-means-example-in-python-f2ad05ed5203

TwoDArray = List[Tuple[float, float]]


def displayData(data: TwoDArray):
    plt.scatter(data[:, 0], data[:, 1])
    plt.show()


def checkClusters(data: TwoDArray):
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


def categoriseData(data: TwoDArray):
    kmeans = KMeans(n_clusters=4, init='k-means++',
                    max_iter=300, n_init=10, random_state=0)
    kmeans.fit_predict(data)
    plt.scatter(data[:, 0], data[:, 1])
    plt.scatter(kmeans.cluster_centers_[:, 0],
                kmeans.cluster_centers_[:, 1], s=300, c='red')
    plt.show()
    # plt.savefig("plot.png")


if __name__ == "__main__":
    randomData: TwoDArray
    y: TwoDArray
    randomData, y = make_blobs(n_samples=300, centers=4,
                               cluster_std=0.60, random_state=0)

    # Uncomment to first visualise your data
    # displayData(randomData)

    # Uncomment to calculate the number of features
    # checkClusters(randomData)

    # Finally uncomment to run your k-means and categorise the data
    # categoriseData(randomData)
    print("Done")
