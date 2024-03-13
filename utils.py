import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


def distance_manhattan(centroid, datapoint):
    centroid = np.array(centroid)
    datapoint = np.array(datapoint)
    dist = np.sum(np.abs(centroid - datapoint))
    return dist

def distance_minkowski(centroid, datapoint, p):
    centroid = np.array(centroid)
    datapoint = np.array(datapoint)
    dist = np.sum(np.abs(centroid - datapoint) ** p)
    return (dist ** (1/p))

def distance_euclidean(centroid, datapoint):
    centroid = np.array(centroid)
    datapoint = np.array(datapoint)
    dist = np.sqrt(np.sum((centroid - datapoint)**2))
    return dist

def distance_cosine(centroid, datapoint):
    centroid = np.array(centroid)
    datapoint = np.array(datapoint)
    up = np.sum(centroid * datapoint)
    down1 = np.sum(centroid * centroid)
    down2 = np.sum(datapoint * datapoint)
    dist = 1 - (up / (np.sqrt(down1) * np.sqrt(down2)))
    return dist

def distance_hamming(centroid, datapoint):
    centroid = np.array(centroid)
    datapoint = np.array(datapoint)
    dist =0
    for c,d in zip(centroid, datapoint):
        if c != d:
            dist += 1
    return dist

def normalize_data(X):
        scaler = StandardScaler()
        X_normalized = scaler.fit_transform(X)
        return X_normalized

def load_data(file_name):
    with open(file_name, 'r') as f:
        lines = f.readlines()
    X = []
    for line in lines:
        parts = line.strip().split(',')
        X.append([float(x) for x in parts[:-1]])
    return np.array(X)

def plot_clusters(clusters, centroids):
        plt.figure(figsize=(8, 6))
        for i, cluster in enumerate(clusters.values()):
            cluster_points = np.array(cluster)
            plt.scatter(cluster_points[:, 2],   cluster_points[:, 0], label=f'Cluster {i+1}')
        centroids = np.array(centroids)
        plt.scatter(centroids[:, 2], centroids[:, 0], marker='*', s=200, c='black', label='Centroids')
        plt.title('Clusters and Centroids')
        plt.xlabel('Petal length')
        plt.ylabel('Sepal length')
        plt.legend()
        plt.show()
