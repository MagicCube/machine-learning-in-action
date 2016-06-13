import decimal
import numpy as np

def k_means(k, dataset):
    print("Analyzing dataset...")
    count, dim = dataset.shape
    print("Record count: %d\nDim: %d" % (count, dim))
    print()
    centroids = _select_random_centroids(dataset, k)
    print("Select random centroids from the dataset\n%s" % centroids)
    clusters = _cluster(dataset, centroids)

    changed = True
    iteration = 0
    while changed:
        centroids = _get_centroids(clusters)
        iteration += 1
        print("Start #%d iteration by using centroids\n%s" % (iteration, centroids))
        new_clusters = _cluster(dataset, centroids)
        changed = not _clusters_equal(clusters, new_clusters)
        if changed:
            clusters = new_clusters
    return clusters


def _cluster(dataset, centroids):
    clusters = [];
    for i in range(len(centroids)):
        clusters.append([])
    for point in dataset:
        distance = _calculate_distance(centroids, point);
        cluster = clusters[distance.argmin()]
        cluster.append(point)
    for i, cluster in enumerate(clusters):
        clusters[i] = np.array(cluster)
    return clusters


def _calculate_distance(centroids, point):
    return np.sqrt(np.sum((centroids - point) ** 2, axis=-1))


def _get_centroids(clusters):
    centroids = []
    dim = 0
    for cluster in clusters:
        centroid = []
        if (dim == 0 and len(cluster) > 0):
            dim = cluster[0].shape[0]
        for d in range(dim):
            centroid.append(np.mean(cluster[:, d]))
        centroids.append(centroid)
    return np.array(centroids)


def _select_random_centroids(dataset, k):
    centroids = []
    count = len(dataset)
    for i in range(k):
        index = int(np.random.uniform(0, count))
        centroids.append(dataset[index])
    return np.array(centroids)

def _clusters_equal(a, b):
    for i, cluster in enumerate(a):
        equal = np.array_equal(cluster, b[i])
        if not equal:
            return False
    return True
