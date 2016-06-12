import numpy as np

def k_means(dataset, k = 3):
    print("Analyzing dataset...")
    count, dim = dataset.shape
    print("Record count: %d\nDim: %d" % (count, dim))
    print()
    centroids = _select_random_centroids(dataset, k)
    print("Select random centroids from the dataset\n%s" % centroids)
    clusters = _cluster(dataset, centroids)

    changed = True
    while changed:
        new_centroids = _get_centroids(clusters)
        print("Start new iteration by using centroids\n%s" % new_centroids)
        changed = not np.array_equal(new_centroids, centroids)
        if changed:
            centroids = new_centroids
            clusters = _cluster(dataset, centroids)

    return clusters


def _cluster(dataset, centroids):
    clusters = [];
    for i in range(len(centroids)):
        clusters.append([])
    for point in dataset:
        distance2 = np.sum((centroids - point) ** 2, -1)
        cluster = clusters[distance2.argmin()]
        cluster.append(list(point))
    for i in range(len(centroids)):
        clusters[i] = np.array(clusters[i])
    return np.array(clusters)


def _get_centroids(clusters):
    centroids = []
    for cluster in clusters:
        centroid = (np.mean(cluster[:, 0]), np.mean(cluster[:, 1]))
        centroids.append(centroid)
    return np.array(centroids)


def _select_random_centroids(dataset, k):
    centroids = []
    for i in range(k):
        index = int(np.random.uniform(0, len(dataset)))
        centroids.append(dataset[index])
    return np.array(centroids)
